"""
PH2 Overcooked Runner.

Per update step:
  1. collect_spec_rollout  — spec self-play (spec-match) + spec vs ind (ind-match, masked)
  2. compute spec returns + train_spec
  3. collect_ind_rollout   — ind vs ind (ind-match) + spec vs ind (spec-match for ind)
  4. compute ind  returns + train_ind
  5. log + eval + save
"""
import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from loguru import logger

from ph2.runner.shared.base_runner import PH2BaseRunner, _t2n
from ph2.algorithms.ph2.ph2_buffer import PH2Buffer


class PH2OvercookedRunner(PH2BaseRunner):

    def __init__(self, config: dict):
        super().__init__(config)

        args = self.all_args
        share_obs_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )
        obs_space  = self.envs.observation_space[0]
        act_space  = self.envs.action_space[0]
        history_len = getattr(args, "ph2_history_len", 5)

        # Two separate buffers: spec + ind
        self.spec_buffer = PH2Buffer(
            args, self.num_agents, obs_space, share_obs_space, act_space,
            history_len=history_len,
        )
        self.ind_buffer = PH2Buffer(
            args, self.num_agents, obs_space, share_obs_space, act_space,
            history_len=history_len,
        )

        # Per-env sliding history windows maintained during rollout
        # shape: (n_envs, n_agents, history_len, *obs_shape)
        from zsceval.utils.util import get_shape_from_obs_space
        obs_shape = get_shape_from_obs_space(obs_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        self._obs_shape = obs_shape
        self._history_len = history_len

        # SHAPED_INFOS import (for env logging)
        try:
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            self.shaped_info_keys = SHAPED_INFOS
        except ImportError:
            self.shaped_info_keys = []

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            s_time = time.time()

            if self.use_linear_lr_decay:
                for pol in [self.trainer.spec_policy, self.trainer.ind_policy]:
                    pol.lr_decay(episode, episodes)

            # ---- (A) spec rollout ----
            ind_match_prob = self.trainer.compute_ind_match_prob(episode)
            ind_match_envs = (np.random.rand(self.n_rollout_threads) < ind_match_prob)

            self._collect_spec_rollout(ind_match_envs)
            self._compute_returns(self.spec_buffer, self.trainer.spec_trainer)
            self.trainer.adapt_entropy_coef(total_num_steps)
            spec_info = self.trainer.train_spec(self.spec_buffer)
            self.spec_buffer.after_update()

            # ---- (B) ind rollout ----
            self._collect_ind_rollout(ind_match_envs)
            self._compute_returns(self.ind_buffer, self.trainer.ind_trainer)
            ind_info = self.trainer.train_ind(self.ind_buffer)
            self.ind_buffer.after_update()

            e_time = time.time()
            logger.trace(f"Episode {episode}: rollout+train {e_time - s_time:.2f}s")

            # ---- step count ----
            # Each episode = 2 rollouts (spec + ind), each episode_length * n_rollout_threads steps
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads * 2

            # ---- save ----
            if episode < 50:
                if episode % 2 == 0:
                    self.save(total_num_steps)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps)

            # ---- log ----
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                fps = int(total_num_steps / max(end - start, 1))
                logger.info(
                    f"[PH2] {self.all_args.layout_name} | ep {episode}/{episodes} | "
                    f"steps {total_num_steps}/{int(self.num_env_steps)*2} | FPS {fps}"
                )
                extra = {
                    "train/ind_match_prob": ind_match_prob,
                    "train/reward_shaping_factor": self.all_args.initial_reward_shaping_factor,
                }
                self.log_ph2(spec_info, ind_info, extra, total_num_steps)

                # episode rewards from buffers
                env_infos = defaultdict(list)
                for buf_name, buf in [("spec", self.spec_buffer), ("ind", self.ind_buffer)]:
                    env_infos[f"{buf_name}/avg_reward"].append(
                        float(np.mean(buf.rewards) * self.episode_length)
                    )
                self.log_env(env_infos, total_num_steps)

            # ---- eval ----
            if self.use_eval and (episode % self.eval_interval == 0 or episode == episodes - 1):
                self.eval(total_num_steps)

    # ------------------------------------------------------------------
    # warmup
    # ------------------------------------------------------------------
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)
        if not self.use_centralized_V:
            share_obs = obs
        for buf in [self.spec_buffer, self.ind_buffer]:
            buf.share_obs[0] = share_obs.copy()
            buf.obs[0]       = obs.copy()
            if buf.available_actions is not None:
                buf.available_actions[0] = available_actions.copy()

    # ------------------------------------------------------------------
    # spec rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_spec_rollout(self, ind_match_envs: np.ndarray):
        """
        Collect one episode for the spec policy.

        For spec-match envs (ind_match=False): spec acts as both agents, train_mask=1
        For ind-match envs  (ind_match=True):  spec acts only as agent-0,
                                               ind acts as agent-1, train_mask=0 for spec
        """
        N = self.n_rollout_threads
        M = self.num_agents      # should be 2 for Overcooked
        buf = self.spec_buffer
        spec_pol = self.trainer.spec_policy
        ind_pol  = self.trainer.ind_policy

        # per-env match state can reset when episode ends
        env_ind_match = ind_match_envs.copy()  # (N,)

        # history windows (not used in pred for spec, but kept for buffer compat)
        obs_shape = self._obs_shape
        L = self._history_len
        obs_hist = np.zeros((N, M, L, *obs_shape), dtype=np.float32)
        act_hist = np.zeros((N, M, L),              dtype=np.int64)

        self.trainer.prep_rollout()

        for step in range(self.episode_length):
            # ---- get actions for BOTH slots from spec (all envs) ----
            spec_pol.prep_rollout()
            values_spec, actions_spec, log_probs_spec, rnn_spec, rnn_critic_spec = \
                self._get_actions(spec_pol, buf, step)

            # ---- override agent-1 with ind policy in ind-match envs ----
            if ind_pol is not spec_pol and np.any(env_ind_match):
                ind_pol.prep_rollout()
                # get ind actions for all envs, then select
                values_ind, actions_ind, _, rnn_ind, rnn_critic_ind = \
                    self._get_actions(ind_pol, buf, step)

                # Where ind-match=True, slot-1 uses ind actions
                for n in range(N):
                    if env_ind_match[n]:
                        actions_spec[n, 1]   = actions_ind[n, 1]
                        log_probs_spec[n, 1] = 0.0        # not trained

            # ---- step env ----
            obs, share_obs, rewards, dones, infos, available_actions = \
                self.envs.step(actions_spec)
            obs = np.stack(obs)
            if not self.use_centralized_V:
                share_obs = obs

            # annealing
            total_steps = (buf.step) * N
            self.envs.anneal_reward_shaping_factor([total_steps] * N)

            # ---- build masks ----
            masks = np.ones((N, M, 1), dtype=np.float32)
            dones_flat = np.array(dones)
            if dones_flat.ndim == 1:
                for n in range(N):
                    if dones_flat[n]:
                        masks[n] = 0.0
            else:
                masks[dones_flat == True] = 0.0

            bad_masks = np.array([[[0.0] if info.get("bad_transition", False) else [1.0]] * M
                                   for info in infos])

            # ---- train_mask: 0 for spec in ind-match envs ----
            train_mask_np = np.ones((N, M, 1), dtype=np.float32)
            for n in range(N):
                if env_ind_match[n]:
                    train_mask_np[n] = 0.0

            # ---- partner_actions (partner of spec is ind or spec-copy) ----
            partner_acts = np.zeros((N, M, 1), dtype=np.int64)
            for n in range(N):
                # agent-0's partner is agent-1, agent-1's partner is agent-0
                partner_acts[n, 0, 0] = int(actions_spec[n, 1, 0])
                partner_acts[n, 1, 0] = int(actions_spec[n, 0, 0])

            # ---- update history ----
            cur_obs = buf.obs[step]   # (N, M, *obs_shape)
            for n in range(N):
                for m in range(M):
                    partner_m = 1 - m
                    obs_hist[n, m] = np.roll(obs_hist[n, m], -1, axis=0)
                    obs_hist[n, m, -1] = cur_obs[n, partner_m]
                    act_hist[n, m] = np.roll(act_hist[n, m], -1, axis=0)
                    act_hist[n, m, -1] = int(actions_spec[n, partner_m, 0])

            # reset history for done envs
            if dones_flat.ndim == 1:
                done_envs = np.where(dones_flat)[0]
            else:
                done_envs = np.where(dones_flat.any(axis=-1))[0]
            for n in done_envs:
                obs_hist[n] = 0.0
                act_hist[n] = 0
                # re-sample match type on episode reset
                env_ind_match[n] = (np.random.rand() < self.trainer.compute_ind_match_prob(0))

            # reset rnn states on done
            rnn_spec[dones_flat == True] = 0.0
            rnn_critic_spec[dones_flat == True] = 0.0

            buf.insert(
                share_obs, obs,
                rnn_spec, rnn_critic_spec,
                actions_spec, log_probs_spec,
                values_spec, rewards, masks,
                bad_masks=bad_masks,
                available_actions=available_actions,
                train_masks=train_mask_np,
                partner_actions=partner_acts,
                obs_history=obs_hist.copy(),
                act_history=act_hist.copy(),
            )

    # ------------------------------------------------------------------
    # ind rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_ind_rollout(self, ind_match_envs: np.ndarray):
        """
        Collect one episode for the ind policy.

        For ind-match envs (ind_match=True):   ind acts as both agents, train_mask=1
        For spec-match envs(ind_match=False):  spec acts agent-0, ind acts agent-1,
                                               train_mask=1 only for ind's slot (slot-1)
        """
        N = self.n_rollout_threads
        M = self.num_agents
        buf = self.ind_buffer
        spec_pol = self.trainer.spec_policy
        ind_pol  = self.trainer.ind_policy

        env_ind_match = ind_match_envs.copy()

        obs_shape = self._obs_shape
        L = self._history_len
        obs_hist = np.zeros((N, M, L, *obs_shape), dtype=np.float32)
        act_hist = np.zeros((N, M, L),              dtype=np.int64)

        self.trainer.prep_rollout()

        for step in range(self.episode_length):
            ind_pol.prep_rollout()
            values_ind, actions_ind, log_probs_ind, rnn_ind, rnn_critic_ind = \
                self._get_actions(ind_pol, buf, step)

            # ---- override agent-0 with spec in spec-match envs ----
            if np.any(~env_ind_match):
                spec_pol.prep_rollout()
                values_spec, actions_spec, _, rnn_spec, _ = \
                    self._get_actions(spec_pol, buf, step)

                for n in range(N):
                    if not env_ind_match[n]:
                        actions_ind[n, 0]   = actions_spec[n, 0]
                        log_probs_ind[n, 0] = 0.0   # spec acts, not trained

            obs, share_obs, rewards, dones, infos, available_actions = \
                self.envs.step(actions_ind)
            obs = np.stack(obs)
            if not self.use_centralized_V:
                share_obs = obs

            total_steps = buf.step * N
            self.envs.anneal_reward_shaping_factor([total_steps] * N)

            masks = np.ones((N, M, 1), dtype=np.float32)
            dones_flat = np.array(dones)
            if dones_flat.ndim == 1:
                for n in range(N):
                    if dones_flat[n]:
                        masks[n] = 0.0
            else:
                masks[dones_flat == True] = 0.0

            bad_masks = np.array([[[0.0] if info.get("bad_transition", False) else [1.0]] * M
                                   for info in infos])

            # ---- train_mask ----
            # ind-match: train on all (1)
            # spec-ind: train only on ind's slot (slot 1). For slot 0, spec acted.
            train_mask_np = np.ones((N, M, 1), dtype=np.float32)
            for n in range(N):
                if not env_ind_match[n]:
                    train_mask_np[n, 0] = 0.0  # slot-0 was spec, don't train ind on it

            # ---- partner_actions (for E3T: partner of ind agent) ----
            partner_acts = np.zeros((N, M, 1), dtype=np.int64)
            for n in range(N):
                partner_acts[n, 0, 0] = int(actions_ind[n, 1, 0])
                partner_acts[n, 1, 0] = int(actions_ind[n, 0, 0])

            # ---- history update ----
            cur_obs = buf.obs[step]
            for n in range(N):
                for m in range(M):
                    partner_m = 1 - m
                    obs_hist[n, m] = np.roll(obs_hist[n, m], -1, axis=0)
                    obs_hist[n, m, -1] = cur_obs[n, partner_m]
                    act_hist[n, m] = np.roll(act_hist[n, m], -1, axis=0)
                    act_hist[n, m, -1] = int(actions_ind[n, partner_m, 0])

            if dones_flat.ndim == 1:
                done_envs = np.where(dones_flat)[0]
            else:
                done_envs = np.where(dones_flat.any(axis=-1))[0]
            for n in done_envs:
                obs_hist[n] = 0.0
                act_hist[n] = 0
                env_ind_match[n] = (np.random.rand() < self.trainer.compute_ind_match_prob(0))

            rnn_ind[dones_flat == True] = 0.0
            rnn_critic_ind[dones_flat == True] = 0.0

            buf.insert(
                share_obs, obs,
                rnn_ind, rnn_critic_ind,
                actions_ind, log_probs_ind,
                values_ind, rewards, masks,
                bad_masks=bad_masks,
                available_actions=available_actions,
                train_masks=train_mask_np,
                partner_actions=partner_acts,
                obs_history=obs_hist.copy(),
                act_history=act_hist.copy(),
            )

    # ------------------------------------------------------------------
    # compute returns
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_returns(self, buffer, trainer):
        trainer.prep_rollout()
        next_values = trainer.policy.get_values(
            np.concatenate(buffer.share_obs[-1]),
            np.concatenate(buffer.rnn_states_critic[-1]),
            np.concatenate(buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        buffer.compute_returns(next_values, trainer.value_normalizer)

    # ------------------------------------------------------------------
    # helper: get actions from a policy using current buffer state
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_actions(self, policy, buffer, step):
        (
            values, actions, log_probs,
            rnn_states, rnn_states_critic,
        ) = policy.get_actions(
            np.concatenate(buffer.share_obs[step]),
            np.concatenate(buffer.obs[step]),
            np.concatenate(buffer.rnn_states[step]),
            np.concatenate(buffer.rnn_states_critic[step]),
            np.concatenate(buffer.masks[step]),
            np.concatenate(buffer.available_actions[step])
            if buffer.available_actions is not None else None,
        )
        N = self.n_rollout_threads
        values        = np.array(np.split(_t2n(values), N))
        actions       = np.array(np.split(_t2n(actions), N))
        log_probs     = np.array(np.split(_t2n(log_probs), N))
        rnn_states    = np.array(np.split(_t2n(rnn_states), N))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), N))
        return values, actions, log_probs, rnn_states, rnn_states_critic

    # ------------------------------------------------------------------
    # eval (ind policy by default)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval(self, total_num_steps: int):
        if self.eval_envs is None:
            return
        eval_env_infos = defaultdict(list)
        eval_obs, eval_share_obs, eval_avail = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)
        if not self.use_centralized_V:
            eval_share_obs = eval_obs

        n_eval = self.n_eval_rollout_threads
        M = self.num_agents
        rnn = np.zeros((n_eval, M, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_c = np.zeros_like(rnn)
        masks = np.ones((n_eval, M, 1), dtype=np.float32)

        self.trainer.ind_policy.prep_rollout()
        for _ in range(self.episode_length):
            _, actions, _, rnn, rnn_c = self.trainer.ind_policy.get_actions(
                eval_share_obs.reshape(n_eval * M, *eval_share_obs.shape[2:]),
                eval_obs.reshape(n_eval * M, *eval_obs.shape[2:]),
                rnn.reshape(n_eval * M, self.recurrent_N, self.hidden_size),
                rnn_c.reshape(n_eval * M, self.recurrent_N, self.hidden_size),
                masks.reshape(n_eval * M, 1),
                deterministic=True,
            )
            actions    = _t2n(actions).reshape(n_eval, M, -1)
            rnn        = _t2n(rnn).reshape(n_eval, M, self.recurrent_N, self.hidden_size)
            rnn_c      = _t2n(rnn_c).reshape(n_eval, M, self.recurrent_N, self.hidden_size)

            eval_obs, eval_share_obs, _, dones, infos, eval_avail = self.eval_envs.step(actions)
            eval_obs = np.stack(eval_obs)
            if not self.use_centralized_V:
                eval_share_obs = eval_obs

            dones_flat = np.array(dones)
            masks = np.ones((n_eval, M, 1), dtype=np.float32)
            if dones_flat.ndim == 1:
                for n in range(n_eval):
                    if dones_flat[n]:
                        masks[n] = 0.0
                        rnn[n] = 0.0
                        rnn_c[n] = 0.0
            else:
                masks[dones_flat == True] = 0.0
                rnn[dones_flat == True] = 0.0
                rnn_c[dones_flat == True] = 0.0

            for info in infos:
                if "episode" in info:
                    eval_env_infos["eval/sparse_r"].append(info["episode"]["ep_sparse_r"])
                    eval_env_infos["eval/shaped_r"].append(info["episode"]["ep_shaped_r"])

        if eval_env_infos:
            logger.info(f"eval sparse_r: {np.mean(eval_env_infos['eval/sparse_r']):.3f}")
            self.log_env(eval_env_infos, total_num_steps)
