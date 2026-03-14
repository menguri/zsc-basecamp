"""
PH2 Overcooked Runner.

Per update step:
  1. _collect_spec_rollout  — pure spec self-play (spec-spec), all envs, train_mask=1
  2. compute spec returns + train_spec
  3. _collect_ind_rollout   — ind-match: ind-ind; spec-match: spec-ind (shuffled slots)
  4. compute ind  returns + train_ind
  5. log + eval + save

Key design decisions:
  [1] spec rollout never involves ind policy — spec-spec only.
  [2] In ind rollout spec-match envs, spec slot is randomised per episode.
  [3] Blocked-state penalty uses L2 distance in CNN latent space:
        penalty_k = omega * exp(-sigma * L2(z_cur, z_tilde_k))  (summed over K)
  [4] Blocked states are sampled via V_gap-based softmax:
        V_gap(s) = E_b[V(o_b, normal) - V(o_b, blocked=s)]
        P(s) ∝ exp(-beta * V_gap(s))
      K_ep ~ Uniform(1, K_max); remaining slots marked invalid.
"""
import time
from collections import defaultdict, deque
from typing import Optional

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

        from zsceval.utils.util import get_shape_from_obs_space
        obs_shape = get_shape_from_obs_space(obs_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        self._obs_shape = obs_shape
        self._history_len = history_len

        pred_dim  = act_space.n
        hidden_sz = getattr(args, "hidden_size", 64)
        self._act_dim = int(act_space.n)

        # ---- blocked-state pool (spec only) ----
        self._use_blocked     = getattr(args, "ph2_spec_use_blocked", False)
        self._num_blocked     = max(1, int(getattr(args, "ph2_num_blocked_slots", 1)))   # K_max
        self._blocked_feat_dim = hidden_sz * self._num_blocked
        self._blocked_pool_sz = getattr(args, "ph2_blocked_pool_size", 200)
        self._blocked_omega   = getattr(args, "ph2_blocked_penalty_omega", 10.0)
        self._blocked_sigma   = getattr(args, "ph2_blocked_penalty_sigma", 2.0)
        self._vgap_beta       = getattr(args, "ph2_vgap_beta", 1.0)
        self._vgap_beta_sched_enabled = getattr(args, "ph2_vgap_beta_schedule_enabled", True)
        self._vgap_beta_start = getattr(args, "ph2_vgap_beta_start", 0.0)
        self._vgap_beta_end = getattr(args, "ph2_vgap_beta_end", 1.0)
        self._vgap_beta_horizon_env_steps = int(
            getattr(args, "ph2_vgap_beta_horizon_env_steps", -1)
        )
        self._vgap_beta_current = float(self._vgap_beta_start if self._vgap_beta_sched_enabled else self._vgap_beta)
        self._ph2_epsilon = float(np.clip(getattr(args, "ph2_epsilon", 0.2), 0.0, 1.0))

        self.spec_buffer = PH2Buffer(
            args,
            self.num_agents,
            obs_space,
            share_obs_space,
            act_space,
            history_len=history_len,
            pred_dim=pred_dim,
            blocked_feat_dim=self._blocked_feat_dim,
        )
        self.ind_buffer = PH2Buffer(
            args,
            self.num_agents,
            obs_space,
            share_obs_space,
            act_space,
            history_len=history_len,
            pred_dim=pred_dim,
            blocked_feat_dim=self._blocked_feat_dim,
        )

        N      = self.n_rollout_threads
        K_max  = self._num_blocked

        if self._use_blocked:
            # per-env FIFO pool of recent spec observations
            self._blocked_pool = [deque(maxlen=self._blocked_pool_sz) for _ in range(N)]
            # current episode blocked obs per env: (N, K_max, *obs_shape)
            self._blocked_obs   = np.zeros((N, K_max, *obs_shape), dtype=np.float32)
            # validity mask: which K slots are active (N, K_max)
            self._blocked_valid = np.zeros((N, K_max), dtype=bool)
            # cached latent features of blocked obs for penalty (N, K_max, hidden)
            self._blocked_feats = np.zeros((N, K_max, hidden_sz), dtype=np.float32)
        else:
            self._blocked_pool  = None
            self._blocked_obs   = None
            self._blocked_valid = None
            self._blocked_feats = None

        try:
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            self.shaped_info_keys = SHAPED_INFOS
        except ImportError:
            self.shaped_info_keys = []

    # ======================================================================
    # Helpers: prediction context
    # ======================================================================
    @torch.no_grad()
    def _compute_pred_context(
        self, obs_hist: np.ndarray, act_hist: np.ndarray, use_spec_pred: bool = False
    ) -> np.ndarray:
        """
        Run PartnerPredictionNet on history windows.

        obs_hist      : (N, M, L, *obs_shape)
        act_hist      : (N, M, L)
        use_spec_pred : if True and spec_pred exists (not shared), use spec's predictor
        Returns       : (N, M, pred_dim)
        """
        pred_net = (
            self.trainer.spec_pred
            if (use_spec_pred and not self.trainer.share_pred and self.trainer.spec_pred is not None)
            else self.trainer.ind_pred
        )
        if pred_net is None:
            return np.zeros(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
        N, M, L = act_hist.shape
        obs_shape = obs_hist.shape[3:]
        oh = torch.FloatTensor(obs_hist.reshape(N * M, L, *obs_shape)).to(self.device)
        ah = torch.LongTensor(act_hist.reshape(N * M, L)).to(self.device)
        pred_net.eval()
        logits = pred_net(oh, ah)                        # (N*M, act_dim)
        return _t2n(logits).reshape(N, M, -1).astype(np.float32)

    # ======================================================================
    # Helpers: blocked features (encode obs through spec actor CNN)
    # ======================================================================
    @torch.no_grad()
    def _encode_obs_features(self, obs_np: np.ndarray) -> np.ndarray:
        """
        Encode a batch of obs through spec actor's base CNN.
        obs_np  : (B, *obs_shape)
        Returns : (B, hidden_size)
        """
        t = torch.FloatTensor(obs_np).to(self.device)
        return _t2n(self.trainer.spec_policy.actor.base(t)).astype(np.float32)

    def _build_blocked_input_for_spec(self) -> tuple:
        """
        Returns:
          blk_feats_nm : (N, M, hidden*K_max) — K-slot concat, broadcast to agents
          blk_flat     : (N*M, hidden*K_max)
        Both are None if blocked states not enabled.
        """
        if not self._use_blocked:
            return None, None
        N   = self.n_rollout_threads
        M   = self.num_agents
        h   = self._blocked_feats.shape[-1]
        K   = self._num_blocked
        valid_f = self._blocked_valid.astype(np.float32)   # (N, K)
        slot_feats = self._blocked_feats * valid_f[:, :, np.newaxis]  # (N, K, h)
        concat_feat = slot_feats.reshape(N, K * h)  # (N, K*h)
        blk_feats_nm = np.broadcast_to(
            concat_feat[:, np.newaxis, :], (N, M, K * h)
        ).copy().astype(np.float32)
        blk_flat = blk_feats_nm.reshape(N * M, K * h)
        return blk_feats_nm, blk_flat

    def _update_vgap_beta(self, total_env_steps: int):
        if not self._vgap_beta_sched_enabled:
            self._vgap_beta_current = float(self._vgap_beta)
            return

        if self._vgap_beta_horizon_env_steps > 0:
            horizon = max(1, int(self._vgap_beta_horizon_env_steps))
        else:
            horizon = max(1, int(self.num_env_steps) * 2)

        progress = float(np.clip(total_env_steps / horizon, 0.0, 1.0))
        self._vgap_beta_current = (
            float(self._vgap_beta_start)
            + progress * float(self._vgap_beta_end - self._vgap_beta_start)
        )

    # ======================================================================
    # Helpers: V_gap-based blocked state sampling
    # ======================================================================
    @torch.no_grad()
    def _compute_vgap_for_candidates(
        self,
        obs_sample: np.ndarray,
        candidates: np.ndarray,
        pred_context_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute V_gap(s_tilde) = E_b[V_policy_vhead(o_b, normal) - V_policy_vhead(o_b, s_tilde)]

        Uses the spec actor's policy value head (v_out). If it is unavailable,
        returns all-zero gaps (uniform sampling).

        obs_sample : (B, *obs_shape)  — recent spec observations from pool
        candidates : (P, *obs_shape)  — candidate blocked states
        Returns    : (P,) float32
        """
        actor = self.trainer.spec_policy.actor
        if not getattr(actor, "_use_policy_vhead", False):
            return np.zeros(len(candidates), dtype=np.float32)

        actor.eval()
        B = obs_sample.shape[0]
        P = len(candidates)

        obs_t = torch.FloatTensor(obs_sample).to(self.device)  # (B, *obs_shape)
        pred_ctx = None
        pred_dim = int(getattr(actor, "_pred_dim", 0))
        if pred_dim > 0:
            if pred_context_sample is not None and pred_context_sample.shape[-1] == pred_dim:
                pred_ctx = torch.FloatTensor(pred_context_sample).to(self.device)
            else:
                pred_ctx = torch.zeros((B, pred_dim), device=self.device, dtype=torch.float32)
        blocked_ctx_zero = None
        blocked_dim = int(getattr(actor, "_blocked_feat_dim", 0))
        if blocked_dim > 0:
            blocked_ctx_zero = torch.zeros((B, blocked_dim), device=self.device, dtype=torch.float32)

        # V(o_b, normal): no blocked features
        feats_base = actor.base(obs_t)          # (B, hidden)
        feats_norm = actor._inject(
            feats_base.clone(),
            pred_context=pred_ctx,
            blocked_features=blocked_ctx_zero,
        )
        v_normal   = actor.v_out(feats_norm).squeeze(-1)  # (B,)

        v_gaps = np.empty(P, dtype=np.float32)
        for p in range(P):
            cand_t  = torch.FloatTensor(candidates[p:p+1]).to(self.device)  # (1, *obs_shape)
            blk_f   = actor.base(cand_t)                                     # (1, hidden)
            blk_f_b = blk_f.expand(B, -1)                                   # (B, hidden)
            blk_slots = torch.zeros(
                (B, self._blocked_feat_dim),
                device=self.device,
                dtype=torch.float32,
            )
            blk_slots[:, : blk_f_b.shape[-1]] = blk_f_b
            # recompute base features for obs (no grad, shared backbone)
            feats_b = actor.base(obs_t)
            feats_b = actor._inject(
                feats_b,
                pred_context=pred_ctx,
                blocked_features=blk_slots,
            )
            v_blk   = actor.v_out(feats_b).squeeze(-1)                       # (B,)
            v_gaps[p] = (v_normal - v_blk).mean().item()

        return v_gaps

    def _sample_blocked_states_for_env(
        self, env_idx: int, k_count: int
    ) -> list:
        """
        Sample k_count blocked states from env_idx's pool via V_gap softmax.
        Returns list of k_count obs arrays (may have repeats if pool too small).
        """
        pool = list(self._blocked_pool[env_idx])
        P    = len(pool)
        if P == 0:
            return []

        # Gather reference obs/pred_context for V_gap (up to 32 random from pool)
        ref_size = min(P, 32)
        ref_idxs = np.random.choice(P, ref_size, replace=False)
        obs_sample = np.stack([pool[i][0] for i in ref_idxs])
        pred_sample = np.stack([pool[i][1] for i in ref_idxs])

        candidates = np.stack([entry[0] for entry in pool])  # (P, *obs_shape)
        v_gaps = self._compute_vgap_for_candidates(
            obs_sample,
            candidates,
            pred_context_sample=pred_sample,
        )  # (P,)

        # Softmax with temperature beta
        logits = -self._vgap_beta_current * v_gaps
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()

        if P >= k_count:
            sampled = np.random.choice(P, k_count, replace=False, p=probs)
        else:
            sampled = np.random.choice(P, k_count, replace=True, p=probs)

        return [pool[i][0] for i in sampled]

    def _update_blocked_pool_and_sample(
        self,
        cur_obs: np.ndarray,
        done_envs: np.ndarray,
        pred_ctx_cur: Optional[np.ndarray] = None,
    ):
        """
        Add agent-0 obs to each env's pool.
        At episode end: pick K_ep ~ Uniform(1, K_max), sample K_ep blocked states
        without replacement via V_gap, fill remaining slots as invalid.

        cur_obs : (N, M, *obs_shape)
        """
        N     = self.n_rollout_threads
        K_max = self._num_blocked

        pred_dim = int(getattr(self.trainer.spec_policy.actor, "_pred_dim", 0))
        for n in range(N):
            if pred_ctx_cur is not None and pred_dim > 0:
                pred_n = pred_ctx_cur[n, 0].copy().astype(np.float32)
            else:
                pred_n = np.zeros((pred_dim,), dtype=np.float32)
            self._blocked_pool[n].append((cur_obs[n, 0].copy(), pred_n))

        for n in done_envs:
            pool = self._blocked_pool[n]
            if len(pool) == 0:
                continue
            # Pick number of blocked states uniformly in [1, K_max]
            k_ep  = np.random.randint(1, K_max + 1)
            sampled = self._sample_blocked_states_for_env(n, k_ep)

            # Reset validity
            self._blocked_obs[n]   = 0.0
            self._blocked_valid[n] = False
            self._blocked_feats[n] = 0.0

            for k, obs_k in enumerate(sampled):
                self._blocked_obs[n, k]   = obs_k
                self._blocked_valid[n, k] = True

            # Cache encoded features for valid slots
            if k_ep > 0:
                valid_obs  = np.stack([self._blocked_obs[n, k] for k in range(k_ep)])
                valid_feat = self._encode_obs_features(valid_obs)  # (k_ep, hidden)
                self._blocked_feats[n, :k_ep] = valid_feat

    def _compute_reward_penalty(self, cur_obs: np.ndarray):
        """
        Sum of exp-decay L2 penalties over all valid blocked slots.

          penalty_k = omega * exp(-sigma * ||z_cur - z_tilde_k||_2)   if valid_k
          total_pen = sum_k penalty_k

        cur_obs : (N, M, *obs_shape)
        Returns:
            penalty_broadcast : (N, M, 1) float32
            dist_broadcast    : (N, M, 1) float32
                Mean L2 distance to valid blocked slots per env.
        """
        N         = self.n_rollout_threads
        M         = self.num_agents
        obs_shape = self._obs_shape

        # Encode agent-0 obs as state proxy
        obs_flat = cur_obs[:, 0].reshape(N, *obs_shape)
        z_cur    = self._encode_obs_features(obs_flat)  # (N, hidden)

        penalty = np.zeros(N, dtype=np.float32)
        dist_sum = np.zeros(N, dtype=np.float32)
        dist_cnt = np.zeros(N, dtype=np.float32)
        for k in range(self._num_blocked):
            valid_k = self._blocked_valid[:, k]  # (N,)
            if not valid_k.any():
                continue
            z_tilde_k = self._blocked_feats[:, k]  # (N, hidden) — cached
            l2_dist = np.linalg.norm(z_cur - z_tilde_k, axis=-1)  # (N,)
            pen_k   = self._blocked_omega * np.exp(-self._blocked_sigma * l2_dist)
            valid_f = valid_k.astype(np.float32)
            penalty += pen_k * valid_f
            dist_sum += l2_dist * valid_f
            dist_cnt += valid_f

        mean_dist = np.divide(
            dist_sum,
            np.maximum(1.0, dist_cnt),
            out=np.zeros_like(dist_sum),
        ).astype(np.float32)

        # shape: (N, M, 1) — same penalty for all agents in the env
        penalty_broadcast = np.broadcast_to(
            penalty[:, np.newaxis, np.newaxis], (N, M, 1)
        ).copy().astype(np.float32)
        dist_broadcast = np.broadcast_to(
            mean_dist[:, np.newaxis, np.newaxis], (N, M, 1)
        ).copy().astype(np.float32)
        return penalty_broadcast, dist_broadcast

    # ======================================================================
    # Main run loop
    # ======================================================================
    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            s_time = time.time()
            self._update_vgap_beta(total_num_steps)

            if self.use_linear_lr_decay:
                for pol in [self.trainer.spec_policy, self.trainer.ind_policy]:
                    pol.lr_decay(episode, episodes)

            ind_match_prob = self.trainer.compute_ind_match_prob(episode)
            # ind_match_envs used only for ind rollout
            ind_match_envs = (np.random.rand(self.n_rollout_threads) < ind_match_prob)

            # ---- (A) spec rollout: pure spec-spec ----
            spec_rollout_info = self._collect_spec_rollout()
            _, blk_last_flat = self._build_blocked_input_for_spec()
            self._compute_returns(
                self.spec_buffer,
                self.trainer.spec_trainer,
                blocked_features=blk_last_flat,
            )
            self.trainer.adapt_entropy_coef(total_num_steps)
            spec_info = self.trainer.train_spec(self.spec_buffer)
            self.spec_buffer.after_update()

            # ---- (B) ind rollout: ind-ind or spec-ind ----
            ind_rollout_info = self._collect_ind_rollout(ind_match_envs, episode)
            self._compute_returns(self.ind_buffer, self.trainer.ind_trainer)
            ind_info = self.trainer.train_ind(self.ind_buffer)
            self.ind_buffer.after_update()

            e_time = time.time()
            logger.trace(f"Episode {episode}: rollout+train {e_time - s_time:.2f}s")

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
                    "train/vgap_beta": float(self._vgap_beta_current),
                    "train/ph2_epsilon": float(self._ph2_epsilon),
                    "train/phase1/reward": float(spec_rollout_info.get("phase1_reward", 0.0)),
                    "train/phase1/reward_no_penalty": float(
                        spec_rollout_info.get("phase1_reward_no_penalty", 0.0)
                    ),
                    "train/phase1/penalty": float(spec_rollout_info.get("phase1_penalty", 0.0)),
                    "train/phase1/penalty_mean": float(
                        spec_rollout_info.get("phase1_penalty_mean", 0.0)
                    ),
                    "train/phase1/distance_mean": float(
                        spec_rollout_info.get("phase1_distance_mean", 0.0)
                    ),
                    "train/phase2/reward": float(ind_rollout_info.get("phase2_reward", 0.0)),
                    "train/phase2/distance_mean": float(
                        ind_rollout_info.get("phase2_distance_mean", 0.0)
                    ),
                    "train/phase2/ind-ind/reward": float(
                        ind_rollout_info.get("phase2_ind_ind_reward", 0.0)
                    ),
                    "train/phase2/spec-ind/reward": float(
                        ind_rollout_info.get("phase2_spec_ind_reward", 0.0)
                    ),
                    "train/phase2/ind-ind_ratio": float(
                        ind_rollout_info.get("phase2_ind_ind_ratio", 0.0)
                    ),
                    "train/spec_random_action_rate": float(
                        spec_rollout_info.get("spec_random_action_rate", 0.0)
                    ),
                    "train/ind_random_action_rate": float(
                        ind_rollout_info.get("ind_random_action_rate", 0.0)
                    ),
                }
                self.log_ph2(spec_info, ind_info, extra, total_num_steps)

                env_infos = defaultdict(list)
                for buf_name, buf in [("spec", self.spec_buffer), ("ind", self.ind_buffer)]:
                    env_infos[f"{buf_name}/avg_reward"].append(
                        float(np.mean(buf.rewards) * self.episode_length)
                    )
                self.log_env(env_infos, total_num_steps)

            # ---- eval ----
            if self.use_eval and (episode % self.eval_interval == 0 or episode == episodes - 1):
                self.eval(total_num_steps)

    # ======================================================================
    # Warmup
    # ======================================================================
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

    # ======================================================================
    # [1] Spec rollout: pure spec-spec self-play
    # ======================================================================
    @torch.no_grad()
    def _collect_spec_rollout(self):
        """
        All N envs: spec acts as both agents. No ind policy involved.
        train_mask = 1 everywhere.
        Blocked-state penalty applied to rewards (L2 latent distance).
        pred_context from spec_pred (or shared ind_pred).
        """
        N = self.n_rollout_threads
        M = self.num_agents
        buf = self.spec_buffer
        spec_pol = self.trainer.spec_policy

        obs_shape = self._obs_shape
        L         = self._history_len
        obs_hist  = np.zeros((N, M, L, *obs_shape), dtype=np.float32)
        act_hist  = np.zeros((N, M, L),              dtype=np.int64)

        self.trainer.prep_rollout()
        random_applied = 0
        # PH2 epsilon parity: env-wise trigger, then pick one actor in that env.
        total_candidates = N * self.episode_length
        raw_reward_sum = 0.0
        penalized_reward_sum = 0.0
        penalty_sum = 0.0
        distance_sum = 0.0
        reward_count = 0
        distance_count = 0

        for step in range(self.episode_length):
            # pred context (spec uses spec_pred if available and not shared)
            pred_ctx  = self._compute_pred_context(obs_hist, act_hist, use_spec_pred=True)
            pred_flat = pred_ctx.reshape(N * M, -1)

            # blocked features for spec actor input
            blk_feats_nm, blk_flat = self._build_blocked_input_for_spec()

            spec_pol.prep_rollout()
            values, actions_sampled, log_probs, rnn_s, rnn_c = self._get_actions(
                spec_pol, buf, step,
                pred_context=pred_flat,
                blocked_features=blk_flat,
            )
            actions_exec = actions_sampled.copy()
            if self._ph2_epsilon > 0:
                eps_env_mask = (np.random.rand(N) < self._ph2_epsilon)
                eps_mask = np.zeros((N, M, 1), dtype=bool)
                if eps_env_mask.any():
                    rand_actor = np.random.randint(0, M, size=N)
                    env_ids = np.where(eps_env_mask)[0]
                    eps_mask[env_ids, rand_actor[env_ids], 0] = True
                rand_actions = np.random.randint(
                    0, self._act_dim, size=(N, M, 1)
                ).astype(actions_exec.dtype)
                actions_exec = np.where(eps_mask, rand_actions, actions_exec)
                random_applied += int(eps_mask.sum())

            obs, share_obs, rewards, dones, infos, available_actions = \
                self.envs.step(actions_exec)
            obs = np.stack(obs)
            if not self.use_centralized_V:
                share_obs = obs

            self.envs.anneal_reward_shaping_factor([buf.step * N] * N)
            raw_reward_sum += float(np.sum(rewards))
            reward_count += int(rewards.size)
            penalty = np.zeros_like(rewards, dtype=np.float32)
            distance = np.zeros_like(rewards, dtype=np.float32)

            # L2 blocked-state penalty
            if self._use_blocked:
                cur_obs_before = buf.obs[step]  # obs at this step (before env step)
                penalty, distance = self._compute_reward_penalty(cur_obs_before)  # (N, M, 1)
                rewards = rewards - penalty
            penalty_sum += float(np.sum(penalty))
            distance_sum += float(np.sum(distance))
            distance_count += int(distance.size)
            penalized_reward_sum += float(np.sum(rewards))

            masks = np.ones((N, M, 1), dtype=np.float32)
            dones_flat = np.array(dones)
            if dones_flat.ndim == 1:
                for n in range(N):
                    if dones_flat[n]:
                        masks[n] = 0.0
            else:
                masks[dones_flat == True] = 0.0

            bad_masks = np.array([
                [[[0.0] if info.get("bad_transition", False) else [1.0]] * M]
                for info in infos
            ]).reshape(N, M, 1)

            # train_mask = 1 for all (spec-spec)
            train_mask_np = np.ones((N, M, 1), dtype=np.float32)

            # partner_actions: what the partner (other agent) did
            partner_acts = np.zeros((N, M, 1), dtype=np.int64)
            for n in range(N):
                partner_acts[n, 0, 0] = int(actions_sampled[n, 1, 0])
                partner_acts[n, 1, 0] = int(actions_sampled[n, 0, 0])

            # update history using the obs BEFORE the step
            cur_obs = buf.obs[step]
            for n in range(N):
                for m in range(M):
                    pm = 1 - m
                    obs_hist[n, m] = np.roll(obs_hist[n, m], -1, axis=0)
                    obs_hist[n, m, -1] = cur_obs[n, pm]
                    act_hist[n, m] = np.roll(act_hist[n, m], -1, axis=0)
                    act_hist[n, m, -1] = int(actions_sampled[n, pm, 0])

            if dones_flat.ndim == 1:
                done_envs = np.where(dones_flat)[0]
            else:
                done_envs = np.where(dones_flat.any(axis=-1))[0]

            if self._use_blocked:
                self._update_blocked_pool_and_sample(cur_obs, done_envs, pred_ctx_cur=pred_ctx)

            for n in done_envs:
                obs_hist[n] = 0.0
                act_hist[n] = 0

            rnn_s[dones_flat == True] = 0.0
            rnn_c[dones_flat == True] = 0.0

            buf.insert(
                share_obs, obs, rnn_s, rnn_c,
                actions_sampled, log_probs, values, rewards, masks,
                bad_masks=bad_masks,
                available_actions=available_actions,
                train_masks=train_mask_np,
                partner_actions=partner_acts,
                obs_history=obs_hist.copy(),
                act_history=act_hist.copy(),
                partner_pred_context=pred_ctx,
                blocked_features=blk_feats_nm,
            )

        reward_denom = float(max(1, reward_count))
        return {
            "spec_random_action_rate": float(random_applied) / float(max(1, total_candidates)),
            "phase1_reward": float(penalized_reward_sum / reward_denom) * self.episode_length,
            "phase1_reward_no_penalty": float(raw_reward_sum / reward_denom) * self.episode_length,
            "phase1_penalty": float(penalty_sum / reward_denom) * self.episode_length,
            "phase1_penalty_mean": float(penalty_sum / reward_denom),
            "phase1_distance_mean": float(distance_sum / float(max(1, distance_count))),
        }

    # ======================================================================
    # [1][2] Ind rollout: ind-ind or spec-ind (shuffled spec slot)
    # ======================================================================
    @torch.no_grad()
    def _collect_ind_rollout(self, ind_match_envs: np.ndarray, episode_idx: int):
        """
        ind_match_envs (N,) bool:
          True  → ind-ind self-play, both slots trained (train_mask=1)
          False → spec-ind: spec randomly assigned to one slot (spec_slot[n] ∈ {0,1}),
                  ind acts as the other slot; only ind slot trained (train_mask[spec_slot]=0)

        No blocked-state penalty applied to rewards (ind sees clean rewards).
        """
        N = self.n_rollout_threads
        M = self.num_agents
        buf = self.ind_buffer
        spec_pol = self.trainer.spec_policy
        ind_pol  = self.trainer.ind_policy

        env_ind_match = ind_match_envs.copy()

        # [2] per-env slot assignment for spec in spec-match envs (0 or 1)
        spec_slot_assign = np.random.randint(0, 2, size=N)  # (N,)

        obs_shape = self._obs_shape
        L         = self._history_len
        obs_hist  = np.zeros((N, M, L, *obs_shape), dtype=np.float32)
        act_hist  = np.zeros((N, M, L),              dtype=np.int64)
        spec_rnn = np.zeros(
            (N, M, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        spec_rnn_critic = np.zeros_like(spec_rnn)

        self.trainer.prep_rollout()
        random_applied = 0
        total_candidates = 0
        total_reward_sum = 0.0
        total_reward_count = 0
        ind_ind_reward_sum = 0.0
        ind_ind_reward_count = 0
        spec_ind_reward_sum = 0.0
        spec_ind_reward_count = 0
        phase2_distance_sum = 0.0
        phase2_distance_count = 0

        for step in range(self.episode_length):
            step_ind_match = env_ind_match.copy()
            # pred context (ind uses ind_pred)
            pred_ctx  = self._compute_pred_context(obs_hist, act_hist, use_spec_pred=False)
            pred_flat = pred_ctx.reshape(N * M, -1)

            # ---- ind actions for all envs ----
            ind_pol.prep_rollout()
            val_ind, act_ind, lp_ind, rnn_ind, rnn_c_ind = self._get_actions(
                ind_pol, buf, step, pred_context=pred_flat, blocked_features=None,
            )

            # ---- spec actions for spec-match envs ----
            # spec_pred context (separate or shared)
            spec_pred_ctx = self._compute_pred_context(obs_hist, act_hist, use_spec_pred=True)
            spec_pred_flat = spec_pred_ctx.reshape(N * M, -1)

            blk_feats_nm_spec, blk_flat_spec = self._build_blocked_input_for_spec()

            spec_pol.prep_rollout()
            _, act_spec, _, rnn_spec_all, rnn_c_spec_all = self._get_actions_from_arrays(
                spec_pol,
                share_obs=buf.share_obs[step],
                obs=buf.obs[step],
                rnn_states=spec_rnn,
                rnn_states_critic=spec_rnn_critic,
                masks=buf.masks[step],
                available_actions=buf.available_actions[step] if buf.available_actions is not None else None,
                pred_context=spec_pred_flat,
                blocked_features=blk_flat_spec,
            )

            # ---- combine: build final actions and train_masks ----
            # Start from ind actions; override spec's slot in spec-match envs
            actions_combined_sampled = act_ind.copy()
            train_mask_np    = np.ones((N, M, 1), dtype=np.float32)

            for n in range(N):
                if not env_ind_match[n]:
                    s = spec_slot_assign[n]      # spec's assigned slot
                    actions_combined_sampled[n, s] = act_spec[n, s]
                    train_mask_np[n, s, 0]   = 0.0  # don't train ind on spec's slot
                    # keep spec recurrent memory only for spec slots
                    spec_rnn[n, s] = rnn_spec_all[n, s]
                    spec_rnn_critic[n, s] = rnn_c_spec_all[n, s]

            actions_exec = actions_combined_sampled.copy()
            # PH2 epsilon parity:
            # - ind-ind: env-wise trigger, then one random actor in that env.
            # - spec-ind: env-wise trigger, only spec slot is eligible.
            total_candidates += int(step_ind_match.sum()) + int((~step_ind_match).sum())
            if self._ph2_epsilon > 0:
                eps_env_mask = (np.random.rand(N) < self._ph2_epsilon)
                eps_mask = np.zeros((N, M, 1), dtype=bool)
                if eps_env_mask.any():
                    # ind-ind: pick one random actor
                    ind_env_ids = np.where(step_ind_match & eps_env_mask)[0]
                    if ind_env_ids.size > 0:
                        rand_actor = np.random.randint(0, M, size=ind_env_ids.size)
                        eps_mask[ind_env_ids, rand_actor, 0] = True
                    # spec-ind: only specialist slot is allowed
                    spec_env_ids = np.where((~step_ind_match) & eps_env_mask)[0]
                    if spec_env_ids.size > 0:
                        spec_slots = spec_slot_assign[spec_env_ids].astype(np.int64)
                        eps_mask[spec_env_ids, spec_slots, 0] = True

                rand_actions = np.random.randint(
                    0, self._act_dim, size=(N, M, 1)
                ).astype(actions_exec.dtype)
                actions_exec = np.where(eps_mask, rand_actions, actions_exec)
                random_applied += int(eps_mask.sum())

            obs, share_obs, rewards, dones, infos, available_actions = \
                self.envs.step(actions_exec)
            obs = np.stack(obs)
            if not self.use_centralized_V:
                share_obs = obs

            self.envs.anneal_reward_shaping_factor([buf.step * N] * N)
            # ind always uses clean rewards (no blocked penalty)
            total_reward_sum += float(np.sum(rewards))
            total_reward_count += int(rewards.size)
            if self._use_blocked:
                cur_obs_before = buf.obs[step]  # obs at this step (before env step)
                _, phase2_distance = self._compute_reward_penalty(cur_obs_before)
                phase2_distance_sum += float(np.sum(phase2_distance))
                phase2_distance_count += int(phase2_distance.size)
            if step_ind_match.any():
                ind_ind_reward_sum += float(np.sum(rewards[step_ind_match]))
                ind_ind_reward_count += int(rewards[step_ind_match].size)
            if (~step_ind_match).any():
                spec_ind_reward_sum += float(np.sum(rewards[~step_ind_match]))
                spec_ind_reward_count += int(rewards[~step_ind_match].size)

            masks = np.ones((N, M, 1), dtype=np.float32)
            dones_flat = np.array(dones)
            if dones_flat.ndim == 1:
                for n in range(N):
                    if dones_flat[n]:
                        masks[n] = 0.0
            else:
                masks[dones_flat == True] = 0.0

            bad_masks = np.array([
                [[[0.0] if info.get("bad_transition", False) else [1.0]] * M]
                for info in infos
            ]).reshape(N, M, 1)

            partner_acts = np.zeros((N, M, 1), dtype=np.int64)
            for n in range(N):
                partner_acts[n, 0, 0] = int(actions_combined_sampled[n, 1, 0])
                partner_acts[n, 1, 0] = int(actions_combined_sampled[n, 0, 0])

            cur_obs = buf.obs[step]
            for n in range(N):
                for m in range(M):
                    pm = 1 - m
                    obs_hist[n, m] = np.roll(obs_hist[n, m], -1, axis=0)
                    obs_hist[n, m, -1] = cur_obs[n, pm]
                    act_hist[n, m] = np.roll(act_hist[n, m], -1, axis=0)
                    act_hist[n, m, -1] = int(actions_combined_sampled[n, pm, 0])

            if dones_flat.ndim == 1:
                done_envs = np.where(dones_flat)[0]
            else:
                done_envs = np.where(dones_flat.any(axis=-1))[0]

            for n in done_envs:
                obs_hist[n] = 0.0
                act_hist[n] = 0
                # [2] re-randomise spec_slot and ind_match at episode boundary
                env_ind_match[n]    = (
                    np.random.rand() < self.trainer.compute_ind_match_prob(episode_idx)
                )
                spec_slot_assign[n] = np.random.randint(0, 2)
                spec_rnn[n] = 0.0
                spec_rnn_critic[n] = 0.0

            rnn_ind[dones_flat == True] = 0.0
            rnn_c_ind[dones_flat == True] = 0.0
            spec_rnn[dones_flat == True] = 0.0
            spec_rnn_critic[dones_flat == True] = 0.0

            buf.insert(
                share_obs, obs, rnn_ind, rnn_c_ind,
                actions_combined_sampled, lp_ind, val_ind, rewards, masks,
                bad_masks=bad_masks,
                available_actions=available_actions,
                train_masks=train_mask_np,
                partner_actions=partner_acts,
                obs_history=obs_hist.copy(),
                act_history=act_hist.copy(),
                partner_pred_context=pred_ctx,
                blocked_features=None,
            )

        phase2_denom = float(max(1, total_reward_count))
        ind_ind_ratio = float(ind_ind_reward_count) / float(max(1, total_reward_count))
        return {
            "ind_random_action_rate": float(random_applied) / float(max(1, total_candidates)),
            "phase2_reward": float(total_reward_sum / phase2_denom) * self.episode_length,
            "phase2_ind_ind_reward": (
                float(ind_ind_reward_sum / float(max(1, ind_ind_reward_count))) * self.episode_length
                if ind_ind_reward_count > 0
                else 0.0
            ),
            "phase2_spec_ind_reward": (
                float(spec_ind_reward_sum / float(max(1, spec_ind_reward_count))) * self.episode_length
                if spec_ind_reward_count > 0
                else 0.0
            ),
            "phase2_ind_ind_ratio": ind_ind_ratio,
            "phase2_distance_mean": (
                float(phase2_distance_sum / float(max(1, phase2_distance_count)))
                if self._use_blocked
                else 0.0
            ),
        }

    # ======================================================================
    # compute returns
    # ======================================================================
    @torch.no_grad()
    def _compute_returns(self, buffer, trainer, blocked_features=None):
        trainer.prep_rollout()
        next_values = trainer.policy.get_values(
            np.concatenate(buffer.share_obs[-1]),
            np.concatenate(buffer.rnn_states_critic[-1]),
            np.concatenate(buffer.masks[-1]),
            blocked_features=blocked_features,
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        buffer.compute_returns(next_values, trainer.value_normalizer)

    # ======================================================================
    # helper: get_actions from a policy via buffer state
    # ======================================================================
    @torch.no_grad()
    def _get_actions(self, policy, buffer, step,
                     pred_context=None, blocked_features=None):
        return self._get_actions_from_arrays(
            policy,
            share_obs=buffer.share_obs[step],
            obs=buffer.obs[step],
            rnn_states=buffer.rnn_states[step],
            rnn_states_critic=buffer.rnn_states_critic[step],
            masks=buffer.masks[step],
            available_actions=(
                buffer.available_actions[step]
                if buffer.available_actions is not None
                else None
            ),
            pred_context=pred_context,
            blocked_features=blocked_features,
        )

    @torch.no_grad()
    def _get_actions_from_arrays(
        self,
        policy,
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        masks,
        available_actions=None,
        pred_context=None,
        blocked_features=None,
    ):
        values, actions, log_probs, rnn_states, rnn_states_critic = policy.get_actions(
            np.concatenate(share_obs),
            np.concatenate(obs),
            np.concatenate(rnn_states),
            np.concatenate(rnn_states_critic),
            np.concatenate(masks),
            np.concatenate(available_actions)
            if available_actions is not None else None,
            pred_context=pred_context,
            blocked_features=blocked_features,
        )
        N = self.n_rollout_threads
        values            = np.array(np.split(_t2n(values), N))
        actions           = np.array(np.split(_t2n(actions), N))
        log_probs         = np.array(np.split(_t2n(log_probs), N))
        rnn_states        = np.array(np.split(_t2n(rnn_states), N))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), N))
        return values, actions, log_probs, rnn_states, rnn_states_critic

    # ======================================================================
    # eval (ind policy by default)
    # ======================================================================
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
        M      = self.num_agents
        rnn    = np.zeros((n_eval, M, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_c  = np.zeros_like(rnn)
        masks  = np.ones((n_eval, M, 1), dtype=np.float32)

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
            actions = _t2n(actions).reshape(n_eval, M, -1)
            rnn     = _t2n(rnn).reshape(n_eval, M, self.recurrent_N, self.hidden_size)
            rnn_c   = _t2n(rnn_c).reshape(n_eval, M, self.recurrent_N, self.hidden_size)

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
                        rnn[n]   = 0.0
                        rnn_c[n] = 0.0
            else:
                masks[dones_flat == True] = 0.0
                rnn[dones_flat == True]   = 0.0
                rnn_c[dones_flat == True] = 0.0

            for info in infos:
                if "episode" in info:
                    eval_env_infos["eval/sparse_r"].append(info["episode"]["ep_sparse_r"])
                    eval_env_infos["eval/shaped_r"].append(info["episode"]["ep_shaped_r"])

        if eval_env_infos:
            logger.info(f"eval sparse_r: {np.mean(eval_env_infos['eval/sparse_r']):.3f}")
            self.log_env(eval_env_infos, total_num_steps)
