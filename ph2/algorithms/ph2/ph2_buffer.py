"""
PH2Buffer — extends ZSC-EVAL's SharedReplayBuffer with:
  - train_masks   : which samples the learner should actually update on
  - partner_actions: actual partner action (for E3T CE loss)
  - obs_history   : ego obs history window (for E3T encoder)
  - act_history   : ego action history window (for E3T encoder)

The generators also yield these extra fields as trailing elements so
PH2Trainer can access them without touching the base PPO update path.
"""
import numpy as np
import torch
from zsceval.utils.shared_buffer import SharedReplayBuffer
from zsceval.utils.util import get_shape_from_obs_space


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


class PH2Buffer(SharedReplayBuffer):
    """
    Extends SharedReplayBuffer with PH2-specific fields.

    Extra fields (all stored for episode_length steps, no +1):
      train_masks          : (T, n_envs, n_agents, 1)              float32
      partner_actions      : (T, n_envs, n_agents, 1)              int64
      obs_history          : (T, n_envs, n_agents, L, *obs_shape)  float32
      act_history          : (T, n_envs, n_agents, L)              int64
      partner_pred_context : (T, n_envs, n_agents, pred_dim)       float32
      blocked_features     : (T, n_envs, n_agents, blocked_feat_dim) float32 (spec only, K-slot concat)
    """

    def __init__(
        self,
        args,
        num_agents,
        obs_space,
        share_obs_space,
        act_space,
        history_len: int = 5,
        pred_dim: int = 0,
        blocked_feat_dim: int = 64,
        n_rollout_threads=None,
    ):
        super().__init__(args, num_agents, obs_space, share_obs_space, act_space,
                         n_rollout_threads=n_rollout_threads)
        self.history_len = history_len
        T = self.episode_length
        N = self.n_rollout_threads
        M = num_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        self.train_masks = np.ones((T, N, M, 1), dtype=np.float32)
        self.partner_actions = np.zeros((T, N, M, 1), dtype=np.int64)
        self.obs_history = np.zeros((T, N, M, history_len, *obs_shape), dtype=np.float32)
        self.act_history = np.zeros((T, N, M, history_len), dtype=np.int64)
        self.partner_pred_context = np.zeros((T, N, M, max(pred_dim, 1)), dtype=np.float32)
        self.blocked_features = np.zeros((T, N, M, blocked_feat_dim), dtype=np.float32)
        self._pred_dim = pred_dim
        self._blocked_feat_dim = blocked_feat_dim

    # ------------------------------------------------------------------
    # insert override
    # ------------------------------------------------------------------
    def insert(
        self,
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
        # PH2-specific
        train_masks=None,
        partner_actions=None,
        obs_history=None,
        act_history=None,
        partner_pred_context=None,
        blocked_features=None,
    ):
        super().insert(
            share_obs, obs, rnn_states, rnn_states_critic,
            actions, action_log_probs, value_preds, rewards, masks,
            bad_masks=bad_masks, active_masks=active_masks,
            available_actions=available_actions,
        )
        step = (self.step - 1) % self.episode_length  # step was already advanced by super()
        if train_masks is not None:
            self.train_masks[step] = train_masks.copy()
        if partner_actions is not None:
            self.partner_actions[step] = partner_actions.copy()
        if obs_history is not None:
            self.obs_history[step] = obs_history.copy()
        if act_history is not None:
            self.act_history[step] = act_history.copy()
        if partner_pred_context is not None:
            self.partner_pred_context[step] = partner_pred_context.copy()
        if blocked_features is not None:
            self.blocked_features[step] = blocked_features.copy()

    # ------------------------------------------------------------------
    # after_update: reset PH2 fields
    # ------------------------------------------------------------------
    def after_update(self):
        super().after_update()
        self.train_masks[:] = 1.0
        self.partner_actions[:] = 0
        self.partner_pred_context[:] = 0.0
        self.blocked_features[:] = 0.0

    # ------------------------------------------------------------------
    # recurrent_generator override: appends PH2 fields at the end
    # ------------------------------------------------------------------
    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yields all base fields PLUS (train_masks_batch, partner_actions_batch,
        obs_history_batch, act_history_batch) as the last 4 elements.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size]
                   for i in range(num_mini_batch)]

        # ---- reshape all arrays to (N*M*T, ...) ----
        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        # PH2 fields: (T, N, M, ...) → (N*M*T, ...)
        train_masks = self.train_masks.transpose(1, 2, 0, 3).reshape(-1, 1)
        partner_actions = self.partner_actions.transpose(1, 2, 0, 3).reshape(-1, 1)

        # obs_history: (T, N, M, L, H, W, C) → (N*M*T, L, H, W, C)
        oh = self.obs_history
        oh_shape = oh.shape[3:]  # (L, *obs_shape)
        obs_history = oh.transpose(1, 2, 0, *range(3, oh.ndim)).reshape(-1, *oh_shape)

        # act_history: (T, N, M, L) → (N*M*T, L)
        act_history = self.act_history.transpose(1, 2, 0, 3).reshape(-1, self.history_len)

        # partner_pred_context: (T, N, M, pred_dim) → (N*M*T, pred_dim)
        pred_dim = self.partner_pred_context.shape[-1]
        partner_pred_context = self.partner_pred_context.transpose(1, 2, 0, 3).reshape(-1, pred_dim)

        # blocked_features: (T, N, M, hidden) → (N*M*T, hidden)
        h_dim = self.blocked_features.shape[-1]
        blocked_features = self.blocked_features.transpose(1, 2, 0, 3).reshape(-1, h_dim)

        for indices in sampler:
            share_obs_batch, obs_batch = [], []
            rnn_states_batch, rnn_states_critic_batch = [], []
            actions_batch, available_actions_batch = [], []
            value_preds_batch, return_batch, masks_batch = [], [], []
            active_masks_batch, old_action_log_probs_batch, adv_targ = [], [], []
            train_masks_batch, partner_actions_batch = [], []
            obs_history_batch, act_history_batch = [], []
            partner_pred_context_batch, blocked_features_batch = [], []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                # PH2
                train_masks_batch.append(train_masks[ind:ind + data_chunk_length])
                partner_actions_batch.append(partner_actions[ind:ind + data_chunk_length])
                obs_history_batch.append(obs_history[ind:ind + data_chunk_length])
                act_history_batch.append(act_history[ind:ind + data_chunk_length])
                partner_pred_context_batch.append(partner_pred_context[ind:ind + data_chunk_length])
                blocked_features_batch.append(blocked_features[ind:ind + data_chunk_length])

            L, N = data_chunk_length, mini_batch_size

            share_obs_batch = _flatten(L, N, np.stack(share_obs_batch, axis=1))
            obs_batch = _flatten(L, N, np.stack(obs_batch, axis=1))
            actions_batch = _flatten(L, N, np.stack(actions_batch, axis=1))
            available_actions_batch = (
                _flatten(L, N, np.stack(available_actions_batch, axis=1))
                if self.available_actions is not None else None
            )
            value_preds_batch = _flatten(L, N, np.stack(value_preds_batch, axis=1))
            return_batch = _flatten(L, N, np.stack(return_batch, axis=1))
            masks_batch = _flatten(L, N, np.stack(masks_batch, axis=1))
            active_masks_batch = _flatten(L, N, np.stack(active_masks_batch, axis=1))
            old_action_log_probs_batch = _flatten(L, N, np.stack(old_action_log_probs_batch, axis=1))
            adv_targ = _flatten(L, N, np.stack(adv_targ, axis=1))
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            # PH2
            train_masks_batch = _flatten(L, N, np.stack(train_masks_batch, axis=1))
            partner_actions_batch = _flatten(L, N, np.stack(partner_actions_batch, axis=1))
            obs_history_batch = _flatten(L, N, np.stack(obs_history_batch, axis=1))
            act_history_batch = _flatten(L, N, np.stack(act_history_batch, axis=1))
            partner_pred_context_batch = _flatten(L, N, np.stack(partner_pred_context_batch, axis=1))
            blocked_features_batch = _flatten(L, N, np.stack(blocked_features_batch, axis=1))

            yield (
                share_obs_batch, obs_batch,
                rnn_states_batch, rnn_states_critic_batch,
                actions_batch, value_preds_batch, return_batch, masks_batch,
                active_masks_batch, old_action_log_probs_batch, adv_targ,
                available_actions_batch, None,  # other_policy_id placeholder
                # PH2 extras:
                train_masks_batch, partner_actions_batch,
                obs_history_batch, act_history_batch,
                partner_pred_context_batch, blocked_features_batch,
            )
