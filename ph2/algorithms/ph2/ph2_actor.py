"""
PH2Actor — R_Actor with optional context injection after the RNN.

Two extra inputs may be provided (both are optional):
  pred_context    : (B, pred_dim)   — L2-normalised logits from PartnerPredictionNet
  blocked_features: (B, hidden_size * K_max) — slot-wise concatenated blocked latents
                    (spec policy only; ind leaves this as None)

After the RNN produces actor_features (B, hidden_size), the extras are concatenated
and projected back to hidden_size before the action head.
"""
import torch
import torch.nn as nn

from zsceval.algorithms.utils.util import check, init
from ph2.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor


class PH2Actor(R_Actor):
    """
    Extends R_Actor with optional (pred_context, blocked_features) injection.

    Args:
        pred_dim        : size of partner-prediction context vector (0 = disabled)
        use_blocked     : whether this actor accepts blocked_features
        blocked_feat_dim: dimensionality of blocked_features (typically hidden_size)
    """

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device,
        pred_dim: int = 0,
        use_blocked: bool = False,
        blocked_feat_dim: int = 0,
    ):
        super().__init__(args, obs_space, action_space, device)

        self._pred_dim = pred_dim
        self._use_blocked = use_blocked
        self._blocked_feat_dim = blocked_feat_dim if use_blocked else 0

        extra = pred_dim + self._blocked_feat_dim
        if extra > 0:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))

            self._ph2_proj = init_(nn.Linear(self.hidden_size + extra, self.hidden_size))
        else:
            self._ph2_proj = None

        self.to(device)

    # ------------------------------------------------------------------
    def _inject(self, features, pred_context=None, blocked_features=None):
        """Concatenate extras and project back to hidden_size (in-place-like)."""
        parts = [features]
        if pred_context is not None and self._pred_dim > 0:
            parts.append(check(pred_context).to(**self.tpdv))
        if blocked_features is not None and self._use_blocked:
            parts.append(check(blocked_features).to(**self.tpdv))
        if len(parts) > 1 and self._ph2_proj is not None:
            features = torch.relu(self._ph2_proj(torch.cat(parts, dim=-1)))
        return features

    # ------------------------------------------------------------------
    # forward (rollout — returns new rnn_states)
    # ------------------------------------------------------------------
    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
        pred_context=None,
        blocked_features=None,
    ):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        # --- PH2 context injection ---
        actor_features = self._inject(actor_features, pred_context, blocked_features)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    # ------------------------------------------------------------------
    # evaluate_actions (training — no rnn_states output)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        pred_context=None,
        blocked_features=None,
    ):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        # --- PH2 context injection ---
        actor_features = self._inject(actor_features, pred_context, blocked_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )
        values = self.v_out(actor_features) if self._use_policy_vhead else None
        return action_log_probs, dist_entropy, values
