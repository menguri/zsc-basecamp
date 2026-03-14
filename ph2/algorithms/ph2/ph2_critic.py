"""
PH2Critic — R_Critic with optional blocked-feature injection after RNN trunk.

blocked_features: (B, blocked_feat_dim)
"""
import torch
import torch.nn as nn

from zsceval.algorithms.utils.util import check, init
from ph2.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic


class PH2Critic(R_Critic):
    def __init__(
        self,
        args,
        share_obs_space,
        device,
        use_blocked: bool = False,
        blocked_feat_dim: int = 0,
    ):
        super().__init__(args, share_obs_space, device)

        self._use_blocked = use_blocked
        self._blocked_feat_dim = blocked_feat_dim if use_blocked else 0

        self._trunk_dim = getattr(self.v_out, "in_features", self.hidden_size)

        if self._blocked_feat_dim > 0:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))

            self._ph2_proj = init_(
                nn.Linear(self._trunk_dim + self._blocked_feat_dim, self._trunk_dim)
            )
        else:
            self._ph2_proj = None

        self.to(device)

    def _inject(self, features, blocked_features=None):
        if (
            blocked_features is not None
            and self._use_blocked
            and self._blocked_feat_dim > 0
            and self._ph2_proj is not None
        ):
            blk = check(blocked_features).to(**self.tpdv)
            features = torch.relu(self._ph2_proj(torch.cat([features, blk], dim=-1)))
        return features

    def forward(self, share_obs, rnn_states, masks, task_id=None, blocked_features=None):
        if self._mixed_obs:
            for key in share_obs.keys():
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_share_obs = self.mlp(share_obs)
            critic_features = torch.cat([critic_features, mlp_share_obs], dim=1)

        if self._layer_after_N > 0:
            critic_features = self.mlp_after(critic_features)

        critic_features = self._inject(critic_features, blocked_features=blocked_features)
        values = self.v_out(critic_features)

        if self._num_v_out > 1 and task_id is not None:
            values = torch.gather(values, -1, task_id.long())

        return values, rnn_states
