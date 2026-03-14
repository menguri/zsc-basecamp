"""
PH2Trainer — manages spec + ind dual policies and their training.

Algorithm:
  Each update step:
    1. spec_rollout collected externally (runner calls collect_spec_step)
    2. train_spec(spec_buffer) → PPO update on spec policy
    3. ind_rollout  collected externally
    4. train_ind(ind_buffer)  → PPO update on ind policy + E3T pred loss

Match schedule:
  ind_match_prob = fixed (ph2_fixed_ind_prob ≥ 0) or staged
  - spec_match env  → spec self-play, spec trains (train_mask=1)
  - ind_match  env  → ind  self-play, spec doesn't train (train_mask=0 for spec)
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from ph2.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from ph2.algorithms.r_mappo.r_mappo import R_MAPPO
from ph2.algorithms.ph2.e3t import PartnerPredictionNet
from zsceval.algorithms.utils.util import check
from zsceval.utils.util import get_gard_norm
from zsceval.utils.valuenorm import ValueNorm


def _t2n(x):
    return x.detach().cpu().numpy()


class PH2Trainer:
    """
    Dual-policy PPO trainer for PH2.

    Exposes:
      spec_policy, ind_policy  : R_MAPPOPolicy
      partner_pred             : PartnerPredictionNet (owned by ind)
      value_normalizer         : shared ValueNorm (or None)
    """

    def __init__(self, args, obs_space, share_obs_space, act_space, device):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # ---- policies ----
        self.spec_policy = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device)
        self.ind_policy  = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device)

        # ---- PPO trainers ----
        self.spec_trainer = R_MAPPO(args, self.spec_policy, device)
        self.ind_trainer  = R_MAPPO(args, self.ind_policy,  device)

        # ---- partner predictor (ind branch) ----
        self.use_partner_pred = getattr(args, "ph2_use_partner_pred", True)
        if self.use_partner_pred:
            from zsceval.utils.util import get_shape_from_obs_space
            obs_shape = get_shape_from_obs_space(obs_space)
            # obs_shape is (H, W, C) for image obs
            if len(obs_shape) == 3:
                obs_channels = obs_shape[2]
            else:
                obs_channels = obs_shape[0]
            act_dim = act_space.n
            history_len = getattr(args, "ph2_history_len", 5)
            self.partner_pred = PartnerPredictionNet(obs_channels, act_dim, history_len).to(device)
            self.pred_optimizer = torch.optim.Adam(
                self.partner_pred.parameters(),
                lr=args.lr,
                eps=getattr(args, "opti_eps", 1e-5),
                weight_decay=getattr(args, "weight_decay", 0),
            )
            self.pred_loss_coef = getattr(args, "ph2_pred_loss_coef", 1.0)
        else:
            self.partner_pred = None
            self.pred_optimizer = None
            self.pred_loss_coef = 0.0

        # ---- value normalizer (shared) ----
        if args.use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=device)
            # point both trainers at the same normalizer
            self.spec_trainer.value_normalizer = self.value_normalizer
            self.ind_trainer.value_normalizer  = self.value_normalizer
        else:
            self.value_normalizer = None

        # ---- PH2 schedule params ----
        self.ph2_fixed_ind_prob = getattr(args, "ph2_fixed_ind_prob", 0.5)
        self.ph2_ratio_stage1   = getattr(args, "ph2_ratio_stage1",   2)
        self.ph2_ratio_stage2   = getattr(args, "ph2_ratio_stage2",   1)
        self.ph2_ratio_stage3   = getattr(args, "ph2_ratio_stage3",   2)
        self.num_updates = max(
            int(args.num_env_steps) // args.episode_length // args.n_rollout_threads,
            1,
        )

    # ------------------------------------------------------------------
    # schedule helpers
    # ------------------------------------------------------------------
    def compute_ind_match_prob(self, update_step: int) -> float:
        """Return probability that a given env is an ind-match episode."""
        if self.ph2_fixed_ind_prob >= 0:
            return float(self.ph2_fixed_ind_prob)
        progress = update_step / max(self.num_updates, 1)
        if progress < 1 / 3:
            ratio = self.ph2_ratio_stage1
        elif progress < 2 / 3:
            ratio = self.ph2_ratio_stage2
        else:
            ratio = self.ph2_ratio_stage3
        return ratio / (ratio + 1)

    # ------------------------------------------------------------------
    # train_spec: PPO update with train_mask
    # ------------------------------------------------------------------
    def train_spec(self, spec_buffer) -> dict:
        """Run PPO update on spec policy using spec_buffer.train_masks."""
        self.spec_trainer.prep_training()
        if self.spec_trainer._use_popart or self.spec_trainer._use_valuenorm:
            advantages = spec_buffer.returns[:-1] - self.spec_trainer.value_normalizer.denormalize(
                spec_buffer.value_preds[:-1]
            )
        else:
            advantages = spec_buffer.returns[:-1] - spec_buffer.value_preds[:-1]

        # mask advantages by train_mask before normalisation
        adv_copy = advantages.copy()
        # where train_mask==0, set nan so nanmean/nanstd ignores them
        adv_copy[spec_buffer.train_masks == 0.0] = np.nan
        mean_adv = np.nanmean(adv_copy) if not np.all(np.isnan(adv_copy)) else 0.0
        std_adv  = np.nanstd(adv_copy)  if not np.all(np.isnan(adv_copy)) else 1.0
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        train_info = defaultdict(float)
        for _ in range(self.spec_trainer.ppo_epoch):
            data_gen = spec_buffer.recurrent_generator(
                advantages, self.spec_trainer.num_mini_batch, self.spec_trainer.data_chunk_length
            )
            for sample in data_gen:
                # sample has base fields + PH2 extras at indices 13-16
                base_sample = sample[:13]
                train_masks_batch = sample[13]
                info = self._ppo_update_with_mask(
                    self.spec_trainer, base_sample, train_masks_batch
                )
                for k, v in info.items():
                    train_info[k] += v

        num_updates = self.spec_trainer.ppo_epoch * self.spec_trainer.num_mini_batch
        return {f"spec/{k}": v / num_updates for k, v in train_info.items()}

    # ------------------------------------------------------------------
    # train_ind: PPO update + E3T pred loss
    # ------------------------------------------------------------------
    def train_ind(self, ind_buffer) -> dict:
        """Run PPO + E3T update on ind policy."""
        self.ind_trainer.prep_training()
        if self.ind_trainer._use_popart or self.ind_trainer._use_valuenorm:
            advantages = ind_buffer.returns[:-1] - self.ind_trainer.value_normalizer.denormalize(
                ind_buffer.value_preds[:-1]
            )
        else:
            advantages = ind_buffer.returns[:-1] - ind_buffer.value_preds[:-1]

        adv_copy = advantages.copy()
        adv_copy[ind_buffer.active_masks[:-1] == 0.0] = np.nan
        mean_adv = np.nanmean(adv_copy) if not np.all(np.isnan(adv_copy)) else 0.0
        std_adv  = np.nanstd(adv_copy)  if not np.all(np.isnan(adv_copy)) else 1.0
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        train_info = defaultdict(float)
        for _ in range(self.ind_trainer.ppo_epoch):
            data_gen = ind_buffer.recurrent_generator(
                advantages, self.ind_trainer.num_mini_batch, self.ind_trainer.data_chunk_length
            )
            for sample in data_gen:
                base_sample    = sample[:13]
                train_masks_b  = sample[13]
                partner_acts_b = sample[14]  # (B, 1)
                obs_hist_b     = sample[15]  # (B, L, H, W, C)
                act_hist_b     = sample[16]  # (B, L)

                ppo_info = self._ppo_update_with_mask(
                    self.ind_trainer, base_sample, train_masks_b
                )
                for k, v in ppo_info.items():
                    train_info[k] += v

                if self.use_partner_pred and self.partner_pred is not None:
                    pred_info = self._pred_update(
                        obs_hist_b, act_hist_b, partner_acts_b, train_masks_b
                    )
                    for k, v in pred_info.items():
                        train_info[k] += v

        num_updates = self.ind_trainer.ppo_epoch * self.ind_trainer.num_mini_batch
        result = {f"ind/{k}": v / num_updates for k, v in train_info.items()}
        return result

    # ------------------------------------------------------------------
    # internal: masked PPO update
    # ------------------------------------------------------------------
    def _ppo_update_with_mask(self, trainer, base_sample, train_masks_batch) -> dict:
        """Run one PPO mini-batch update, masking loss by train_mask."""
        (
            share_obs_batch, obs_batch,
            rnn_states_batch, rnn_states_critic_batch,
            actions_batch, value_preds_batch, return_batch, masks_batch,
            active_masks_batch, old_action_log_probs_batch,
            adv_targ, available_actions_batch, _other_id,
        ) = base_sample

        tpdv = self.tpdv
        old_lp  = check(old_action_log_probs_batch).to(**tpdv)
        adv     = check(adv_targ).to(**tpdv)
        vp      = check(value_preds_batch).to(**tpdv)
        ret     = check(return_batch).to(**tpdv)
        amask   = check(active_masks_batch).to(**tpdv)
        tmask   = check(train_masks_batch).to(**tpdv)        # (B, 1)

        policy  = trainer.policy

        values, action_log_probs, dist_entropy, _ = policy.evaluate_actions(
            share_obs_batch, obs_batch,
            rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch,
            available_actions_batch, active_masks_batch,
        )

        # ---- actor loss (masked) ----
        ratio  = torch.exp(action_log_probs - old_lp)
        surr1  = ratio * adv
        surr2  = torch.clamp(ratio, 1 - trainer.clip_param, 1 + trainer.clip_param) * adv
        raw_loss = -torch.min(surr1, surr2).sum(-1, keepdim=True)  # (B, 1)

        denom = tmask.sum().clamp(min=1.0)
        policy_action_loss = (raw_loss * tmask).sum() / denom
        entropy = (dist_entropy * tmask.squeeze(-1)).sum() / denom

        policy.actor_optimizer.zero_grad()
        actor_loss = policy_action_loss - entropy * trainer.entropy_coef
        actor_loss.backward()
        if trainer._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), trainer.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
        policy.actor_optimizer.step()

        # ---- value loss (masked) ----
        value_loss = trainer.cal_value_loss(
            trainer.value_normalizer, values, vp, ret, amask
        )
        policy.critic_optimizer.zero_grad()
        (value_loss * trainer.value_loss_coef).backward()
        if trainer._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), trainer.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.critic_optimizer.step()

        upper_rate = ((ratio > (1 + trainer.clip_param)).float() * tmask).sum() / denom
        lower_rate = ((ratio < (1 - trainer.clip_param)).float() * tmask).sum() / denom

        return {
            "value_loss":       value_loss.item(),
            "policy_loss":      policy_action_loss.item(),
            "dist_entropy":     entropy.item(),
            "actor_grad_norm":  actor_grad_norm.item() if hasattr(actor_grad_norm, "item") else float(actor_grad_norm),
            "critic_grad_norm": critic_grad_norm.item() if hasattr(critic_grad_norm, "item") else float(critic_grad_norm),
            "ratio":            ratio.mean().item(),
            "upper_clip_rate":  upper_rate.item(),
            "lower_clip_rate":  lower_rate.item(),
            "entropy_coef":     trainer.entropy_coef,
        }

    # ------------------------------------------------------------------
    # internal: partner predictor update
    # ------------------------------------------------------------------
    def _pred_update(self, obs_hist_b, act_hist_b, partner_acts_b, train_masks_b) -> dict:
        """E3T CE loss update for partner predictor."""
        tpdv = self.tpdv
        obs_h = torch.as_tensor(obs_hist_b, **tpdv)      # (B, L, H, W, C)
        act_h = torch.as_tensor(act_hist_b, dtype=torch.long, device=self.device)  # (B, L)
        p_act = torch.as_tensor(partner_acts_b.squeeze(-1), dtype=torch.long, device=self.device)  # (B,)
        tmask = torch.as_tensor(train_masks_b, **tpdv)   # (B, 1)

        self.partner_pred.train()
        self.pred_optimizer.zero_grad()
        loss, info = self.partner_pred.predict_and_loss(
            obs_h, act_h, p_act, train_mask=tmask, pred_loss_coef=self.pred_loss_coef
        )
        loss.backward()
        if self.args.use_max_grad_norm:
            nn.utils.clip_grad_norm_(self.partner_pred.parameters(), self.args.max_grad_norm)
        self.pred_optimizer.step()
        return info

    # ------------------------------------------------------------------
    # entropy scheduling
    # ------------------------------------------------------------------
    def adapt_entropy_coef(self, num_steps: int):
        self.spec_trainer.adapt_entropy_coef(num_steps)
        self.ind_trainer.adapt_entropy_coef(num_steps)

    # ------------------------------------------------------------------
    # prep helpers
    # ------------------------------------------------------------------
    def prep_rollout(self):
        self.spec_trainer.prep_rollout()
        self.ind_trainer.prep_rollout()
        if self.partner_pred is not None:
            self.partner_pred.eval()

    def prep_training(self):
        self.spec_trainer.prep_training()
        self.ind_trainer.prep_training()

    # ------------------------------------------------------------------
    # save / restore
    # ------------------------------------------------------------------
    def save(self, save_dir: str, step: int = 0):
        import os
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_periodic_{step}" if step > 0 else ""
        torch.save(self.spec_policy.actor.state_dict(),  f"{save_dir}/spec_actor{suffix}.pt")
        torch.save(self.spec_policy.critic.state_dict(), f"{save_dir}/spec_critic{suffix}.pt")
        torch.save(self.ind_policy.actor.state_dict(),   f"{save_dir}/ind_actor{suffix}.pt")
        torch.save(self.ind_policy.critic.state_dict(),  f"{save_dir}/ind_critic{suffix}.pt")
        # eval compat: ind is the default evaluation policy
        torch.save(self.ind_policy.actor.state_dict(),   f"{save_dir}/actor{suffix}.pt")
        torch.save(self.ind_policy.critic.state_dict(),  f"{save_dir}/critic{suffix}.pt")
        if self.partner_pred is not None:
            torch.save(self.partner_pred.state_dict(), f"{save_dir}/partner_pred{suffix}.pt")

    def restore(self, model_dir: str):
        import os
        map_loc = self.device

        def _try_load(path, module):
            if os.path.exists(path):
                module.load_state_dict(torch.load(path, map_location=map_loc))
                logger.info(f"Loaded {path}")

        _try_load(f"{model_dir}/spec_actor.pt",  self.spec_policy.actor)
        _try_load(f"{model_dir}/spec_critic.pt", self.spec_policy.critic)
        _try_load(f"{model_dir}/ind_actor.pt",   self.ind_policy.actor)
        _try_load(f"{model_dir}/ind_critic.pt",  self.ind_policy.critic)
        if self.partner_pred is not None:
            _try_load(f"{model_dir}/partner_pred.pt", self.partner_pred)
