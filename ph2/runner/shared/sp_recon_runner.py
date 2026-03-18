"""
SpReconRunner — Self-Play runner with StateReconNet trained in parallel.

Extends ZSC-EVAL's OvercookedRunner with minimal changes:
  - Instantiates StateReconNet in __init__
  - Overrides train() to also run _train_recon_from_buffer()
  - Overrides save() to also save state_recon weights

Usage (via train_sp.py with --use_state_recon):
  USE_STATE_RECON=1 bash sh_scripts/zsceval/sp.sh cramped_room

StateReconNet trains on (obs_hist, act_hist) → (target_obs, partner_action)
where obs_hist = [obs_{t-L}, ..., obs_{t-1}] (does NOT include obs_t).
This keeps the reconstruction task non-trivial in OV1,
and naturally extends to OV2 partial-obs reconstruction.
"""
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb

from zsceval.runner.shared.overcooked_runner import OvercookedRunner
from zsceval.utils.util import get_shape_from_obs_space

from ph2.algorithms.ph2.state_recon import StateReconNet


class SpReconRunner(OvercookedRunner):
    """Self-Play runner with StateReconNet co-training."""

    def __init__(self, config):
        super().__init__(config)

        args = self.all_args
        obs_space = self.envs.observation_space[0]
        act_space = self.envs.action_space[0]

        obs_shape = get_shape_from_obs_space(obs_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        self._recon_obs_shape = obs_shape          # e.g. (H, W, C)

        # OV1 lossless obs: channels-last → C is last dim
        obs_channels = obs_shape[-1] if len(obs_shape) == 3 else obs_shape[0]
        act_dim = act_space.n

        self._recon_history_len = int(getattr(args, "recon_history_len", 5))
        self._recon_coef        = float(getattr(args, "recon_coef", 1.0))
        self._pred_coef         = float(getattr(args, "pred_coef", 1.0))
        self._recon_batch_size  = int(getattr(args, "recon_batch_size", 512))
        self._recon_grad_steps  = int(getattr(args, "recon_grad_steps", 5))
        self._recon_log_interval = int(getattr(args, "recon_log_interval", 25))

        self.state_recon = StateReconNet(
            obs_channels=obs_channels,
            action_dim=act_dim,
            history_len=self._recon_history_len,
        ).to(self.device)

        # Force lazy init (StepWiseEncoder._mlp, decoder) before optimizer creation
        # so that ALL parameters are registered and tracked by the optimizer.
        _dummy_obs = torch.zeros(
            1, self._recon_history_len, *obs_shape, device=self.device
        )
        _dummy_act = torch.zeros(
            1, self._recon_history_len, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            self.state_recon(_dummy_obs, _dummy_act)

        recon_lr = float(getattr(args, "recon_lr", 1e-3))
        self.recon_optimizer = torch.optim.Adam(
            self.state_recon.parameters(), lr=recon_lr
        )

    # ------------------------------------------------------------------
    # Override train(): PPO update + StateReconNet update
    # ------------------------------------------------------------------
    def train(self, total_num_steps: int):
        # ── standard PPO update ───────────────────────────────────────
        train_infos = super().train(total_num_steps)

        # ── StateReconNet update ──────────────────────────────────────
        recon_infos = self._train_recon_from_buffer(total_num_steps)
        train_infos.update(recon_infos)
        return train_infos

    # ------------------------------------------------------------------
    # StateReconNet training from the filled replay buffer
    # ------------------------------------------------------------------
    def _train_recon_from_buffer(self, total_num_steps: int) -> dict:
        """
        Build (obs_hist, act_hist, target_obs, partner_action) batches from
        self.buffer and run recon_grad_steps gradient updates.

        obs_hist window: obs[t-L : t]  (does NOT include obs_t)
        target_obs     : obs[t]
        partner_action : actions[t, n, 1-m]
        """
        L  = self._recon_history_len
        T  = self.episode_length           # e.g. 400
        N  = self.n_rollout_threads        # e.g. 100
        M  = self.num_agents               # 2

        # ── Pull arrays from buffer ───────────────────────────────────
        # obs   : (T+1, N, M, H, W, C)
        # acts  : (T,   N, M)            (squeeze action dim)
        # masks : (T+1, N, M, 1)         1=continue, 0=episode start
        obs   = self.buffer.obs                          # (T+1, N, M, *obs_shape)
        acts  = self.buffer.actions[:, :, :, 0]         # (T, N, M)
        masks = self.buffer.masks                        # (T+1, N, M, 1)

        # ── Valid sample discovery ────────────────────────────────────
        # Step t is valid iff no episode boundary in obs_hist window.
        # masks[t, n, m, 0] = 0 means step t is the START of a new episode.
        # For obs_hist = obs[t-L:t], boundary checks = masks[t-L+1 : t+1].
        # Build (T-L, L, N, M) tensor of relevant masks.
        if T - L <= 0:
            return {}

        # Stack mask windows: shape (T-L, L, N, M)
        mask_wins = np.stack(
            [masks[t - L + 1 : t + 1, :, :, 0] for t in range(L, T)],
            axis=0,
        )                                               # (T-L, L, N, M)
        valid_mask = mask_wins.all(axis=1)              # (T-L, N, M)  bool

        # (t_offset, n, m) where t_offset = t - L, so t = t_offset + L
        valid_indices = np.argwhere(valid_mask)         # (K, 3)
        if len(valid_indices) == 0:
            return {}

        B = min(self._recon_batch_size, len(valid_indices))

        self.state_recon.train()
        acc_info: dict[str, float] = defaultdict(float)

        for grad_step in range(self._recon_grad_steps):
            # Sample without replacement per grad step
            chosen = np.random.choice(len(valid_indices), B, replace=False)
            idx    = valid_indices[chosen]              # (B, 3)
            t_off  = idx[:, 0]                         # t - L
            n_idx  = idx[:, 1]
            m_idx  = idx[:, 2]
            t_idx  = t_off + L                         # actual t

            # ── Build obs_hist: (B, L, *obs_shape) ───────────────────
            # t_range[i, j] = t_idx[i] - L + j  → time indices [t-L, ..., t-1]
            t_range = (
                t_idx[:, np.newaxis] - L + np.arange(L)[np.newaxis, :]
            )                                           # (B, L)
            obs_hist_np = obs[t_range, n_idx[:, np.newaxis], m_idx[:, np.newaxis]]
            # → (B, L, H, W, C)
            act_hist_np = acts[t_range, n_idx[:, np.newaxis], m_idx[:, np.newaxis]]
            # → (B, L)

            # ── target_obs: obs[t] ────────────────────────────────────
            target_np = obs[t_idx, n_idx, m_idx]       # (B, H, W, C)

            # ── partner_action: acts[t, n, 1-m] ──────────────────────
            partner_m      = 1 - m_idx                 # (B,)
            partner_act_np = acts[t_idx, n_idx, partner_m]  # (B,)

            # ── To tensors ────────────────────────────────────────────
            obs_hist_t   = torch.FloatTensor(obs_hist_np).to(self.device)
            act_hist_t   = torch.LongTensor(act_hist_np).to(self.device)
            target_t     = torch.FloatTensor(target_np).to(self.device)
            partner_act_t = torch.LongTensor(partner_act_np.astype(np.int64)).to(self.device)

            # ── Forward + loss ────────────────────────────────────────
            loss, info = self.state_recon.compute_loss(
                obs_hist_t, act_hist_t, target_t, partner_act_t,
                recon_coef=self._recon_coef,
                pred_coef=self._pred_coef,
            )

            self.recon_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.state_recon.parameters(), 10.0)
            self.recon_optimizer.step()

            for k, v in info.items():
                acc_info[k] += v

        # ── Average over grad steps ───────────────────────────────────
        result = {
            f"state_recon/{k}": v / self._recon_grad_steps
            for k, v in acc_info.items()
        }

        # ── Wandb logging ─────────────────────────────────────────────
        if self.use_wandb:
            wandb.log(result, step=total_num_steps)
            # Periodic heatmap: log channel images every recon_log_interval episodes
            ep = total_num_steps // (self.episode_length * self.n_rollout_threads)
            if ep % self._recon_log_interval == 0:
                self._log_recon_heatmap(
                    obs_hist_t[:4], act_hist_t[:4], target_t[:4],
                    total_num_steps,
                )

        return result

    # ------------------------------------------------------------------
    # Heatmap logging (side-by-side target vs recon per channel)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _log_recon_heatmap(
        self,
        obs_hist: torch.Tensor,   # (B, L, H, W, C)  small B for vis
        act_hist: torch.Tensor,   # (B, L)
        target:   torch.Tensor,   # (B, H, W, C)
        total_num_steps: int,
        channels_to_log: tuple = (0, 1, 10),  # player0_loc, player1_loc, pot_loc
    ):
        """Log side-by-side target vs sigmoid(recon) for key channels."""
        self.state_recon.eval()
        recon_logits, _ = self.state_recon(obs_hist, act_hist)  # (B, H, W, C)
        recon_prob = torch.sigmoid(recon_logits)                 # (B, H, W, C)

        C = target.shape[-1]
        imgs = {}
        for ch in channels_to_log:
            if ch >= C:
                continue
            # Take first sample
            t_ch = target[0, :, :, ch].cpu().numpy()       # (H, W)
            r_ch = recon_prob[0, :, :, ch].cpu().numpy()   # (H, W) soft
            r_bin = (r_ch > 0.5).astype(np.float32)        # (H, W) thresholded

            # side-by-side: target | soft | binary  →  (H, 3W)
            vis = np.concatenate([t_ch, r_ch, r_bin], axis=1)
            # Normalise to [0, 1] for wandb
            vis = np.clip(vis, 0.0, 1.0)
            imgs[f"state_recon/ch{ch:02d}_target_soft_binary"] = wandb.Image(
                (vis * 255).astype(np.uint8),
                caption=f"ch{ch} | target | soft | binary",
            )
        if imgs:
            wandb.log(imgs, step=total_num_steps)
        self.state_recon.train()

    # ------------------------------------------------------------------
    # Override save(): also save StateReconNet
    # ------------------------------------------------------------------
    def save(self, step: int = 0, save_critic: bool = False):
        super().save(step, save_critic)
        suffix = f"_periodic_{step}" if step > 0 else ""
        save_path = Path(self.save_dir) / f"state_recon{suffix}.pt"
        torch.save(self.state_recon.state_dict(), str(save_path))
