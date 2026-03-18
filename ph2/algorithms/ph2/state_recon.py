"""
StateReconNet — Simultaneous state reconstruction + partner action prediction.

In OV1, lossless_state_encoding gives the full game state to every agent,
so obs[0] == obs[1] == state. We reconstruct the state (= obs[0]) from the
ego agent's own (obs, action) history.

Architecture:
  Shared encoder  : StepWiseEncoder (E3T, same structure) × T steps → trunk MLP → z (64-dim)
  Recon head      : z → FC decoder → raw logits (B, H, W, C)
                    Loss: BCE-with-logits  (state is binary 0/1)
                    Eval: sigmoid → threshold 0.5 → pixel accuracy
  Action pred head: z → Linear + Tanh → Linear → L2-norm logits (B, action_dim)
                    Loss: CrossEntropy
                    Eval: argmax accuracy

Independence:
  - Completely separate module; does NOT touch PartnerPredictionNet.
  - Instantiated only when ph2_use_state_recon=True.
  - Removing it has zero effect on the original action-pred flow.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ph2.algorithms.ph2.e3t import StepWiseEncoder


class StateReconNet(nn.Module):
    """
    Shared-trunk module for state reconstruction + partner action prediction.
    Target: state = obs[0] (full game state via lossless_state_encoding in OV1).

    Args:
        obs_channels : C dimension of lossless state  (B, H, W, C)
        action_dim   : number of discrete actions (6 for Overcooked)
        history_len  : T, number of (obs, action) history steps
        decoder_hidden: width of the hidden layer in the FC decoder
    """

    def __init__(
        self,
        obs_channels: int,
        action_dim: int,
        history_len: int = 5,
        decoder_hidden: int = 256,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.history_len = history_len
        self._decoder_hidden = decoder_hidden

        # ── Shared encoder (same structure as PartnerPredictionNet) ───────
        self.encoder = StepWiseEncoder(obs_channels, action_dim)

        flat_dim = history_len * 64

        # ── Shared trunk MLP → z ──────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),       nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),       nn.LeakyReLU(inplace=True),
        )

        # ── Action prediction head (mirrors PartnerPredictionNet tail) ────
        self.action_head = nn.Sequential(
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # ── Reconstruction decoder (lazy-initialised on first forward) ────
        # obs shape (H, W, C) is not known at construction time.
        self._decoder: Optional[nn.Sequential] = None
        self._obs_shape: Optional[tuple] = None   # (H, W, C)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
        for m in self.action_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # action head last layer: small init (same as PartnerPredictionNet)
        nn.init.orthogonal_(self.action_head[-1].weight, gain=0.01)

    # ------------------------------------------------------------------
    # Lazy decoder build
    # ------------------------------------------------------------------
    def _build_decoder(self, H: int, W: int, C: int):
        obs_flat = H * W * C
        self._obs_shape = (H, W, C)
        self._decoder = nn.Sequential(
            nn.Linear(64, 128),                        nn.LeakyReLU(inplace=True),
            nn.Linear(128, self._decoder_hidden),      nn.LeakyReLU(inplace=True),
            nn.Linear(self._decoder_hidden, obs_flat),
            # No final activation: output = raw logits for BCE-with-logits
        ).to(next(self.trunk.parameters()).device)
        # orthogonal init for decoder too
        for m in self._decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Encode history → z
    # ------------------------------------------------------------------
    def _encode(
        self,
        obs_history: torch.Tensor,   # (B, T, H, W, C)
        act_history: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:               # (B, 64)
        B, T, H, W, C = obs_history.shape
        if self._decoder is None:
            self._build_decoder(H, W, C)

        obs_flat = obs_history.reshape(B * T, H, W, C)
        act_flat = act_history.reshape(B * T)
        features = self.encoder(obs_flat, act_flat)   # (B*T, 64)
        features = features.reshape(B, T * 64)
        return self.trunk(features)                    # (B, 64)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs_history: torch.Tensor,   # (B, T, H, W, C)
        act_history: torch.Tensor,   # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon_logits: (B, H, W, C)    raw logits (apply sigmoid for probabilities)
            pred_logits : (B, action_dim) L2-normalised action prediction logits
        """
        z = self._encode(obs_history, act_history)     # (B, 64)

        H, W, C = self._obs_shape
        recon_logits = self._decoder(z).view(-1, H, W, C)

        pred_logits = self.action_head(z)              # (B, action_dim)
        pred_logits = F.normalize(pred_logits, p=2, dim=-1)

        return recon_logits, pred_logits

    # ------------------------------------------------------------------
    # Loss + metrics
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        obs_history: torch.Tensor,       # (B, T, H, W, C)
        act_history: torch.Tensor,       # (B, T)
        target_obs: torch.Tensor,        # (B, H, W, C)  binary 0/1
        partner_action: torch.Tensor,    # (B,)  int64
        train_mask: Optional[torch.Tensor] = None,   # (B, 1) float32
        recon_coef: float = 1.0,
        pred_coef: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute joint loss and evaluation metrics.

        Reconstruction:
            Loss : BCE-with-logits (treats each pixel as binary classification)
            Eval : pixel accuracy after sigmoid → threshold 0.5

        Action prediction:
            Loss : CrossEntropy
            Eval : argmax accuracy

        Returns (total_loss, info_dict).
        """
        recon_logits, pred_logits = self(obs_history, act_history)

        target_obs = target_obs.float()

        # ── Reconstruction ────────────────────────────────────────────────
        # BCE per pixel: (B, H, W, C)
        bce_map = F.binary_cross_entropy_with_logits(
            recon_logits, target_obs, reduction="none"
        )
        # mean over spatial dims → (B,)
        bce_per_sample = bce_map.mean(dim=(1, 2, 3))

        # ── Action prediction ─────────────────────────────────────────────
        ce_vec = F.cross_entropy(
            pred_logits, partner_action.long(), reduction="none"
        )   # (B,)

        # ── Masking ───────────────────────────────────────────────────────
        if train_mask is not None:
            m = train_mask.squeeze(-1).float()    # (B,)
            denom = m.sum().clamp(min=1.0)
            recon_loss = (bce_per_sample * m).sum() / denom
            pred_loss  = (ce_vec * m).sum() / denom
        else:
            recon_loss = bce_per_sample.mean()
            pred_loss  = ce_vec.mean()

        total_loss = recon_coef * recon_loss + pred_coef * pred_loss

        # ── Eval metrics (no grad needed) ─────────────────────────────────
        with torch.no_grad():
            # pixel accuracy: sigmoid → 0.5 threshold → compare with binary target
            recon_binary = (torch.sigmoid(recon_logits) > 0.5).float()
            if train_mask is not None:
                # (B, H, W, C) → mean over spatial → (B,) → masked mean
                per_sample_acc = recon_binary.eq(target_obs).float().mean(dim=(1, 2, 3))
                pixel_acc = (per_sample_acc * m).sum() / denom
            else:
                pixel_acc = recon_binary.eq(target_obs).float().mean()

            # action prediction accuracy
            pred_labels = pred_logits.argmax(dim=-1)
            if train_mask is not None:
                pred_acc = (
                    (pred_labels == partner_action.long()).float() * m
                ).sum() / denom
            else:
                pred_acc = (pred_labels == partner_action.long()).float().mean()

        info = {
            "recon/bce_loss":     recon_loss.item(),
            "recon/pixel_acc":    pixel_acc.item(),
            "pred/ce_loss":       pred_loss.item(),
            "pred/accuracy":      pred_acc.item(),
            "total_loss":         total_loss.item(),
        }
        return total_loss, info
