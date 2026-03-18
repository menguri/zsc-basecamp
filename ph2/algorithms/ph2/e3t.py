"""
PyTorch port of the E3T Partner Prediction Module.

Architecture (action-prediction mode, matching E3T paper Table 4):
  StepWiseEncoder:
    CNN (25-filter 5×5 → 25-filter 3×3 → 25-filter 3×3) + LeakyReLU at each layer
    Flatten → concat with action embedding (25-dim)
    3 × (Linear 64 + LeakyReLU)
    output: 64-dim feature per timestep

  PartnerPredictionNet:
    Apply StepWiseEncoder to each of T history steps → (B, T, 64)
    Flatten → (B, T*64)
    3 × (Linear 64 + LeakyReLU) → Linear 64 + Tanh → Linear action_dim
    L2 normalise → (B, action_dim) logits
    Loss: cross-entropy vs. actual partner action
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepWiseEncoder(nn.Module):
    """Encode a single (obs, action) pair into a 64-dim feature."""

    def __init__(self, obs_channels: int, action_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 25, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(25, 25, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(25, 25, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.act_embed = nn.Embedding(action_dim, 25)
        # MLP: conv_flat + act_embed → 3 × (64 + LeakyReLU)
        # conv_flat_size is determined at first forward pass (lazy init)
        self._mlp = None
        self._conv_flat_size = None
        self.action_dim = action_dim

    def _build_mlp(self, conv_flat_size: int):
        in_size = conv_flat_size + 25
        self._mlp = nn.Sequential(
            nn.Linear(in_size, 64), nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),     nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),     nn.LeakyReLU(inplace=True),
        ).to(next(self.conv.parameters()).device)
        self._conv_flat_size = conv_flat_size

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        obs : (B, H, W, C)  float32  — channels-last from env
        act : (B,)           int64
        returns: (B, 64)
        """
        # channels-last → channels-first for Conv2d
        x = obs.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.conv(x)                            # (B, 25, H, W)
        x = x.flatten(1)                            # (B, 25*H*W)

        if self._mlp is None:
            self._build_mlp(x.shape[1])

        a = self.act_embed(act.long())              # (B, 25)
        x = torch.cat([x, a], dim=-1)              # (B, 25*H*W + 25)
        return self._mlp(x)                         # (B, 64)


class PartnerPredictionNet(nn.Module):
    """
    Full partner prediction module.
    Takes history of T (obs, action) pairs and predicts the partner's action logits.
    """

    def __init__(self, obs_channels: int, action_dim: int, history_len: int = 5):
        super().__init__()
        self.action_dim = action_dim
        self.history_len = history_len
        self.encoder = StepWiseEncoder(obs_channels, action_dim)

        flat_dim = history_len * 64
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),       nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),       nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),       nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # final layer: small init
        nn.init.orthogonal_(self.mlp[-1].weight, gain=0.01)

    def forward(
        self,
        obs_history: torch.Tensor,
        act_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        obs_history : (B, T, H, W, C)  float32
        act_history : (B, T)            int64
        returns     : (B, action_dim)   logits (L2 normalised)
        """
        B, T, H, W, C = obs_history.shape
        obs_flat = obs_history.reshape(B * T, H, W, C)
        act_flat = act_history.reshape(B * T)

        features = self.encoder(obs_flat, act_flat)  # (B*T, 64)
        features = features.reshape(B, T * 64)       # (B, T*64)

        logits = self.mlp(features)                  # (B, action_dim)
        # L2 normalise
        logits = F.normalize(logits, p=2, dim=-1)
        return logits

    def predict_and_loss(
        self,
        obs_history: torch.Tensor,
        act_history: torch.Tensor,
        partner_action: torch.Tensor,
        train_mask: torch.Tensor | None = None,
        pred_loss_coef: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute forward pass, CE loss and accuracy.

        partner_action : (B,) int64
        train_mask     : (B, 1) float32 or None
        returns        : (loss, info_dict)
        """
        logits = self(obs_history, act_history)      # (B, action_dim)
        loss_vec = F.cross_entropy(logits, partner_action.long(), reduction="none")  # (B,)

        if train_mask is not None:
            m = train_mask.squeeze(-1).float()       # (B,)
            denom = m.sum().clamp(min=1.0)
            loss = (loss_vec * m).sum() / denom
            pred_labels = logits.argmax(dim=-1)
            acc = ((pred_labels == partner_action.long()).float() * m).sum() / denom
        else:
            loss = loss_vec.mean()
            pred_labels = logits.argmax(dim=-1)
            acc = (pred_labels == partner_action.long()).float().mean()

        loss = loss * pred_loss_coef
        return loss, {"pred_loss": loss.item(), "pred_accuracy": acc.item()}
