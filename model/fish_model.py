"""Fish presence detection and length/weight prediction model.

Architecture depends on whether physics features are present in the dataset:

    With physics features (physics_dim > 0) — Dual-branch:
        Branch 1 — Temporal CNN
            Processes the first input_channels raw sensor channels at 40 Hz.
        Branch 2 — Physics MLP
            Processes the remaining physics_dim scalar features tiled across
            the time axis (constant per detection).
        Fusion: concatenated representations -> shared FC -> two output heads.

    Without physics features (physics_dim = 0) — Single-branch:
        Only the Temporal CNN branch is used.  All input_channels are treated
        as time-varying raw signals.  No physics MLP is instantiated.

Input layout — X has shape (batch, T, C):
    physics_dim > 0:  C = input_channels + physics_dim
        X[:, :, :input_channels]  time-varying waveform channels
        X[:, 0,  input_channels:] physics features (constant over time)
    physics_dim == 0: C = input_channels
        X[:, :, :]                all channels are time-varying

Outputs:
    logit_presence : (batch,)  raw logit  -> BCEWithLogitsLoss / sigmoid
    pred_regression: (batch,)  length (cm) or weight (g), unconstrained scalar

Hyperparameters are loaded from model/model_config.yaml.
"""

from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU with an optional pooling layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: nn.Module = None,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,          # bias absorbed by BatchNorm
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class FishModel(nn.Module):
    """
    Model for joint fish presence detection and regression (length or weight).

    When physics_dim > 0: dual-branch (Temporal CNN + Physics MLP).
    When physics_dim == 0: single-branch (Temporal CNN only) — used with the
    v3 dataset where all features are raw time-series from the recording files.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        n_sensor   = cfg["input_channels"]          # waveform channels (17 for v3)
        n_physics  = cfg.get("physics_dim", 0)      # 0 for v3 (no processed features)
        hidden     = cfg["hidden_dim"]              # shared FC width
        phys_h     = cfg.get("physics_hidden", 64)  # physics MLP hidden width
        self._use_physics = n_physics > 0

        # --- Temporal CNN (40 Hz, 39-sample window) -------------------
        #   time axis:  39 -> 19 -> 9 -> 1
        self.temporal_branch = nn.Sequential(
            ConvBlock(n_sensor, 32,  kernel_size=7, pool=nn.MaxPool1d(2)),          # (B, 32,  19)
            ConvBlock(32,       64,  kernel_size=5, pool=nn.MaxPool1d(2)),          # (B, 64,   9)
            ConvBlock(64,       128, kernel_size=3, pool=nn.AdaptiveAvgPool1d(1)),  # (B, 128,  1)
        )
        self.temporal_dropout = nn.Dropout(cfg["dropout_temporal"])

        # --- Physics MLP (only when physics_dim > 0) ------------------
        if self._use_physics:
            self.physics_branch = nn.Sequential(
                nn.Linear(n_physics, phys_h),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.get("dropout_physics", 0.2)),
                nn.Linear(phys_h, phys_h // 2),
                nn.ReLU(inplace=True),
            )
            fusion_dim = 128 + phys_h // 2
        else:
            fusion_dim = 128

        # --- Fusion FC neck ------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout_fc"]),
        )

        # --- Task heads ----------------------------------------------
        self.presence_head    = nn.Linear(hidden, 1)   # raw logit
        self.regression_head  = nn.Linear(hidden, 1)   # length (cm) or weight (g)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, T, C) input tensor.
               physics_dim > 0: first input_channels cols are waveform,
                                 remaining cols are tiled physics features.
               physics_dim == 0: all cols are time-varying raw signals.

        Returns:
            logit_presence  : (batch,)
            pred_regression : (batch,)  length in cm or weight in g
        """
        n_sensor = self.temporal_branch[0].conv[0].in_channels

        # Temporal branch — always uses the first n_sensor channels
        x_wave = x[:, :, :n_sensor]                   # (B, T, n_sensor)
        x_wave = x_wave.transpose(1, 2)               # (B, n_sensor, T) for Conv1d
        x_wave = self.temporal_branch(x_wave)         # (B, 128, 1)
        x_wave = x_wave.squeeze(-1)                   # (B, 128)
        x_wave = self.temporal_dropout(x_wave)

        # Physics branch (optional)
        if self._use_physics:
            x_feat = x[:, 0, n_sensor:]               # (B, n_physics)
            x_feat = self.physics_branch(x_feat)      # (B, phys_h // 2)
            x_wave = torch.cat([x_wave, x_feat], dim=1)

        x = self.fusion(x_wave)                       # (B, hidden)

        logit_presence  = self.presence_head(x).squeeze(-1)    # (B,)
        pred_regression = self.regression_head(x).squeeze(-1)  # (B,)

        return logit_presence, pred_regression
