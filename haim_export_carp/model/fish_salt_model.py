"""
FishSaltModel — joint fish detection, weight regression, and salinity estimation.

This model eliminates the need for a labelled salt channel in the input.
Instead it receives three scalar physics features computed once per recording
from the raw electrode baseline and uses a dedicated salt-prediction head to
make salinity estimation an explicit, self-supervised sub-task.

Physical background:
    Water electrical conductivity scales approximately linearly with dissolved
    NaCl concentration (Kohlrausch's law for dilute electrolytes, ~25 °C).
    Higher conductivity raises the resting electrode voltage at all 12 channels.

    Empirically measured on this hardware:
        Fresh water  (0 g/L NaCl):   grand baseline mean ~ 11 010 ADC
        Salt  water  (400 g/L NaCl): grand baseline mean ~ 19 731 ADC
        Gap: 8 721 ADC — zero overlap between the two conditions.

    Physics-based salt estimate:
        salt_norm = clip( (baseline_mean - FRESH_BASELINE) / CONDUCTIVITY_SLOPE, 0, 1 )
        salt_g/L  = salt_norm * 400

See compute_physics_features() for the three recording-level features.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from model.fish_model import ConvBlock

# Hardware calibration constants (re-calibrate if installation or hardware changes)
FRESH_BASELINE     = 11_010.0              # ADC, 0 g/L NaCl
SALT_BASELINE      = 19_731.0              # ADC, 400 g/L NaCl
CONDUCTIVITY_SLOPE = SALT_BASELINE - FRESH_BASELINE   # 8 721 ADC per 400 g/L
SALT_MAX_G_PER_L   = 400.0

PHYSICS_DIM = 3


def compute_physics_features(raw: np.ndarray) -> np.ndarray:
    """
    Compute recording-level physics features from raw electrode ADC values.

    Call once per CSV file; the returned (3,) vector is identical for every
    39-sample window extracted from that recording.

    Args:
        raw:  (N, 12) float32 array of raw ADC readings (all channels, all rows).

    Returns:
        (3,) float32 array:
            [0] baseline_mean_norm  — grand mean / SALT_BASELINE
            [1] baseline_std_norm   — grand std  / SALT_BASELINE
            [2] physics_salt_est    — Kohlrausch estimate in [0, 1]  (* 400 = g/L)
    """
    mean = float(np.nanmean(raw))
    std  = float(np.nanstd(raw))
    salt_est = float(np.clip(
        (mean - FRESH_BASELINE) / CONDUCTIVITY_SLOPE, 0.0, 1.0
    ))
    return np.array(
        [mean / SALT_BASELINE, std / SALT_BASELINE, salt_est],
        dtype=np.float32,
    )


class FishSaltModel(nn.Module):
    """Joint fish detection, weight regression, and salinity estimation."""

    def __init__(self, cfg: dict):
        super().__init__()

        n_sensor = cfg["input_channels"]     # 12 acoustic channels
        hidden   = cfg["hidden_dim"]
        phys_h   = cfg.get("physics_hidden", 64)

        self.temporal_branch = nn.Sequential(
            ConvBlock(n_sensor, 32,  kernel_size=7, pool=nn.MaxPool1d(2)),
            ConvBlock(32,       64,  kernel_size=5, pool=nn.MaxPool1d(2)),
            ConvBlock(64,       128, kernel_size=3, pool=nn.AdaptiveAvgPool1d(1)),
        )
        self.temporal_dropout = nn.Dropout(cfg["dropout_temporal"])

        self.physics_branch = nn.Sequential(
            nn.Linear(PHYSICS_DIM, phys_h),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.get("dropout_physics", 0.2)),
            nn.Linear(phys_h, phys_h // 2),
            nn.ReLU(inplace=True),
        )

        self.salt_head = nn.Sequential(
            nn.Linear(phys_h // 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        fusion_dim = 128 + phys_h // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout_fc"]),
        )

        self.presence_head = nn.Linear(hidden, 1)
        self.weight_head   = nn.Linear(hidden, 1)

    def forward(
        self,
        x_wave: torch.Tensor,   # (B, T, 12)
        x_phys: torch.Tensor,   # (B, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_w = x_wave.transpose(1, 2)
        x_w = self.temporal_branch(x_w).squeeze(-1)
        x_w = self.temporal_dropout(x_w)

        x_p = self.physics_branch(x_phys)
        pred_salt_norm = self.salt_head(x_p).squeeze(-1)

        x = torch.cat([x_w, x_p], dim=1)
        x = self.fusion(x)

        return (
            self.presence_head(x).squeeze(-1),
            self.weight_head(x).squeeze(-1),
            pred_salt_norm,
        )
