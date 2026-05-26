"""
FishSaltModel — joint fish detection, weight regression, and salinity estimation.

────────────────────────────────────────────────────────────────────────────────
Why this model exists
────────────────────────────────────────────────────────────────────────────────
The previous FishModel received water salinity as a hard-coded 13th channel taken
directly from the recording filename (e.g. "_S400" → 1.0, fresh water → 0.0).
That approach breaks whenever a recording is mislabelled or the filename convention
changes.

FishSaltModel eliminates the labelled salt channel entirely.  Instead it receives
three scalar *physics features* computed once per recording from the raw electrode
baseline and uses a dedicated salt-prediction head to make salinity estimation an
explicit, supervised sub-task.

────────────────────────────────────────────────────────────────────────────────
Physical background — how salinity is encoded in the raw signal
────────────────────────────────────────────────────────────────────────────────
The 12 electrodes apply a weak electric field across the fish-pass tube and measure
the resulting voltage.  Water electrical conductivity σ scales approximately linearly
with dissolved NaCl concentration c at the concentrations we use (~25 °C):

    σ  ≈  k · c          (Kohlrausch's law for dilute electrolytes)

Higher conductivity raises the resting (no-fish) electrode voltage at all 12
channels.  We measure this through the raw ADC baseline.

Empirically measured on our hardware:

    Fresh water  (0 g/L NaCl):   grand baseline mean ≈ 11 010 ADC
    Salt  water  (400 g/L NaCl): grand baseline mean ≈ 19 731 ADC
    Ratio: 1.79 ×,  gap: 8 721 ADC — zero overlap between the two conditions.

The physics-based salt estimate therefore is:

    salt_norm  =  clip( (baseline_mean − FRESH_BASELINE) / CONDUCTIVITY_SLOPE,  0, 1 )
    salt_g/L   =  salt_norm × 400

Because per-file z-score normalisation (required so the CNN focuses on wave shape,
not absolute level) removes this baseline, the conductivity information must be
passed as a separate input — the three physics features below.

────────────────────────────────────────────────────────────────────────────────
Physics features (computed once per recording CSV, then constant for all windows)
────────────────────────────────────────────────────────────────────────────────
  feat 0  baseline_mean_norm  =  grand_mean / SALT_BASELINE
              ~0.56 for fresh water,  ~1.00 for 400 g/L salt water
              Encodes absolute conductivity level.

  feat 1  baseline_std_norm   =  grand_std / SALT_BASELINE
              Noise-level proxy; slightly lower in salt water (lower CV).

  feat 2  physics_salt_est    =  clip((grand_mean − FRESH_BASELINE) / CONDUCTIVITY_SLOPE, 0, 1)
              Kohlrausch linear-conductivity formula applied directly.
              This is the model's hard prior: ~0 → fresh, ~1 → salt.
              Values outside [0, 1] are clipped (handles other salt concentrations).

Use  compute_physics_features(raw_ndarray)  to produce this vector from any CSV.

────────────────────────────────────────────────────────────────────────────────
Architecture
────────────────────────────────────────────────────────────────────────────────

  x_wave  (B, 39, 12)          x_phys  (B, 3)
       |                              |
  Temporal CNN                 Physics MLP
  Conv1d(k=7)+MaxPool           FC(phys_h) → ReLU → Dropout
  Conv1d(k=5)+MaxPool           FC(phys_h/2) → ReLU
  Conv1d(k=3)+AvgPool           |
  Dropout                       ├──► Salt head: FC(16)→ReLU→FC(1)→Sigmoid
  128-d feature                 |    pred_salt_norm ∈ [0,1]  (×400 = g/L)
       |                    (phys_h/2)-d feature
       └─────────── cat ────────┘
               (128 + phys_h/2)-d
                     |
              FC(hidden) → ReLU → Dropout
                     |
          ┌──────────┴──────────┐
     Presence head          Weight head
     FC(1) → logit          FC(1) → weight (g) or log-weight

────────────────────────────────────────────────────────────────────────────────
Inputs / Outputs
────────────────────────────────────────────────────────────────────────────────
  forward(x_wave, x_phys) →
      logit_presence : (B,)  raw logit for BCEWithLogitsLoss / sigmoid
      pred_weight    : (B,)  weight in grams (log-grams if log_scale training)
      pred_salt_norm : (B,)  salt estimate in [0, 1]; multiply by 400 for g/L

────────────────────────────────────────────────────────────────────────────────
Model config keys (model/model_config_salt.yaml)
────────────────────────────────────────────────────────────────────────────────
  input_channels   (int)    acoustic waveform channels — must be 12
  hidden_dim       (int)    shared FC neck width (e.g. 128)
  physics_hidden   (int)    physics MLP hidden width (e.g. 64)
  dropout_temporal (float)  dropout after temporal branch
  dropout_physics  (float)  dropout inside physics MLP
  dropout_fc       (float)  dropout in fusion FC
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from model.fish_model import ConvBlock

# ── Hardware calibration constants ───────────────────────────────────────────
# These were measured empirically from still-water recordings on our hardware.
# They relate water conductivity to raw ADC and should be re-calibrated if the
# installation or hardware changes.

FRESH_BASELINE     = 11_010.0              # ADC, 0 g/L NaCl   (grand mean, all 12 channels)
SALT_BASELINE      = 19_731.0              # ADC, 400 g/L NaCl
CONDUCTIVITY_SLOPE = SALT_BASELINE - FRESH_BASELINE   # 8 721 ADC per 400 g/L NaCl
SALT_MAX_G_PER_L   = 400.0                 # maximum salt concentration in our data

PHYSICS_DIM        = 3   # default; override with cfg["physics_dim"]


# ── Physics feature helper ────────────────────────────────────────────────────

def compute_physics_features(raw: np.ndarray) -> np.ndarray:
    """
    Compute the three recording-level physics features from raw electrode data.

    Call this ONCE per CSV file before sliding-window inference.  The returned
    vector is the same for every window extracted from that recording.

    Args:
        raw:  (N, 12) float32 array of raw ADC readings (all channels, all rows).
              May contain NaN; they are ignored in the statistics.

    Returns:
        (3,) float32 array:
            [0] baseline_mean_norm   — grand mean / SALT_BASELINE
                                       (~0.56 fresh water, ~1.00 salt water)
            [1] baseline_std_norm    — grand std / SALT_BASELINE
            [2] physics_salt_est     — Kohlrausch-based salt estimate in [0, 1]
                                       Multiply by SALT_MAX_G_PER_L (400) for g/L.
    """
    mean = float(np.nanmean(raw))
    std  = float(np.nanstd(raw))
    salt_est = float(np.clip(
        (mean - FRESH_BASELINE) / CONDUCTIVITY_SLOPE, 0.0, 1.0
    ))
    return np.array(
        [mean / SALT_BASELINE,
         std  / SALT_BASELINE,
         salt_est],
        dtype=np.float32,
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class FishSaltModel(nn.Module):
    """
    Joint fish detection, weight regression, and salinity estimation.
    See module docstring for architecture details and physical motivation.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        n_sensor   = cfg["input_channels"]          # 12 acoustic channels (no salt in X)
        hidden     = cfg["hidden_dim"]
        phys_h     = cfg.get("physics_hidden", 64)
        physics_dim = cfg.get("physics_dim", PHYSICS_DIM)

        # ── Temporal CNN (identical architecture to FishModel) ─────────────
        # Processes the 12-channel per-file z-normalised waveform.
        # Z-normalisation removes the absolute baseline, so this branch has NO
        # access to salinity — it focuses purely on wave shape.
        self.temporal_branch = nn.Sequential(
            ConvBlock(n_sensor, 32,  kernel_size=7, pool=nn.MaxPool1d(2)),   # (B,32,19)
            ConvBlock(32,       64,  kernel_size=5, pool=nn.MaxPool1d(2)),   # (B,64,9)
            ConvBlock(64,       128, kernel_size=3, pool=nn.AdaptiveAvgPool1d(1)),  # (B,128,1)
        )
        self.temporal_dropout = nn.Dropout(cfg["dropout_temporal"])

        # ── Physics MLP ────────────────────────────────────────────────────
        # Processes [baseline_mean_norm, baseline_std_norm, physics_salt_est].
        # The baseline mean already encodes salinity via Kohlrausch's law, so
        # this small MLP easily learns to separate fresh from salt water.
        self.physics_branch = nn.Sequential(
            nn.Linear(physics_dim, phys_h),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.get("dropout_physics", 0.2)),
            nn.Linear(phys_h, phys_h // 2),
            nn.ReLU(inplace=True),
        )

        # ── Salt head (branches before fusion) ─────────────────────────────
        # Uses only the physics branch output because salt information lives
        # entirely in the baseline level, not in the normalised waveform.
        # Sigmoid constrains output to [0,1]; multiply by 400 to get g/L.
        self.salt_head = nn.Sequential(
            nn.Linear(phys_h // 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # ── Fusion FC neck ─────────────────────────────────────────────────
        # Concatenates temporal features (128-d) with physics features (phys_h/2-d).
        # This lets the presence/weight heads be conditioned on conductivity, which
        # affects fish signal amplitude and requires different calibration thresholds.
        fusion_dim = 128 + phys_h // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout_fc"]),
        )

        # ── Task heads ─────────────────────────────────────────────────────
        self.presence_head = nn.Linear(hidden, 1)   # raw logit
        self.weight_head   = nn.Linear(hidden, 1)   # grams or log-grams

    def forward(
        self,
        x_wave: torch.Tensor,   # (B, T, 12)  per-file z-normalised waveform
        x_phys: torch.Tensor,   # (B, 3)      recording-level physics features
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_wave: (B, 39, 12) — acoustic signal, per-file z-normalised.
            x_phys: (B, 3)      — physics features from compute_physics_features().

        Returns:
            logit_presence : (B,)  raw logit
            pred_weight    : (B,)  weight prediction
            pred_salt_norm : (B,)  salt estimate in [0, 1]  (× 400 = g/L)
        """
        # Temporal branch — wave shape only, no conductivity information
        x_w = x_wave.transpose(1, 2)          # (B, 12, T) for Conv1d
        x_w = self.temporal_branch(x_w)       # (B, 128, 1)
        x_w = x_w.squeeze(-1)                 # (B, 128)
        x_w = self.temporal_dropout(x_w)

        # Physics branch — conductivity / salinity features
        x_p = self.physics_branch(x_phys)     # (B, phys_h // 2)

        # Salt prediction (derived from physics features alone)
        pred_salt_norm = self.salt_head(x_p).squeeze(-1)    # (B,)

        # Fuse and predict presence / weight
        x = torch.cat([x_w, x_p], dim=1)      # (B, 128 + phys_h // 2)
        x = self.fusion(x)                     # (B, hidden)

        logit_presence = self.presence_head(x).squeeze(-1)  # (B,)
        pred_weight    = self.weight_head(x).squeeze(-1)     # (B,)

        return logit_presence, pred_weight, pred_salt_norm
