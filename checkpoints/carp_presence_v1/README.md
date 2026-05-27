# carp_presence_v1

**Baseline presence detector — 12-channel temporal CNN.**

## Architecture
Single-branch temporal CNN with 3 ConvBlocks (Conv1d → BN → ReLU → Pool),
AdaptiveAvgPool to a 128-dim vector, FC fusion neck, presence logit head.
Input: `(batch, 39, 12)` — 39 samples @ 40 Hz, 12 electrodes (F15 F37 F18 F32 F45 F67 / B15 B37 B18 B32 B45 B67).

## Dataset
- Single data source: ACQ_5_3_2026 recordings only.
- Normalisation: standard per-channel z-score (no sigma floor).
- Negatives sampled from gap regions using center ± WINDOW_SIZE//2 guard zone.
- No GT=0 recordings included as negatives.

## Known issues fixed in v2
- Flatline recordings (σ~30) had noise amplified by the z-score → false positives on quiet files.
- No GT=0 negative source forced the model to rely on relative amplitude alone.
- Limited data variety reduced generalisation across environments.
