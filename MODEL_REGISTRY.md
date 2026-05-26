# Model Registry

Tracks checkpoints trained in this repo and their deployment status.

---

## Currently Deployed — AquaSense Cloud App

**Deployment:** `C:\Users\Admin\repos\AquaSenseCloudApp` → Render (backend) + HF Spaces  
**Pipeline:** Two-stage inference (`backend/scripts/infer_twostage.py`)

---

### Stage 1 — Presence Detection

| Field | Value |
|---|---|
| Name | `bream_carp_presence` |
| Checkpoints | `checkpoints/bream_carp_presence/20260430_213813/` |
| | `checkpoints/bream_carp_presence/20260430_213921/` |
| | `checkpoints/bream_carp_presence/20260430_214039/` |
| Calibration | `checkpoints/bream_carp_presence/platt.npy` (a=1.0, b=0.0) |
| Architecture | FishModel — 12-ch Conv1d × 3 → AdaptiveAvgPool → FC, 54,722 params |
| Input | 39-sample window, 12 channels, z-normalised per recording |
| Output | Probability fish is present in window |
| Trained | 2026-04-30 |
| Ensemble size | 3 runs |
| Best val loss | 0.1484 / 0.1426 / 0.1523 (early stop ~70 epochs) |

**Training data:**

| Source | Events | Water type |
|---|---|---|
| `data/raw/carp/carp_salted/salted_water/` | 731 | Salted |
| `data/raw/carp/carp_salted/salted_water/test/` | 89 | Salted |
| `data/raw/bream/recordings/` + `test/` | ~1284 | Conditioned |

Negatives: gap regions from the same recordings (80/20 neg/pos ratio).  
Split: 70/20/10 stratified by species (carp vs. bream), seeds 42/43.  
**Fresh-water carp (`data/raw/carp/ACQ_5_3_2026/`) was excluded** — different acoustic environment causes excessive false positives.

**Evaluation:**

| Dataset | Type | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Held-out test split (carp only, salted water) | Window-level | 0.857 | 0.805 | 0.830 | 0.980 |
| ACQ_5_3_2026 (fresh water, unseen) | Event-level | 0.047 | 0.703 | 0.088 | — |

> Note: The poor precision on fresh-water data is expected — the model was never exposed to fresh-water background noise. It finds ~70% of real fish events but generates massive false positives from the different electrical environment.

---

### Stage 2 — Weight Regression

| Field | Value |
|---|---|
| Name | `carp_weight_combined_model` |
| Checkpoints | `checkpoints/carp_weight_combined_model/20260430_205005/` |
| | `checkpoints/carp_weight_combined_model/20260430_205106/` |
| | `checkpoints/carp_weight_combined_model/20260430_205142/` |
| Architecture | FishModel (same as Stage 1), 54,722 params |
| Input | 39-sample window at NMS peak, 12 channels, z-normalised per recording |
| Output | Fish weight in grams (minimum clipped to 100g) |
| Trained | 2026-04-30 |
| Ensemble size | 3 runs |
| Best val loss | 9370 / 10067 / 9371 g² (MSE; early stop ~200–400 epochs) |

**Training data:**

| Source | Events | Weight range |
|---|---|---|
| `data/raw/carp/output_report.csv` | 85 | 280–1100g |
| `data/raw/carp/carp_salted/salted_water/output_report.csv` | 731 | 420–1580g |
| `data/raw/carp/carp_salted/salted_water/test/output_report.csv` | 89 | mostly 1580g |

Weight labels parsed from filename (e.g. `880gr` → 880g). Events without weight in filename skipped.  
Split: 70/20/10 stratified by weight class, seed 42.

---

## Inference Parameters (both stages)

| Parameter | Value |
|---|---|
| Window size | 39 samples |
| Stride (Stage 1) | 5 samples |
| NMS merge gap | 39 samples |
| Detection threshold | 0.5 |
| Sample rate | 40 Hz |
| Batch size | 256 |

---

## Older / Experimental Checkpoints (not deployed)

| Folder | Notes |
|---|---|
| `checkpoints/v3/` | Early bream-only model, 4 runs, March 2026 |
| `checkpoints/v4/` | Mixed bream+carp v4 architecture |
| `checkpoints/v5/` | v5 architecture experiment |
| `checkpoints/carp/` | Carp-only presence, 2 runs |
| `checkpoints/salted_water/` | Salted-water presence experiments, multiple runs |
| `checkpoints/salted_water_salt_model/` | Salt-conditioned model variants |
| `checkpoints/salted_water_weight*/` | Weight regression experiments |
| `checkpoints/s0/`, `checkpoints/s400/` | Salt-level specific models (0 ppm / 400 ppm) |
| `checkpoints/carp_weight/` | Early single carp weight model |
| `checkpoints/v3_weight/`, `checkpoints/v4_weight/` | Weight models paired with v3/v4 |
