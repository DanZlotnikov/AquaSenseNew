# AquaSense — Bream Evaluation

Evaluates trained length and weight models against a labelled bream dataset.
Builds a clean test NPZ from raw recordings and a ground-truth report, runs
both models, and produces per-sample predictions alongside summary metrics.

---

## Folder structure

```
haim_export/
├── checkpoints/
│   ├── v4/20260318_171244/best_model.pt          <- length model
│   └── v4_weight/20260318_171628/best_model.pt   <- weight model
├── data/
│   └── v4/
│       └── normalizer.npz                        <- training-set normalizer
├── model/
│   ├── __init__.py
│   └── fish_model.py
├── results/                                       <- output written here
├── scripts/
│   └── evaluate_custom_test.py
├── requirements.txt
└── README.md
```

---

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

---

## Input data format

Your data folder must contain **two things**:

### 1. `output_report.csv`

A detection report with one row per labelled fish event.
Required columns:

| Column | Description |
|--------|-------------|
| `raw_data_file_name` | Filename of the recording CSV the detection came from |
| `start_index` | Sample index where the detection starts |
| `end_index` | Sample index where the detection ends |
| `length_from_filename` | Ground-truth fish length in cm |
| `weight_from_filename` | Ground-truth fish weight in g |

All other columns in the report are ignored.

### 2. Recording CSV files

Place the raw recording CSVs either directly in the data folder or inside a
`recordings/` subfolder. Each CSV must contain the 12 acoustic sensor columns:

| Column | Description |
|--------|-------------|
| `F15` | Front sensor, 15 kHz band |
| `F37` | Front sensor, 37 kHz band |
| `F18` | Front sensor, 18 kHz band |
| `F32` | Front sensor, 32 kHz band |
| `F45` | Front sensor, 45 kHz band |
| `F67` | Front sensor, 67 kHz band |
| `B15` | Back sensor, 15 kHz band |
| `B37` | Back sensor, 37 kHz band |
| `B18` | Back sensor, 18 kHz band |
| `B32` | Back sensor, 32 kHz band |
| `B45` | Back sensor, 45 kHz band |
| `B67` | Back sensor, 67 kHz band |

All other columns (`IMP13`, `Distance_cm`, `Manual_button`, `Date`, `Time`, etc.)
are silently ignored and never used as model input.

### Example layout

```
my_data/
├── output_report.csv
└── recordings/
    ├── session_01.csv
    ├── session_02.csv
    └── ...
```

---

## Usage

Run from inside the `haim_export/` folder:

```bash
cd haim_export
python scripts/evaluate_custom_test.py  path/to/my_data
```

### All options

```
positional arguments:
  folder                Folder containing output_report.csv and recording CSVs

optional arguments:
  --checkpoint          Length model checkpoint
                        (default: checkpoints/v4/20260318_171244/best_model.pt)
  --weight_checkpoint   Weight model checkpoint
                        (default: checkpoints/v4_weight/20260318_171628/best_model.pt)
  --normalizer          Training-set normalizer.npz
                        (default: data/v4/normalizer.npz)
  --threshold           Presence probability threshold  (default: 0.5)
  --output              Directory to write output files (default: results/)
  --batch_size          Windows per forward pass        (default: 512)
```

### Example

```bash
python scripts/evaluate_custom_test.py  bream/
```

---

## What the script does

**Phase 1 — Build NPZ**

Reads the report and recording CSVs and constructs a test dataset:
- One 39-sample acoustic window per detection (positive sample)
- 4x as many background windows from non-detection regions (negative samples)
- Normalised using the training-set normalizer
- Saved to `results/<timestamp>/test_dataset.npz`

Only the 12 raw acoustic channels go into the model input. `Distance_cm`,
IMP columns, and all report-derived features are excluded.

**Phase 2 — Evaluate**

Runs both models over the NPZ and saves results to `results/<timestamp>/`.

---

## Output files

All output is written to a timestamped subfolder: `results/YYYYMMDD_HHMMSS/`

| File | Description |
|------|-------------|
| `test_dataset.npz` | The built test dataset (X, y_presence, y_length, y_weight) |
| `provenance.csv` | Maps each NPZ row back to its recording file and sample index |
| `build_report.txt` | Leakage audit — lists exactly what went into X and what was excluded |
| `predictions.csv` | All samples: GT and predicted length/weight per row |
| `true_positives.csv` | Only rows where GT fish present AND model predicted present |
| `model_detections.csv` | All rows where model predicted present (true + false positives) |
| `metrics.json` | MAE, RMSE, MAPE for length and weight; F1/accuracy for presence |
| `evaluation.png` | 6-panel diagnostic figure |

### `predictions.csv` columns

| Column | Description |
|--------|-------------|
| `recording_file` | Source recording CSV name |
| `sample_mid` | Centre sample index of the window |
| `is_positive` | 1 = GT fish present, 0 = background window |
| `gt_length_cm` | Ground-truth length (0 for background) |
| `pred_length_cm` | Model-predicted length |
| `gt_weight_g` | Ground-truth weight (0 for background) |
| `pred_weight_g` | Model-predicted weight |
| `presence_prob` | Model confidence (0-1) |
| `pred_present` | 1 if presence_prob >= threshold |

---

## Model details

| | Length model | Weight model |
|---|---|---|
| Input | (batch, 39, 12) | same |
| Architecture | Temporal CNN -> FC -> regression head | same |
| Validation MAE | 0.91 cm | 38.3 g |
| Validation MAPE | 3.20% | 9.15% |
| Parameters | 54,722 | 54,722 |

Both models were trained on bream recordings at 40 Hz.
Model architecture is stored inside the `.pt` file — no separate config needed.
