import pandas as pd
import numpy as np
from pathlib import Path

ACQ = Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\ACQ_5_3_2026")
TRAIN = Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water")
SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]

rep = pd.read_csv(ACQ / "output_report.csv")

print("=" * 70)
print("ROOT CAUSE INVESTIGATION: why GT=0 files generate ~200 ML detections")
print("=" * 70)

# Per-channel std for a sample of files
print("\n--- Per-file F15 std (proxy for signal richness) ---")
checks = [
    ("AQUA1375.csv",   0, "GT=0, ~202 ML"),
    ("AQUA1380.csv",   0, "GT=0, ~187 ML"),
    ("AQUA1395.csv",   0, "GT=0, ~206 ML"),
    ("AQUA1410.csv",   0, "GT=0, ~167 ML"),
    ("AQUA1411.csv",  10, "GT=10, ~16 ML"),
    ("AQUA1416.csv",  41, "GT=41, ~55 ML"),
    ("AQUA1417.csv",  39, "GT=39, ~61 ML"),
    ("AQUA1423.csv",  21, "GT=21, ~21 ML  (perfect match)"),
    ("AQUA1429.csv",   1, "GT=1,  ~86 ML"),
]
for fname, gt, note in checks:
    df = pd.read_csv(ACQ / fname)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[SIGNAL_CHANNELS].values.astype(np.float32)
    per_ch_std = raw.std(axis=0)
    mean_std = per_ch_std.mean()
    print(f"  {fname}  mean_ch_std={mean_std:>7.1f}   ({note})")

print()
print("--- Training file signal richness ---")
for fname in sorted(TRAIN.glob("carp_*.csv")):
    df = pd.read_csv(fname)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[SIGNAL_CHANNELS].values.astype(np.float32)
    mean_std = raw.std(axis=0).mean()
    rep_f = rep[rep["raw_data_file_name"] == fname.name]
    n_valid = (rep_f["is_valid"] == True).sum() if len(rep_f) else "N/A"
    print(f"  {fname.name:<35}  mean_ch_std={mean_std:>7.1f}")
