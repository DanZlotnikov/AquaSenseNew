"""List every recording file across all carp sources with GT count and sig_std."""
import pandas as pd
import numpy as np
from pathlib import Path

SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]
SKIP = ("output", "report", "axis", "summary", "function", "points", "updated")

SOURCES = [
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water"), "salted_water"),
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\test\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\test"), "salted_water_test"),
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\ACQ_5_3_2026\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\ACQ_5_3_2026"), "ACQ_5_3_2026"),
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_old\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_old"), "carp_old"),
]

def sig_std(path):
    df = pd.read_csv(path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns: df[col] = 0.0
    return df[SIGNAL_CHANNELS].values.astype(np.float32).std(axis=0).mean()

all_rows = []
for report_path, rec_dir, source in SOURCES:
    rep = pd.read_csv(report_path)
    valid = rep[rep["is_valid"] == True] if "is_valid" in rep.columns else pd.DataFrame()
    gt_map = valid.groupby("raw_data_file_name").size().to_dict() if len(valid) else {}
    for f in sorted(rec_dir.glob("*.csv")):
        if any(x in f.name.lower() for x in SKIP): continue
        gt = gt_map.get(f.name, 0)
        std = sig_std(f)
        all_rows.append((source, f.name, gt, round(std, 1)))

print(f"{'Source':<20} {'File':<40} {'GT':>5}  {'sig_std':>8}")
print("-" * 80)
cur_source = None
for source, fname, gt, std in all_rows:
    if source != cur_source:
        if cur_source is not None: print()
        cur_source = source
    print(f"{source:<20} {fname:<40} {gt:>5}  {std:>8.1f}")

print()
print(f"Total recordings : {len(all_rows)}")
print(f"Total GT>0 files : {sum(1 for r in all_rows if r[2]>0)}")
print(f"Total GT=0 files : {sum(1 for r in all_rows if r[2]==0)}")
print(f"Total GT events  : {sum(r[2] for r in all_rows)}")
