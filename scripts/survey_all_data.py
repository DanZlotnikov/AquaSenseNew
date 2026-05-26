import pandas as pd
import numpy as np
from pathlib import Path

SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]

sources = [
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water"), "salted_water"),
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\test\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\test"), "salted_water/test"),
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\ACQ_5_3_2026\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\ACQ_5_3_2026"), "ACQ_5_3_2026"),
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_old\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_old"), "carp_old"),
]

SKIP = ("output", "report", "axis", "summary", "function", "points", "updated")

total_pos = 0
total_neg_files = 0

for report_path, rec_dir, name in sources:
    rep = pd.read_csv(report_path)
    valid = rep[rep["is_valid"] == True] if "is_valid" in rep.columns else pd.DataFrame()
    all_rec_files = [f for f in rec_dir.glob("*.csv") if not any(x in f.name.lower() for x in SKIP)]
    files_with_gt = set(valid["raw_data_file_name"].unique()) if len(valid) else set()
    files_no_gt = [f for f in all_rec_files if f.name not in files_with_gt]
    n_pos = len(valid)
    total_pos += n_pos
    total_neg_files += len(files_no_gt)

    # Per-file std sample to characterise background
    stds = []
    for f in all_rec_files[:5]:
        df = pd.read_csv(f)
        for col in SIGNAL_CHANNELS:
            if col not in df.columns: df[col] = 0.0
        stds.append(df[SIGNAL_CHANNELS].values.astype(float).std(axis=0).mean())

    print(f"[{name}]")
    print(f"  Recording files     : {len(all_rec_files)}")
    print(f"  Files with GT>0     : {len(files_with_gt)}")
    print(f"  Files with GT=0     : {len(files_no_gt)}  (pure negative recordings)")
    print(f"  Valid (pos) events  : {n_pos}")
    print(f"  Sample sig_std (first 5 files): {[round(s,1) for s in stds]}")
    print()

print(f"COMBINED: {total_pos} positive events  |  {total_neg_files} zero-GT files available as pure negatives")
