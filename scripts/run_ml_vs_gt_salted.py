"""Run ML inference on salted_water recordings and compare against GT."""
import sys, json
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

sys.path.insert(0, r"C:\Users\Admin\repos\AquaSenseCloudApp\backend\model")
from fish_model import FishModel

WINDOW_SIZE = 39; STRIDE = 5; MERGE_GAP = 39; THRESHOLD = 0.5
SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]

CKPT_DIR = Path(r"C:\Users\Admin\repos\AquaSenseNew\checkpoints\carp_presence")
CFG_PATH = Path(r"C:\Users\Admin\repos\AquaSenseCloudApp\backend\model\model_config_12ch.yaml")

# Both report files
SOURCES = [
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water")),
    (Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\test\output_report.csv"),
     Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\test")),
]

_SKIP = ("output", "report", "axis", "summary", "function", "points", "updated")

# Load models
cfg = yaml.safe_load(open(CFG_PATH)); cfg.setdefault("physics_dim", 0)
device = torch.device("cpu")
models = []
for run in sorted(CKPT_DIR.glob("2*")):
    pt = run / "best_model.pt"
    if not pt.exists(): continue
    m = FishModel(cfg).to(device)
    ck = torch.load(pt, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"]); m.eval(); models.append(m)
platt = np.load(str(CKPT_DIR / "platt.npy")); pa, pb = float(platt[0]), float(platt[1])
print(f"Loaded {len(models)} models  |  Platt a={pa}, b={pb}")

def score_windows(wins):
    logits = np.zeros(len(wins), np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), 256):
            X = torch.from_numpy(wins[b:b+256]).to(device)
            s = sum(m(X)[0].cpu().numpy() for m in models) / len(models)
            logits[b:b+len(s)] = s
    return (1 / (1 + np.exp(-(pa * logits + pb)))).astype(np.float32)

def nms_count(starts, probs):
    pos = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    if not pos: return 0
    groups, cur = [], [pos[0]]
    for i in pos[1:]:
        if starts[i] - starts[cur[-1]] <= MERGE_GAP: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    return len(groups)

def infer_file(path):
    df = pd.read_csv(path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns: df[col] = 0.0
    raw = df[SIGNAL_CHANNELS].values.astype(np.float32)
    mu = np.nanmean(raw, axis=0); s = np.nanstd(raw, axis=0)
    s[(s < 1e-8) | np.isnan(s)] = 1.0; mu = np.where(np.isnan(mu), 0, mu)
    norm = np.where(np.isnan((raw - mu) / s), 0, (raw - mu) / s).astype(np.float32)
    starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
    if not starts: return 0, raw.std(axis=0).mean()
    wins = np.stack([norm[i:i+WINDOW_SIZE] for i in starts])
    probs = score_windows(wins)
    return nms_count(starts, probs), raw.std(axis=0).mean()

# Build GT from both reports
gt = {}
for report_path, rec_dir in SOURCES:
    if not report_path.exists():
        print(f"WARNING: {report_path} not found"); continue
    rep = pd.read_csv(report_path)
    valid = rep[rep["is_valid"] == True]
    for fname, cnt in valid.groupby("raw_data_file_name").size().items():
        gt[fname] = gt.get(fname, 0) + cnt

# Run inference on all recording files across both dirs
print("\nRunning inference...")
ml_counts = {}
signal_stds = {}
seen = set()
for _, rec_dir in SOURCES:
    for f in sorted(rec_dir.glob("*.csv")):
        if f.name in seen: continue
        if any(x in f.name.lower() for x in _SKIP): continue
        seen.add(f.name)
        ml_n, sig_std = infer_file(f)
        ml_counts[f.name] = ml_n
        signal_stds[f.name] = sig_std
        gt_n = gt.get(f.name, 0)
        print(f"  {f.name:<38} GT={gt_n:>4}  ML={ml_n:>4}  sig_std={sig_std:>6.1f}")

# Build comparison
all_files = sorted(set(list(ml_counts.keys()) + list(gt.keys())))
rows = []
for fname in all_files:
    ml_n = ml_counts.get(fname, 0)
    gt_n = gt.get(fname, 0)
    diff = ml_n - gt_n
    pct  = (diff / gt_n * 100) if gt_n > 0 else (float("inf") if ml_n > 0 else 0.0)
    rows.append((fname, gt_n, ml_n, diff, pct, signal_stds.get(fname, 0)))

total_gt   = sum(r[1] for r in rows)
total_ml   = sum(r[2] for r in rows)
total_diff = total_ml - total_gt
overall_pct = total_diff / total_gt * 100 if total_gt else 0

# Write output file
out_lines = [
    "ML vs Ground Truth -- salted_water recordings",
    "Detection model: carp-only ensemble (20260517_*), threshold=0.50",
    "GT source: output_report.csv, is_valid==True",
    "=" * 82,
    f"  {'File':<36}  {'GT':>5}  {'ML':>5}  {'Diff':>6}  {'Diff %':>8}  {'sig_std':>8}",
    "-" * 82,
]
for fname, gt_n, ml_n, diff, pct, sig_std in rows:
    pct_str = f"{pct:+7.1f}%" if pct != float("inf") else "   inf %"
    out_lines.append(f"  {fname:<36}  {gt_n:>5}  {ml_n:>5}  {diff:>+6}  {pct_str}  {sig_std:>8.1f}")

out_lines += [
    "-" * 82,
    f"  {'TOTAL':<36}  {total_gt:>5}  {total_ml:>5}  {total_diff:>+6}  {overall_pct:+.1f}%",
    "=" * 82,
    "",
    f"Files: {len(rows)}",
    f"Files with GT>0 : {sum(1 for r in rows if r[1]>0)}",
    f"Files with ML>0 : {sum(1 for r in rows if r[2]>0)}",
    f"GT=0, ML>0 (FP files): {sum(1 for r in rows if r[1]==0 and r[2]>0)}",
    f"GT>0, ML=0 (missed):   {sum(1 for r in rows if r[1]>0 and r[2]==0)}",
]

out_text = "\n".join(out_lines)
out_path = Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\carp_salted\salted_water\ml_vs_gt_comparison.txt")
out_path.write_text(out_text, encoding="utf-8")
print()
print(out_text)
print(f"\nSaved to {out_path}")
