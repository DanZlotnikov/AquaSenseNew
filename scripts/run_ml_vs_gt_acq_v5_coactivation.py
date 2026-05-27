"""ML vs GT comparison on ACQ_5_3_2026 — carp_presence_v5_coactivation ensemble."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from model.fish_model import FishModel

DATA_DIR   = _ROOT / "data" / "raw" / "carp" / "ACQ_5_3_2026"
REPORT_CSV = DATA_DIR / "output_report.csv"
CKPT_DIR   = _ROOT / "checkpoints" / "carp_presence_v5_coactivation"
CFG_PATH   = _ROOT / "model" / "model_config_v5_coact.yaml"
OUT_TXT    = DATA_DIR / "ml_vs_gt_comparison_v5_coactivation.txt"

SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]
SKIP  = ("output","report","axis","summary","function","points","updated")

WINDOW_SIZE = 39
STRIDE      = 5
SIGMA_FLOOR = 150.0
THRESHOLD   = 0.5
MERGE_GAP   = 79
BATCH_SIZE  = 512


def load_ensemble(device):
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(CKPT_DIR.glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(device)
        ck = torch.load(pt, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} ensemble model(s).")
    return models


def normalize(raw):
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    out = (raw - mu) / sigma_eff
    return np.where(np.isnan(out), 0.0, out).astype(np.float32)


def infer(csv_path, models, device):
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw    = df[SIGNAL_CHANNELS].values.astype(np.float32)
    norm   = normalize(raw)
    starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
    if not starts:
        return 0
    wins = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
    logits = np.zeros(len(wins), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), BATCH_SIZE):
            X = torch.from_numpy(wins[b: b + BATCH_SIZE]).to(device)
            s = None
            for m in models:
                lv, _ = m(X)
                lv = lv.cpu().numpy()
                s  = lv if s is None else s + lv
            s /= len(models)
            logits[b: b + len(s)] = s
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    pos_idx = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    if not pos_idx:
        return 0
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= MERGE_GAP:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    return len(groups)


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models  = load_ensemble(device)

    report  = pd.read_csv(REPORT_CSV)
    valid   = report[report["is_valid"] == True] if "is_valid" in report.columns else pd.DataFrame()
    gt_map  = valid.groupby("raw_data_file_name").size().to_dict() if len(valid) else {}

    all_files = sorted(f for f in DATA_DIR.glob("*.csv")
                       if not any(x in f.name.lower() for x in SKIP))

    rows = []
    for idx, f in enumerate(all_files, 1):
        gt_n = gt_map.get(f.name, 0)
        ml_n = infer(f, models, device)
        diff = ml_n - gt_n
        pct  = (diff / gt_n * 100) if gt_n > 0 else float("inf") if ml_n > 0 else 0.0
        rows.append((f.name, gt_n, ml_n, diff, pct))
        pct_str = f"{pct:+.1f}%" if pct != float("inf") else "inf%"
        print(f"[{idx:3d}/{len(all_files)}] {f.name:<25}  GT={gt_n:3d}  ML={ml_n:3d}  {diff:+4d}  {pct_str}")

    total_gt   = sum(r[1] for r in rows)
    total_ml   = sum(r[2] for r in rows)
    total_diff = total_ml - total_gt
    overall_pct = (total_diff / total_gt * 100) if total_gt > 0 else 0.0

    lines = [
        "ML vs Ground Truth — ACQ_5_3_2026",
        f"Detection model : carp_presence_v5_coactivation ensemble",
        f"Normalization   : per-recording z-score, SIGMA_FLOOR={SIGMA_FLOOR}",
        f"Architecture    : 12-ch temporal CNN + additive co-activation gate on min(F_peak, B_peak)",
        f"NMS             : threshold={THRESHOLD}, merge_gap={MERGE_GAP}",
        "GT source       : output_report.csv, is_valid==True",
        "=" * 72,
        f"  {'File':<25}  {'GT':>5}  {'ML':>5}  {'Diff':>6}  {'Diff%':>8}",
        "-" * 72,
    ]
    for fname, gt_n, ml_n, diff, pct in rows:
        pct_str = f"{pct:+7.1f}%" if pct != float("inf") else "     inf%"
        lines.append(f"  {fname:<25}  {gt_n:>5}  {ml_n:>5}  {diff:>+6}  {pct_str}")
    lines.append("-" * 72)
    lines.append(f"  {'TOTAL':<25}  {total_gt:>5}  {total_ml:>5}  {total_diff:>+6}  {overall_pct:+.1f}%")
    lines.append("=" * 72)
    lines += [
        "",
        f"Files processed : {len(rows)}",
        f"Files GT>0      : {sum(1 for r in rows if r[1]>0)}",
        f"Files GT=0,ML=0 : {sum(1 for r in rows if r[1]==0 and r[2]==0)}  (clean negatives)",
        f"Files GT=0,ML>0 : {sum(1 for r in rows if r[1]==0 and r[2]>0)}  (false positives)",
        f"Files GT>0,ML=0 : {sum(1 for r in rows if r[1]>0 and r[2]==0)}  (missed)",
    ]

    out = "\n".join(lines)
    OUT_TXT.write_text(out, encoding="utf-8")
    print("\n" + out)
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
