"""
Event-level detection evaluation on salted-water carp recordings.

These are the in-domain recordings the model was trained on.
Ground truth comes from output_report.csv in each source directory.
Only is_valid==True events are used.

Outputs:
  results/event_detection_eval_salted/summary.txt
  results/event_detection_eval_salted/per_file.csv
"""

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

CKPT_DIR  = _ROOT / "checkpoints" / "bream_carp_presence"
CFG_PATH  = _ROOT / "model" / "model_config_12ch.yaml"
PLATT_PATH = CKPT_DIR / "platt.npy"
OUT_DIR   = _ROOT / "results" / "event_detection_eval_salted"

SOURCES = [
    (
        _ROOT / "data/raw/carp/carp_salted/salted_water/output_report.csv",
        _ROOT / "data/raw/carp/carp_salted/salted_water",
    ),
    (
        _ROOT / "data/raw/carp/carp_salted/salted_water/test/output_report.csv",
        _ROOT / "data/raw/carp/carp_salted/salted_water/test",
    ),
]

WINDOW_SIZE = 39
STRIDE      = 5
MERGE_GAP   = 39
THRESHOLD   = 0.5
BATCH_SIZE  = 512

SIGNAL_CHANNELS = [
    "F15","F37","F18","F32","F45","F67",
    "B15","B37","B18","B32","B45","B67",
]


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


def load_platt():
    if PLATT_PATH.exists():
        ab = np.load(str(PLATT_PATH))
        return float(ab[0]), float(ab[1])
    return 1.0, 0.0


def score_windows(wins, models, device, platt_a, platt_b):
    logits = np.zeros(len(wins), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), BATCH_SIZE):
            X = torch.from_numpy(wins[b: b + BATCH_SIZE]).to(device)
            s = None
            for m in models:
                lv, _ = m(X)
                lv = lv.cpu().numpy()
                s = lv if s is None else s + lv
            s /= len(models)
            logits[b: b + len(s)] = s
    return (1.0 / (1.0 + np.exp(-(platt_a * logits + platt_b)))).astype(np.float32)


def nms_detections(starts, probs):
    pos_idx = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    if not pos_idx:
        return []
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= MERGE_GAP:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    dets = []
    for g in groups:
        pk = g[int(np.argmax(probs[g]))]
        dets.append({
            "span_start":    int(starts[g[0]]),
            "span_end":      int(starts[g[-1]]) + WINDOW_SIZE,
            "center_sample": int(starts[pk]) + WINDOW_SIZE // 2,
            "peak_prob":     float(probs[pk]),
        })
    return dets


def run_inference(csv_path, models, device, platt_a, platt_b):
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[SIGNAL_CHANNELS].values.astype(np.float32)
    mu = np.nanmean(raw, axis=0); s = np.nanstd(raw, axis=0)
    s[(s < 1e-8) | np.isnan(s)] = 1.0
    mu = np.where(np.isnan(mu), 0.0, mu)
    norm = np.where(np.isnan((raw - mu) / s), 0.0, (raw - mu) / s).astype(np.float32)
    starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
    if not starts:
        return []
    wins  = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
    probs = score_windows(wins, models, device, platt_a, platt_b)
    return nms_detections(starts, probs)


def match_detections(gt_events, detections):
    matched_gt = set()
    tp = 0
    for det in sorted(detections, key=lambda d: -d["peak_prob"]):
        ds, de = det["span_start"], det["span_end"]
        for j, ev in enumerate(gt_events):
            if j in matched_gt:
                continue
            if ds < ev["end_index"] and de > ev["start_index"]:
                matched_gt.add(j)
                tp += 1
                break
    return tp, len(detections) - tp, len(gt_events) - tp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = load_ensemble(device)
    platt_a, platt_b = load_platt()

    # Collect all GT events and recording paths
    gt_by_file: dict[str, list] = {}
    rec_paths:  dict[str, Path] = {}

    for report_path, rec_dir in SOURCES:
        df = pd.read_csv(report_path)
        valid = df[df["is_valid"] == True]
        print(f"{report_path.parent.name}/{report_path.name}: "
              f"{len(valid)} valid / {len(df)} total events, "
              f"{valid['raw_data_file_name'].nunique()} files")
        for fname, grp in valid.groupby("raw_data_file_name"):
            evs = [{"start_index": int(r["start_index"]),
                    "end_index":   int(r["end_index"])}
                   for _, r in grp.iterrows()]
            if fname not in gt_by_file:
                gt_by_file[fname] = []
            gt_by_file[fname].extend(evs)
            # Try to locate the recording CSV
            for d in [rec_dir, rec_dir.parent / "salted_water"]:
                p = d / fname
                if p.exists():
                    rec_paths[fname] = p
                    break

    print(f"\nTotal: {sum(len(v) for v in gt_by_file.values())} valid GT events "
          f"across {len(gt_by_file)} files\n")

    rows = []
    total_tp = total_fp = total_fn = 0

    for fname in sorted(gt_by_file):
        gt_events = gt_by_file[fname]
        csv_path  = rec_paths.get(fname)
        if csv_path is None:
            print(f"  WARNING: recording not found for {fname}, skipping")
            continue

        dets = run_inference(csv_path, models, device, platt_a, platt_b)
        tp, fp, fn = match_detections(gt_events, dets)
        total_tp += tp; total_fp += fp; total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if not any(np.isnan([prec, rec])) and prec + rec > 0
                else float("nan"))

        rows.append({
            "file":      fname,
            "gt_events": len(gt_events),
            "model_det": len(dets),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 4) if not np.isnan(prec) else "",
            "recall":    round(rec,  4) if not np.isnan(rec)  else "",
            "F1":        round(f1,   4) if not np.isnan(f1)   else "",
        })

        p_str  = f"{prec:.3f}" if not np.isnan(prec) else "n/a"
        r_str  = f"{rec:.3f}"  if not np.isnan(rec)  else "n/a"
        f1_str = f"{f1:.3f}"   if not np.isnan(f1)   else "n/a"
        print(f"  {fname:35s}  GT={len(gt_events):3d}  Det={len(dets):3d}  "
              f"TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
              f"P={p_str}  R={r_str}  F1={f1_str}")

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    overall_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")
    overall_f1   = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                    if overall_prec + overall_rec > 0 else float("nan"))

    summary = (
        f"Event-level detection evaluation — salted-water carp (in-domain recordings)\n"
        f"{'=' * 65}\n"
        f"NOTE: These recordings were used during training (in-domain eval).\n"
        f"Model: bream_carp_presence ensemble (3 runs, Platt-calibrated, thr=0.5)\n"
        f"\n"
        f"Overall\n"
        f"  TP={total_tp}  FP={total_fp}  FN={total_fn}\n"
        f"  Precision : {overall_prec:.4f}\n"
        f"  Recall    : {overall_rec:.4f}\n"
        f"  F1        : {overall_f1:.4f}\n"
    )
    print("\n" + summary)

    (OUT_DIR / "summary.txt").write_text(summary)
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_file.csv", index=False)
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
