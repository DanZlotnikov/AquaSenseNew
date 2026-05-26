"""
Event-level detection evaluation on ACQ_5_3_2026 (unseen fresh-water carp data).

Ground truth: data/raw/carp/ACQ_5_3_2026/output_report.csv
  - Only is_valid==True events used (427 / 1823 total)
  - Columns: raw_data_file_name, start_index, end_index

Model: bream_carp_presence 3-ensemble + Platt calibration
  - Sliding window (size=39, stride=5) → NMS (threshold=0.5, merge_gap=39)
  - Detection span: [span_start, span_end)

Matching criterion: a model detection and a GT event overlap if their sample
ranges intersect.  Matching is greedy: detections sorted by peak_prob
(descending), each matched to at most one GT event.

Outputs:
  results/event_detection_eval/summary.txt
  results/event_detection_eval/per_file.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_HERE = Path(__file__).resolve().parent   # scripts/
_ROOT = _HERE.parent                      # AquaSenseNew/

sys.path.insert(0, str(_ROOT))
from model.fish_model import FishModel

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR      = _ROOT / "data" / "raw" / "carp" / "ACQ_5_3_2026"
REPORT_CSV    = DATA_DIR / "output_report.csv"
CKPT_DIR      = _ROOT / "checkpoints" / "bream_carp_presence"
CFG_PATH      = _ROOT / "model" / "model_config_12ch.yaml"
PLATT_PATH    = CKPT_DIR / "platt.npy"
OUT_DIR       = _ROOT / "results" / "event_detection_eval"

# ── inference constants ───────────────────────────────────────────────────────
WINDOW_SIZE = 39
STRIDE      = 5
MERGE_GAP   = 39
THRESHOLD   = 0.5
BATCH_SIZE  = 512

SIGNAL_CHANNELS = [
    "F15","F37","F18","F32","F45","F67",
    "B15","B37","B18","B32","B45","B67",
]


# ── model loading ─────────────────────────────────────────────────────────────
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


# ── inference ─────────────────────────────────────────────────────────────────
def score_windows(wins, models, device, platt_a, platt_b):
    logits = np.zeros(len(wins), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), BATCH_SIZE):
            X = torch.from_numpy(wins[b: b + BATCH_SIZE]).to(device)
            batch_sum = None
            for m in models:
                lv, _ = m(X)
                lv = lv.cpu().numpy()
                batch_sum = lv if batch_sum is None else batch_sum + lv
            batch_sum /= len(models)
            logits[b: b + len(batch_sum)] = batch_sum
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
            groups.append(cur)
            cur = [i]
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
    mu = np.nanmean(raw, axis=0)
    s  = np.nanstd(raw,  axis=0)
    s[(s < 1e-8) | np.isnan(s)] = 1.0
    mu = np.where(np.isnan(mu), 0.0, mu)
    norm = ((raw - mu) / s).astype(np.float32)
    norm = np.where(np.isnan(norm), 0.0, norm)

    starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
    if not starts:
        return []
    wins  = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
    probs = score_windows(wins, models, device, platt_a, platt_b)
    return nms_detections(starts, probs)


# ── matching ──────────────────────────────────────────────────────────────────
def match_detections(gt_events, detections):
    """
    Greedy matching: sort detections by peak_prob desc, match each to the
    first unmatched GT event that overlaps.

    Returns: (tp, fp, fn, matched_gt_indices)
    """
    matched_gt = set()
    tp = 0
    # Sort detections by confidence (best first)
    for det in sorted(detections, key=lambda d: -d["peak_prob"]):
        ds, de = det["span_start"], det["span_end"]
        for j, ev in enumerate(gt_events):
            if j in matched_gt:
                continue
            # Overlap check: ranges [ds,de) and [ev.start, ev.end) intersect
            if ds < ev["end_index"] and de > ev["start_index"]:
                matched_gt.add(j)
                tp += 1
                break
    fp = len(detections) - tp
    fn = len(gt_events) - tp
    return tp, fp, fn


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models         = load_ensemble(device)
    platt_a, platt_b = load_platt()
    print(f"Platt: a={platt_a:.4f}  b={platt_b:.4f}")

    report = pd.read_csv(REPORT_CSV)
    valid  = report[report["is_valid"] == True].copy()
    print(f"\nGround truth: {len(valid)} valid events across "
          f"{valid['raw_data_file_name'].nunique()} files "
          f"(out of {len(report)} total events)")

    # Group GT by filename
    gt_by_file = {}
    for fname, grp in valid.groupby("raw_data_file_name"):
        gt_by_file[fname] = [
            {"start_index": int(r["start_index"]), "end_index": int(r["end_index"])}
            for _, r in grp.iterrows()
        ]

    rows = []
    total_tp = total_fp = total_fn = 0

    all_files = sorted(DATA_DIR.glob("AQUA*.csv"))
    print(f"\nRunning inference on {len(all_files)} recording(s)…\n")

    for idx, csv_path in enumerate(all_files, 1):
        fname = csv_path.name
        gt_events = gt_by_file.get(fname, [])

        dets = run_inference(csv_path, models, device, platt_a, platt_b)
        tp, fp, fn = match_detections(gt_events, dets)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if not (np.isnan(prec) or np.isnan(rec) or prec + rec == 0)
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

        print(f"[{idx:3d}/{len(all_files)}] {fname:15s}  "
              f"GT={len(gt_events):3d}  Det={len(dets):3d}  "
              f"TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}"
              if not np.isnan(f1) else
              f"[{idx:3d}/{len(all_files)}] {fname:15s}  "
              f"GT={len(gt_events):3d}  Det={len(dets):3d}  "
              f"TP={tp:3d} FP={fp:3d} FN={fn:3d}  (no events/dets)")

    # ── overall metrics ───────────────────────────────────────────────────────
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    overall_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")
    overall_f1   = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                    if overall_prec + overall_rec > 0 else float("nan"))

    summary = (
        f"Event-level detection evaluation — ACQ_5_3_2026 (unseen fresh-water carp)\n"
        f"{'=' * 65}\n"
        f"Ground truth: {len(valid)} valid events in {valid['raw_data_file_name'].nunique()} files\n"
        f"Recordings processed: {len(all_files)}\n"
        f"\n"
        f"Overall\n"
        f"  TP={total_tp}  FP={total_fp}  FN={total_fn}\n"
        f"  Precision : {overall_prec:.4f}\n"
        f"  Recall    : {overall_rec:.4f}\n"
        f"  F1        : {overall_f1:.4f}\n"
    )
    print("\n" + summary)

    # ── save outputs ──────────────────────────────────────────────────────────
    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(summary)

    per_file_df = pd.DataFrame(rows)
    per_file_path = OUT_DIR / "per_file.csv"
    per_file_df.to_csv(per_file_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {per_file_path}")


if __name__ == "__main__":
    main()
