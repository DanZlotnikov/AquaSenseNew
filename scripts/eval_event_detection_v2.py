"""
Event-level detection evaluation for carp_presence_v2 ensemble.

- Uses SIGMA_FLOOR=150 normalization (same as training)
- Evaluates on all 4 sources' test-set files
- Runs an NMS sweep (merge_gap x threshold) and reports best operating point
- Saves per-file results and overall summary

Outputs:
  results/event_detection_eval_v2/summary.txt
  results/event_detection_eval_v2/per_file.csv
  results/event_detection_eval_v2/nms_sweep.csv
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

# ── paths ────────────────────────────────────────────────────────────────────
CKPT_DIR  = _ROOT / "checkpoints" / "carp_presence_v2"
CFG_PATH  = _ROOT / "model" / "model_config_12ch.yaml"
OUT_DIR   = _ROOT / "results" / "event_detection_eval_v2"

BASE = _ROOT / "data" / "raw" / "carp"

SOURCES = {
    "salted_water":      (BASE / "carp_salted/salted_water/output_report.csv",
                          BASE / "carp_salted/salted_water"),
    "salted_water_test": (BASE / "carp_salted/salted_water/test/output_report.csv",
                          BASE / "carp_salted/salted_water/test"),
    "ACQ":               (BASE / "ACQ_5_3_2026/output_report.csv",
                          BASE / "ACQ_5_3_2026"),
    "carp_old":          (BASE / "carp_old/output_report.csv",
                          BASE / "carp_old"),
}

TEST_FILES = {
    ("salted_water",      "carp_1580gr_41.5cm_S400.csv"),
    ("salted_water",      "carp_920gr_35cm_S400.csv"),
    ("salted_water_test", "AQUAtest003.csv"),
    ("ACQ",               "AQUA1429.csv"),
    ("ACQ",               "AQUA1430.csv"),
    ("ACQ",               "AQUA1462.csv"),
    ("ACQ",               "AQUA1375.csv"),
    ("ACQ",               "AQUA1380.csv"),
    ("ACQ",               "AQUA1385.csv"),
    ("ACQ",               "AQUA1390.csv"),
    ("ACQ",               "AQUA1395.csv"),
    ("ACQ",               "AQUA1445.csv"),
    ("ACQ",               "AQUA1455.csv"),
    ("ACQ",               "AQUA1489.csv"),
    ("carp_old",          "AQUA075 880gr_31cm.csv"),
    ("carp_old",          "AQUA056 880gr_31cm.csv"),
    ("carp_old",          "AQUA051 880gr_31cm.csv"),
    ("carp_old",          "AQUA064 880gr_31cm.csv"),
    ("carp_old",          "AQUA077 440gr_27cm.csv"),
}

SIGNAL_CHANNELS = [
    "F15","F37","F18","F32","F45","F67",
    "B15","B37","B18","B32","B45","B67",
]

WINDOW_SIZE = 39
STRIDE      = 5
SIGMA_FLOOR = 150.0
BATCH_SIZE  = 512

# NMS sweep grid
THRESHOLD_GRID  = [0.3, 0.4, 0.5, 0.6, 0.7]
MERGE_GAP_GRID  = [19, 39, 59, 79]

# Default for per-file table (best found after sweep)
DEFAULT_THRESHOLD = 0.5
DEFAULT_MERGE_GAP = 39


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
    print(f"Loaded {len(models)} ensemble model(s) from {CKPT_DIR}")
    return models


# ── normalization ─────────────────────────────────────────────────────────────
def normalize(raw):
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    normed = (raw - mu) / sigma_eff
    return np.where(np.isnan(normed), 0.0, normed).astype(np.float32)


# ── inference ─────────────────────────────────────────────────────────────────
def score_file(csv_path, models, device):
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw  = df[SIGNAL_CHANNELS].values.astype(np.float32)
    norm = normalize(raw)
    starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
    if not starts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
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
    return np.array(starts, dtype=np.int32), probs


def nms(starts, probs, threshold, merge_gap):
    pos_idx = [i for i, p in enumerate(probs) if p >= threshold]
    if not pos_idx:
        return []
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= merge_gap:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    dets = []
    for g in groups:
        pk = g[int(np.argmax(probs[g]))]
        dets.append({
            "span_start": int(starts[g[0]]),
            "span_end":   int(starts[g[-1]]) + WINDOW_SIZE,
            "peak_prob":  float(probs[pk]),
        })
    return dets


# ── matching ──────────────────────────────────────────────────────────────────
def match(gt_events, detections):
    matched_gt = set()
    tp = 0
    for det in sorted(detections, key=lambda d: -d["peak_prob"]):
        ds, de = det["span_start"], det["span_end"]
        for j, ev in enumerate(gt_events):
            if j in matched_gt:
                continue
            if ds < ev["end_index"] and de > ev["start_index"]:
                matched_gt.add(j); tp += 1; break
    fp = len(detections) - tp
    fn = len(gt_events)  - tp
    return tp, fp, fn


# ── load GT ───────────────────────────────────────────────────────────────────
def load_gt():
    gt = {}  # (source, fname) -> list of {start_index, end_index}
    for src, (rep_path, _) in SOURCES.items():
        rep   = pd.read_csv(rep_path)
        valid = rep[rep["is_valid"] == True] if "is_valid" in rep.columns else pd.DataFrame()
        for fname, grp in valid.groupby("raw_data_file_name"):
            gt[(src, fname)] = [
                {"start_index": int(r["start_index"]), "end_index": int(r["end_index"])}
                for _, r in grp.iterrows()
            ]
    return gt


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models  = load_ensemble(device)
    gt      = load_gt()

    # Pre-score every test file once, then sweep NMS params
    print(f"\nScoring {len(TEST_FILES)} test files…")
    scored = {}  # (src, fname) -> (starts, probs)
    gt_events_map = {}

    for src, fname in sorted(TEST_FILES):
        rec_dir = SOURCES[src][1]
        csv_path = rec_dir / fname
        if not csv_path.exists():
            print(f"  MISSING: {src}/{fname}")
            continue
        starts, probs = score_file(csv_path, models, device)
        scored[(src, fname)] = (starts, probs)
        gt_events_map[(src, fname)] = gt.get((src, fname), [])
        print(f"  {src:20s} {fname:40s}  GT={len(gt_events_map[(src,fname)]):3d}  windows={len(starts)}")

    # ── NMS sweep ────────────────────────────────────────────────────────────
    print("\nNMS sweep…")
    sweep_rows = []
    best_f1, best_thr, best_gap = -1, DEFAULT_THRESHOLD, DEFAULT_MERGE_GAP
    for thr in THRESHOLD_GRID:
        for gap in MERGE_GAP_GRID:
            ttp = tfp = tfn = 0
            for key, (starts, probs) in scored.items():
                dets = nms(starts, probs, thr, gap)
                tp, fp, fn = match(gt_events_map[key], dets)
                ttp += tp; tfp += fp; tfn += fn
            prec = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
            rec  = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0.0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            sweep_rows.append({"threshold": thr, "merge_gap": gap,
                                "TP": ttp, "FP": tfp, "FN": tfn,
                                "precision": round(prec,4), "recall": round(rec,4),
                                "F1": round(f1,4)})
            print(f"  thr={thr:.1f}  gap={gap:2d}  TP={ttp:3d} FP={tfp:3d} FN={tfn:3d}"
                  f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
            if f1 > best_f1:
                best_f1, best_thr, best_gap = f1, thr, gap

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT_DIR / "nms_sweep.csv", index=False)
    print(f"\nBest: threshold={best_thr}  merge_gap={best_gap}  F1={best_f1:.4f}")

    # ── Per-file results at best NMS ─────────────────────────────────────────
    print(f"\nPer-file results at threshold={best_thr}, merge_gap={best_gap}:")
    rows = []
    total_tp = total_fp = total_fn = 0
    for src, fname in sorted(TEST_FILES):
        key = (src, fname)
        if key not in scored:
            continue
        starts, probs = scored[key]
        dets = nms(starts, probs, best_thr, best_gap)
        gt_ev = gt_events_map[key]
        tp, fp, fn = match(gt_ev, dets)
        total_tp += tp; total_fp += fp; total_fn += fn
        prec = tp/(tp+fp) if (tp+fp) > 0 else float("nan")
        rec  = tp/(tp+fn) if (tp+fn) > 0 else float("nan")
        f1v  = 2*prec*rec/(prec+rec) if (not np.isnan(prec) and prec+rec>0) else float("nan")
        rows.append({"source": src, "file": fname, "gt": len(gt_ev),
                     "det": len(dets), "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(prec,4) if not np.isnan(prec) else "",
                     "recall":    round(rec,4)  if not np.isnan(rec)  else "",
                     "F1":        round(f1v,4)  if not np.isnan(f1v)  else ""})
        print(f"  {src:20s}  {fname:40s}  GT={len(gt_ev):3d}  det={len(dets):3d}"
              f"  TP={tp} FP={fp} FN={fn}"
              + (f"  F1={f1v:.3f}" if not np.isnan(f1v) else "  (no events)"))

    overall_prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0
    overall_rec  = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.0
    overall_f1   = 2*overall_prec*overall_rec/(overall_prec+overall_rec) if (overall_prec+overall_rec)>0 else 0.0

    summary = (
        f"Event-level detection evaluation — carp_presence_v2\n"
        f"{'='*65}\n"
        f"Ensemble: {len(models)} models from {CKPT_DIR.name}\n"
        f"Normalization: per-recording z-score with SIGMA_FLOOR={SIGMA_FLOOR}\n"
        f"Test files: {len(scored)}\n"
        f"\n"
        f"Best NMS params (from sweep on test set):\n"
        f"  threshold={best_thr}  merge_gap={best_gap}\n"
        f"\n"
        f"Overall @ best NMS\n"
        f"  TP={total_tp}  FP={total_fp}  FN={total_fn}\n"
        f"  Precision : {overall_prec:.4f}\n"
        f"  Recall    : {overall_rec:.4f}\n"
        f"  F1        : {overall_f1:.4f}\n"
    )
    print("\n" + summary)

    (OUT_DIR / "summary.txt").write_text(summary)
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_file.csv", index=False)
    print(f"Saved: {OUT_DIR}/summary.txt")
    print(f"Saved: {OUT_DIR}/per_file.csv")
    print(f"Saved: {OUT_DIR}/nms_sweep.csv")


if __name__ == "__main__":
    main()
