"""
Event-level evaluation using find_peaks instead of NMS.

find_peaks(probs, height=threshold, distance=D) finds true local maxima
in the probability time series. D is in units of the stride (STRIDE=5 samples),
so distance=D enforces a minimum of D*5 samples between counted peaks.

No gap-accumulation problem: two high-prob bursts with a small dip between them
produce two peaks (separate events) rather than one merged group, as long as the
dip goes below the threshold. This is closer to what we want for dense salted_water
recordings.

Outputs:
  results/find_peaks_eval_v3/summary.txt
  results/find_peaks_eval_v3/distance_sweep.csv
  results/find_peaks_eval_v3/per_file.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from model.fish_model import FishModel

CKPT_DIR = _ROOT / "checkpoints" / "carp_presence_v3"
CFG_PATH = _ROOT / "model" / "model_config_12ch.yaml"
OUT_DIR  = _ROOT / "results" / "find_peaks_eval_v3"

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

# Sweep: distance in stride units → actual sample separation = D * STRIDE
# Also sweep height (threshold) for peaks
HEIGHT_GRID   = [0.3, 0.5, 0.7]
# distance values in stride units (multiply by STRIDE=5 for samples)
DISTANCE_GRID = [4, 8, 12, 16, 20, 24, 32, 40, 50, 60]
# in samples: 20, 40, 60, 80, 100, 120, 160, 200, 250, 300


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
    print(f"Loaded {len(models)} ensemble model(s) from {CKPT_DIR.name}")
    return models


def normalize(raw):
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    return np.where(np.isnan((raw-mu)/sigma_eff), 0.0, (raw-mu)/sigma_eff).astype(np.float32)


def score_file(csv_path, models, device):
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw  = df[SIGNAL_CHANNELS].values.astype(np.float32)
    norm = normalize(raw)
    starts = np.array(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE), dtype=np.int32)
    if len(starts) == 0:
        return starts, np.array([], dtype=np.float32)
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
    return starts, probs


def detect_peaks(starts, probs, height, distance):
    """
    Find peaks in the probability signal.
    Returns list of dicts with span_start, span_end, peak_prob.
    distance is in array-index units (each index = STRIDE samples).
    """
    if len(probs) == 0:
        return []
    peak_idx, props = find_peaks(probs, height=height, distance=distance)
    dets = []
    for idx in peak_idx:
        dets.append({
            "span_start": int(starts[idx]),
            "span_end":   int(starts[idx]) + WINDOW_SIZE,
            "peak_prob":  float(probs[idx]),
        })
    return dets


def match(gt_events, detections):
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
    fp = len(detections) - tp
    fn = len(gt_events) - tp
    return tp, fp, fn


def load_gt():
    gt = {}
    for src, (rep_path, _) in SOURCES.items():
        rep   = pd.read_csv(rep_path)
        valid = rep[rep["is_valid"] == True] if "is_valid" in rep.columns else pd.DataFrame()
        for fname, grp in valid.groupby("raw_data_file_name"):
            gt[(src, fname)] = [
                {"start_index": int(r["start_index"]), "end_index": int(r["end_index"])}
                for _, r in grp.iterrows()
            ]
    return gt


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models  = load_ensemble(device)
    gt      = load_gt()

    # Score all test files once
    print(f"\nScoring {len(TEST_FILES)} test files...")
    scored = {}
    gt_events_map = {}
    for src, fname in sorted(TEST_FILES):
        rec_dir  = SOURCES[src][1]
        csv_path = rec_dir / fname
        if not csv_path.exists():
            print(f"  MISSING: {src}/{fname}")
            continue
        starts, probs = score_file(csv_path, models, device)
        scored[(src, fname)] = (starts, probs)
        gt_events_map[(src, fname)] = gt.get((src, fname), [])
        print(f"  {src:20s}  {fname:40s}  GT={len(gt_events_map[(src,fname)]):3d}")

    # Sweep
    print("\nfind_peaks sweep (height x distance):")
    print(f"  {'height':>6}  {'dist':>5}  {'samp':>5}  {'TP':>4}  {'FP':>4}  {'FN':>3}"
          f"  {'P':>6}  {'R':>6}  {'F1':>6}  sw_det(GT=123)")
    sweep_rows = []
    best_f1, best_h, best_d = -1, 0.5, 16

    for height in HEIGHT_GRID:
        for dist in DISTANCE_GRID:
            ttp = tfp = tfn = 0
            sw_det = 0
            for key, (starts, probs) in scored.items():
                dets = detect_peaks(starts, probs, height, dist)
                tp, fp, fn = match(gt_events_map[key], dets)
                ttp += tp; tfp += fp; tfn += fn
                if key[0] == "salted_water":
                    sw_det += len(dets)
            prec = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
            rec  = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0.0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            samp = dist * STRIDE
            print(f"  {height:>6.1f}  {dist:>5}  {samp:>5}  {ttp:>4}  {tfp:>4}  {tfn:>3}"
                  f"  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {sw_det}")
            sweep_rows.append({"height": height, "distance_idx": dist,
                                "distance_samples": samp,
                                "TP": ttp, "FP": tfp, "FN": tfn,
                                "precision": round(prec,4), "recall": round(rec,4),
                                "F1": round(f1,4), "sw_det": sw_det})
            if f1 > best_f1:
                best_f1, best_h, best_d = f1, height, dist

    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "distance_sweep.csv", index=False)
    best_samp = best_d * STRIDE
    print(f"\nBest: height={best_h}  distance={best_d} ({best_samp} samples)  F1={best_f1:.4f}")

    # Per-file at best params
    print(f"\nPer-file results @ height={best_h}, distance={best_d} ({best_samp} samples):")
    rows = []
    total_tp = total_fp = total_fn = 0
    for src, fname in sorted(TEST_FILES):
        key = (src, fname)
        if key not in scored:
            continue
        starts, probs = scored[key]
        dets  = detect_peaks(starts, probs, best_h, best_d)
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
        print(f"  {src:20s}  {fname:38s}  GT={len(gt_ev):3d}  det={len(dets):3d}"
              f"  TP={tp} FP={fp} FN={fn}"
              + (f"  F1={f1v:.3f}" if not np.isnan(f1v) else "  (no events)"))

    overall_prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0
    overall_rec  = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.0
    overall_f1   = 2*overall_prec*overall_rec/(overall_prec+overall_rec) if (overall_prec+overall_rec)>0 else 0.0

    summary = (
        f"find_peaks evaluation — carp_presence_v3\n"
        f"{'='*60}\n"
        f"Model     : {CKPT_DIR.name} ({len(models)} ensemble members)\n"
        f"Norm      : per-recording z-score, SIGMA_FLOOR={SIGMA_FLOOR}\n"
        f"Detection : scipy.signal.find_peaks(height, distance)\n"
        f"Test files: {len(scored)}\n"
        f"\n"
        f"Best params: height={best_h}  distance={best_d} ({best_samp} samples)\n"
        f"\nOverall @ best params\n"
        f"  TP={total_tp}  FP={total_fp}  FN={total_fn}\n"
        f"  Precision : {overall_prec:.4f}\n"
        f"  Recall    : {overall_rec:.4f}\n"
        f"  F1        : {overall_f1:.4f}\n"
    )
    print("\n" + summary)
    (OUT_DIR / "summary.txt").write_text(summary)
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_file.csv", index=False)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
