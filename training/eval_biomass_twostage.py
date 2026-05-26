"""Biomass evaluation of the two-stage pipeline on data/raw/carp.

Two evaluations:
  1. Full data  — all 84 recordings in data/raw/carp
  2. Test set   — recordings that contain the 9 weight-model test events
                  (same stratified split as build_carp_weight.py)

For each recording we compare:
  - True biomass   : sum of weight_from_filename for labeled events in that recording
  - Pipeline biomass: sum of weight predictions from the two-stage pipeline

Metrics reported:
  - Detection recall  : fraction of true events detected (within MATCH_RADIUS samples)
  - Detection precision: fraction of detections that match a true event
  - Weight MAE / MAPE : on matched detections only
  - Total biomass diff: (pred_total - true_total) / true_total

Run from project root:
    python training/eval_biomass_twostage.py [--threshold 0.5]
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fish_model import FishModel

# ---------------------------------------------------------------------------
WINDOW_SIZE  = 39
STRIDE       = 5
MERGE_GAP    = WINDOW_SIZE
MATCH_RADIUS = WINDOW_SIZE * 2   # samples: detection within this of true event = hit

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH = len(ACOUSTIC_CHANNELS)

PRESENCE_CFG_DIR = Path("checkpoints/bream_carp_presence")
WEIGHT_CFG_DIR   = Path("checkpoints/carp_weight_model")   # overridable via --weight_ckpt
MODEL_CFG_PATH   = Path("model/model_config_12ch.yaml")
REPORT_PATH      = Path("data/raw/carp/carp_old/output_report.csv")
RECORDINGS_DIR   = Path("data/raw/carp/carp_old")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------


def _load_models(ckpt_dir, cfg):
    models = []
    for run_dir in sorted(ckpt_dir.glob("2*")):
        pt = run_dir / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(DEVICE)
        ckpt = torch.load(pt, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        m.eval()
        models.append(m)
    return models


def load_models():
    with open(MODEL_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["physics_dim"] = 0
    pm = _load_models(PRESENCE_CFG_DIR, cfg)
    wm = _load_models(WEIGHT_CFG_DIR,   cfg)
    print(f"  Presence ensemble: {len(pm)}  Weight ensemble: {len(wm)}")
    return pm, wm


def load_and_normalise(csv_path):
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)
    m = np.nanmean(raw, axis=0).astype(np.float32)
    s = np.nanstd(raw,  axis=0).astype(np.float32)
    s[s < 1e-8] = 1.0
    mask = np.isnan(raw)
    if mask.any():
        raw = np.where(mask, m[np.newaxis, :], raw)
    return ((raw - m) / s).astype(np.float32)


def infer_presence_batch(models, wins):
    probs_list = []
    with torch.no_grad():
        for m in models:
            t = torch.from_numpy(wins).to(DEVICE)
            logits, _ = m(t)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
    return np.mean(probs_list, axis=0)


def infer_weight_single(models, win):
    t = torch.from_numpy(win[np.newaxis]).to(DEVICE)
    preds = []
    with torch.no_grad():
        for m in models:
            _, pred = m(t)
            preds.append(pred.item())
    return max(100.0, float(np.mean(preds)))


def sliding_windows(norm):
    n = len(norm)
    starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
    wins   = np.stack([norm[s: s + WINDOW_SIZE] for s in starts])
    return wins, np.array(starts)


def nms_merge(starts, probs, threshold):
    pos_idx = np.where(probs >= threshold)[0]
    if len(pos_idx) == 0:
        return []
    groups, current = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[current[-1]] <= MERGE_GAP:
            current.append(i)
        else:
            groups.append(current); current = [i]
    groups.append(current)
    dets = []
    for grp in groups:
        gp      = probs[grp]
        peak    = grp[int(np.argmax(gp))]
        ps      = starts[peak]
        dets.append({
            "center": ps + WINDOW_SIZE // 2,
            "start":  ps,
            "prob":   float(gp.max()),
        })
    return dets


def run_recording(norm, pm, wm, threshold):
    """Returns list of {center, prob, weight_pred}."""
    wins, starts = sliding_windows(norm)
    probs = infer_presence_batch(pm, wins)
    dets  = nms_merge(starts, probs, threshold)
    for d in dets:
        ps  = d["start"]
        win = norm[ps: ps + WINDOW_SIZE]
        if len(win) < WINDOW_SIZE:
            win = np.concatenate([win, np.zeros((WINDOW_SIZE - len(win), N_CH), np.float32)])
        d["weight_pred"] = infer_weight_single(wm, win)
    return dets


def match_detections(dets, true_centers):
    """
    Greedy nearest-neighbour matching.
    Returns (matched_det_indices, matched_true_indices, unmatched_det_indices, unmatched_true_indices)
    """
    matched_d, matched_t = [], []
    used_t = set()
    for di, d in enumerate(dets):
        best_dist, best_t = float("inf"), -1
        for ti, tc in enumerate(true_centers):
            if ti in used_t:
                continue
            dist = abs(d["center"] - tc)
            if dist < best_dist:
                best_dist, best_t = dist, ti
        if best_t >= 0 and best_dist <= MATCH_RADIUS:
            matched_d.append(di)
            matched_t.append(best_t)
            used_t.add(best_t)
    unmatched_d = [i for i in range(len(dets))  if i not in matched_d]
    unmatched_t = [i for i in range(len(true_centers)) if i not in matched_t]
    return matched_d, matched_t, unmatched_d, unmatched_t


def evaluate_recordings(recording_list, pm, wm, threshold, report_df):
    """
    recording_list: list of (csv_path, [list of (true_center, true_weight)])
    """
    total_true_biomass   = 0.0
    total_pred_biomass   = 0.0
    total_true_events    = 0
    total_detections     = 0
    total_tp             = 0
    total_fp             = 0
    total_fn             = 0
    weight_errors        = []

    rows = []
    for csv_path, true_events in recording_list:
        norm = load_and_normalise(csv_path)
        dets = run_recording(norm, pm, wm, threshold)

        true_centers  = [e[0] for e in true_events]
        true_weights  = [e[1] for e in true_events]
        true_biomass  = sum(true_weights)
        pred_biomass  = sum(d["weight_pred"] for d in dets)

        md, mt, ud, ut = match_detections(dets, true_centers)
        tp = len(md);  fp = len(ud);  fn = len(ut)

        for di, ti in zip(md, mt):
            weight_errors.append(abs(dets[di]["weight_pred"] - true_weights[ti]))

        total_true_biomass  += true_biomass
        total_pred_biomass  += pred_biomass
        total_true_events   += len(true_events)
        total_detections    += len(dets)
        total_tp += tp; total_fp += fp; total_fn += fn

        rows.append({
            "file":         csv_path.name,
            "true_events":  len(true_events),
            "detections":   len(dets),
            "tp": tp, "fp": fp, "fn": fn,
            "true_biomass": true_biomass,
            "pred_biomass": pred_biomass,
        })

    return {
        "rows":               rows,
        "total_true_biomass": total_true_biomass,
        "total_pred_biomass": total_pred_biomass,
        "total_true_events":  total_true_events,
        "total_detections":   total_detections,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "weight_errors":      weight_errors,
    }


def print_results(label, res):
    tp = res["tp"]; fp = res["fp"]; fn = res["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    errs = res["weight_errors"]
    mae  = np.mean(errs) if errs else float("nan")

    tb = res["total_true_biomass"]
    pb = res["total_pred_biomass"]
    bio_diff = (pb - tb) / tb * 100 if tb > 0 else float("nan")

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Recordings evaluated : {len(res['rows'])}")
    print(f"  True events          : {res['total_true_events']}")
    print(f"  Pipeline detections  : {res['total_detections']}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Weight MAE (matched) : {mae:.0f} g" if not np.isnan(mae) else "  Weight MAE: n/a")
    print(f"  True total biomass   : {tb:>10.0f} g  ({tb/1000:.3f} kg)")
    print(f"  Pred total biomass   : {pb:>10.0f} g  ({pb/1000:.3f} kg)")
    print(f"  Biomass diff         : {bio_diff:+.1f}%")

    # Per-recording detail (recordings with events only)
    active = [r for r in res["rows"] if r["true_events"] > 0 or r["detections"] > 0]
    if active:
        print(f"\n  {'File':<32} {'True':>5} {'Det':>4} {'TP':>3} {'FP':>3} {'FN':>3} "
              f"{'TrueW':>7} {'PredW':>7} {'Diff':>7}")
        print("  " + "-" * 78)
        for r in active:
            diff_pct = (r["pred_biomass"] - r["true_biomass"]) / r["true_biomass"] * 100 \
                       if r["true_biomass"] > 0 else float("nan")
            diff_str = f"{diff_pct:+.0f}%" if not np.isnan(diff_pct) else "   n/a"
            print(f"  {r['file']:<32} {r['true_events']:>5} {r['detections']:>4} "
                  f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
                  f"{r['true_biomass']:>7.0f} {r['pred_biomass']:>7.0f} {diff_str:>7}")


def _parse_weight_g(fname):
    m = re.search(r"(\d+)gr", fname)
    return float(m.group(1)) if m else None


def get_test_recording_set():
    """
    Reproduce the build_carp_weight.py split and return the set of (fname, event_row_index)
    for the 9 test events.
    """
    report = pd.read_csv(REPORT_PATH)
    waves_meta = []   # (fname, start_idx, end_idx, weight)
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        rec   = RECORDINGS_DIR / fname
        w     = _parse_weight_g(fname)
        if w is None or not rec.exists():
            continue
        waves_meta.append((fname, int(row["start_index"]), int(row["end_index"]), w))

    n      = len(waves_meta)
    classes = [str(int(m[3])) for m in waves_meta]
    # merge rare classes (<3) same as build script
    from collections import Counter
    counts = Counter(classes)
    all_vals = sorted(set(int(c) for c in classes))
    merged = list(classes)
    for i, c in enumerate(merged):
        if counts[c] < 3:
            idx   = all_vals.index(int(c))
            nbr   = all_vals[idx - 1] if idx > 0 else all_vals[idx + 1]
            merged[i] = str(nbr)
    merged = np.array(merged)

    sss1 = StratifiedShuffleSplit(1, test_size=0.10, random_state=42)
    _, test_idx = next(sss1.split(np.zeros(n), merged))

    return [waves_meta[i] for i in test_idx]   # list of (fname, start, end, weight)


def main():
    global WEIGHT_CFG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight_ckpt", type=str, default=None,
                        help="Override weight model checkpoint dir")
    args = parser.parse_args()
    thr = args.threshold
    if args.weight_ckpt:
        WEIGHT_CFG_DIR = Path(args.weight_ckpt)

    print("=" * 60)
    print("Two-stage biomass evaluation — data/raw/carp")
    print(f"Threshold: {thr}")
    print("=" * 60)

    print("\nLoading models ...")
    pm, wm = load_models()

    report = pd.read_csv(REPORT_PATH)
    all_csvs = sorted([p for p in RECORDINGS_DIR.glob("*.csv")
                       if "output" not in p.name.lower()])

    # Build per-recording truth from output_report
    truth: dict[str, list] = defaultdict(list)
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        rec   = RECORDINGS_DIR / fname
        if not rec.exists():
            continue
        center = int((row["start_index"] + row["end_index"]) // 2)
        w      = float(row["weight_from_filename"])
        truth[fname].append((center, w))

    # -----------------------------------------------------------------------
    # 1. FULL DATA EVALUATION — all 84 recordings
    # -----------------------------------------------------------------------
    full_list = []
    for csv_path in all_csvs:
        full_list.append((csv_path, truth.get(csv_path.name, [])))

    print(f"\nRunning full-data evaluation ({len(full_list)} recordings) ...")
    full_res = evaluate_recordings(full_list, pm, wm, thr, report)
    print_results(f"FULL DATA  (thr={thr})", full_res)

    # -----------------------------------------------------------------------
    # 2. TEST SET EVALUATION — recordings containing the 9 held-out test events
    # -----------------------------------------------------------------------
    test_events = get_test_recording_set()   # (fname, start, end, weight)
    test_fnames = sorted(set(e[0] for e in test_events))

    print(f"\nTest-set recordings ({len(test_fnames)} files, {len(test_events)} events):")
    for fname in test_fnames:
        events_here = [(e[1], e[2], e[3]) for e in test_events if e[0] == fname]
        print(f"  {fname}  — {len(events_here)} test event(s): "
              f"{[int(e[2]) for e in events_here]}g")

    test_list = []
    for fname in test_fnames:
        csv_path   = RECORDINGS_DIR / fname
        # true events for this recording (ALL events, not just test-split ones,
        # since the pipeline sees the full recording)
        test_list.append((csv_path, truth.get(fname, [])))

    print(f"\nRunning test-set evaluation ({len(test_list)} recordings) ...")
    test_res = evaluate_recordings(test_list, pm, wm, thr, report)
    print_results(f"TEST SET   (thr={thr})", test_res)

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("  SUMMARY")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<30} {'Full data':>12} {'Test set':>12}")
    print("  " + "-" * 56)
    for label, fv, tv in [
        ("True total biomass (g)",
         full_res["total_true_biomass"], test_res["total_true_biomass"]),
        ("Pred total biomass (g)",
         full_res["total_pred_biomass"], test_res["total_pred_biomass"]),
        ("Biomass diff (%)",
         (full_res["total_pred_biomass"] - full_res["total_true_biomass"])
          / full_res["total_true_biomass"] * 100,
         (test_res["total_pred_biomass"] - test_res["total_true_biomass"])
          / test_res["total_true_biomass"] * 100 if test_res["total_true_biomass"] > 0 else float("nan")),
        ("True events",  full_res["total_true_events"], test_res["total_true_events"]),
        ("Detections",   full_res["total_detections"],  test_res["total_detections"]),
        ("Precision",
         full_res["tp"] / max(1, full_res["tp"] + full_res["fp"]),
         test_res["tp"]  / max(1, test_res["tp"]  + test_res["fp"])),
        ("Recall",
         full_res["tp"] / max(1, full_res["tp"] + full_res["fn"]),
         test_res["tp"]  / max(1, test_res["tp"]  + test_res["fn"])),
    ]:
        if isinstance(fv, float):
            print(f"  {label:<30} {fv:>12.1f} {tv:>12.1f}")
        else:
            print(f"  {label:<30} {int(fv):>12d} {int(tv):>12d}")


if __name__ == "__main__":
    main()
