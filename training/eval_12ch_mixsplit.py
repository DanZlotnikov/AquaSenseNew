"""Evaluate the 12-channel mixsplit ensemble on the test NPZ and full CSV files.

Prints:
  - NPZ test metrics (F1, MAPE, MAE, accuracy)
  - Full-file biomass per recording and total diff vs ground truth

Run from project root:
    python training/eval_12ch_mixsplit.py --ckpt_dir checkpoints/mixsplit_12ch/
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fish_model import FishModel
from utils.metrics import compute_all_metrics

WINDOW_SIZE = 39
STRIDE = 10
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
DATA_DIR    = Path("data/carp/salted_water")
REPORT_PATH = Path("data/carp/salted_water/output_report.csv")
TEST_NPZ    = Path("data/carp/salted_water_mixsplit/test_salt.npz")


def _load_ensemble(ckpt_dir: Path, device):
    runs = sorted(ckpt_dir.iterdir())
    models, log_scales = [], []
    for run in runs:
        ckpt_path = run / "best_model.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = FishModel(ckpt["cfg_model"]).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        log_scales.append(ckpt.get("log_scale_weight", False))
        print(f"  loaded {ckpt_path}  (epoch={ckpt['epoch']+1}  val_f1={ckpt['val_f1']:.3f})")
    return models, log_scales


def _infer_npz(models, log_scales, npz_path, device, batch_size=256):
    d   = np.load(npz_path)
    X   = d["X"].astype(np.float32)
    yp  = d["y_presence"].astype(np.float32)
    yw  = d["y_weight"].astype(np.float32)

    all_logits = np.zeros(len(X), np.float32)
    all_weights = np.zeros(len(X), np.float32)

    for b in range(0, len(X), batch_size):
        Xb = torch.tensor(X[b: b + batch_size]).to(device)
        logit_sum = weight_sum = None
        with torch.no_grad():
            for model, log_scale in zip(models, log_scales):
                lp, pw = model(Xb)
                lv = lp.cpu().numpy()
                pv = pw.cpu().numpy()
                if log_scale:
                    pv = np.exp(pv)
                logit_sum  = lv if logit_sum  is None else logit_sum  + lv
                weight_sum = pv if weight_sum is None else weight_sum + pv
        all_logits[b: b + batch_size]  = logit_sum  / len(models)
        all_weights[b: b + batch_size] = weight_sum / len(models)

    return all_logits, all_weights, yp, yw


def _dedup(positions, probs, weights, gap=WINDOW_SIZE):
    if len(positions) == 0:
        return np.array([]), np.array([]), 0
    pos = np.array(positions)
    order = np.argsort(pos)
    pos = pos[order]; probs = probs[order]; weights = weights[order]
    ev_w, ev_p = [], []
    gs = pos[0]; gp, gw = [probs[0]], [weights[0]]
    for i in range(1, len(pos)):
        if pos[i] - gs <= gap:
            gp.append(probs[i]); gw.append(weights[i])
        else:
            b = int(np.argmax(gp))
            ev_w.append(gw[b]); ev_p.append(gp[b])
            gs = pos[i]; gp = [probs[i]]; gw = [weights[i]]
    b = int(np.argmax(gp))
    ev_w.append(gw[b]); ev_p.append(gp[b])
    return np.array(ev_w), np.array(ev_p), len(ev_w)


def _full_file_biomass(models, log_scales, device, threshold, batch_size=256):
    report = pd.read_csv(REPORT_PATH)

    def parse_weight(fname):
        m = re.search(r"(\d+)gr", fname)
        return float(m.group(1)) if m else 0.0

    csv_files = sorted([p for p in DATA_DIR.glob("*.csv") if "output" not in p.name.lower()])
    rows = []
    for csv_path in csv_files:
        fname      = csv_path.name
        true_weight = parse_weight(fname)
        n_events    = len(report[report["raw_data_file_name"] == fname])
        true_bio    = n_events * true_weight

        df = pd.read_csv(csv_path)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)
        mn  = np.nanmean(raw, axis=0).astype(np.float32)
        st  = np.nanstd(raw,  axis=0).astype(np.float32)
        st[st < 1e-8] = 1.0
        raw_norm  = (raw - mn) / st
        positions = list(range(0, max(0, len(raw_norm) - WINDOW_SIZE + 1), STRIDE))
        if not positions:
            rows.append({"file": fname, "true_bio": true_bio, "pred_bio": 0.0,
                         "n_events": n_events, "n_det": 0})
            continue
        windows = np.stack([raw_norm[p: p + WINDOW_SIZE].astype(np.float32) for p in positions])

        all_logits = all_wts = None
        for b in range(0, len(windows), batch_size):
            Xb = torch.tensor(windows[b: b + batch_size]).to(device)
            ls = ws = None
            with torch.no_grad():
                for model, log_scale in zip(models, log_scales):
                    lp, pw = model(Xb)
                    lv = lp.cpu().numpy(); pv = pw.cpu().numpy()
                    if log_scale: pv = np.exp(pv)
                    ls = lv if ls is None else ls + lv
                    ws = pv if ws is None else ws + pv
            ls /= len(models); ws /= len(models)
            all_logits = ls if all_logits is None else np.concatenate([all_logits, ls])
            all_wts    = ws if all_wts    is None else np.concatenate([all_wts,    ws])

        probs    = 1.0 / (1.0 + np.exp(-all_logits))
        det_mask = probs >= threshold
        ev_w, _, n_det = _dedup(
            [positions[i] for i in range(len(positions)) if det_mask[i]],
            probs[det_mask], all_wts[det_mask],
        )
        pred_bio = float(ev_w.sum()) if len(ev_w) > 0 else 0.0
        rows.append({"file": fname, "true_bio": true_bio, "pred_bio": pred_bio,
                     "n_events": n_events, "n_det": n_det,
                     "avg_pred_w": float(ev_w.mean()) if len(ev_w) > 0 else 0.0})

    return rows


def main(ckpt_dir: Path, threshold: float = 0.50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading ensemble from {ckpt_dir}")
    models, log_scales = _load_ensemble(ckpt_dir, device)
    print(f"Ensemble size: {len(models)}")

    # ── NPZ test metrics ───────────────────────────────────────────────────
    print(f"\n--- NPZ test set ({TEST_NPZ}) ---")
    logits, weights, yp, yw = _infer_npz(models, log_scales, TEST_NPZ, device)
    metrics = compute_all_metrics(logits, weights, yp, yw)
    print(f"  F1        : {metrics['f1']:.3f}")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  Accuracy  : {metrics['accuracy']:.3f}")
    print(f"  MAPE      : {metrics['mape']:.2f}%")
    print(f"  MAE       : {metrics['mae']:.1f} g")

    # NPZ biomass diff
    probs   = 1.0 / (1.0 + np.exp(-logits))
    pos_pred = probs >= threshold
    pred_biomass_npz = float(weights[pos_pred & (yp == 1)].sum()) if (pos_pred & (yp == 1)).any() else 0.0
    true_biomass_npz = float(yw[yp == 1].sum())
    # Use all detected windows (TP+FP) vs all true windows
    pred_bio_all = float(weights[pos_pred].sum())
    diff_npz = 100 * (pred_bio_all - true_biomass_npz) / true_biomass_npz if true_biomass_npz > 0 else float("nan")
    print(f"\n  NPZ biomass (thr={threshold}):")
    print(f"    predicted (all detections) : {pred_bio_all:,.0f} g")
    print(f"    true (all positive windows): {true_biomass_npz:,.0f} g")
    print(f"    diff                       : {diff_npz:+.1f}%")

    # ── Full-file biomass ──────────────────────────────────────────────────
    print(f"\n--- Full-file biomass (thr={threshold}) ---")
    rows = _full_file_biomass(models, log_scales, device, threshold)
    print(f"{'recording':<42s}  true_bio   n_ev  pred_bio   n_det  diff%")
    print("-" * 95)
    total_true = total_pred = 0
    for r in rows:
        diff = 100 * (r["pred_bio"] - r["true_bio"]) / r["true_bio"] if r["true_bio"] > 0 else float("nan")
        print(f"{r['file']:<42s}  {r['true_bio']:8.0f}g  {r['n_events']:4d}  "
              f"{r['pred_bio']:8.0f}g  {r['n_det']:5d}  {diff:+6.1f}%")
        total_true += r["true_bio"]
        total_pred += r["pred_bio"]
    total_diff = 100 * (total_pred - total_true) / total_true
    print("-" * 95)
    print(f"{'TOTAL':<42s}  {total_true:8.0f}g        {total_pred:8.0f}g         {total_diff:+6.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",  default="checkpoints/mixsplit_12ch/")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    main(Path(args.ckpt_dir), args.threshold)
