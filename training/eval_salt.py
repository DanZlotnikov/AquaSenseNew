"""Evaluate FishSaltModel (single or ensemble) on val and test splits.

Usage (from project root):
    # Auto-find latest single checkpoint
    python training/eval_salt.py

    # Explicit single checkpoint
    python training/eval_salt.py --ckpt checkpoints/salted_water_salt_model/<run>/best_model.pt

    # Ensemble: average logits across multiple checkpoints
    python training/eval_salt.py \\
        --ckpt checkpoints/salted_water_salt_model/run1/best_model.pt \\
               checkpoints/salted_water_salt_model/run2/best_model.pt \\
               checkpoints/salted_water_salt_model/run3/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fish_dataset_salt import make_salt_dataloader
from model.fish_salt_model import FishSaltModel
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_ensemble(models, log_scales, loader, device):
    """Average logits across all ensemble members; use first member's weight head."""
    for m in models:
        m.eval()

    all_logits, all_pred_w, all_y_pres, all_y_weight = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            X        = batch["X"].to(device)
            X_phys   = batch["X_phys"].to(device)
            y_pres   = batch["presence"]
            y_weight = batch["weight"]

            avg_logit  = None
            avg_weight = None

            for model, log_scale in zip(models, log_scales):
                logit_pres, pred_weight, _ = model(X, X_phys)
                lv = logit_pres.cpu().numpy()
                pw = pred_weight.cpu().numpy()
                if log_scale:
                    pw = np.exp(pw)

                avg_logit  = lv if avg_logit  is None else avg_logit  + lv
                avg_weight = pw if avg_weight is None else avg_weight + pw

            avg_logit  /= len(models)
            avg_weight /= len(models)

            yw = y_weight.numpy()
            # Use first model's log_scale for round-trip of ground truth
            if log_scales[0]:
                yw = np.exp(np.log(yw.clip(1.0)))

            all_logits.append(avg_logit)
            all_pred_w.append(avg_weight)
            all_y_pres.append(y_pres.numpy())
            all_y_weight.append(yw)

    return (
        np.concatenate(all_logits),
        np.concatenate(all_pred_w),
        np.concatenate(all_y_pres),
        np.concatenate(all_y_weight),
    )


def _biomass(pred_w, y_pres, threshold, logits):
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Hard biomass: sum weights where prob >= threshold
    detected = probs >= threshold
    hard_pred  = float(pred_w[detected].sum())
    hard_true  = float(pred_w[y_pres.astype(bool)].sum())   # uses true weights for reference
    # Actually hard_true should use actual y_weight of true positives
    # We use y_weight (ground truth weight) for the true total
    return hard_pred, hard_true, probs


def _biomass_stats(pred_w, y_weight, y_pres, logits, threshold):
    probs     = 1.0 / (1.0 + np.exp(-logits))
    mask_pred = probs >= threshold
    mask_true = y_pres.astype(bool)

    hard_pred = float(pred_w[mask_pred].sum())
    hard_true = float(y_weight[mask_true].sum())
    hard_diff = (hard_pred - hard_true) / hard_true * 100.0 if hard_true > 0 else 0.0

    soft_pred = float((pred_w * probs).sum())
    soft_diff = (soft_pred - hard_true) / hard_true * 100.0 if hard_true > 0 else 0.0

    tp = int((mask_pred & mask_true).sum())
    fp = int((mask_pred & ~mask_true).sum())
    fn = int((~mask_pred & mask_true).sum())
    tn = int((~mask_pred & ~mask_true).sum())
    acc  = (tp + tn) / len(y_pres)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Weight metrics (fish-present only)
    if mask_true.sum() > 0:
        errors = np.abs(pred_w[mask_true] - y_weight[mask_true])
        mae  = float(errors.mean())
        rmse = float(np.sqrt((errors ** 2).mean()))
        mape = float((errors / y_weight[mask_true].clip(1.0)).mean()) * 100.0
    else:
        mae = rmse = mape = 0.0

    return {
        "threshold":  threshold,
        "n_samples":  len(y_pres),
        "n_pos":      int(mask_true.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "mae":        mae,
        "rmse":       rmse,
        "mape":       mape,
        "hard_pred":  hard_pred,
        "hard_true":  hard_true,
        "hard_diff":  hard_diff,
        "soft_pred":  soft_pred,
        "soft_diff":  soft_diff,
    }


def _print_stats(label, s):
    print(f"\n{'='*60}")
    print(f"  {label}  (n={s['n_samples']}, pos={s['n_pos']}, threshold={s['threshold']:.2f})")
    print(f"{'='*60}")
    print(f"  Presence | Acc={s['accuracy']:.4f}  Prec={s['precision']:.4f}  "
          f"Rec={s['recall']:.4f}  F1={s['f1']:.4f}")
    print(f"           | TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  TN={s['tn']}")
    print(f"  Weight   | MAE={s['mae']:.1f}g  RMSE={s['rmse']:.1f}g  MAPE={s['mape']:.1f}%")
    print(f"  Hard Bio | True={s['hard_true']:.0f}g  Pred={s['hard_pred']:.0f}g  "
          f"Diff={s['hard_diff']:+.2f}%")
    print(f"  Soft Bio | Pred={s['soft_pred']:.0f}g  "
          f"Diff={s['soft_diff']:+.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, nargs="+",
                        help="One or more checkpoint paths (ensemble). "
                             "Auto-finds the 3 latest runs if omitted.")
    parser.add_argument("--val_file",   default="data/carp/salted_water_split/val_salt.npz")
    parser.add_argument("--test_file",  default="data/carp/salted_water_split/test_salt.npz")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint(s) ---
    if args.ckpt:
        ckpt_paths = [Path(p) for p in args.ckpt]
    else:
        all_runs = sorted(Path("checkpoints/salted_water_salt_model").glob("*/best_model.pt"))
        if not all_runs:
            print("No checkpoint found under checkpoints/salted_water_salt_model/")
            sys.exit(1)
        # Use the 3 most recent runs (the ensemble members just trained)
        ckpt_paths = all_runs[-3:]

    print(f"Ensemble size: {len(ckpt_paths)}")
    models, log_scales = [], []
    for ckpt_path in ckpt_paths:
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        log_scale = ckpt.get("log_scale_weight", False)
        model     = FishSaltModel(ckpt["cfg_model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
        log_scales.append(log_scale)
        print(f"  Loaded epoch {ckpt['epoch']+1:3d}  val_loss={ckpt['val_loss']:.4f}  {ckpt_path}")

    print(f"Device: {device}")

    val_loader  = make_salt_dataloader(args.val_file,  args.batch_size, shuffle=False)
    test_loader = make_salt_dataloader(args.test_file, args.batch_size, shuffle=False)

    val_logits,  val_pw,  val_yp,  val_yw  = _predict_ensemble(
        models, log_scales, val_loader,  device)
    test_logits, test_pw, test_yp, test_yw = _predict_ensemble(
        models, log_scales, test_loader, device)

    # --- Threshold sweep on val ---
    print("\nThreshold sweep on VAL (minimising |hard_diff|):")
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thr, best_abs = 0.5, float("inf")
    for thr in thresholds:
        s = _biomass_stats(val_pw, val_yw, val_yp, val_logits, thr)
        if abs(s["hard_diff"]) < best_abs:
            best_abs = abs(s["hard_diff"])
            best_thr = thr
    print(f"  Best threshold = {best_thr:.2f}  (|hard_diff| = {best_abs:.2f}%)")

    # --- Final eval ---
    val_stats  = _biomass_stats(val_pw,  val_yw,  val_yp,  val_logits,  best_thr)
    test_stats = _biomass_stats(test_pw, test_yw, test_yp, test_logits, best_thr)

    _print_stats("VAL", val_stats)
    _print_stats("TEST", test_stats)

    test_50 = _biomass_stats(test_pw, test_yw, test_yp, test_logits, 0.50)
    _print_stats("TEST (thr=0.50 reference)", test_50)

    print()


if __name__ == "__main__":
    main()
