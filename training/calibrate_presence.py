"""Platt scaling calibration for the bream_carp_presence ensemble.

Fits sigmoid(a * logit + b) on the val set, evaluates on test set,
saves calibration parameters to checkpoints/bream_carp_presence/platt.npy.

Usage:
    python training/calibrate_presence.py
"""

import sys
import numpy as np
import torch
import yaml
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit   # sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_model import FishModel

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR    = Path("checkpoints/bream_carp_presence")
CFG_PATH    = Path("model/model_config_12ch.yaml")
VAL_NPZ     = Path("data/presence_bream_carp/val_dataset.npz")
TEST_NPZ    = Path("data/presence_bream_carp/test_dataset.npz")
PLATT_OUT   = CKPT_DIR / "platt.npy"


def load_ensemble():
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(CKPT_DIR.glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(DEVICE)
        ck = torch.load(pt, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
    return models


def get_logits(models, npz_path: Path) -> tuple:
    """Return (averaged_logits, labels) from an NPZ split."""
    data = np.load(npz_path)
    X    = torch.from_numpy(data["X"]).to(DEVICE)
    y    = data["y_presence"].astype(np.int32)

    sum_logits = None
    with torch.no_grad():
        for m in models:
            logits, _ = m(X)
            lv = logits.cpu().numpy()
            sum_logits = lv if sum_logits is None else sum_logits + lv
    avg_logits = (sum_logits / len(models)).ravel()
    return avg_logits, y


def nll_loss(params, logits, labels):
    """Negative log-likelihood for Platt calibration."""
    a, b = params
    p = expit(a * logits + b)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def best_f1_threshold(logits, labels, steps=200):
    """Sweep thresholds on raw logits and return the F1-maximising one."""
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(-4, 4, steps):          # logit space
        preds = (logits >= thr).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    return best_thr, best_f1


def eval_at_threshold(logits_or_probs, labels, threshold, prob_space=True):
    preds = (logits_or_probs >= threshold).astype(int)
    return {
        "f1":   f1_score(labels, preds, zero_division=0),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec":  recall_score(labels, preds, zero_division=0),
    }


def main():
    print("Loading ensemble …")
    models = load_ensemble()
    print(f"  {len(models)} models from {CKPT_DIR}")

    # ── Val set logits ────────────────────────────────────────────────
    print("\nCollecting val-set logits …")
    val_logits, val_labels = get_logits(models, VAL_NPZ)
    print(f"  val samples: {len(val_labels)}  pos={val_labels.sum()}  neg={(val_labels==0).sum()}")

    # Baseline: best threshold in logit space (for comparison)
    raw_best_thr, raw_best_f1 = best_f1_threshold(val_logits, val_labels)
    print(f"\n  Best raw threshold (val, logit space): {raw_best_thr:+.3f}  F1={raw_best_f1:.4f}")

    # ── Fit Platt scaling on val set ──────────────────────────────────
    print("\nFitting Platt scaling on val set …")
    res = minimize(
        nll_loss,
        x0=[1.0, 0.0],
        args=(val_logits, val_labels),
        method="L-BFGS-B",
    )
    a, b = res.x
    print(f"  a={a:.4f}  b={b:.4f}  (converged={res.success})")

    val_probs_cal = expit(a * val_logits + b)
    val_m_cal     = eval_at_threshold(val_probs_cal, val_labels, 0.5)
    val_probs_raw = expit(val_logits)
    val_m_raw     = eval_at_threshold(val_probs_raw, val_labels, 0.5)

    print(f"\n  Val  (thr=0.5)  raw probs:  F1={val_m_raw['f1']:.4f}  "
          f"P={val_m_raw['prec']:.4f}  R={val_m_raw['rec']:.4f}")
    print(f"  Val  (thr=0.5)  calibrated: F1={val_m_cal['f1']:.4f}  "
          f"P={val_m_cal['prec']:.4f}  R={val_m_cal['rec']:.4f}")

    # ── Evaluate on test set ──────────────────────────────────────────
    print("\nEvaluating on test set …")
    test_logits, test_labels = get_logits(models, TEST_NPZ)
    print(f"  test samples: {len(test_labels)}  pos={test_labels.sum()}  neg={(test_labels==0).sum()}")

    test_probs_raw = expit(test_logits)
    test_probs_cal = expit(a * test_logits + b)

    # Best threshold sweep on test raw logits (oracle — for reference only)
    test_best_thr, _ = best_f1_threshold(test_logits, test_labels)
    test_oracle = eval_at_threshold(test_logits, test_labels, test_best_thr, prob_space=False)

    test_raw_05 = eval_at_threshold(test_probs_raw, test_labels, 0.5)
    test_cal_05 = eval_at_threshold(test_probs_cal, test_labels, 0.5)

    print(f"\n  {'Method':<30}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}")
    print("  " + "-" * 58)
    for label, m in [
        ("Raw probs  (thr=0.50)",     test_raw_05),
        ("Calibrated (thr=0.50)",     test_cal_05),
        ("Oracle best thr (test)",    test_oracle),
    ]:
        print(f"  {label:<30}  {m['f1']:>7.4f}  {m['prec']:>7.4f}  {m['rec']:>7.4f}")

    # ── Reliability diagram (probability bins) ────────────────────────
    print("\n  Calibration check (raw vs calibrated, mean predicted prob per bin):")
    print(f"  {'Bin':>12}  {'Raw mean p':>12}  {'Cal mean p':>12}  {'True frac':>12}")
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask_r = (test_probs_raw >= lo) & (test_probs_raw < hi)
        mask_c = (test_probs_cal >= lo) & (test_probs_cal < hi)
        frac_r = test_labels[mask_r].mean() if mask_r.sum() > 0 else float("nan")
        frac_c = test_labels[mask_c].mean() if mask_c.sum() > 0 else float("nan")
        mp_r   = test_probs_raw[mask_r].mean() if mask_r.sum() > 0 else float("nan")
        mp_c   = test_probs_cal[mask_c].mean() if mask_c.sum() > 0 else float("nan")
        print(f"  [{lo:.1f}, {hi:.1f})   raw: p={mp_r:>5.3f} true={frac_r:>5.3f}   "
              f"cal: p={mp_c:>5.3f} true={frac_c:>5.3f}")

    # ── Save ──────────────────────────────────────────────────────────
    np.save(PLATT_OUT, np.array([a, b], dtype=np.float32))
    print(f"\nSaved Platt params to {PLATT_OUT}  (a={a:.4f}, b={b:.4f})")
    print("\nUse in inference:  p_cal = sigmoid(a * ensemble_logit + b)  then threshold at 0.5")


if __name__ == "__main__":
    main()
