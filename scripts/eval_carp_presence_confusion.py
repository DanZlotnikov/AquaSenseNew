"""Evaluate the bream_carp_presence ensemble on carp-only held-out test data.

Reproduces the exact 70/20/10 split from build_presence_all.py, isolates carp
test positives, pairs with carp-only negatives, applies Platt calibration, and
reports a confusion matrix plus precision/recall/F1 metrics.

Run from project root:
    python scripts/eval_carp_presence_confusion.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_model import FishModel

# ---------------------------------------------------------------------------
WINDOW_SIZE       = 39
GUARD             = WINDOW_SIZE
N_CH              = 12
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]

# Same SOURCES as build_presence_all.py (salted carp only, no fresh carp)
CARP_SOURCES = [
    (
        Path("data/raw/carp/carp_salted/salted_water/output_report.csv"),
        [Path("data/raw/carp/carp_salted/salted_water")],
        "carp",
    ),
    (
        Path("data/raw/carp/carp_salted/salted_water/test/output_report.csv"),
        [Path("data/raw/carp/carp_salted/salted_water/test"),
         Path("data/raw/carp/carp_salted/salted_water")],
        "carp",
    ),
]
BREAM_SOURCES = [
    (
        Path("data/raw/bream/output_report.csv"),
        [Path("data/raw/bream/recordings")],
        "bream",
    ),
    (
        Path("data/raw/bream/test/output_report_F.csv"),
        [Path("data/raw/bream/test"), Path("data/raw/bream/recordings")],
        "bream",
    ),
]
ALL_SOURCES = CARP_SOURCES + BREAM_SOURCES

CKPT_DIR       = Path("checkpoints/bream_carp_presence")
MODEL_CFG_PATH = Path("model/model_config_12ch.yaml")
PLATT_PATH     = CKPT_DIR / "platt.npy"
PRESENCE_RATIO = 0.20
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("results/carp_presence_confusion")
# ---------------------------------------------------------------------------


class RecordingCache:
    def __init__(self):
        self._raw  = {}
        self._mean = {}
        self._std  = {}

    def load(self, csv_path: Path) -> bool:
        key = str(csv_path)
        if key in self._raw:
            return True
        if not csv_path.exists():
            return False
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
        self._raw[key]  = raw
        self._mean[key] = m
        self._std[key]  = s
        return True

    def get_centered_window(self, key: str, mid: int):
        if key not in self._raw:
            return None
        raw  = self._raw[key]
        half = WINDOW_SIZE // 2
        start = mid - half; end = start + WINDOW_SIZE; n = len(raw)
        pad_l = max(0, -start); pad_r = max(0, end - n)
        seg   = raw[max(0, start): min(n, end)]
        if pad_l: seg = np.concatenate([np.zeros((pad_l, N_CH), np.float32), seg])
        if pad_r: seg = np.concatenate([seg, np.zeros((pad_r, N_CH), np.float32)])
        seg = seg[:WINDOW_SIZE]
        return ((seg - self._mean[key]) / self._std[key]).astype(np.float32)

    def get_segment(self, key: str, start: int):
        if key not in self._raw:
            return None
        raw = self._raw[key]
        seg = raw[start: start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            seg = np.concatenate([seg, np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)])
        return ((seg[:WINDOW_SIZE] - self._mean[key]) / self._std[key]).astype(np.float32)

    def n_rows(self, key: str) -> int:
        return len(self._raw.get(key, []))

    def all_keys(self) -> list:
        return list(self._raw.keys())


def _find_recording(fname: str, search_dirs) -> Path | None:
    for d in search_dirs:
        p = d / fname
        if p.exists():
            return p
    return None


def rebuild_all_positives(cache: RecordingCache):
    """Reproduce exact positives list from build_presence_all.py."""
    waves, species_labels = [], []
    for report_path, search_dirs, species in ALL_SOURCES:
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping")
            continue
        df = pd.read_csv(report_path)
        n = 0
        for _, row in df.iterrows():
            fname = row["raw_data_file_name"]
            rec   = _find_recording(fname, search_dirs)
            if rec is None:
                continue
            cache.load(rec)
            key = str(rec)
            mid = int((row["start_index"] + row["end_index"]) // 2)
            win = cache.get_centered_window(key, mid)
            if win is None:
                continue
            waves.append(win)
            species_labels.append(species)
            n += 1
        print(f"  {report_path.name:<45s}  +{n} positives ({species})")
    return np.stack(waves), np.array(species_labels)


def build_carp_negatives(cache: RecordingCache, n_total: int, rng: np.random.Generator):
    """Sample negatives only from carp recordings, avoiding event regions."""
    occupied = defaultdict(list)
    for report_path, search_dirs, _ in CARP_SOURCES:
        if not report_path.exists():
            continue
        df = pd.read_csv(report_path)
        for _, row in df.iterrows():
            fname = row["raw_data_file_name"]
            rec   = _find_recording(fname, search_dirs)
            if rec is None:
                continue
            key = str(rec)
            s = max(0, int(row["start_index"]) - GUARD)
            e = int(row["end_index"]) + GUARD
            occupied[key].append((s, e))

    carp_keys = [k for k in cache.all_keys()
                 if "carp_salted" in k or "salted_water" in k]
    if not carp_keys:
        carp_keys = cache.all_keys()

    candidates = []
    n_keys = max(len(carp_keys), 1)
    for key in carp_keys:
        n_rows = cache.n_rows(key)
        occ    = sorted(occupied.get(key, []))
        free_starts = [0]; free_ends = []
        for s, e in occ:
            free_ends.append(max(0, s - 1))
            free_starts.append(min(n_rows, e + 1))
        free_ends.append(n_rows)
        for fs, fe in zip(free_starts, free_ends):
            max_start = fe - WINDOW_SIZE
            if max_start > fs:
                step = max(1, (max_start - fs) // max(1, n_total // n_keys))
                for pos in range(fs, max_start, step):
                    candidates.append((key, pos))

    if not candidates:
        raise RuntimeError("No negative candidates found in carp recordings!")

    while len(candidates) < n_total:
        key, pos = candidates[int(rng.integers(len(candidates)))]
        jitter  = int(rng.integers(-5, 6))
        new_pos = max(0, min(pos + jitter, cache.n_rows(key) - WINDOW_SIZE))
        candidates.append((key, new_pos))

    idx      = rng.choice(len(candidates), size=n_total, replace=len(candidates) < n_total)
    selected = [candidates[i] for i in idx]
    waves    = [cache.get_segment(k, st) for k, st in selected]
    return np.stack(waves)


def load_ensemble():
    with open(MODEL_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
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
        print(f"  Loaded {pt}")
    return models


def get_ensemble_logits(models, X: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(X).to(DEVICE)
    sum_logits = None
    with torch.no_grad():
        for m in models:
            logits, _ = m(t)
            lv = logits.cpu().numpy()
            sum_logits = lv if sum_logits is None else sum_logits + lv
    return (sum_logits / len(models)).ravel()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print("=" * 65)
    print("Carp Presence Detection — Confusion Matrix & Metrics")
    print("Model: bream_carp_presence ensemble (Platt-calibrated)")
    print("=" * 65)

    # ── Load ensemble ────────────────────────────────────────────────
    print(f"\nLoading ensemble from {CKPT_DIR} ...")
    models = load_ensemble()
    if not models:
        raise RuntimeError(f"No model checkpoints found in {CKPT_DIR}")
    print(f"Ensemble size: {len(models)}")

    # ── Load Platt calibration ───────────────────────────────────────
    platt_ab = np.load(PLATT_PATH)
    a_platt, b_platt = float(platt_ab[0]), float(platt_ab[1])
    print(f"Platt params: a={a_platt:.4f}  b={b_platt:.4f}")

    # ── Rebuild positives (same seed as build_presence_all.py) ───────
    print("\nRebuilding positives ...")
    cache = RecordingCache()
    waves_all, species_all = rebuild_all_positives(cache)
    n_pos = len(species_all)
    n_carp  = int((species_all == "carp").sum())
    n_bream = int((species_all == "bream").sum())
    print(f"\nTotal positives: {n_pos}  (carp={n_carp}, bream={n_bream})")

    # ── Reproduce exact 70/20/10 split ──────────────────────────────
    sss = StratifiedShuffleSplit(1, test_size=0.10, random_state=42)
    trainval_idx, test_idx = next(sss.split(np.zeros(n_pos), species_all))

    carp_test_mask = species_all[test_idx] == "carp"
    carp_test_idx  = test_idx[carp_test_mask]
    n_carp_test    = len(carp_test_idx)
    print(f"\nCarp test positives (held-out 10%): {n_carp_test}")
    print(f"Bream test positives (not used here): {int((~carp_test_mask).sum())}")

    X_pos = waves_all[carp_test_idx]   # (n_carp_test, 39, 12)

    # ── Build carp-only negatives ─────────────────────────────────────
    n_neg = int(n_carp_test / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    print(f"\nSampling {n_neg} carp negatives (target {PRESENCE_RATIO*100:.0f}% presence ratio) ...")
    X_neg = build_carp_negatives(cache, n_neg, rng)

    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(n_carp_test, dtype=np.int64),
                        np.zeros(n_neg,       dtype=np.int64)])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    print(f"\nCarp test set: {len(X)} samples  "
          f"({int(y.sum())} positive / {int((y==0).sum())} negative)")

    # ── Run ensemble inference ────────────────────────────────────────
    print("\nRunning ensemble inference ...")
    avg_logits = get_ensemble_logits(models, X)

    # Raw probs and Platt-calibrated probs
    probs_raw = expit(avg_logits)
    probs_cal = expit(a_platt * avg_logits + b_platt)

    # Binary predictions at threshold 0.5 (calibrated)
    preds = (probs_cal >= 0.5).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(y)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    ap  = average_precision_score(y, probs_cal)
    try:
        auc = roc_auc_score(y, probs_cal)
    except Exception:
        auc = float("nan")

    print("\n" + "=" * 65)
    print("RESULTS (threshold = 0.50, Platt-calibrated probabilities)")
    print("=" * 65)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted NO   Predicted YES")
    print(f"  Actual NO         {tn:5d}           {fp:5d}   (TN / FP)")
    print(f"  Actual YES        {fn:5d}           {tp:5d}   (FN / TP)")
    print(f"\nMetrics:")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  Specificity     : {specificity:.4f}")
    print(f"  Avg Precision   : {ap:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")

    print("\nFull classification report:")
    print(classification_report(y, preds, target_names=["No Fish", "Carp Present"], digits=4))

    # ── Threshold sweep ───────────────────────────────────────────────
    print("Threshold sweep (Platt-calibrated probabilities):")
    print(f"  {'thr':>5}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")
    print("  " + "-" * 65)
    best_f1_row = None
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = (probs_cal >= thr).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(y, p).ravel()
        _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        _f1   = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0
        print(f"  {thr:>5.1f}  {_prec:>7.4f}  {_rec:>7.4f}  {_f1:>7.4f}  "
              f"{_tp:>5d}  {_fp:>5d}  {_fn:>5d}  {_tn:>5d}")
        if best_f1_row is None or _f1 > best_f1_row[3]:
            best_f1_row = (thr, _prec, _rec, _f1)

    print(f"\n  Best F1={best_f1_row[3]:.4f} at thr={best_f1_row[0]:.1f}  "
          f"(Prec={best_f1_row[1]:.4f}, Recall={best_f1_row[2]:.4f})")

    # ── Figures ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Carp Presence Detection — Held-Out Test Set\n"
        "bream_carp_presence ensemble (Platt-calibrated, thr=0.5)",
        fontsize=13,
    )

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fish", "Carp"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(
        f"Confusion Matrix\nPrec={precision:.3f}  Rec={recall:.3f}  F1={f1:.3f}",
        fontsize=11,
    )

    # Precision-Recall curve
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y, probs_cal)
    axes[1].step(rec_curve, prec_curve, where="post", color="steelblue", linewidth=2)
    axes[1].scatter([recall], [precision], color="red", zorder=5, s=80,
                    label=f"thr=0.5  F1={f1:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve\nAP={ap:.3f}", fontsize=11)
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "carp_presence_confusion.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")

    # ── Save text summary ──────────────────────────────────────────────
    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Carp Presence Detection — Held-Out Test Set\n")
        f.write("Model: bream_carp_presence ensemble (Platt-calibrated, thr=0.5)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Test set: {len(X)} samples  ({int(y.sum())} pos / {int((y==0).sum())} neg)\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}\n\n")
        f.write(f"Accuracy    : {accuracy:.4f}\n")
        f.write(f"Precision   : {precision:.4f}\n")
        f.write(f"Recall      : {recall:.4f}\n")
        f.write(f"F1 Score    : {f1:.4f}\n")
        f.write(f"Specificity : {specificity:.4f}\n")
        f.write(f"Avg Prec    : {ap:.4f}\n")
        f.write(f"ROC-AUC     : {auc:.4f}\n\n")
        f.write(classification_report(y, preds, target_names=["No Fish", "Carp Present"], digits=4))
    print(f"Summary saved to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
