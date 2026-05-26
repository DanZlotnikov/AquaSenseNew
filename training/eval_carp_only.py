"""Evaluate fish_presence_model on carp-only test data.

Reproduces the same split as build_presence_all.py, isolates carp test
positives, pairs them with carp-only negatives, and reports full
Precision / Recall / F1 at several thresholds.

Run from project root:
    python training/eval_carp_only.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fish_model import FishModel

# ---------------------------------------------------------------------------
WINDOW_SIZE = 39
GUARD       = WINDOW_SIZE
N_CH        = 12
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]

CARP_SOURCES = [
    (
        Path("data/raw/carp/carp_salted/salted_water/output_report.csv"),
        [Path("data/raw/carp/carp_salted/salted_water")],
    ),
    (
        Path("data/raw/carp/carp_salted/salted_water/test/output_report.csv"),
        [Path("data/raw/carp/carp_salted/salted_water/test"),
         Path("data/raw/carp/carp_salted/salted_water")],
    ),
    (
        Path("data/raw/carp/output_report.csv"),
        [Path("data/raw/carp")],
    ),
]

ALL_SOURCES = CARP_SOURCES + [
    (
        Path("data/raw/bream/output_report.csv"),
        [Path("data/raw/bream/recordings")],
    ),
    (
        Path("data/raw/bream/test/output_report_F.csv"),
        [Path("data/raw/bream/test"), Path("data/raw/bream/recordings")],
    ),
]

CHECKPOINT_DIRS = sorted(Path("checkpoints/fish_presence_model").glob("2*"))
MODEL_CFG_PATH  = Path("model/model_config_12ch.yaml")

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
import pandas as pd


def _find_recording(fname: str, search_dirs):
    for d in search_dirs:
        p = d / fname
        if p.exists():
            return p
    return None


class RecordingCache:
    def __init__(self):
        self._raw  = {}
        self._mean = {}
        self._std  = {}

    def load(self, csv_path: Path):
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

    def get_centered_window(self, key, mid):
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

    def get_segment(self, key, start):
        if key not in self._raw:
            return None
        raw = self._raw[key]
        seg = raw[start: start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            seg = np.concatenate([seg, np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)])
        return ((seg[:WINDOW_SIZE] - self._mean[key]) / self._std[key]).astype(np.float32)

    def n_rows(self, key):
        return len(self._raw.get(key, []))

    def all_keys(self):
        return list(self._raw.keys())


def rebuild_all_positives(cache):
    """Reproduce the exact positives list from build_presence_all.py."""
    import pandas as pd
    waves, species_labels = [], []
    for report_path, search_dirs, species in [
        (Path("data/raw/carp/carp_salted/salted_water/output_report.csv"),
         [Path("data/raw/carp/carp_salted/salted_water")], "carp"),
        (Path("data/raw/carp/carp_salted/salted_water/test/output_report.csv"),
         [Path("data/raw/carp/carp_salted/salted_water/test"),
          Path("data/raw/carp/carp_salted/salted_water")], "carp"),
        (Path("data/raw/carp/output_report.csv"), [Path("data/raw/carp")], "carp"),
        (Path("data/raw/bream/output_report.csv"), [Path("data/raw/bream/recordings")], "bream"),
        (Path("data/raw/bream/test/output_report_F.csv"),
         [Path("data/raw/bream/test"), Path("data/raw/bream/recordings")], "bream"),
    ]:
        if not report_path.exists():
            continue
        df = pd.read_csv(report_path)
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

    return np.stack(waves), np.array(species_labels)


def build_carp_negatives(cache, n_total, rng):
    """Sample negatives only from carp recordings, avoiding event regions."""
    import pandas as pd
    occupied = defaultdict(list)
    for report_path, search_dirs, _ in [
        (Path("data/raw/carp/carp_salted/salted_water/output_report.csv"),
         [Path("data/raw/carp/carp_salted/salted_water")], "carp"),
        (Path("data/raw/carp/carp_salted/salted_water/test/output_report.csv"),
         [Path("data/raw/carp/carp_salted/salted_water/test"),
          Path("data/raw/carp/carp_salted/salted_water")], "carp"),
        (Path("data/raw/carp/output_report.csv"), [Path("data/raw/carp")], "carp"),
    ]:
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

    # Only use carp keys
    carp_keys = [k for k in cache.all_keys()
                 if any(str(d) in k for d in [
                     "carp_salted", "raw/carp", "raw\\carp"])]
    if not carp_keys:
        carp_keys = cache.all_keys()  # fallback

    candidates = []
    n_keys = len(carp_keys)
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
                step = max(1, (max_start - fs) // max(1, n_total // max(n_keys, 1)))
                for pos in range(fs, max_start, step):
                    candidates.append((key, pos))

    if not candidates:
        raise RuntimeError("No negative candidates found for carp recordings!")

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
    import yaml
    with open(MODEL_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["physics_dim"] = 0

    models = []
    for ckpt_dir in CHECKPOINT_DIRS:
        pt = ckpt_dir / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(DEVICE)
        ckpt = torch.load(pt, map_location=DEVICE)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        print(f"  Loaded {pt}")
    return models


def infer_batch(models, X: np.ndarray) -> np.ndarray:
    """Run ensemble and return averaged probabilities."""
    t = torch.from_numpy(X).to(DEVICE)  # (N, T, C)
    probs_list = []
    with torch.no_grad():
        for m in models:
            logits, _ = m(t)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
    return np.mean(probs_list, axis=0)


def pr_metrics(y_true, probs, thr):
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn


def main():
    rng = np.random.default_rng(42)

    print("=" * 65)
    print("Evaluating fish_presence_model — CARP-ONLY test set")
    print("=" * 65)

    print("\nLoading models ...")
    models = load_ensemble()
    print(f"Ensemble size: {len(models)}")

    print("\nRebuilding positives (same seed as build_presence_all.py) ...")
    cache = RecordingCache()
    waves_all, species_all = rebuild_all_positives(cache)
    n_pos = len(species_all)
    print(f"  Total positives: {n_pos}  "
          f"(carp={(species_all=='carp').sum()}, bream={(species_all=='bream').sum()})")
    print(f"  Recordings loaded: {len(cache.all_keys())}")

    # Reproduce the exact split
    sss1 = StratifiedShuffleSplit(1, test_size=0.10, random_state=42)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n_pos), species_all))

    carp_test_mask = species_all[test_idx] == "carp"
    carp_test_idx  = test_idx[carp_test_mask]
    n_carp_test    = len(carp_test_idx)
    print(f"\nCarp test positives: {n_carp_test}")
    print(f"Bream test positives: {(~carp_test_mask).sum()}")

    X_pos = waves_all[carp_test_idx]   # (n_carp_test, 39, 12)

    # Build matching carp negatives (~4× to mimic 20% presence ratio)
    n_neg = n_carp_test * 4
    print(f"\nSampling {n_neg} carp negatives ...")
    X_neg = build_carp_negatives(cache, n_neg, rng)

    # Combine
    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(n_carp_test, dtype=np.int64),
                        np.zeros(n_neg,       dtype=np.int64)])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    print(f"Carp test set: {len(X)} samples  ({int(y.sum())} pos / {int((y==0).sum())} neg)")

    print("\nRunning inference ...")
    probs = infer_batch(models, X)

    print("\nThreshold sweep (carp-only):")
    print(f"{'thr':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("-" * 50)
    best_f1 = 0.0
    best_row = None
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        prec, rec, f1, tp, fp, fn = pr_metrics(y, probs, thr)
        row = f"{thr:>5.1f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {tp:>5d}  {fp:>5d}  {fn:>5d}"
        print(row)
        if f1 > best_f1:
            best_f1 = f1
            best_row = (thr, prec, rec, f1)

    print(f"\nBest carp F1 = {best_row[3]:.3f}  at thr={best_row[0]:.1f}  "
          f"(Prec={best_row[1]:.3f}, Rec={best_row[2]:.3f})")

    # Also show overall test_dataset.npz numbers for comparison
    print("\n--- Reference: full test_dataset.npz (all species) ---")
    test_npz = np.load("data/presence_all/test_dataset.npz")
    X_all = test_npz["X"]
    y_all = test_npz["y_presence"]
    probs_all = infer_batch(models, X_all)
    print(f"{'thr':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    for thr in [0.3, 0.5, 0.6, 0.7]:
        prec, rec, f1, *_ = pr_metrics(y_all, probs_all, thr)
        print(f"{thr:>5.1f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")


if __name__ == "__main__":
    main()
