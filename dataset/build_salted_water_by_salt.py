"""Build separate train/val/test npz splits for S0 (sweet) and S400 (salty) carp.

Each dataset uses only its own recordings, 12 acoustic channels, per-file
z-scored — identical pipeline to v5 but single-species, single salt condition.

Output
------
    data/carp/s0_split/    train/val/test_dataset.npz  (sweet water, 4 fish)
    data/carp/s400_split/  train/val/test_dataset.npz  (salty water, 3 fish)

Run from the project root:
    python dataset/build_salted_water_by_salt.py
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE    = 39
PRESENCE_RATIO = 0.20
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
TEST_FRAC      = 0.15
RANDOM_SEED    = 42

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_ACOUSTIC = len(ACOUSTIC_CHANNELS)

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_meta(fname: str):
    m_w    = re.search(r"(\d+)gr",           fname)
    m_l    = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    weight = float(m_w.group(1)) if m_w else None
    length = float(m_l.group(1)) if m_l else None
    salt   = 400.0 if "_S400" in fname else 0.0
    return length, weight, salt


class RecordingCache:
    def __init__(self):
        self._raw:  dict[str, np.ndarray] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._std:  dict[str, np.ndarray] = {}

    def load(self, fname: str, path: Path) -> bool:
        if not path.exists():
            return False
        df = pd.read_csv(path)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)
        m   = np.nanmean(raw, axis=0).astype(np.float32)
        s   = np.nanstd(raw,  axis=0).astype(np.float32)
        s[s < 1e-8] = 1.0
        if np.isnan(raw).any():
            raw = np.where(np.isnan(raw), m[np.newaxis, :], raw)
        self._raw[fname]  = raw
        self._mean[fname] = m
        self._std[fname]  = s
        return True

    def normalised_window(self, fname: str, mid: int) -> np.ndarray | None:
        raw = self._raw.get(fname)
        if raw is None:
            return None
        win = _extract_window(raw, mid)
        return ((win - self._mean[fname]) / self._std[fname]).astype(np.float32)

    def normalised_segment(self, fname: str, start: int) -> np.ndarray | None:
        raw = self._raw.get(fname)
        if raw is None:
            return None
        seg = raw[start : start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            seg = np.concatenate(
                [seg, np.zeros((WINDOW_SIZE - len(seg), N_ACOUSTIC), np.float32)]
            )
        return ((seg[:WINDOW_SIZE] - self._mean[fname]) / self._std[fname]).astype(np.float32)

    def n_rows(self, fname: str) -> int:
        return len(self._raw.get(fname, []))

    def all_files(self) -> list[str]:
        return list(self._raw.keys())


def _extract_window(signal: np.ndarray, mid: int) -> np.ndarray:
    half      = WINDOW_SIZE // 2
    start     = mid - half
    end       = start + WINDOW_SIZE
    n         = len(signal)
    pad_left  = max(0, -start)
    pad_right = max(0, end - n)
    s_start   = max(0, start)
    s_end     = min(n, end)
    window    = signal[s_start:s_end]
    if pad_left > 0:
        window = np.concatenate(
            [np.zeros((pad_left,  N_ACOUSTIC), np.float32), window]
        )
    if pad_right > 0:
        window = np.concatenate(
            [window, np.zeros((pad_right, N_ACOUSTIC), np.float32)]
        )
    return window[:WINDOW_SIZE]


def build_positive_samples(report: pd.DataFrame, cache: RecordingCache):
    waves, lengths, weights, strata = [], [], [], []
    skipped = 0
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        mid   = int((row["start_index"] + row["end_index"]) // 2)
        win   = cache.normalised_window(fname, mid)
        if win is None:
            skipped += 1
            continue
        waves.append(win)
        lengths.append(row["_length"])
        weights.append(row["_weight"])
        strata.append(str(int(row["_length"])))
    if skipped:
        print(f"  Skipped {skipped} detections")
    return (np.stack(waves),
            np.array(lengths, np.float64),
            np.array(weights, np.float64),
            np.array(strata))


def build_negative_samples(report: pd.DataFrame, cache: RecordingCache,
                            n_total: int, rng: np.random.Generator):
    GUARD = WINDOW_SIZE
    occupied: dict[str, list] = defaultdict(list)
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        occupied[fname].append((
            max(0, int(row["start_index"]) - GUARD),
            int(row["end_index"]) + GUARD,
        ))

    candidates = []
    n_files    = len(cache.all_files())
    for fname in cache.all_files():
        n_rows = cache.n_rows(fname)
        occ    = sorted(occupied.get(fname, []))
        free_starts, free_ends = [0], []
        for s, e in occ:
            free_ends.append(max(0, s - 1))
            free_starts.append(min(n_rows, e + 1))
        free_ends.append(n_rows)
        for fs, fe in zip(free_starts, free_ends):
            max_start = fe - WINDOW_SIZE
            if max_start > fs:
                step = max(1, (max_start - fs) // max(1, n_total // n_files))
                for pos in range(fs, max_start, step):
                    candidates.append((fname, pos))

    while len(candidates) < n_total:
        fname, pos = candidates[int(rng.integers(len(candidates)))]
        jitter  = int(rng.integers(-5, 6))
        new_pos = max(0, min(pos + jitter, cache.n_rows(fname) - WINDOW_SIZE))
        candidates.append((fname, new_pos))

    idx   = rng.choice(len(candidates), size=n_total,
                       replace=len(candidates) < n_total)
    return np.stack([
        cache.normalised_segment(candidates[i][0], candidates[i][1])
        for i in idx
    ])


def combine(pos_waves, pos_lengths, pos_weights, neg_waves, rng):
    n_pos = len(pos_waves)
    n_neg = len(neg_waves)
    X          = np.concatenate([pos_waves, neg_waves])
    y_presence = np.concatenate([np.ones(n_pos, np.int64),  np.zeros(n_neg, np.int64)])
    y_length   = np.concatenate([pos_lengths,               np.zeros(n_neg, np.float64)])
    y_weight   = np.concatenate([pos_weights,               np.zeros(n_neg, np.float64)])
    perm       = rng.permutation(len(X))
    return X[perm], y_presence[perm], y_length[perm], y_weight[perm]


# ---------------------------------------------------------------------------
# Build one dataset
# ---------------------------------------------------------------------------

def build_dataset(report: pd.DataFrame, out_dir: Path, label: str,
                  rng: np.random.Generator) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label}  ({len(report)} detections)")
    print(f"{'='*60}")

    # Load recordings for this subset
    cache = RecordingCache()
    for fname in report["raw_data_file_name"].unique():
        p = RECORDINGS_DIR / fname
        ok = cache.load(fname, p)
        status = f"{cache.n_rows(fname):6d} rows" if ok else "NOT FOUND"
        print(f"  {fname:35s}  {status}")

    loaded = set(cache.all_files())
    report = report[report["raw_data_file_name"].isin(loaded)].reset_index(drop=True)

    # Positive samples
    pos_waves, pos_lengths, pos_weights, pos_strata = build_positive_samples(
        report, cache
    )
    n_pos = len(pos_waves)
    print(f"\n  Positives: {n_pos}")
    for st in sorted(np.unique(pos_strata)):
        print(f"    {st} cm  {(pos_strata == st).sum():3d}")

    # Stratified split
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRAC,
        random_state=int(rng.integers(0, 10_000))
    )
    trainval_idx, test_idx = next(sss1.split(np.zeros(n_pos), pos_strata))

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=round(VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), 4),
        random_state=int(rng.integers(0, 10_000))
    )
    train_rel, val_rel = next(
        sss2.split(np.zeros(len(trainval_idx)), pos_strata[trainval_idx])
    )
    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]

    print(f"\n  Split — train: {len(train_idx)}  "
          f"val: {len(val_idx)}  test: {len(test_idx)}")

    # Negatives
    n_neg_train = int(len(train_idx) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_idx)   / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_idx)  / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_total = n_neg_train + n_neg_val + n_neg_test

    neg_all   = build_negative_samples(report, cache, n_neg_total, rng)
    neg_train = neg_all[:n_neg_train]
    neg_val   = neg_all[n_neg_train : n_neg_train + n_neg_val]
    neg_test  = neg_all[n_neg_train + n_neg_val :]

    # Save
    splits = {
        "train": (train_idx, neg_train),
        "val":   (val_idx,   neg_val),
        "test":  (test_idx,  neg_test),
    }
    print()
    for name, (pos_idx, neg_waves) in splits.items():
        X, y_pres, y_len, y_wt = combine(
            pos_waves[pos_idx], pos_lengths[pos_idx],
            pos_weights[pos_idx], neg_waves, rng
        )
        path = out_dir / f"{name}_dataset.npz"
        np.savez(path, X=X.astype(np.float32),
                 y_presence=y_pres, y_length=y_len, y_weight=y_wt)
        n = len(y_pres); pos_n = int(y_pres.sum())
        print(f"  {name:5s}: {n:5d} total  "
              f"({pos_n} pos / {n - pos_n} neg)  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Load and fix report
    report = pd.read_csv(REPORT_PATH)
    lengths, weights, salts = [], [], []
    for fname in report["raw_data_file_name"]:
        l, w, s = _parse_meta(fname)
        lengths.append(l); weights.append(w); salts.append(s)
    report["_length"] = lengths
    report["_weight"] = weights
    report["_salt"]   = salts
    report = report.dropna(subset=["_length", "_weight"]).reset_index(drop=True)

    s0_report   = report[report["_salt"] == 0.0].reset_index(drop=True)
    s400_report = report[report["_salt"] == 400.0].reset_index(drop=True)

    build_dataset(s0_report,   Path("data/carp/s0_split"),   "S0  — sweet water (0 g/L salt)",   rng)
    build_dataset(s400_report, Path("data/carp/s400_split"), "S400 — salty water (400 g/L salt)", rng)

    print("\nDone.")


if __name__ == "__main__":
    main()
