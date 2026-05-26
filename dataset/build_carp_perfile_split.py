"""Build train / val / test npz files from data/raw/carp/ using per-file
z-score normalisation and a 13th channel fixed to salt=0.0.

This makes the carp dataset compatible with the salted-water weight model
(which expects 12 acoustic + 1 salt channel, per-file z-scored).

Run from the project root:
    python dataset/build_carp_perfile_split.py

Output
------
    data/carp/carp_perfile_split/
        train_dataset.npz   keys: X (N,39,13), y_presence, y_length, y_weight
        val_dataset.npz
        test_dataset.npz
        build_report.txt
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
SALT_VALUE     = 0.0   # carp recordings are fresh water

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_ACOUSTIC  = len(ACOUSTIC_CHANNELS)   # 12
TOTAL_CHANS = N_ACOUSTIC + 1           # 13

REPORT_PATH    = Path("data/raw/carp/output_report.csv")
RECORDINGS_DIR = Path("data/raw/carp")
OUT_DIR        = Path("data/carp/carp_perfile_split")


# ---------------------------------------------------------------------------
# Recording cache with per-file normalisation
# ---------------------------------------------------------------------------

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
        m = np.nanmean(raw, axis=0).astype(np.float32)
        s = np.nanstd(raw,  axis=0).astype(np.float32)
        s[s < 1e-8] = 1.0
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            raw = np.where(nan_mask, m[np.newaxis, :], raw)
        self._raw[fname]  = raw
        self._mean[fname] = m
        self._std[fname]  = s
        return True

    def normalised_window(self, fname: str, mid: int) -> np.ndarray | None:
        raw = self._raw.get(fname)
        if raw is None:
            return None
        win  = _extract_window(raw, mid)
        win  = ((win - self._mean[fname]) / self._std[fname]).astype(np.float32)
        salt = np.full((WINDOW_SIZE, 1), SALT_VALUE, dtype=np.float32)
        return np.concatenate([win, salt], axis=1)   # (39, 13)

    def normalised_segment(self, fname: str, start: int) -> np.ndarray | None:
        raw = self._raw.get(fname)
        if raw is None:
            return None
        seg = raw[start : start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(seg), N_ACOUSTIC), np.float32)
            seg = np.concatenate([seg, pad])
        seg  = seg[:WINDOW_SIZE]
        seg  = ((seg - self._mean[fname]) / self._std[fname]).astype(np.float32)
        salt = np.full((WINDOW_SIZE, 1), SALT_VALUE, dtype=np.float32)
        return np.concatenate([seg, salt], axis=1)   # (39, 13)

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


# ---------------------------------------------------------------------------
# Positive / negative sample builders
# ---------------------------------------------------------------------------

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
        print(f"  Skipped {skipped} detections (file not in cache)")
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
    n_files = len(cache.all_files())
    for fname in cache.all_files():
        n_rows = cache.n_rows(fname)
        occ = sorted(occupied.get(fname, []))
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
    waves = []
    for i in idx:
        fname, start = candidates[i]
        win = cache.normalised_segment(fname, start)
        waves.append(win)
    return np.stack(waves)


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
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building carp per-file split (12 acoustic + 1 salt=0.0)")
    print("=" * 60)

    # Load report and parse ground truth from filenames
    report = pd.read_csv(REPORT_PATH)
    lengths, weights = [], []
    for fname in report["raw_data_file_name"]:
        mw = re.search(r"(\d+)gr",           fname)
        ml = re.search(r"(\d+(?:\.\d+)?)cm", fname)
        weights.append(float(mw.group(1)) if mw else None)
        lengths.append(float(ml.group(1)) if ml else None)
    report["_length"] = lengths
    report["_weight"] = weights
    report = report.dropna(subset=["_length", "_weight"]).reset_index(drop=True)

    print(f"\nDetections with valid ground truth: {len(report)}")

    # Load recordings
    print("\nLoading recordings ...")
    cache = RecordingCache()
    for p in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in p.name.lower():
            continue
        ok = cache.load(p.name, p)
        if ok:
            print(f"  {p.name:40s}  {cache.n_rows(p.name):6d} rows")

    loaded = set(cache.all_files())
    report = report[report["raw_data_file_name"].isin(loaded)].reset_index(drop=True)

    # Positive samples
    print("\nExtracting positive samples ...")
    pos_waves, pos_lengths, pos_weights, pos_strata = build_positive_samples(
        report, cache
    )
    n_pos = len(pos_waves)
    print(f"  Total: {n_pos}")
    for st in sorted(np.unique(pos_strata)):
        print(f"    {st:>4} cm  {(pos_strata == st).sum():3d}")

    # Stratified split
    print("\nStratified split (70 / 15 / 15) ...")
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
    print(f"  train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    # Negative samples
    n_neg_train = int(len(train_idx) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_idx)   / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_idx)  / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_total = n_neg_train + n_neg_val + n_neg_test
    print(f"  negatives — train: {n_neg_train}  val: {n_neg_val}  test: {n_neg_test}")

    print("\nGenerating negative samples ...")
    neg_all   = build_negative_samples(report, cache, n_neg_total, rng)
    neg_train = neg_all[:n_neg_train]
    neg_val   = neg_all[n_neg_train : n_neg_train + n_neg_val]
    neg_test  = neg_all[n_neg_train + n_neg_val :]

    # Combine and save
    print("\nSaving datasets ...")
    splits = {
        "train": (train_idx, neg_train),
        "val":   (val_idx,   neg_val),
        "test":  (test_idx,  neg_test),
    }
    summary = {}
    for name, (pos_idx, neg_waves) in splits.items():
        X, y_pres, y_len, y_wt = combine(
            pos_waves[pos_idx], pos_lengths[pos_idx],
            pos_weights[pos_idx], neg_waves, rng
        )
        path = OUT_DIR / f"{name}_dataset.npz"
        np.savez(path, X=X.astype(np.float32),
                 y_presence=y_pres, y_length=y_len, y_weight=y_wt)
        n = len(y_pres); pos_n = int(y_pres.sum())
        summary[name] = (n, pos_n)
        print(f"  {name:5s}: {n:5d} total  ({pos_n} pos / {n - pos_n} neg)  -> {path}")

    lines = [
        "Carp Per-File Split Dataset Build Report",
        "=" * 60,
        f"Source        : {RECORDINGS_DIR}",
        f"Window size   : {WINDOW_SIZE} samples at 40 Hz",
        f"Channels      : {N_ACOUSTIC} acoustic (per-file z-scored) + 1 salt=0.0",
        f"X shape       : (N, {WINDOW_SIZE}, {TOTAL_CHANS})",
        f"Presence ratio: {PRESENCE_RATIO:.0%} positive",
        f"Split         : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / {TEST_FRAC:.0%}",
        "",
        "Compatible with salted-water weight model (input_channels=12, physics_dim=1)",
        "",
        "Dataset sizes:",
    ]
    for name, (n, pos_n) in summary.items():
        lines.append(f"  {name:5s}: {n:5d} total  ({pos_n} pos / {n - pos_n} neg)")
    (OUT_DIR / "build_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Build report -> {OUT_DIR / 'build_report.txt'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
