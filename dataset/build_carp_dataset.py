"""Build train / val / test .npz files from the carp recordings.

Run from the project root:
    python dataset/build_carp_dataset.py

Architecture is identical to build_v4_dataset.py (12 acoustic channels,
39-sample windows at 40 Hz).  The only difference is the data source:
carp recordings from data/raw/carp/ instead of bream recordings.

The only information extracted from output_report.csv is:
    - raw_data_file_name        : which recording file the detection is in
    - start_index / end_index   : where in the recording the detection occurs
    - length_from_filename      : regression label (y_length)
    - weight_from_filename      : regression label (y_weight)

Output
------
    data/carp/
        train_dataset.npz   keys: X (N,39,12), y_presence, y_length, y_weight
        val_dataset.npz
        test_dataset.npz
        normalizer.npz      keys: mean, std  (shape: 12)
        build_report.txt
"""

import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE    = 39
PRESENCE_RATIO = 0.20
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.20
RANDOM_SEED    = 42

RAW_CHANNELS = [
    "F15",  "F37",  "F18",  "F32",  "F45",  "F67",
    "B15",  "B37",  "B18",  "B32",  "B45",  "B67",
]
INPUT_CHANNELS = len(RAW_CHANNELS)   # 12

REPORT_PATH    = Path("data/raw/carp/output_report.csv")
RECORDINGS_DIR = Path("data/raw/carp")
OUT_DIR        = Path("data/carp")


# ---------------------------------------------------------------------------
# Recording cache
# ---------------------------------------------------------------------------

class RecordingCache:
    def __init__(self, rec_lookup: dict[str, Path]):
        self._lookup = rec_lookup
        self._cache: dict[str, np.ndarray] = {}

    def get(self, fname: str) -> np.ndarray | None:
        if fname not in self._lookup:
            return None
        if fname not in self._cache:
            df = pd.read_csv(self._lookup[fname])
            for col in RAW_CHANNELS:
                if col not in df.columns:
                    df[col] = 0.0
            self._cache[fname] = df[RAW_CHANNELS].values.astype(np.float32)
        return self._cache[fname]

    def all_files(self) -> list[str]:
        return list(self._lookup.keys())


def _extract_window(signal: np.ndarray, mid: int) -> np.ndarray:
    half      = WINDOW_SIZE // 2
    start     = mid - half
    end       = start + WINDOW_SIZE
    n         = len(signal)
    pad_left  = max(0, -start)
    pad_right = max(0, end - n)
    s_start   = max(0, start)
    s_end     = min(n, end)

    window = signal[s_start:s_end]
    if pad_left > 0:
        window = np.concatenate(
            [np.zeros((pad_left, INPUT_CHANNELS), dtype=np.float32), window]
        )
    if pad_right > 0:
        window = np.concatenate(
            [window, np.zeros((pad_right, INPUT_CHANNELS), dtype=np.float32)]
        )
    return window[:WINDOW_SIZE]


# ---------------------------------------------------------------------------
# Positive samples
# ---------------------------------------------------------------------------

def build_positive_samples(detections: pd.DataFrame, cache: RecordingCache) -> dict:
    waves, lengths, weights, classes = [], [], [], []
    skipped = 0

    for _, row in detections.iterrows():
        sig = cache.get(row["raw_data_file_name"])
        if sig is None:
            skipped += 1
            continue
        mid    = int((row["start_index"] + row["end_index"]) // 2)
        window = _extract_window(sig, mid)
        waves.append(window)
        lengths.append(float(row["length_from_filename"]))
        weights.append(float(row["weight_from_filename"]))
        classes.append(str(int(row["length_from_filename"])))

    if skipped:
        print(f"  Skipped {skipped} detections (recording file unavailable)")

    return {
        "wave":         np.stack(waves),
        "y_presence":   np.ones(len(waves), dtype=np.int64),
        "y_length":     np.array(lengths, dtype=np.float64),
        "y_weight":     np.array(weights, dtype=np.float64),
        "length_class": np.array(classes),
    }


# ---------------------------------------------------------------------------
# Negative samples
# ---------------------------------------------------------------------------

def build_negative_samples(
    detections: pd.DataFrame,
    cache: RecordingCache,
    n_total: int,
    rng: np.random.Generator,
) -> dict:
    GUARD = WINDOW_SIZE

    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for _, row in detections.iterrows():
        fname = row["raw_data_file_name"]
        s = max(0, int(row["start_index"]) - GUARD)
        e = int(row["end_index"]) + GUARD
        occupied[fname].append((s, e))

    candidates: list[tuple[str, int]] = []
    for fname in cache.all_files():
        sig = cache.get(fname)
        if sig is None:
            continue
        n_rows = len(sig)
        occ = sorted(occupied.get(fname, []))
        free_starts = [0]
        free_ends   = []
        for s, e in occ:
            free_ends.append(max(0, s - 1))
            free_starts.append(min(n_rows, e + 1))
        free_ends.append(n_rows)

        for fs, fe in zip(free_starts, free_ends):
            max_start = fe - WINDOW_SIZE
            if max_start > fs:
                step = max(1, (max_start - fs) // max(1, n_total // len(cache.all_files())))
                for pos in range(fs, max_start, step):
                    candidates.append((fname, pos))

    if len(candidates) < n_total:
        rng.shuffle(candidates)
        while len(candidates) < n_total:
            fname, pos = random.choice(candidates)
            jitter  = int(rng.integers(-5, 6))
            sig     = cache.get(fname)
            new_pos = max(0, min(pos + jitter, len(sig) - WINDOW_SIZE))
            candidates.append((fname, new_pos))

    idx      = rng.choice(len(candidates), size=n_total, replace=len(candidates) < n_total)
    selected = [candidates[i] for i in idx]

    waves = []
    for fname, start in selected:
        sig    = cache.get(fname)
        window = sig[start : start + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            pad    = np.zeros((WINDOW_SIZE - len(window), INPUT_CHANNELS), dtype=np.float32)
            window = np.concatenate([window, pad])
        waves.append(window[:WINDOW_SIZE])

    n = len(waves)
    return {
        "wave":         np.stack(waves),
        "y_presence":   np.zeros(n, dtype=np.int64),
        "y_length":     np.zeros(n, dtype=np.float64),
        "y_weight":     np.zeros(n, dtype=np.float64),
        "length_class": np.array(["0"] * n),
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(pos: dict, rng: np.random.Generator):
    labels = pos["length_class"]
    n      = len(labels)

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.15, random_state=int(rng.integers(0, 10_000))
    )
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), labels))

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=round(VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), 4),
        random_state=int(rng.integers(0, 10_000)),
    )
    train_rel, val_rel = next(
        sss2.split(np.zeros(len(trainval_idx)), labels[trainval_idx])
    )
    return trainval_idx[train_rel], trainval_idx[val_rel], test_idx


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------

def oversample_by_class(pos: dict, idx: np.ndarray, rng: np.random.Generator):
    classes   = pos["length_class"][idx]
    unique_cl = np.unique(classes)
    max_count = max((classes == c).sum() for c in unique_cl)

    new_idx = list(idx)
    for c in unique_cl:
        c_idx   = idx[classes == c]
        deficit = max_count - len(c_idx)
        if deficit > 0:
            new_idx.extend(rng.choice(c_idx, size=deficit, replace=True).tolist())

    new_idx = np.array(new_idx)
    rng.shuffle(new_idx)
    return new_idx


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_normalizer(X_train: np.ndarray):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def normalise(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def _subset(d: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in d.items() if k != "length_class"}


def combine(pos_subset: dict, neg: dict) -> dict:
    wave       = np.concatenate([pos_subset["wave"],       neg["wave"]])
    y_presence = np.concatenate([pos_subset["y_presence"], neg["y_presence"]])
    y_length   = np.concatenate([pos_subset["y_length"],   neg["y_length"]])
    y_weight   = np.concatenate([pos_subset["y_weight"],   neg["y_weight"]])

    rng  = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(y_presence))
    return {
        "X":          wave[perm],
        "y_presence": y_presence[perm],
        "y_length":   y_length[perm],
        "y_weight":   y_weight[perm],
    }


def _subset_pos(pos: dict, idx: np.ndarray) -> dict:
    return {
        "wave":       pos["wave"][idx],
        "y_presence": pos["y_presence"][idx],
        "y_length":   pos["y_length"][idx],
        "y_weight":   pos["y_weight"][idx],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building carp dataset  (12 acoustic channels)")
    print("=" * 60)

    # Load report
    report = pd.read_csv(REPORT_PATH)
    report["length_from_filename"] = pd.to_numeric(
        report["length_from_filename"], errors="coerce")
    report["weight_from_filename"] = pd.to_numeric(
        report["weight_from_filename"], errors="coerce")
    report = report.dropna(subset=["length_from_filename", "weight_from_filename"])
    report = report[["raw_data_file_name", "start_index", "end_index",
                     "length_from_filename", "weight_from_filename"]].copy()
    report = report.reset_index(drop=True)

    print(f"\nDetections with ground truth : {len(report)}")
    print("  Length class distribution:")
    for lv, cnt in report["length_from_filename"].value_counts().sort_index().items():
        print(f"    {lv:4.0f} cm  {cnt:4d}")

    # Build recording lookup
    rec_lookup: dict[str, Path] = {}
    for fname in report["raw_data_file_name"].unique():
        p = RECORDINGS_DIR / fname
        if p.exists():
            rec_lookup[fname] = p
        else:
            print(f"  WARNING: {p} not found")

    print(f"\nRecording files found : {len(rec_lookup)}")

    # Cache recordings
    cache = RecordingCache(rec_lookup)
    for fname in rec_lookup:
        cache.get(fname)
    print(f"  Loaded {len(rec_lookup)} recording files")

    # Only keep detections whose recording file was found
    report = report[report["raw_data_file_name"].isin(rec_lookup)].reset_index(drop=True)

    # Positive samples
    pos   = build_positive_samples(report, cache)
    n_pos = len(pos["y_presence"])
    print(f"  Positive samples: {n_pos}")

    # Split
    print("\nSplitting positives (stratified by length class) ...")
    train_idx, val_idx, test_idx = stratified_split(pos, rng)
    print(f"  Raw split  -  train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    train_idx_bal = oversample_by_class(pos, train_idx, rng)
    print(f"  After oversampling  -  train: {len(train_idx_bal)}")
    tc = pos["length_class"][train_idx_bal]
    for lv in sorted(np.unique(tc)):
        print(f"    {lv:>4} cm  {(tc == lv).sum():4d}")

    # Negative samples
    print("\nGenerating negative samples ...")
    n_neg_train = int(len(train_idx_bal) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_idx)       / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_idx)      / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_total = n_neg_train + n_neg_val + n_neg_test
    print(f"  Negatives needed: train={n_neg_train}  val={n_neg_val}  test={n_neg_test}")

    neg_all   = build_negative_samples(report, cache, n_neg_total, rng)
    neg_train = _subset_pos(neg_all, np.arange(0, n_neg_train))
    neg_val   = _subset_pos(neg_all, np.arange(n_neg_train, n_neg_train + n_neg_val))
    neg_test  = _subset_pos(neg_all, np.arange(n_neg_train + n_neg_val, n_neg_total))

    # Build final arrays
    print("\nBuilding X arrays ...")
    train_data = combine(_subset_pos(pos, train_idx_bal), neg_train)
    val_data   = combine(_subset_pos(pos, val_idx),       neg_val)
    test_data  = combine(_subset_pos(pos, test_idx),      neg_test)
    print(f"  X shape: {train_data['X'].shape}")

    # Normalise
    print("\nNormalising ...")
    mean, std = compute_normalizer(train_data["X"])
    train_data["X"] = normalise(train_data["X"], mean, std)
    val_data["X"]   = normalise(val_data["X"],   mean, std)
    test_data["X"]  = normalise(test_data["X"],  mean, std)
    np.savez(OUT_DIR / "normalizer.npz", mean=mean, std=std)
    print(f"  Saved normalizer -> {OUT_DIR / 'normalizer.npz'}")

    # Save
    print("\nSaving datasets ...")
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = OUT_DIR / f"{name}_dataset.npz"
        np.savez(path, X=data["X"], y_presence=data["y_presence"],
                 y_length=data["y_length"], y_weight=data["y_weight"])
        n     = len(data["y_presence"])
        pos_n = int(data["y_presence"].sum())
        print(f"  {name:5s}: {n:5d} samples  ({pos_n} pos / {n - pos_n} neg)")

    # Build report
    report_lines = [
        "Carp Dataset Build Report  (12 acoustic channels)",
        "=" * 60,
        f"Source report   : {REPORT_PATH}",
        f"Recordings dir  : {RECORDINGS_DIR}",
        f"Window size     : {WINDOW_SIZE} samples at 40 Hz",
        f"Input channels  : {INPUT_CHANNELS}",
        f"Presence ratio  : {PRESENCE_RATIO:.0%} positive",
        f"Split           : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / 15%",
        "",
        "Raw channels:",
    ]
    for i, ch in enumerate(RAW_CHANNELS):
        report_lines.append(f"  [{i+1:2d}] {ch}")

    report_lines += [
        "",
        f"Total positives : {n_pos}",
        "",
        "Final dataset sizes",
    ]
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        n     = len(data["y_presence"])
        pos_n = int(data["y_presence"].sum())
        report_lines.append(
            f"  {name:5s}: {n:5d} total  ({pos_n} positive / {n - pos_n} negative)"
        )

    report_lines += [
        "",
        "Length class distribution (original):",
    ]
    for lv, cnt in report["length_from_filename"].value_counts().sort_index().items():
        report_lines.append(f"  {lv:4.0f} cm  {cnt:4d}")

    report_lines += [
        "",
        "To train with this dataset:",
        "  python training/train.py --config training/train_config_carp.yaml",
        "  python training/train_weight.py --config training/train_config_carp_weight.yaml",
    ]

    (OUT_DIR / "build_report.txt").write_text("\n".join(report_lines))
    print(f"\n  Build report -> {OUT_DIR / 'build_report.txt'}")
    print("\n" + "=" * 60)
    print("Done.  Carp dataset saved to data/carp/")
    print("=" * 60)


if __name__ == "__main__":
    main()
