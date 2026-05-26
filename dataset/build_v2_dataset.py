"""Build improved train / val / test .npz files (v2 dataset).

Run from the project root:
    python dataset/build_v2_dataset.py

Design
------
Sources
    Detection reports are merged from two sources so that all 10 length
    classes are well-represented even in the training split:
        data/raw/output_report.csv          – 1 284 rows, 86 cols (richer features)
        data/raw/test/output_report_F.csv   – 1 795 rows, 45 cols (more detections)
    Detections that appear in both are de-duplicated; the main-report version
    is kept because it carries more physics features.

Raw waveform (X_wave)
    Each detection maps to a row range [start_index, end_index] inside a
    recording CSV file (12 sensor channels).  A fixed-length window of
    WINDOW_SIZE = 39 samples is extracted centred on the detection midpoint.
    Windows that fall outside the file are zero-padded.

Physics features (X_feat)
    35 scalar features per detection (amplitude, geometry, power, ring
    statistics, sensor peaks …) are read from the reports.  Columns not
    present in a given report are filled with 0.  For negative (no-fish)
    samples all physics features are 0.

Final feature array (X)
    X_feat is tiled across the WINDOW_SIZE time axis and concatenated with
    X_wave, giving a final shape of (N, 39, 12 + 35) = (N, 39, 47).
    This lets the existing model architecture work without code changes —
    only model_config.yaml needs input_channels updated to 47.

Negative samples
    39-sample windows are drawn from the recording files outside all
    detection windows (genuine background signal).  The positive-to-negative
    ratio is fixed at PRESENCE_RATIO = 0.20 (20 % positives, 80 % negatives)
    after class balancing.

Split
    Stratified 70 / 20 / 10 at the detection level, stratified by length
    class so every class appears in all three splits.

Class balancing
    Training positives are oversampled (with replacement) so every length
    class contributes equally.  Val and test keep the natural distribution.

Normalisation
    Per-channel mean / std computed from the training set and applied to
    all splits.  Saved to data/v2/normalizer.npz.

Output
    data/v2/
        train_dataset.npz       keys: X, y_presence, y_length, y_weight
        val_dataset.npz
        test_dataset.npz
        normalizer.npz          keys: mean, std  (shape: 47)
        build_report.txt        summary statistics
"""

import os
import sys
import random
import textwrap
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE     = 39          # time steps per sample (matches existing model)
PRESENCE_RATIO  = 0.20        # fraction of the dataset that is fish-present
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.20
# TEST_FRAC     = 0.10  (remainder)

RANDOM_SEED     = 42

SENSOR_COLS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]

# 35 physics features; columns missing in a report are filled with 0
PHYSICS_FEATURES = [
    "front_amplitude",        "front_relative_strength",
    "back_amplitude",         "back_relative_strength",
    "front_back_correlation",
    "rms",                    "duration",
    "area_below_reference",
    "width",                  "height",                  "volume",
    "distance_from_center_cm",
    "ring_distance_samples",
    "adaptive_measure",       "total_power",
    "ring_overlap_ratio",     "ring_gap_ratio",
    "max_amplitude_F15",      "max_amplitude_F37",
    "max_amplitude_B15",      "max_amplitude_B37",
    "max_amplitude_F18",      "max_amplitude_F32",
    "max_amplitude_F45",      "max_amplitude_F67",
    "max_amplitude_B18",      "max_amplitude_B32",
    "max_amplitude_B45",      "max_amplitude_B67",
    "peak_std_peripheral_sensors",  "peak_mean_peripheral_sensors",
    "peak_ratio_60_5",        "peak_ratio_30_5",
    "sensor_intersection_point",    "intersection_to_peak_ratio",
]

INPUT_CHANNELS = len(SENSOR_COLS) + len(PHYSICS_FEATURES)  # 47

# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find_recording_files(report_df: pd.DataFrame) -> dict[str, Path]:
    """Return {filename: full_path} for every raw_data_file_name in the report.

    Searches in data/raw/recordings/ and data/raw/test/.
    """
    search_dirs = [
        Path("data/raw/recordings"),
        Path("data/raw/test"),
    ]
    lookup: dict[str, Path] = {}
    for fname in report_df["raw_data_file_name"].unique():
        for d in search_dirs:
            p = d / fname
            if p.exists():
                lookup[fname] = p
                break
    return lookup


# ---------------------------------------------------------------------------
# Report loading and merging
# ---------------------------------------------------------------------------

def _load_reports(rec_lookup: dict[str, Path]) -> pd.DataFrame:
    """Load, filter, and merge detection reports.

    Priority: main report carries richer physics features and takes
    precedence over F-report entries for the same (file, start_index).
    F-report adds unique detections not present in the main report.
    Only rows whose recording file is available on disk are kept.
    """
    available_files = set(rec_lookup.keys())

    main = pd.read_csv("data/raw/output_report.csv")
    main = main[main["raw_data_file_name"].isin(available_files)].copy()
    main["_source"] = "main"

    f = pd.read_csv("data/raw/test/output_report_F.csv")
    f = f[f["raw_data_file_name"].isin(available_files)].copy()
    f["_source"] = "f"

    # Ensure the 35 physics-feature columns exist in both (fill missing with NaN)
    for df in (main, f):
        for col in PHYSICS_FEATURES + ["length_from_filename", "weight_from_filename"]:
            if col not in df.columns:
                df[col] = np.nan

    # Keep core columns
    keep = (
        ["raw_data_file_name", "start_index", "end_index",
         "length_from_filename", "weight_from_filename", "_source"]
        + PHYSICS_FEATURES
    )
    main = main[keep].copy()
    f    = f[[c for c in keep if c in f.columns] +
             [c for c in keep if c not in f.columns and c not in
              ["raw_data_file_name", "start_index", "end_index"]]].copy()

    # Rebuild F with all columns (fill missing with 0)
    for col in keep:
        if col not in f.columns:
            f[col] = np.nan
    f = f[keep].copy()

    # De-duplicate: build main-report key set, keep only F-unique detections
    main_keys = set(zip(main["raw_data_file_name"], main["start_index"]))
    f_unique  = f[~f.apply(
        lambda r: (r["raw_data_file_name"], r["start_index"]) in main_keys, axis=1
    )]

    merged = pd.concat([main, f_unique], ignore_index=True)

    # Drop rows with invalid length / weight
    merged["length_from_filename"] = pd.to_numeric(
        merged["length_from_filename"], errors="coerce"
    )
    merged["weight_from_filename"] = pd.to_numeric(
        merged["weight_from_filename"], errors="coerce"
    )
    merged = merged.dropna(subset=["length_from_filename", "weight_from_filename"])
    merged["length_from_filename"] = merged["length_from_filename"].astype(float)
    merged["weight_from_filename"] = merged["weight_from_filename"].astype(float)

    # Fill remaining NaN physics features with 0
    merged[PHYSICS_FEATURES] = merged[PHYSICS_FEATURES].fillna(0.0)

    merged = merged.reset_index(drop=True)
    print(f"  Merged detections  : {len(merged)}")
    print(f"    from main report : {(merged['_source']=='main').sum()}")
    print(f"    from F report    : {(merged['_source']=='f').sum()}")
    print(f"  Length class distribution:")
    for lv, cnt in merged["length_from_filename"].value_counts().sort_index().items():
        print(f"    {lv:4.0f} cm  {cnt:4d}")
    return merged


# ---------------------------------------------------------------------------
# Waveform extraction
# ---------------------------------------------------------------------------

class RecordingCache:
    """Lazy-loads recording CSV files and caches them in memory."""

    def __init__(self, rec_lookup: dict[str, Path]):
        self._lookup = rec_lookup
        self._cache: dict[str, np.ndarray] = {}

    def get(self, fname: str) -> np.ndarray | None:
        """Return the signal array (n_rows, 12) for the recording, or None."""
        if fname not in self._lookup:
            return None
        if fname not in self._cache:
            df = pd.read_csv(self._lookup[fname])
            available = [c for c in SENSOR_COLS if c in df.columns]
            if len(available) < len(SENSOR_COLS):
                return None
            self._cache[fname] = df[SENSOR_COLS].values.astype(np.float32)
        return self._cache[fname]

    def all_files(self) -> list[str]:
        return list(self._lookup.keys())


def _extract_window(signal: np.ndarray, mid: int) -> np.ndarray:
    """Extract a WINDOW_SIZE window centred on mid, zero-padding at edges.

    Returns shape (WINDOW_SIZE, 12).
    """
    half  = WINDOW_SIZE // 2
    start = mid - half
    end   = start + WINDOW_SIZE
    n     = len(signal)

    # Crop to valid range, then pad
    pad_left  = max(0, -start)
    pad_right = max(0, end - n)
    s_start   = max(0, start)
    s_end     = min(n, end)

    window = signal[s_start:s_end]
    if pad_left > 0:
        window = np.concatenate([np.zeros((pad_left, 12), dtype=np.float32), window])
    if pad_right > 0:
        window = np.concatenate([window, np.zeros((pad_right, 12), dtype=np.float32)])
    return window[:WINDOW_SIZE]


# ---------------------------------------------------------------------------
# Positive sample extraction
# ---------------------------------------------------------------------------

def build_positive_samples(
    detections: pd.DataFrame,
    cache: RecordingCache,
) -> dict:
    """Extract waveform + features for every valid detection row.

    Returns dict with keys: wave (N,39,12), feat (N,35),
    y_presence, y_length, y_weight, length_class (str label for stratification).
    """
    waves, feats, lengths, weights, classes = [], [], [], [], []
    skipped = 0

    for _, row in detections.iterrows():
        sig = cache.get(row["raw_data_file_name"])
        if sig is None:
            skipped += 1
            continue

        mid = int((row["start_index"] + row["end_index"]) // 2)
        window = _extract_window(sig, mid)

        phys = np.array(
            [float(row[c]) for c in PHYSICS_FEATURES], dtype=np.float32
        )

        waves.append(window)
        feats.append(phys)
        lengths.append(float(row["length_from_filename"]))
        weights.append(float(row["weight_from_filename"]))
        classes.append(str(int(row["length_from_filename"])))

    if skipped:
        print(f"  Skipped {skipped} detections (recording file not cached)")

    return {
        "wave":     np.stack(waves),                            # (N, 39, 12)
        "feat":     np.stack(feats),                            # (N, 35)
        "y_presence": np.ones(len(waves), dtype=np.int64),
        "y_length":   np.array(lengths, dtype=np.float64),
        "y_weight":   np.array(weights, dtype=np.float64),
        "length_class": np.array(classes),
    }


# ---------------------------------------------------------------------------
# Negative sample extraction
# ---------------------------------------------------------------------------

def build_negative_samples(
    detections: pd.DataFrame,
    cache: RecordingCache,
    n_total: int,
    rng: np.random.Generator,
) -> dict:
    """Sample n_total background windows from non-detection regions.

    For each recording file, detection windows (with a small guard margin)
    are marked as occupied.  Random WINDOW_SIZE-length windows are drawn
    from the remaining signal.
    """
    GUARD = WINDOW_SIZE  # extra buffer around each detection

    # Build occupied intervals per file
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for _, row in detections.iterrows():
        fname = row["raw_data_file_name"]
        s = max(0, int(row["start_index"]) - GUARD)
        e = int(row["end_index"]) + GUARD
        occupied[fname].append((s, e))

    # Collect candidate (file, start_pos) tuples for negative windows
    candidates: list[tuple[str, int]] = []
    for fname in cache.all_files():
        sig = cache.get(fname)
        if sig is None:
            continue
        n_rows = len(sig)
        # Merge and sort occupied intervals
        occ = sorted(occupied.get(fname, []))
        # Build free regions
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
        # Not enough unique positions — allow duplicates (different jitter)
        rng.shuffle(candidates)
        while len(candidates) < n_total:
            fname, pos = random.choice(candidates)
            jitter = int(rng.integers(-5, 6))
            sig = cache.get(fname)
            new_pos = max(0, min(pos + jitter, len(sig) - WINDOW_SIZE))
            candidates.append((fname, new_pos))

    # Sample without replacement (or with if unavoidable)
    idx      = rng.choice(len(candidates), size=n_total, replace=len(candidates) < n_total)
    selected = [candidates[i] for i in idx]

    waves = []
    for fname, start in selected:
        sig    = cache.get(fname)
        window = sig[start : start + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            pad    = np.zeros((WINDOW_SIZE - len(window), 12), dtype=np.float32)
            window = np.concatenate([window, pad])
        waves.append(window[:WINDOW_SIZE])

    n = len(waves)
    return {
        "wave":       np.stack(waves),                               # (N, 39, 12)
        "feat":       np.zeros((n, len(PHYSICS_FEATURES)), dtype=np.float32),
        "y_presence": np.zeros(n, dtype=np.int64),
        "y_length":   np.zeros(n, dtype=np.float64),
        "y_weight":   np.zeros(n, dtype=np.float64),
        "length_class": np.array(["0"] * n),
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    pos: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) for the positive samples."""
    labels = pos["length_class"]
    n      = len(labels)

    # First pass: split off test (10 %)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=int(rng.integers(0, 10_000))
    )
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), labels))

    # Second pass: split trainval into train (70 %) and val (20 %)
    # val_size relative to trainval = 0.20 / 0.90 ≈ 0.222
    labels_trainval = labels[trainval_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=round(VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), 4),
        random_state=int(rng.integers(0, 10_000)),
    )
    train_rel, val_rel = next(sss2.split(np.zeros(len(trainval_idx)), labels_trainval))

    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Class balancing (oversampling)
# ---------------------------------------------------------------------------

def oversample_by_class(
    pos: dict,
    idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return oversampled index so all length classes are equally sized."""
    classes   = pos["length_class"][idx]
    unique_cl = np.unique(classes)
    max_count = max((classes == c).sum() for c in unique_cl)

    new_idx = list(idx)
    for c in unique_cl:
        c_idx   = idx[classes == c]
        deficit = max_count - len(c_idx)
        if deficit > 0:
            extra = rng.choice(c_idx, size=deficit, replace=True)
            new_idx.extend(extra.tolist())

    new_idx = np.array(new_idx)
    rng.shuffle(new_idx)
    return new_idx


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_normalizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std computed from the training set."""
    # X_train: (N, 39, C) — compute stats over (N, T) for each channel C
    flat  = X_train.reshape(-1, X_train.shape[-1])  # (N*39, C)
    mean  = flat.mean(axis=0).astype(np.float32)
    std   = flat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0  # prevent division by zero
    return mean, std


def normalise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Build final sample array
# ---------------------------------------------------------------------------

def combine_and_build_X(pos_subset: dict, neg: dict) -> dict:
    """Concatenate positives and negatives; tile physics features across time."""
    wave = np.concatenate([pos_subset["wave"], neg["wave"]], axis=0)  # (N, 39, 12)
    feat = np.concatenate([pos_subset["feat"], neg["feat"]], axis=0)  # (N, 35)

    # Tile physics features: (N, 35) -> (N, 39, 35) then concat on channel dim
    feat_tiled = np.tile(feat[:, np.newaxis, :], (1, WINDOW_SIZE, 1))  # (N, 39, 35)
    X = np.concatenate([wave, feat_tiled], axis=2)                      # (N, 39, 47)

    y_presence = np.concatenate([pos_subset["y_presence"], neg["y_presence"]])
    y_length   = np.concatenate([pos_subset["y_length"],   neg["y_length"]])
    y_weight   = np.concatenate([pos_subset["y_weight"],   neg["y_weight"]])

    # Shuffle
    rng  = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(y_presence))
    return {
        "X":          X[perm],
        "y_presence": y_presence[perm],
        "y_length":   y_length[perm],
        "y_weight":   y_weight[perm],
    }


def _subset_pos(pos: dict, idx: np.ndarray) -> dict:
    return {
        "wave":       pos["wave"][idx],
        "feat":       pos["feat"][idx],
        "y_presence": pos["y_presence"][idx],
        "y_length":   pos["y_length"][idx],
        "y_weight":   pos["y_weight"][idx],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    out_dir = Path("data/v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building v2 dataset")
    print("=" * 60)

    # --- Step 1: Find recording files ---------------------------------
    # Load temporarily to build the file lookup
    _tmp = pd.read_csv("data/raw/output_report.csv")
    _tmp2 = pd.read_csv("data/raw/test/output_report_F.csv")
    rec_lookup = _find_recording_files(pd.concat([_tmp, _tmp2]))
    print(f"\nRecording files found : {len(rec_lookup)}")

    # --- Step 2: Load and merge detection reports ---------------------
    print("\nLoading detection reports …")
    detections = _load_reports(rec_lookup)

    # --- Step 3: Cache recordings and extract positive samples --------
    print("\nCaching recordings and extracting positive waveforms …")
    cache = RecordingCache(rec_lookup)
    # Pre-load all files
    for fname in rec_lookup:
        cache.get(fname)
    print(f"  Loaded {len(rec_lookup)} recording files")

    pos = build_positive_samples(detections, cache)
    n_pos = len(pos["y_presence"])
    print(f"  Positive samples     : {n_pos}")

    # --- Step 4: Stratified split on positives ------------------------
    print("\nSplitting positives (stratified by length class) …")
    train_idx, val_idx, test_idx = stratified_split(pos, rng)
    print(f"  Raw split  —  train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    # --- Step 5: Oversample train positives ---------------------------
    train_idx_bal = oversample_by_class(pos, train_idx, rng)
    print(f"  After oversampling  —  train: {len(train_idx_bal)}")
    # Verify class balance
    tc = pos["length_class"][train_idx_bal]
    for lv in sorted(np.unique(tc)):
        print(f"    {lv:>4} cm  {(tc == lv).sum():4d}")

    # --- Step 6: Generate negative samples ----------------------------
    print("\nGenerating negative samples …")
    n_neg_train = int(len(train_idx_bal) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_idx)       / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_idx)      / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_total = n_neg_train + n_neg_val + n_neg_test
    print(f"  Negatives needed: train={n_neg_train}  val={n_neg_val}  test={n_neg_test}")

    neg_all = build_negative_samples(detections, cache, n_neg_total, rng)

    neg_train = _subset_pos(neg_all, np.arange(0, n_neg_train))
    neg_val   = _subset_pos(neg_all, np.arange(n_neg_train, n_neg_train + n_neg_val))
    neg_test  = _subset_pos(neg_all, np.arange(n_neg_train + n_neg_val, n_neg_total))

    # --- Step 7: Combine and build raw X arrays -----------------------
    print("\nBuilding raw X arrays (tiling physics features) …")
    train_data = combine_and_build_X(_subset_pos(pos, train_idx_bal), neg_train)
    val_data   = combine_and_build_X(_subset_pos(pos, val_idx),        neg_val)
    test_data  = combine_and_build_X(_subset_pos(pos, test_idx),       neg_test)

    # --- Step 8: Normalise --------------------------------------------
    print("\nNormalising …")
    mean, std = compute_normalizer(train_data["X"])
    train_data["X"] = normalise(train_data["X"], mean, std)
    val_data["X"]   = normalise(val_data["X"],   mean, std)
    test_data["X"]  = normalise(test_data["X"],  mean, std)

    np.savez(out_dir / "normalizer.npz", mean=mean, std=std)
    print(f"  Saved normalizer -> {out_dir / 'normalizer.npz'}")

    # --- Step 9: Save npz files ---------------------------------------
    print("\nSaving datasets …")
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = out_dir / f"{name}_dataset.npz"
        np.savez(
            path,
            X=data["X"],
            y_presence=data["y_presence"],
            y_length=data["y_length"],
            y_weight=data["y_weight"],
        )
        n   = len(data["y_presence"])
        pos_n = data["y_presence"].sum()
        print(f"  {name:5s}: {n:5d} samples  ({pos_n} pos / {n - pos_n} neg)")

    # --- Step 10: Build report ----------------------------------------
    report_lines = [
        "v2 Dataset Build Report",
        "=" * 60,
        f"Window size          : {WINDOW_SIZE} samples",
        f"Input channels       : {INPUT_CHANNELS}  (12 waveform + 35 physics)",
        f"Presence ratio       : {PRESENCE_RATIO:.0%} positive",
        f"Split                : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / 10%",
        "",
        "Detection sources",
        f"  main report        : {(detections['_source']=='main').sum()} rows",
        f"  F report (unique)  : {(detections['_source']=='f').sum()} rows",
        f"  Total positives    : {n_pos}",
        "",
        "Final dataset sizes",
    ]
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        n     = len(data["y_presence"])
        pos_n = int(data["y_presence"].sum())
        report_lines.append(f"  {name:5s}: {n:5d} total  ({pos_n} positive / {n - pos_n} negative)")

    report_lines += [
        "",
        "Training length class distribution (after oversampling)",
    ]
    for lv in sorted(np.unique(pos["length_class"][train_idx_bal])):
        cnt = (pos["length_class"][train_idx_bal] == lv).sum()
        report_lines.append(f"  {lv:>4} cm  {cnt:4d}")

    report_lines += [
        "",
        "Physics features included",
    ]
    for i, f_name in enumerate(PHYSICS_FEATURES):
        report_lines.append(f"  [{i+1:2d}] {f_name}")

    report_lines += [
        "",
        "To retrain with this dataset, update training/train_config.yaml:",
        "    data_dir: data/v2/",
        "And update model/model_config.yaml:",
        "    input_channels: 47",
    ]

    report_text = "\n".join(report_lines)
    report_path = out_dir / "build_report.txt"
    report_path.write_text(report_text)
    print(f"\n  Build report -> {report_path}")

    print("\n" + "=" * 60)
    print("Done.  v2 dataset saved to data/v2/")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Set data_dir: data/v2/ in training/train_config.yaml")
    print("  2. Set input_channels: 47 in model/model_config.yaml")
    print("  3. python main.py --mode train")


if __name__ == "__main__":
    main()
