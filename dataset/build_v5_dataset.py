"""Build train / val / test .npz files — v5: combined bream + carp, length only.

Run from the project root:
    python dataset/build_v5_dataset.py

Key differences from v4
-----------------------
* Combines bream recordings (data/raw/recordings/, data/raw/test/) with
  carp recordings (data/raw/carp/).
* Per-file normalisation: every window is z-scored using its own recording's
  mean and std BEFORE being stored.  This is necessary because bream and carp
  sensors operate at very different DC baselines (~27 000 vs ~13 000).
* No global normaliser is applied on top — normalizer.npz stores the identity
  (mean=0, std=1) as a signal that data is already per-file normalised.
* Only y_presence and y_length are stored (no weight task).
* Stratification uses species+length_class labels to guarantee both species
  appear in every split.
* Length class oversampling balances across the combined length distribution.

Output
------
    data/v5/
        train_dataset.npz   keys: X (N,39,12), y_presence, y_length
        val_dataset.npz
        test_dataset.npz
        normalizer.npz      keys: mean (zeros,12), std (ones,12)  — identity
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
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
INPUT_CHANNELS = len(RAW_CHANNELS)   # 12

# Data sources
BREAM_REPORT_MAIN = Path("data/raw/output_report.csv")
BREAM_REPORT_F    = Path("data/raw/test/output_report_F.csv")
BREAM_DIRS        = [Path("data/raw/recordings"), Path("data/raw/test")]

CARP_REPORT       = Path("data/raw/carp/output_report.csv")
CARP_DIR          = Path("data/raw/carp")

OUT_DIR = Path("data/v5")


# ---------------------------------------------------------------------------
# Recording cache with per-file normalisation
# ---------------------------------------------------------------------------

class RecordingCache:
    """Loads recording CSVs and caches (raw_signal, per_file_mean, per_file_std)."""

    def __init__(self):
        self._raw:   dict[str, np.ndarray] = {}
        self._mean:  dict[str, np.ndarray] = {}
        self._std:   dict[str, np.ndarray] = {}

    def load(self, fname: str, path: Path) -> bool:
        if not path.exists():
            return False
        if fname not in self._raw:
            df = pd.read_csv(path)
            for col in RAW_CHANNELS:
                if col not in df.columns:
                    df[col] = 0.0
            raw = df[RAW_CHANNELS].values.astype(np.float32)
            # Compute stats ignoring NaN, then fill NaN with column mean
            m = np.nanmean(raw, axis=0).astype(np.float32)
            s = np.nanstd(raw,  axis=0).astype(np.float32)
            s[s < 1e-8] = 1.0
            # Replace NaN entries with column mean so windows are clean
            nan_mask = np.isnan(raw)
            if nan_mask.any():
                raw = np.where(nan_mask, m[np.newaxis, :], raw)
            self._raw[fname]  = raw
            self._mean[fname] = m
            self._std[fname]  = s
        return True

    def get_normalised_window(self, fname: str, mid: int) -> np.ndarray | None:
        if fname not in self._raw:
            return None
        raw   = self._raw[fname]
        win   = _extract_window_raw(raw, mid)
        normed = (win - self._mean[fname]) / self._std[fname]
        return normed.astype(np.float32)

    def get_normalised_segment(self, fname: str, start: int) -> np.ndarray | None:
        if fname not in self._raw:
            return None
        raw = self._raw[fname]
        seg = raw[start : start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(seg), INPUT_CHANNELS), dtype=np.float32)
            seg = np.concatenate([seg, pad])
        seg = seg[:WINDOW_SIZE]
        normed = (seg - self._mean[fname]) / self._std[fname]
        return normed.astype(np.float32)

    def all_files(self) -> list[str]:
        return list(self._raw.keys())

    def n_rows(self, fname: str) -> int:
        return len(self._raw.get(fname, []))


def _extract_window_raw(signal: np.ndarray, mid: int) -> np.ndarray:
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
            [np.zeros((pad_left, INPUT_CHANNELS), dtype=np.float32), window]
        )
    if pad_right > 0:
        window = np.concatenate(
            [window, np.zeros((pad_right, INPUT_CHANNELS), dtype=np.float32)]
        )
    return window[:WINDOW_SIZE]


# ---------------------------------------------------------------------------
# Detection report loading
# ---------------------------------------------------------------------------

def _load_bream_detections() -> pd.DataFrame:
    keep = ["raw_data_file_name", "start_index", "end_index", "length_from_filename"]

    main = pd.read_csv(BREAM_REPORT_MAIN)
    for col in ["length_from_filename"]:
        if col not in main.columns:
            main[col] = np.nan
    main = main[keep].copy()
    main["_source"] = "bream_main"

    f = pd.read_csv(BREAM_REPORT_F)
    for col in ["length_from_filename"]:
        if col not in f.columns:
            f[col] = np.nan
    f = f[keep].copy()
    f["_source"] = "bream_f"

    # De-duplicate: main takes priority
    main_keys = set(zip(main["raw_data_file_name"], main["start_index"]))
    f_unique  = f[~f.apply(
        lambda r: (r["raw_data_file_name"], r["start_index"]) in main_keys, axis=1
    )]

    merged = pd.concat([main, f_unique], ignore_index=True)
    merged["length_from_filename"] = pd.to_numeric(
        merged["length_from_filename"], errors="coerce"
    )
    merged = merged.dropna(subset=["length_from_filename"]).reset_index(drop=True)
    merged["species"] = "bream"
    return merged


def _load_carp_detections() -> pd.DataFrame:
    df = pd.read_csv(CARP_REPORT)
    df["length_from_filename"] = pd.to_numeric(
        df["length_from_filename"], errors="coerce"
    )
    df = df.dropna(subset=["length_from_filename"])
    df = df[["raw_data_file_name", "start_index", "end_index",
             "length_from_filename"]].copy()
    df["_source"] = "carp"
    df["species"] = "carp"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Positive samples (with per-file normalisation)
# ---------------------------------------------------------------------------

def build_positive_samples(detections: pd.DataFrame, cache: RecordingCache) -> dict:
    waves, lengths, classes = [], [], []
    skipped = 0

    for _, row in detections.iterrows():
        fname = row["raw_data_file_name"]
        mid   = int((row["start_index"] + row["end_index"]) // 2)
        win   = cache.get_normalised_window(fname, mid)
        if win is None:
            skipped += 1
            continue
        waves.append(win)
        lengths.append(float(row["length_from_filename"]))
        # stratum = species + length bucket (e.g. "bream_21", "carp_31")
        classes.append(f"{row['species']}_{int(row['length_from_filename'])}")

    if skipped:
        print(f"  Skipped {skipped} detections (recording not found)")

    return {
        "wave":         np.stack(waves),
        "y_presence":   np.ones(len(waves), dtype=np.int64),
        "y_length":     np.array(lengths, dtype=np.float64),
        "length_class": np.array(classes),
    }


# ---------------------------------------------------------------------------
# Negative samples (with per-file normalisation)
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
    n_files = len(cache.all_files())
    for fname in cache.all_files():
        n_rows = cache.n_rows(fname)
        occ    = sorted(occupied.get(fname, []))
        free_starts = [0]
        free_ends   = []
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

    if len(candidates) < n_total:
        while len(candidates) < n_total:
            fname, pos = random.choice(candidates)
            jitter  = int(rng.integers(-5, 6))
            new_pos = max(0, min(pos + jitter, cache.n_rows(fname) - WINDOW_SIZE))
            candidates.append((fname, new_pos))

    idx      = rng.choice(len(candidates), size=n_total, replace=len(candidates) < n_total)
    selected = [candidates[i] for i in idx]

    waves = []
    for fname, start in selected:
        win = cache.get_normalised_segment(fname, start)
        waves.append(win)

    n = len(waves)
    return {
        "wave":       np.stack(waves),
        "y_presence": np.zeros(n, dtype=np.int64),
        "y_length":   np.zeros(n, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(pos: dict, rng: np.random.Generator):
    labels = pos["length_class"]
    n      = len(labels)

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=int(rng.integers(0, 10_000))
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
    """Oversample by length_class (coarse cm bucket, ignoring species prefix)."""
    # Use just the cm number as the oversample unit so bream+carp at same
    # length are balanced together, not species separately
    classes   = np.array([c.split("_")[1] for c in pos["length_class"][idx]])
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
# Combine pos + neg
# ---------------------------------------------------------------------------

def _subset_pos(pos: dict, idx: np.ndarray) -> dict:
    return {
        "wave":       pos["wave"][idx],
        "y_presence": pos["y_presence"][idx],
        "y_length":   pos["y_length"][idx],
    }


def combine(pos_subset: dict, neg: dict) -> dict:
    wave       = np.concatenate([pos_subset["wave"],       neg["wave"]])
    y_presence = np.concatenate([pos_subset["y_presence"], neg["y_presence"]])
    y_length   = np.concatenate([pos_subset["y_length"],   neg["y_length"]])

    rng  = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(y_presence))
    return {
        "X":          wave[perm],
        "y_presence": y_presence[perm],
        "y_length":   y_length[perm],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building v5 dataset  (bream + carp combined, length only)")
    print("Per-file normalisation applied at window extraction time")
    print("=" * 60)

    # --- Load detection reports ---
    print("\nLoading detection reports ...")
    bream = _load_bream_detections()
    carp  = _load_carp_detections()
    print(f"  Bream detections : {len(bream)}")
    print(f"  Carp detections  : {len(carp)}")

    all_detections = pd.concat([bream, carp], ignore_index=True)

    # --- Build recording cache ---
    print("\nLoading recordings (per-file normalisation) ...")
    cache = RecordingCache()

    # Bream recordings
    n_bream_loaded = 0
    for fname in bream["raw_data_file_name"].unique():
        for d in BREAM_DIRS:
            if cache.load(fname, d / fname):
                n_bream_loaded += 1
                break

    # Carp recordings (all CSVs in CARP_DIR, including files with no detections
    # — they are valid background sources)
    n_carp_loaded = 0
    for p in CARP_DIR.glob("*.csv"):
        if "output" in p.name.lower():
            continue
        if cache.load(p.name, p):
            n_carp_loaded += 1

    print(f"  Bream recording files loaded : {n_bream_loaded}")
    print(f"  Carp  recording files loaded : {n_carp_loaded}")
    print(f"  Total recording files        : {len(cache.all_files())}")

    # Filter detections to only those whose recording was loaded
    loaded = set(cache.all_files())
    all_detections = all_detections[
        all_detections["raw_data_file_name"].isin(loaded)
    ].reset_index(drop=True)

    # --- Positive samples ---
    print("\nExtracting positive samples ...")
    pos   = build_positive_samples(all_detections, cache)
    n_pos = len(pos["y_presence"])
    print(f"  Total positives: {n_pos}")

    bream_pos = (np.array([c.startswith("bream") for c in pos["length_class"]])).sum()
    carp_pos  = n_pos - bream_pos
    print(f"    Bream : {bream_pos}")
    print(f"    Carp  : {carp_pos}")
    print("  Length class distribution:")
    for lv, cnt in sorted(
        [(c, (pos["length_class"] == c).sum()) for c in np.unique(pos["length_class"])]
    ):
        print(f"    {lv:>12}  {cnt:4d}")

    # --- Stratified split ---
    print("\nSplitting positives (stratified by species+length) ...")
    train_idx, val_idx, test_idx = stratified_split(pos, rng)
    print(f"  Raw split — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    train_idx_bal = oversample_by_class(pos, train_idx, rng)
    print(f"  After oversampling — train: {len(train_idx_bal)}")
    tc = np.array([c.split("_")[1] for c in pos["length_class"][train_idx_bal]])
    for lv in sorted(np.unique(tc)):
        print(f"    {lv:>4} cm  {(tc == lv).sum():4d}")

    # --- Negative samples ---
    print("\nGenerating negative samples ...")
    n_neg_train = int(len(train_idx_bal) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_idx)       / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_idx)      / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_total = n_neg_train + n_neg_val + n_neg_test
    print(f"  Negatives needed: train={n_neg_train}  val={n_neg_val}  test={n_neg_test}")

    neg_all   = build_negative_samples(all_detections, cache, n_neg_total, rng)
    neg_train = {"wave": neg_all["wave"][:n_neg_train],
                 "y_presence": neg_all["y_presence"][:n_neg_train],
                 "y_length":   neg_all["y_length"][:n_neg_train]}
    neg_val   = {"wave": neg_all["wave"][n_neg_train:n_neg_train+n_neg_val],
                 "y_presence": neg_all["y_presence"][n_neg_train:n_neg_train+n_neg_val],
                 "y_length":   neg_all["y_length"][n_neg_train:n_neg_train+n_neg_val]}
    neg_test  = {"wave": neg_all["wave"][n_neg_train+n_neg_val:],
                 "y_presence": neg_all["y_presence"][n_neg_train+n_neg_val:],
                 "y_length":   neg_all["y_length"][n_neg_train+n_neg_val:]}

    # --- Build final arrays ---
    print("\nBuilding X arrays ...")
    train_data = combine(_subset_pos(pos, train_idx_bal), neg_train)
    val_data   = combine(_subset_pos(pos, val_idx),       neg_val)
    test_data  = combine(_subset_pos(pos, test_idx),      neg_test)
    print(f"  Train X shape: {train_data['X'].shape}")
    print(f"  Val   X shape: {val_data['X'].shape}")
    print(f"  Test  X shape: {test_data['X'].shape}")

    # --- Save datasets ---
    print("\nSaving datasets ...")
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = OUT_DIR / f"{name}_dataset.npz"
        np.savez(path, X=data["X"], y_presence=data["y_presence"],
                 y_length=data["y_length"],
                 # store dummy y_weight (zeros) so the dataloader is compatible
                 y_weight=np.zeros(len(data["y_presence"]), dtype=np.float64))
        n     = len(data["y_presence"])
        pos_n = int(data["y_presence"].sum())
        print(f"  {name:5s}: {n:5d} samples  ({pos_n} pos / {n - pos_n} neg)")

    # --- Identity normalizer (data is already per-file normalised) ---
    mean_id = np.zeros(INPUT_CHANNELS, dtype=np.float32)
    std_id  = np.ones(INPUT_CHANNELS,  dtype=np.float32)
    np.savez(OUT_DIR / "normalizer.npz", mean=mean_id, std=std_id)
    print(f"\n  Normalizer (identity) -> {OUT_DIR / 'normalizer.npz'}")
    print("  NOTE: data is per-file normalised at build time; normalizer is identity.")

    # --- Build report ---
    report_lines = [
        "v5 Dataset Build Report  (bream + carp combined, length only)",
        "=" * 60,
        f"Window size        : {WINDOW_SIZE} samples at 40 Hz",
        f"Input channels     : {INPUT_CHANNELS}  (acoustic only)",
        f"Normalisation      : per-file (each window z-scored by its recording stats)",
        f"Presence ratio     : {PRESENCE_RATIO:.0%} positive",
        f"Split              : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / 10%",
        "",
        "Sources:",
        f"  Bream main report : {BREAM_REPORT_MAIN}  ({len(bream)} detections)",
        f"  Bream F report    : {BREAM_REPORT_F}",
        f"  Bream recordings  : {BREAM_DIRS}  ({n_bream_loaded} files)",
        f"  Carp report       : {CARP_REPORT}  ({len(carp)} detections)",
        f"  Carp recordings   : {CARP_DIR}  ({n_carp_loaded} files, incl. no-detection files)",
        "",
        f"Total positives    : {n_pos}  (bream={bream_pos}, carp={carp_pos})",
        "",
        "Final dataset sizes:",
    ]
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        n     = len(data["y_presence"])
        pos_n = int(data["y_presence"].sum())
        report_lines.append(
            f"  {name:5s}: {n:5d} total  ({pos_n} positive / {n - pos_n} negative)"
        )
    report_lines += [
        "",
        "Training length class distribution (after oversampling by cm bucket):",
    ]
    for lv in sorted(np.unique(tc)):
        report_lines.append(f"  {lv:>4} cm  {(tc == lv).sum():4d}")
    report_lines += [
        "",
        "To train:",
        "  python training/train.py --config training/train_config_v5.yaml",
    ]

    (OUT_DIR / "build_report.txt").write_text("\n".join(report_lines))
    print(f"  Build report -> {OUT_DIR / 'build_report.txt'}")
    print("\n" + "=" * 60)
    print("Done.  v5 dataset saved to data/v5/")
    print("=" * 60)


if __name__ == "__main__":
    main()
