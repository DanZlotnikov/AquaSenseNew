"""Build a test-only npz dataset from data/carp/salted_water/ recordings.

Length and weight are parsed directly from filenames (the report's
length_from_filename column has a bug for ".5cm" files, e.g. 41.5 → 5).

Two output files are produced so both trained models can be evaluated:

  data/carp/salted_water/test_v5.npz   — per-file normalised  (v5 length model)
  data/carp/salted_water/test_v4w.npz  — v4 globally normalised (v4 weight model)

Run from the project root:
    python dataset/build_salted_water_dataset.py
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE    = 39
PRESENCE_RATIO = 0.20
RANDOM_SEED    = 42

RAW_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
INPUT_CHANNELS = len(RAW_CHANNELS)

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")
OUT_DIR        = Path("data/carp/salted_water")
V4_NORMALIZER  = Path("data/v4/normalizer.npz")


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_length_weight(fname: str):
    """Return (length_cm, weight_g) parsed from filename, or (None, None)."""
    m_w = re.search(r"(\d+)gr",           fname)
    m_l = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    weight = float(m_w.group(1)) if m_w else None
    length = float(m_l.group(1)) if m_l else None
    return length, weight


# ---------------------------------------------------------------------------
# Recording cache
# ---------------------------------------------------------------------------

class RecordingCache:
    def __init__(self):
        self._raw:  dict[str, np.ndarray] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._std:  dict[str, np.ndarray] = {}

    def load(self, fname: str, path: Path) -> bool:
        if not path.exists():
            return False
        df  = pd.read_csv(path)
        for col in RAW_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[RAW_CHANNELS].values.astype(np.float32)
        m   = np.nanmean(raw, axis=0).astype(np.float32)
        s   = np.nanstd(raw,  axis=0).astype(np.float32)
        s[s < 1e-8] = 1.0
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            raw = np.where(nan_mask, m[np.newaxis, :], raw)
        self._raw[fname]  = raw
        self._mean[fname] = m
        self._std[fname]  = s
        return True

    def raw(self, fname: str) -> np.ndarray | None:
        return self._raw.get(fname)

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
                [seg, np.zeros((WINDOW_SIZE - len(seg), INPUT_CHANNELS), np.float32)]
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
            [np.zeros((pad_left,  INPUT_CHANNELS), np.float32), window]
        )
    if pad_right > 0:
        window = np.concatenate(
            [window, np.zeros((pad_right, INPUT_CHANNELS), np.float32)]
        )
    return window[:WINDOW_SIZE]


# ---------------------------------------------------------------------------
# Positive samples
# ---------------------------------------------------------------------------

def build_positive_samples(report: pd.DataFrame, cache: RecordingCache):
    waves, lengths, weights = [], [], []
    skipped = 0
    for _, row in report.iterrows():
        fname        = row["raw_data_file_name"]
        length, weight = row["_length"], row["_weight"]
        mid          = int((row["start_index"] + row["end_index"]) // 2)
        win          = cache.normalised_window(fname, mid)
        if win is None:
            skipped += 1
            continue
        waves.append(win)
        lengths.append(length)
        weights.append(weight)
    if skipped:
        print(f"  Skipped {skipped} detections (recording not in cache)")
    return np.stack(waves), np.array(lengths, np.float64), np.array(weights, np.float64)


# ---------------------------------------------------------------------------
# Negative samples
# ---------------------------------------------------------------------------

def build_negative_samples(report: pd.DataFrame, cache: RecordingCache,
                            n_total: int, rng: np.random.Generator):
    GUARD = WINDOW_SIZE
    occupied: dict[str, list] = defaultdict(list)
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        s = max(0, int(row["start_index"]) - GUARD)
        e = int(row["end_index"]) + GUARD
        occupied[fname].append((s, e))

    candidates = []
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

    while len(candidates) < n_total:
        fname, pos = candidates[int(rng.integers(len(candidates)))]
        jitter  = int(rng.integers(-5, 6))
        new_pos = max(0, min(pos + jitter, cache.n_rows(fname) - WINDOW_SIZE))
        candidates.append((fname, new_pos))

    idx      = rng.choice(len(candidates), size=n_total, replace=len(candidates) < n_total)
    selected = [candidates[i] for i in idx]

    waves = []
    for fname, start in selected:
        win = cache.normalised_segment(fname, start)
        waves.append(win)
    return np.stack(waves)


# ---------------------------------------------------------------------------
# Assemble dataset
# ---------------------------------------------------------------------------

def assemble(pos_waves, pos_lengths, pos_weights, neg_waves, rng):
    n_pos = len(pos_waves)
    n_neg = len(neg_waves)
    X          = np.concatenate([pos_waves, neg_waves])
    y_presence = np.concatenate([np.ones(n_pos, np.int64), np.zeros(n_neg, np.int64)])
    y_length   = np.concatenate([pos_lengths, np.zeros(n_neg, np.float64)])
    y_weight   = np.concatenate([pos_weights, np.zeros(n_neg, np.float64)])
    perm       = rng.permutation(len(X))
    return X[perm], y_presence[perm], y_length[perm], y_weight[perm]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building salted-water test dataset")
    print("=" * 60)

    # --- Load report, fix length/weight from filename ---
    report = pd.read_csv(REPORT_PATH)
    lengths_parsed, weights_parsed = [], []
    for fname in report["raw_data_file_name"]:
        l, w = _parse_length_weight(fname)
        lengths_parsed.append(l)
        weights_parsed.append(w)
    report["_length"] = lengths_parsed
    report["_weight"] = weights_parsed
    report = report.dropna(subset=["_length", "_weight"]).reset_index(drop=True)

    print(f"\nDetections with valid ground truth: {len(report)}")
    print("  Per-fish breakdown:")
    for fname, grp in report.groupby("raw_data_file_name"):
        l = grp["_length"].iloc[0]
        w = grp["_weight"].iloc[0]
        print(f"    {fname:35s}  {len(grp):3d} detections  "
              f"length={l:.1f}cm  weight={w:.0f}g")

    # --- Load recordings ---
    print("\nLoading recordings ...")
    cache = RecordingCache()
    for p in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in p.name.lower():
            continue
        ok = cache.load(p.name, p)
        if ok:
            print(f"  Loaded {p.name}  ({cache.n_rows(p.name)} rows)")

    # Filter to detections whose recording was loaded
    loaded = set(cache.all_files())
    report = report[report["raw_data_file_name"].isin(loaded)].reset_index(drop=True)
    print(f"\nDetections after recording filter: {len(report)}")

    # --- Positive samples ---
    print("\nExtracting positive samples (per-file normalised) ...")
    pos_waves, pos_lengths, pos_weights = build_positive_samples(report, cache)
    n_pos = len(pos_waves)
    print(f"  Positive samples: {n_pos}")

    # --- Negative samples ---
    n_neg = int(n_pos / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    print(f"\nGenerating {n_neg} negative samples ...")
    neg_waves = build_negative_samples(report, cache, n_neg, rng)
    print(f"  Negative samples: {n_neg}")

    # --- Assemble (per-file-normalised X — correct for v5) ---
    X, y_presence, y_length, y_weight = assemble(
        pos_waves, pos_lengths, pos_weights, neg_waves, rng
    )
    total = len(X)
    print(f"\nTotal samples: {total}  "
          f"({int(y_presence.sum())} pos / {total - int(y_presence.sum())} neg)")

    # --- Save v5 variant (per-file normalised, already done) ---
    out_v5 = OUT_DIR / "test_v5.npz"
    np.savez(out_v5,
             X=X.astype(np.float32),
             y_presence=y_presence,
             y_length=y_length,
             y_weight=y_weight)
    print(f"\nSaved (per-file normalised)  -> {out_v5}")

    # --- Save v4w variant (apply v4 global normalizer on top) ---
    norm = np.load(V4_NORMALIZER)
    v4_mean = norm["mean"].astype(np.float32)   # shape (12,)
    v4_std  = norm["std"].astype(np.float32)    # shape (12,)

    # X is already per-file z-scored; undo that and re-apply v4 stats would be
    # incorrect.  Instead, build raw windows and apply v4 normalization directly.
    print("\nRe-extracting windows for v4 global normalisation ...")
    pos_raw, neg_raw = _build_raw_windows(report, cache, n_neg, rng)
    X_raw, y_pres_raw, y_len_raw, y_wt_raw = assemble(
        pos_raw, pos_lengths, pos_weights, neg_raw, np.random.default_rng(RANDOM_SEED)
    )
    X_v4 = ((X_raw - v4_mean) / v4_std).astype(np.float32)

    out_v4w = OUT_DIR / "test_v4w.npz"
    np.savez(out_v4w,
             X=X_v4,
             y_presence=y_pres_raw,
             y_length=y_len_raw,
             y_weight=y_wt_raw)
    print(f"Saved (v4 global normalised) -> {out_v4w}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Raw-window helpers (no per-file normalisation)
# ---------------------------------------------------------------------------

def _build_raw_windows(report, cache, n_neg, rng):
    """Same as build_positive/negative_samples but returns raw (un-normalised) windows."""
    # Positives
    pos_waves = []
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        mid   = int((row["start_index"] + row["end_index"]) // 2)
        raw   = cache.raw(fname)
        if raw is None:
            continue
        pos_waves.append(_extract_window(raw, mid))

    # Negatives
    GUARD = WINDOW_SIZE
    occupied = defaultdict(list)
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
        occ    = sorted(occupied.get(fname, []))
        free_starts, free_ends = [0], []
        for s, e in occ:
            free_ends.append(max(0, s - 1))
            free_starts.append(min(n_rows, e + 1))
        free_ends.append(n_rows)
        for fs, fe in zip(free_starts, free_ends):
            max_start = fe - WINDOW_SIZE
            if max_start > fs:
                step = max(1, (max_start - fs) // max(1, n_neg // n_files))
                for pos in range(fs, max_start, step):
                    candidates.append((fname, pos))

    while len(candidates) < n_neg:
        fname, pos = candidates[int(rng.integers(len(candidates)))]
        jitter  = int(rng.integers(-5, 6))
        new_pos = max(0, min(pos + jitter, cache.n_rows(fname) - WINDOW_SIZE))
        candidates.append((fname, new_pos))

    idx      = rng.choice(len(candidates), size=n_neg, replace=len(candidates) < n_neg)
    neg_waves = []
    for i in idx:
        fname, start = candidates[i]
        raw  = cache.raw(fname)
        seg  = raw[start : start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            seg = np.concatenate(
                [seg, np.zeros((WINDOW_SIZE - len(seg), INPUT_CHANNELS), np.float32)]
            )
        neg_waves.append(seg[:WINDOW_SIZE])

    return np.stack(pos_waves), np.stack(neg_waves)


if __name__ == "__main__":
    main()
