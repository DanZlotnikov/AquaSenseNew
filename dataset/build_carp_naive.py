"""Build train / val / test NPZ dataset for the carp_naive_model.

Same logic as the bream length model (build_v5_dataset.py):
  - ONE centered window per detected event  → positive sample
  - Negatives strictly from gap regions between events (guard = WINDOW_SIZE)
  - Per-recording z-normalisation (removes salinity amplitude difference)
  - Stratified split by length class
  - 70 / 20 / 10 split

Length is parsed from the filename (e.g. carp_920gr_35cm.csv → 35.0 cm).
All 731 detected events across 7 recordings are used (valid + invalid).

Output:
    data/carp/naive/
        train.npz   keys: X (N,39,12), y_presence, y_length
        val.npz
        test.npz
        build_report.txt

Run from the project root:
    python dataset/build_carp_naive.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
WINDOW_SIZE    = 39
PRESENCE_RATIO = 0.20
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.20
RANDOM_SEED    = 42
GUARD          = WINDOW_SIZE      # samples masked around each event for negatives

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH = len(ACOUSTIC_CHANNELS)

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")
OUT_DIR        = Path("data/carp/naive")
# ---------------------------------------------------------------------------


def _parse_length_cm(fname: str) -> float | None:
    """Extract length in cm from filename, e.g. 'carp_920gr_35cm.csv' -> 35.0."""
    m = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    return float(m.group(1)) if m else None


def _parse_weight_g(fname: str) -> float | None:
    m = re.search(r"(\d+)gr", fname)
    return float(m.group(1)) if m else None


class RecordingCache:
    """Loads and caches per-recording z-normalised signals."""

    def __init__(self):
        self._raw:  dict[str, np.ndarray] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._std:  dict[str, np.ndarray] = {}

    def load(self, csv_path: Path) -> bool:
        fname = csv_path.name
        if fname in self._raw:
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
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            raw = np.where(nan_mask, m[np.newaxis, :], raw)
        self._raw[fname]  = raw
        self._mean[fname] = m
        self._std[fname]  = s
        return True

    def get_centered_window(self, fname: str, mid: int) -> np.ndarray | None:
        if fname not in self._raw:
            return None
        raw  = self._raw[fname]
        half = WINDOW_SIZE // 2
        start = mid - half
        end   = start + WINDOW_SIZE
        n     = len(raw)
        pad_l = max(0, -start)
        pad_r = max(0, end - n)
        seg   = raw[max(0, start): min(n, end)]
        if pad_l > 0:
            seg = np.concatenate([np.zeros((pad_l, N_CH), np.float32), seg])
        if pad_r > 0:
            seg = np.concatenate([seg, np.zeros((pad_r, N_CH), np.float32)])
        seg = seg[:WINDOW_SIZE]
        return ((seg - self._mean[fname]) / self._std[fname]).astype(np.float32)

    def get_segment(self, fname: str, start: int) -> np.ndarray | None:
        if fname not in self._raw:
            return None
        raw = self._raw[fname]
        seg = raw[start: start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)
            seg = np.concatenate([seg, pad])
        return ((seg[:WINDOW_SIZE] - self._mean[fname]) / self._std[fname]).astype(np.float32)

    def n_rows(self, fname: str) -> int:
        return len(self._raw.get(fname, []))

    def all_files(self) -> list[str]:
        return list(self._raw.keys())


# ---------------------------------------------------------------------------

def build_positives(report: pd.DataFrame, cache: RecordingCache) -> dict:
    waves, lengths, weights, classes = [], [], [], []
    for _, row in report.iterrows():
        fname  = row["raw_data_file_name"]
        mid    = int((row["start_index"] + row["end_index"]) // 2)
        win    = cache.get_centered_window(fname, mid)
        if win is None:
            continue
        length_cm = _parse_length_cm(fname)
        weight_g  = _parse_weight_g(fname)
        if length_cm is None:
            continue
        waves.append(win)
        lengths.append(length_cm)
        weights.append(weight_g or 0.0)
        classes.append(f"{int(length_cm * 10)}")   # e.g. "275", "320", "350", "415"

    return {
        "wave":         np.stack(waves),
        "y_presence":   np.ones(len(waves),  dtype=np.int64),
        "y_length":     np.array(lengths,    dtype=np.float64),
        "y_weight":     np.array(weights,    dtype=np.float64),
        "length_class": np.array(classes),
    }


def build_negatives(
    report: pd.DataFrame,
    cache: RecordingCache,
    n_total: int,
    rng: np.random.Generator,
) -> dict:
    # Build occupied intervals (event regions + guard)
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for _, row in report.iterrows():
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

    # If not enough candidates, jitter existing ones
    while len(candidates) < n_total:
        fname, pos = candidates[int(rng.integers(len(candidates)))]
        jitter  = int(rng.integers(-5, 6))
        new_pos = max(0, min(pos + jitter, cache.n_rows(fname) - WINDOW_SIZE))
        candidates.append((fname, new_pos))

    idx      = rng.choice(len(candidates), size=n_total, replace=len(candidates) < n_total)
    selected = [candidates[i] for i in idx]

    waves = [cache.get_segment(fname, start) for fname, start in selected]
    n = len(waves)
    return {
        "wave":       np.stack(waves),
        "y_presence": np.zeros(n, dtype=np.int64),
        "y_length":   np.zeros(n, dtype=np.float64),
        "y_weight":   np.zeros(n, dtype=np.float64),
    }


def _subset(pos: dict, idx: np.ndarray) -> dict:
    return {k: pos[k][idx] for k in ["wave", "y_presence", "y_length", "y_weight"]}


def combine(pos_sub: dict, neg: dict, rng: np.random.Generator) -> dict:
    X  = np.concatenate([pos_sub["wave"],       neg["wave"]])
    yp = np.concatenate([pos_sub["y_presence"], neg["y_presence"]])
    yl = np.concatenate([pos_sub["y_length"],   neg["y_length"]])
    yw = np.concatenate([pos_sub["y_weight"],   neg["y_weight"]])
    perm = rng.permutation(len(X))
    return {"X": X[perm], "y_presence": yp[perm], "y_length": yl[perm], "y_weight": yw[perm]}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("Building carp_naive dataset")
    print("One centered window per event | gap-only negatives")
    print("=" * 60)

    report = pd.read_csv(REPORT_PATH)
    # Keep only events from recordings with parseable length
    report = report[
        report["raw_data_file_name"].apply(lambda f: _parse_length_cm(f) is not None)
    ].reset_index(drop=True)
    print(f"\nEvents with parseable length: {len(report)}")

    # Load all recordings
    cache = RecordingCache()
    for csv_path in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in csv_path.name.lower():
            continue
        cache.load(csv_path)
        print(f"  loaded {csv_path.name}  ({cache.n_rows(csv_path.name):,} rows)")

    # Positives
    print("\nExtracting positive samples (centered windows) ...")
    pos = build_positives(report, cache)
    print(f"  Total positives: {len(pos['y_presence'])}")
    for lc in sorted(np.unique(pos["length_class"])):
        n = (pos["length_class"] == lc).sum()
        l = float(lc) / 10
        print(f"    {l} cm : {n} events")

    # Stratified split on positives
    labels = pos["length_class"]
    sss1 = StratifiedShuffleSplit(1, test_size=0.10, random_state=42)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))
    sss2 = StratifiedShuffleSplit(1, test_size=round(VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), 4), random_state=43)
    train_rel, val_rel = next(sss2.split(np.zeros(len(trainval_idx)), labels[trainval_idx]))
    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]

    print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)} positives")

    # Negatives
    n_neg_train = int(len(train_idx) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_idx)   / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_idx)  / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    print(f"Negatives: train={n_neg_train}  val={n_neg_val}  test={n_neg_test}")

    neg_all = build_negatives(report, cache, n_neg_train + n_neg_val + n_neg_test, rng)
    neg_train = {k: neg_all[k][:n_neg_train]                       for k in neg_all}
    neg_val   = {k: neg_all[k][n_neg_train: n_neg_train + n_neg_val] for k in neg_all}
    neg_test  = {k: neg_all[k][n_neg_train + n_neg_val:]            for k in neg_all}

    # Save
    splits = {
        "train": combine(_subset(pos, train_idx), neg_train, rng),
        "val":   combine(_subset(pos, val_idx),   neg_val,   rng),
        "test":  combine(_subset(pos, test_idx),  neg_test,  rng),
    }

    print()
    for name, data in splits.items():
        path = OUT_DIR / f"{name}_dataset.npz"
        np.savez(path,
                 X=data["X"],
                 y_presence=data["y_presence"],
                 y_length=data["y_length"],
                 y_weight=data["y_weight"])
        n   = len(data["y_presence"])
        pos_n = int(data["y_presence"].sum())
        print(f"  {name:5s}: {n:5d} samples  ({pos_n} pos / {n - pos_n} neg)  -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
