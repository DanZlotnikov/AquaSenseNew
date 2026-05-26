"""Build train / val / test npz files from data/carp/salted_water/ recordings.

Salt water (S400 files) → salt_level=400 g/L, sweet water → salt_level=0.
Salt is stored as the 13th channel (index 12) in X, constant across time,
normalised to [0, 1] (0 → 0.0, 400 → 1.0).

The 12 acoustic channels are per-file z-scored (same as v5).

Stratification is by (length_bucket, salt_level) so every fish size and every
salt condition appears in all three splits.

Run from the project root:
    python dataset/build_salted_water_split.py

Output
------
    data/carp/salted_water_split/
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
GUARD_SAMPLES  = 117          # 3× window (~3 sec at 40 Hz) — excludes near-miss windows
PRESENCE_RATIO = 0.10
TRAIN_FRAC     = 0.75
VAL_FRAC       = 0.10
TEST_FRAC      = 0.15
RANDOM_SEED    = 42

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_ACOUSTIC   = len(ACOUSTIC_CHANNELS)   # 12
TOTAL_CHANS  = N_ACOUSTIC + 1           # 13 (12 acoustic + 1 salt)
SALT_MAX     = 400.0                    # normalise salt to [0, 1]

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")
OUT_DIR        = Path("data/carp/salted_water_split")


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _parse_meta(fname: str):
    """Return (length_cm, weight_g, salt_level) from filename."""
    m_w    = re.search(r"(\d+)gr",           fname)
    m_l    = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    weight = float(m_w.group(1)) if m_w else None
    length = float(m_l.group(1)) if m_l else None
    salt   = 400.0 if "_S400" in fname else 0.0
    return length, weight, salt


# ---------------------------------------------------------------------------
# Recording cache with per-file normalisation
# ---------------------------------------------------------------------------

class RecordingCache:
    def __init__(self):
        self._raw:  dict[str, np.ndarray] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._std:  dict[str, np.ndarray] = {}
        self._salt: dict[str, float]      = {}

    def load(self, fname: str, path: Path, salt: float) -> bool:
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
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            raw = np.where(nan_mask, m[np.newaxis, :], raw)
        self._raw[fname]  = raw
        self._mean[fname] = m
        self._std[fname]  = s
        self._salt[fname] = salt
        return True

    def normalised_window(self, fname: str, mid: int) -> np.ndarray | None:
        """Return (39, 13) window: 12 z-scored acoustic + 1 normalised salt."""
        raw = self._raw.get(fname)
        if raw is None:
            return None
        win   = _extract_window(raw, mid)                              # (39, 12)
        win   = ((win - self._mean[fname]) / self._std[fname]).astype(np.float32)
        salt  = np.full((WINDOW_SIZE, 1),
                        self._salt[fname] / SALT_MAX, dtype=np.float32)
        return np.concatenate([win, salt], axis=1)                     # (39, 13)

    def normalised_segment(self, fname: str, start: int) -> np.ndarray | None:
        raw = self._raw.get(fname)
        if raw is None:
            return None
        seg = raw[start : start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(seg), N_ACOUSTIC), np.float32)
            seg = np.concatenate([seg, pad])
        seg   = seg[:WINDOW_SIZE]
        seg   = ((seg - self._mean[fname]) / self._std[fname]).astype(np.float32)
        salt  = np.full((WINDOW_SIZE, 1),
                        self._salt[fname] / SALT_MAX, dtype=np.float32)
        return np.concatenate([seg, salt], axis=1)                     # (39, 13)

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
        # stratum = "length_salt" for stratified splitting
        salt_tag = "s400" if row["_salt"] > 0 else "s0"
        strata.append(f"{int(row['_length'])}_{salt_tag}")
    if skipped:
        print(f"  Skipped {skipped} detections (file not in cache)")
    return (np.stack(waves),
            np.array(lengths, np.float64),
            np.array(weights, np.float64),
            np.array(strata))


def build_negative_samples(report: pd.DataFrame, cache: RecordingCache,
                            n_total: int, rng: np.random.Generator):
    GUARD = GUARD_SAMPLES
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
    waves = []
    for i in idx:
        fname, start = candidates[i]
        win = cache.normalised_segment(fname, start)
        waves.append(win)
    return np.stack(waves)


# ---------------------------------------------------------------------------
# Combine + shuffle
# ---------------------------------------------------------------------------

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
    print("Building salted-water split dataset  (12 acoustic + 1 salt)")
    print("Per-file normalisation on acoustic channels")
    print("=" * 60)

    # --- Load and fix report ---
    report = pd.read_csv(REPORT_PATH)
    lengths, weights, salts = [], [], []
    for fname in report["raw_data_file_name"]:
        l, w, s = _parse_meta(fname)
        lengths.append(l); weights.append(w); salts.append(s)
    report["_length"] = lengths
    report["_weight"] = weights
    report["_salt"]   = salts
    report = report.dropna(subset=["_length", "_weight"]).reset_index(drop=True)

    print(f"\nDetections with valid ground truth: {len(report)}")
    print("  Per-file breakdown:")
    for fname, grp in report.groupby("raw_data_file_name"):
        l = grp["_length"].iloc[0]; w = grp["_weight"].iloc[0]
        s = grp["_salt"].iloc[0]
        print(f"    {fname:35s}  {len(grp):3d} det  "
              f"{l:.1f}cm  {w:.0f}g  salt={s:.0f}")

    # --- Load recordings ---
    print("\nLoading recordings ...")
    cache = RecordingCache()
    for p in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in p.name.lower():
            continue
        _, _, salt = _parse_meta(p.name)
        ok = cache.load(p.name, p, salt)
        if ok:
            print(f"  {p.name:35s}  {cache.n_rows(p.name):6d} rows  "
                  f"salt={salt:.0f}")

    loaded = set(cache.all_files())
    report = report[report["raw_data_file_name"].isin(loaded)].reset_index(drop=True)

    # --- Positive samples ---
    print("\nExtracting positive samples ...")
    pos_waves, pos_lengths, pos_weights, pos_strata = build_positive_samples(
        report, cache
    )
    n_pos = len(pos_waves)
    print(f"  Total: {n_pos}")
    print("  Strata:")
    for st in sorted(np.unique(pos_strata)):
        print(f"    {st:15s}  {(pos_strata == st).sum():3d}")

    # --- Stratified split of positives ---
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

    print(f"  Positives — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    # --- Negative samples ---
    n_neg_train = int(len(train_idx) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_idx)   / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_idx)  / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_total = n_neg_train + n_neg_val + n_neg_test
    print(f"  Negatives  — train: {n_neg_train}  val: {n_neg_val}  test: {n_neg_test}")

    print("\nGenerating negative samples ...")
    neg_all   = build_negative_samples(report, cache, n_neg_total, rng)
    neg_train = neg_all[:n_neg_train]
    neg_val   = neg_all[n_neg_train : n_neg_train + n_neg_val]
    neg_test  = neg_all[n_neg_train + n_neg_val :]

    # --- Combine and save ---
    print("\nBuilding and saving datasets ...")
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
        n   = len(y_pres); pos_n = int(y_pres.sum())
        summary[name] = (n, pos_n)
        print(f"  {name:5s}: {n:5d} total  ({pos_n} pos / {n - pos_n} neg)  "
              f"-> {path}")

    # --- Build report ---
    lines = [
        "Salted-Water Split Dataset Build Report",
        "=" * 60,
        f"Source        : {RECORDINGS_DIR}",
        f"Window size   : {WINDOW_SIZE} samples at 40 Hz",
        f"Channels      : {N_ACOUSTIC} acoustic (per-file z-scored) + 1 salt (0 or 1)",
        f"X shape       : (N, {WINDOW_SIZE}, {TOTAL_CHANS})",
        f"Presence ratio: {PRESENCE_RATIO:.0%} positive",
        f"Split         : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / {TEST_FRAC:.0%}",
        "",
        "Salt encoding : 0 g/L → 0.0   |   400 g/L → 1.0  (channel index 12)",
        "",
        "Files:",
    ]
    for fname, grp in report.groupby("raw_data_file_name"):
        l = grp["_length"].iloc[0]; w = grp["_weight"].iloc[0]
        s = grp["_salt"].iloc[0]
        lines.append(f"  {fname:35s}  {len(grp):3d} det  "
                     f"{l:.1f}cm  {w:.0f}g  salt={s:.0f}")
    lines += ["", "Dataset sizes:"]
    for name, (n, pos_n) in summary.items():
        lines.append(f"  {name:5s}: {n:5d} total  "
                     f"({pos_n} positive / {n - pos_n} negative)")
    lines += [
        "",
        "Model config required:",
        "  input_channels: 12",
        "  physics_dim: 1   (salt feature via physics branch)",
        "",
        "To train:",
        "  python training/train.py --config training/train_config_salted.yaml "
        "--model_config model/model_config_salted.yaml",
    ]
    (OUT_DIR / "build_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Build report -> {OUT_DIR / 'build_report.txt'}")
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
