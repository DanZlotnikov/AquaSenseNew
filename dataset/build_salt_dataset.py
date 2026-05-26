"""Build train / val / test NPZ datasets for FishSaltModel.

Differences from build_salted_water_split.py:
  - X has shape (N, 39, 12)  — 12 acoustic channels only (NO salt channel).
  - X_phys has shape (N, 3)  — per-recording physics features, constant within
                               a recording but varying across recordings:
                                   [baseline_mean_norm, baseline_std_norm,
                                    physics_salt_estimate]
                               Computed via compute_physics_features() from
                               model/fish_salt_model.py.
  - y_salt has shape (N,)    — ground-truth salt concentration in g/L (0 or 400).
                               Used to supervise the salt-prediction head during
                               training.

The stratified split (75 / 10 / 15) and negative sampling are identical to the
existing builder.  Hard-negative mining can be run on top of this dataset using
mine_hard_negatives.py with an adapted model.

Run from the project root:
    python dataset/build_salt_dataset.py

Output files (saved alongside the existing split):
    data/carp/salted_water_split/
        train_salt.npz   keys: X, X_phys, y_presence, y_weight, y_salt
        val_salt.npz
        test_salt.npz
        build_salt_report.txt
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_salt_model import compute_physics_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE    = 39
GUARD_SAMPLES  = 117          # 3× window — same guard zone as existing builder
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
SALT_MAX     = 400.0

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")
OUT_DIR        = Path("data/carp/salted_water_split")


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _parse_meta(fname: str):
    m_w    = re.search(r"(\d+)gr",           fname)
    m_l    = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    weight = float(m_w.group(1)) if m_w else None
    length = float(m_l.group(1)) if m_l else None
    salt   = 400.0 if "_S400" in fname else 0.0
    return length, weight, salt


# ---------------------------------------------------------------------------
# Recording cache — per-file z-normalisation, physics feature computation
# ---------------------------------------------------------------------------

class RecordingCache:
    def __init__(self):
        self._raw:    dict[str, np.ndarray] = {}
        self._mean:   dict[str, np.ndarray] = {}   # per-channel mean (12,)
        self._std:    dict[str, np.ndarray] = {}   # per-channel std  (12,)
        self._phys:   dict[str, np.ndarray] = {}   # physics features (3,)
        self._salt_g: dict[str, float]      = {}   # ground-truth salt in g/L

    def load(self, fname: str, path: Path, salt_g: float) -> bool:
        if not path.exists():
            return False
        df = pd.read_csv(path)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)

        # Fill NaN with per-channel mean before computing stats
        m   = np.nanmean(raw, axis=0).astype(np.float32)
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            raw = np.where(nan_mask, m[np.newaxis, :], raw)

        s = np.nanstd(raw, axis=0).astype(np.float32)
        s[s < 1e-8] = 1.0

        self._raw[fname]    = raw
        self._mean[fname]   = m
        self._std[fname]    = s
        self._salt_g[fname] = salt_g

        # Physics features use the full recording (grand mean/std across all channels).
        # Using the full recording instead of a "quiet baseline" only is fine because
        # fish-pass windows are ~10 % of samples and barely move the mean.
        self._phys[fname] = compute_physics_features(raw)
        return True

    def normalised_window(self, fname: str, mid: int):
        """Return (39, 12) z-scored waveform window centred at mid."""
        raw = self._raw.get(fname)
        if raw is None:
            return None
        win = _extract_window(raw, mid)                            # (39, 12)
        return ((win - self._mean[fname]) / self._std[fname]).astype(np.float32)

    def normalised_segment(self, fname: str, start: int):
        """Return (39, 12) z-scored waveform window starting at start."""
        raw = self._raw.get(fname)
        if raw is None:
            return None
        seg = raw[start: start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            seg = np.concatenate(
                [seg, np.zeros((WINDOW_SIZE - len(seg), N_ACOUSTIC), np.float32)]
            )
        return ((seg[:WINDOW_SIZE] - self._mean[fname]) / self._std[fname]).astype(np.float32)

    def physics(self, fname: str) -> np.ndarray:
        """Return (3,) physics feature vector for this recording."""
        return self._phys[fname]

    def salt_g(self, fname: str) -> float:
        return self._salt_g[fname]

    def n_rows(self, fname: str) -> int:
        return len(self._raw.get(fname, []))

    def all_files(self) -> list:
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
# Positive sample builder
# ---------------------------------------------------------------------------

def build_positive_samples(report: pd.DataFrame, cache: RecordingCache):
    waves, phys_feats, weights, salts_g, strata = [], [], [], [], []
    skipped = 0
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        mid   = int((row["start_index"] + row["end_index"]) // 2)
        win   = cache.normalised_window(fname, mid)
        if win is None:
            skipped += 1
            continue
        waves.append(win)
        phys_feats.append(cache.physics(fname))
        weights.append(row["_weight"])
        salts_g.append(cache.salt_g(fname))
        salt_tag = "s400" if row["_salt"] > 0 else "s0"
        strata.append(f"{int(row['_length'])}_{salt_tag}")

    if skipped:
        print(f"  Skipped {skipped} detections (file not in cache)")
    return (
        np.stack(waves),
        np.stack(phys_feats),
        np.array(weights, np.float64),
        np.array(salts_g, np.float32),
        np.array(strata),
    )


# ---------------------------------------------------------------------------
# Negative sample builder
# ---------------------------------------------------------------------------

def build_negative_samples(report: pd.DataFrame, cache: RecordingCache,
                             n_total: int, rng: np.random.Generator):
    occupied: dict[str, list] = defaultdict(list)
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        occupied[fname].append((
            max(0, int(row["start_index"]) - GUARD_SAMPLES),
            int(row["end_index"]) + GUARD_SAMPLES,
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

    idx        = rng.choice(len(candidates), size=n_total,
                            replace=len(candidates) < n_total)
    waves, phys_feats, salts_g = [], [], []
    for i in idx:
        fname, start = candidates[i]
        win = cache.normalised_segment(fname, start)
        waves.append(win)
        phys_feats.append(cache.physics(fname))
        salts_g.append(cache.salt_g(fname))

    return np.stack(waves), np.stack(phys_feats), np.array(salts_g, np.float32)


# ---------------------------------------------------------------------------
# Combine and shuffle
# ---------------------------------------------------------------------------

def combine(pos_waves, pos_phys, pos_weights, pos_salts,
            neg_waves, neg_phys, neg_salts, rng):
    n_pos = len(pos_waves)
    n_neg = len(neg_waves)

    X          = np.concatenate([pos_waves, neg_waves])
    X_phys     = np.concatenate([pos_phys,  neg_phys])
    y_presence = np.concatenate([np.ones(n_pos, np.int64), np.zeros(n_neg, np.int64)])
    y_weight   = np.concatenate([pos_weights, np.zeros(n_neg, np.float64)])
    y_salt     = np.concatenate([pos_salts,   neg_salts])

    perm = rng.permutation(len(X))
    return (X[perm], X_phys[perm], y_presence[perm],
            y_weight[perm], y_salt[perm])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building salt-model dataset  (12 acoustic + 3 physics features)")
    print("No salt channel in X — salinity inferred from baseline stats")
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
    for fname, grp in report.groupby("raw_data_file_name"):
        l = grp["_length"].iloc[0]; w = grp["_weight"].iloc[0]
        s = grp["_salt"].iloc[0]
        print(f"  {fname:35s}  {len(grp):3d} det  "
              f"{l:.1f}cm  {w:.0f}g  salt={s:.0f}")

    # --- Load recordings ---
    print("\nLoading recordings and computing physics features ...")
    cache = RecordingCache()
    for p in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in p.name.lower():
            continue
        _, _, salt = _parse_meta(p.name)
        ok = cache.load(p.name, p, salt)
        if ok:
            phys = cache.physics(p.name)
            print(f"  {p.name:35s}  "
                  f"phys=[{phys[0]:.3f}, {phys[1]:.3f}, salt_est={phys[2]:.3f}]")

    loaded = set(cache.all_files())
    report = report[report["raw_data_file_name"].isin(loaded)].reset_index(drop=True)

    # --- Positive samples ---
    print("\nExtracting positive samples ...")
    pos_waves, pos_phys, pos_weights, pos_salts, pos_strata = build_positive_samples(
        report, cache
    )
    n_pos = len(pos_waves)
    print(f"  Total positive samples: {n_pos}")

    # --- Stratified split of positives (same fractions as existing builder) ---
    print(f"\nStratified split ({TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / {TEST_FRAC:.0%}) ...")
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
    neg_waves_all, neg_phys_all, neg_salts_all = build_negative_samples(
        report, cache, n_neg_total, rng
    )
    neg_waves_t = neg_waves_all[:n_neg_train]
    neg_phys_t  = neg_phys_all[:n_neg_train]
    neg_salts_t = neg_salts_all[:n_neg_train]
    neg_waves_v = neg_waves_all[n_neg_train: n_neg_train + n_neg_val]
    neg_phys_v  = neg_phys_all[n_neg_train: n_neg_train + n_neg_val]
    neg_salts_v = neg_salts_all[n_neg_train: n_neg_train + n_neg_val]
    neg_waves_e = neg_waves_all[n_neg_train + n_neg_val:]
    neg_phys_e  = neg_phys_all[n_neg_train + n_neg_val:]
    neg_salts_e = neg_salts_all[n_neg_train + n_neg_val:]

    # --- Combine and save ---
    print("\nBuilding and saving datasets ...")
    splits = {
        "train": (train_idx, neg_waves_t, neg_phys_t, neg_salts_t),
        "val":   (val_idx,   neg_waves_v, neg_phys_v, neg_salts_v),
        "test":  (test_idx,  neg_waves_e, neg_phys_e, neg_salts_e),
    }
    summary = {}
    for name, (pos_idx, nw, np_, ns) in splits.items():
        X, X_phys, y_pres, y_wt, y_salt = combine(
            pos_waves[pos_idx], pos_phys[pos_idx],
            pos_weights[pos_idx], pos_salts[pos_idx],
            nw, np_, ns, rng,
        )
        path = OUT_DIR / f"{name}_salt.npz"
        np.savez(path,
                 X=X.astype(np.float32),
                 X_phys=X_phys.astype(np.float32),
                 y_presence=y_pres,
                 y_weight=y_wt.astype(np.float64))
        n     = len(y_pres)
        pos_n = int(y_pres.sum())
        summary[name] = (n, pos_n)
        print(f"  {name:5s}: {n:5d} total  ({pos_n} pos / {n - pos_n} neg)  -> {path}")

    # --- Build report ---
    lines = [
        "Salt-Model Dataset Build Report",
        "=" * 60,
        f"X shape    : (N, {WINDOW_SIZE}, {N_ACOUSTIC})  — acoustic only, per-file z-scored",
        f"X_phys     : (N, 3)  — [baseline_mean_norm, baseline_std_norm, physics_salt_est]",
        f"y_salt     : (N,)    — ground-truth salt in g/L (0 or 400)",
        f"Presence   : {PRESENCE_RATIO:.0%} positive",
        f"Split      : {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%} / {TEST_FRAC:.0%}",
        "",
        "Files:",
    ]
    for fname, grp in report.groupby("raw_data_file_name"):
        l = grp["_length"].iloc[0]; w = grp["_weight"].iloc[0]
        s = grp["_salt"].iloc[0]; phys = cache.physics(fname)
        lines.append(f"  {fname:35s}  {len(grp):3d} det  "
                     f"{l:.1f}cm  {w:.0f}g  salt={s:.0f}  "
                     f"phys_salt_est={phys[2]:.3f}")
    lines += ["", "Dataset sizes:"]
    for name, (n, pos_n) in summary.items():
        lines.append(f"  {name:5s}: {n} total  ({pos_n} pos / {n-pos_n} neg)")
    lines += ["", "To train:", "  python training/train_salt.py"]
    (OUT_DIR / "build_salt_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report -> {OUT_DIR / 'build_salt_report.txt'}")
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
