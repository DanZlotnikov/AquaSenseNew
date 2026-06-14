"""Build joint bream+carp presence dataset — v1.

Key design decisions:
  - Bream files are split into train / val / test at file level.
  - All carp sources go to train only (supply positives + rich GT=0 pool).
  - Knockdown augmentation (F-kd / B-kd) applied to ALL train positives
    from both species.
  - Val and test contain only bream files so metrics reflect bream performance.

Training set composition (approx):
  ~500 bream positives  +  ~540 carp positives
  + knockdown negatives (2x all train positives)
  + gap negatives + GT=0-file negatives

Output: data/bream_presence_v1/{train,val,test}_dataset.npz
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants (identical to carp v5) ─────────────────────────────────────────
WINDOW_SIZE    = 39
GUARD          = WINDOW_SIZE
STRIDE         = 5
PRESENCE_RATIO = 0.20   # 20 % positives → 1:4 ratio
GT0_NEG_FRAC   = 0.50   # half of negatives from GT=0 files
SIGMA_FLOOR    = 150.0
RANDOM_SEED    = 42

SIGNAL_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH  = len(SIGNAL_CHANNELS)
F_IDX = list(range(0, 6))
B_IDX = list(range(6, 12))

SKIP = ("output", "report", "axis", "summary", "function", "points", "updated")

BASE_BREAM = Path("data/raw/bream")
BASE_CARP  = Path("data/raw/carp")

# ── Bream file-level split ────────────────────────────────────────────────────
# Test: 5 files, ~298 events, weights 150/390/620/680/800g
BREAM_TEST = {
    "ARDG_AQUA bream 150gr 21 sm  281025 1132002.csv",
    "ARDG_AQUA bream 390gr 26cm  281025 1325003.csv",
    "ARDG_AQUA bream620 gr31cm  281025 1618003.csv",
    "ARDG_AQUA bream 680gr 33 sm  281025 1208003.csv",
    "ARDG_AQUA bream 800gr 36cm  281025 1453002.csv",
}

# Val: 5 files, ~269 events, weights 150/280/500/580/630g
BREAM_VAL = {
    "ARDG_AQUA bream 150gr 21 sm  281025 1132004.csv",
    "ARDG_AQUA bream 280gr 24cm  281025 1425002.csv",
    "ARDG_AQUA bream500gr 30cm  281025 1500003.csv",
    "ARDG_AQUA bream 580gr 30 sm  281025 1254002.csv",
    "ARDG_AQUA bream 630gr 31cm  281025 1400002.csv",
}

OUT_DIR = Path("data/bream_presence_v1")


# ── Sources ───────────────────────────────────────────────────────────────────
# (report_csv, rec_dir, source_tag, force_split)
# force_split=None → use bream split logic; force_split="train" → always train
SOURCES = [
    (BASE_BREAM / "output_report.csv",
     BASE_BREAM / "recordings",
     "bream", None),
    (BASE_CARP / "carp_salted/salted_water/output_report.csv",
     BASE_CARP / "carp_salted/salted_water",
     "salted_water", "train"),
    (BASE_CARP / "carp_salted/salted_water/test/output_report.csv",
     BASE_CARP / "carp_salted/salted_water/test",
     "salted_water_test", "train"),
    (BASE_CARP / "ACQ_5_3_2026/output_report.csv",
     BASE_CARP / "ACQ_5_3_2026",
     "ACQ", "train"),
    (BASE_CARP / "carp_old/output_report.csv",
     BASE_CARP / "carp_old",
     "carp_old", "train"),
]


def normalize(raw: np.ndarray) -> np.ndarray:
    mu         = np.nanmean(raw, axis=0)
    sigma      = np.nanstd(raw,  axis=0)
    sigma_eff  = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    return np.where(np.isnan((raw - mu) / sigma_eff), 0.0,
                    (raw - mu) / sigma_eff).astype(np.float32)


class RecordingCache:
    def __init__(self):
        self._norm = {}
        self._n    = {}

    def load(self, path: Path) -> bool:
        key = str(path)
        if key in self._norm:
            return True
        if not path.exists():
            return False
        df = pd.read_csv(path)
        for col in SIGNAL_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[SIGNAL_CHANNELS].values.astype(np.float32)
        if raw.shape[0] < WINDOW_SIZE or np.all(raw == 0) or np.all(np.isnan(raw)):
            return False
        self._norm[key] = normalize(raw)
        self._n[key]    = len(raw)
        return True

    def window(self, key: str, start: int) -> np.ndarray:
        norm = self._norm[key]
        seg  = norm[start: start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            seg = np.concatenate(
                [seg, np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)])
        return seg[:WINDOW_SIZE]

    def n_rows(self, key: str) -> int:
        return self._n.get(key, 0)


def gap_candidates(rec_key, events, cache):
    n        = cache.n_rows(rec_key)
    occupied = sorted((max(0, s - GUARD), min(n, e + GUARD)) for s, e in events)
    free_starts, free_ends = [0], []
    for s, e in occupied:
        free_ends.append(max(0, s - 1))
        free_starts.append(min(n, e + 1))
    free_ends.append(n)
    candidates = []
    for fs, fe in zip(free_starts, free_ends):
        for pos in range(fs, fe - WINDOW_SIZE, STRIDE):
            candidates.append((rec_key, pos))
    return candidates


def full_candidates(rec_key, cache):
    n = cache.n_rows(rec_key)
    return [(rec_key, pos) for pos in range(0, max(0, n - WINDOW_SIZE + 1), STRIDE)]


def build_split(name, events, gt0_keys, cache, rng):
    # ── Positives ─────────────────────────────────────────────────────────────
    pos_X = []
    for rec_key, s, e, mid in events:
        half  = WINDOW_SIZE // 2
        start = max(0, mid - half)
        pos_X.append(cache.window(rec_key, start))

    n_pos       = len(pos_X)
    n_neg_total = int(n_pos / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_gt0   = int(n_neg_total * GT0_NEG_FRAC)
    n_neg_gap   = n_neg_total - n_neg_gt0

    # ── Gap negatives ─────────────────────────────────────────────────────────
    gap_pool = []
    occ = defaultdict(list)
    for rec_key, s, e, mid in events:
        occ[rec_key].append((s, e))
    for rec_key, evts in occ.items():
        gap_pool.extend(gap_candidates(rec_key, evts, cache))

    gt0_pool = []
    for rec_key in gt0_keys:
        gt0_pool.extend(full_candidates(rec_key, cache))

    def sample_pool(pool, n, rng):
        if not pool:
            return []
        while len(pool) < n:
            pool.append(pool[int(rng.integers(len(pool)))])
        return [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]

    neg_X = [cache.window(rk, pos)
             for rk, pos in sample_pool(gap_pool, n_neg_gap, rng)
                          + sample_pool(gt0_pool, n_neg_gt0, rng)]

    # ── Knockdown negatives (training split only) ─────────────────────────────
    kd_X = []
    if name == "train":
        for win in pos_X:
            kd_f = win.copy(); kd_f[:, F_IDX] = 0.0
            kd_b = win.copy(); kd_b[:, B_IDX] = 0.0
            kd_X.extend([kd_f, kd_b])

    all_neg_X = neg_X + kd_X
    X  = np.stack(pos_X + all_neg_X).astype(np.float32)
    yp = np.array([1] * n_pos + [0] * len(all_neg_X), dtype=np.int64)
    perm = rng.permutation(len(X))

    print(f"  {name:5s}: {len(X):5d} samples  "
          f"({n_pos} pos / {len(neg_X)} orig-neg / {len(kd_X)} knockdown-neg)")
    return X[perm], yp[perm]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng   = np.random.default_rng(RANDOM_SEED)
    cache = RecordingCache()

    print("=" * 65)
    print("Building bream_presence_v1 dataset (joint bream+carp)")
    print("=" * 65)

    all_events    = defaultdict(list)   # split -> [(rec_key, s, e, mid)]
    gt0_train     = []                  # GT=0 recording keys for train negatives

    for report_path, rec_dir, source, force_split in SOURCES:
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping")
            continue

        rep   = pd.read_csv(report_path)
        valid = rep[rep["is_valid"] == True] if "is_valid" in rep.columns else pd.DataFrame()
        gt_map = valid.groupby("raw_data_file_name").size().to_dict() if len(valid) else {}

        n_loaded = 0
        for fpath in sorted(rec_dir.glob("*.csv")):
            if any(x in fpath.name.lower() for x in SKIP):
                continue
            if not cache.load(fpath):
                continue
            rec_key = str(fpath)
            gt_count = gt_map.get(fpath.name, 0)

            # Determine split
            if force_split:
                split = force_split
            else:
                split = ("test" if fpath.name in BREAM_TEST else
                         "val"  if fpath.name in BREAM_VAL  else "train")

            if gt_count == 0:
                # GT=0 files only contribute to training negatives pool
                if split == "train":
                    gt0_train.append(rec_key)
            else:
                file_events = valid[valid["raw_data_file_name"] == fpath.name]
                for _, row in file_events.iterrows():
                    s   = int(row["start_index"])
                    e   = int(row["end_index"])
                    mid = (s + e) // 2
                    all_events[split].append((rec_key, s, e, mid))
            n_loaded += 1

        print(f"  [{source}] loaded {n_loaded} files")

    print()
    for sp in ("train", "val", "test"):
        n = len(all_events[sp])
        print(f"  {sp:5s}: {n:4d} positive events")
    print(f"  GT=0 training pool: {len(gt0_train)} files")
    print()

    for split in ("train", "val", "test"):
        # Val and test use their own gap pool but no GT=0 pool
        gt0 = gt0_train if split == "train" else []
        X, yp = build_split(split, all_events[split], gt0, cache, rng)
        out   = OUT_DIR / f"{split}_dataset.npz"
        np.savez(out,
                 X          = X,
                 y_presence = yp,
                 y_length   = np.zeros(len(X), np.float64),
                 y_weight   = np.zeros(len(X), np.float64))
        print(f"    saved -> {out}")

    report = [
        "bream_presence_v1 build report",
        f"SIGMA_FLOOR    = {SIGMA_FLOOR}",
        f"GT0_NEG_FRAC   = {GT0_NEG_FRAC}",
        f"PRESENCE_RATIO = {PRESENCE_RATIO}",
        f"RANDOM_SEED    = {RANDOM_SEED}",
        "",
        "Bream test files:",
    ] + [f"  {f}" for f in sorted(BREAM_TEST)] + [
        "",
        "Bream val files:",
    ] + [f"  {f}" for f in sorted(BREAM_VAL)] + [
        "",
        "Split positive event counts:",
    ] + [f"  {sp:5s}: {len(all_events[sp])}" for sp in ("train", "val", "test")]

    (OUT_DIR / "build_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nDone.  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
