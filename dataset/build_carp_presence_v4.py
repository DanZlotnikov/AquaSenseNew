"""Build carp presence v4 dataset.

Key improvement over v3:
  Add two engineered envelope channels to every window:
    ch 13 — F_env(t) = max absolute z-score across the 6 front electrodes at time t
    ch 14 — B_env(t) = max absolute z-score across the 6 back electrodes at time t

  These channels give the model an explicit, symmetric representation of "how activated
  is each ring right now", without wiring in a hand-crafted decision rule.  The model
  can learn to require BOTH envelopes to be elevated simultaneously — the physically
  expected signature of a real carp crossing — without being forced to by the loss.

  The envelope is computed from the already-normalised signal so its scale is
  comparable to the raw electrode z-scores.

  Window shape: (39, 14) instead of (39, 12).
  All other hyperparameters, splits, and normalisation are identical to v3.

Split (positive events):  Train ~544 (69%) | Val ~103 (13%) | Test ~139 (18%)
Negative ratio: 4:1 (PRESENCE_RATIO=0.20), 50% from GT=0 files / 50% from gap regions.

Output (data/carp_presence_v4/):
  train_dataset.npz, val_dataset.npz, test_dataset.npz
    keys: X (N,39,14) float32, y_presence int64, y_length float64, y_weight float64
  build_report.txt
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 39
GUARD          = WINDOW_SIZE        # samples excluded around each event
STRIDE         = 5                  # stride for negative candidate generation
PRESENCE_RATIO = 0.20               # 20 % positives → 4:1 neg:pos
GT0_NEG_FRAC   = 0.50               # fraction of negatives drawn from GT=0 files
SIGMA_FLOOR    = 150.0              # minimum per-channel std
RANDOM_SEED    = 42

SIGNAL_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH_RAW = len(SIGNAL_CHANNELS)   # 12
N_CH     = N_CH_RAW + 2           # 14 (+ F_env, B_env)
F_IDX    = list(range(0, 6))      # indices of F electrodes
B_IDX    = list(range(6, 12))     # indices of B electrodes

SKIP = ("output", "report", "axis", "summary", "function", "points", "updated")

# ── Data sources ──────────────────────────────────────────────────────────────
BASE = Path("data/raw/carp")

SOURCES = [
    (BASE / "carp_salted/salted_water/output_report.csv",
     BASE / "carp_salted/salted_water",        "salted_water"),
    (BASE / "carp_salted/salted_water/test/output_report.csv",
     BASE / "carp_salted/salted_water/test",   "salted_water_test"),
    (BASE / "ACQ_5_3_2026/output_report.csv",
     BASE / "ACQ_5_3_2026",                    "ACQ"),
    (BASE / "carp_old/output_report.csv",
     BASE / "carp_old",                        "carp_old"),
]

OUT_DIR = Path("data/carp_presence_v4")

# ── Split assignment  (source, filename) — identical to v3 ───────────────────
TEST_FILES = {
    ("salted_water",      "carp_1580gr_41.5cm_S400.csv"),
    ("salted_water",      "carp_920gr_35cm_S400.csv"),
    ("salted_water_test", "AQUAtest003.csv"),
    ("ACQ",               "AQUA1429.csv"),
    ("ACQ",               "AQUA1430.csv"),
    ("ACQ",               "AQUA1462.csv"),
    ("ACQ",               "AQUA1375.csv"),
    ("ACQ",               "AQUA1380.csv"),
    ("ACQ",               "AQUA1385.csv"),
    ("ACQ",               "AQUA1390.csv"),
    ("ACQ",               "AQUA1395.csv"),
    ("ACQ",               "AQUA1445.csv"),
    ("ACQ",               "AQUA1455.csv"),
    ("ACQ",               "AQUA1489.csv"),
    ("carp_old",          "AQUA075 880gr_31cm.csv"),
    ("carp_old",          "AQUA056 880gr_31cm.csv"),
    ("carp_old",          "AQUA051 880gr_31cm.csv"),
    ("carp_old",          "AQUA064 880gr_31cm.csv"),
    ("carp_old",          "AQUA077 440gr_27cm.csv"),
}

VAL_FILES = {
    ("salted_water_test", "carp_1580gr_41.5cm.csv"),
    ("salted_water",      "carp_420gr_27.5cm.csv"),
    ("ACQ",               "AQUA1411.csv"),
    ("ACQ",               "AQUA1421.csv"),
    ("ACQ",               "AQUA1427.csv"),
    ("ACQ",               "AQUA1432.csv"),
    ("ACQ",               "AQUA1400.csv"),
    ("ACQ",               "AQUA1410.csv"),
    ("ACQ",               "AQUA1439.csv"),
    ("ACQ",               "AQUA1440.csv"),
    ("ACQ",               "AQUA1450.csv"),
    ("ACQ",               "AQUA1460.csv"),
    ("ACQ",               "AQUA1470.csv"),
    ("ACQ",               "AQUA1480.csv"),
    ("carp_old",          "AQUA073 880gr_31cm.csv"),
    ("carp_old",          "AQUA119 650gr_29cm.csv"),
    ("carp_old",          "AQUA052 880gr_31cm.csv"),
    ("carp_old",          "AQUA082 440gr_27cm.csv"),
    ("carp_old",          "AQUA090 440gr_27cm.csv"),
}


# ── Normalisation ─────────────────────────────────────────────────────────────
def normalize(raw: np.ndarray) -> np.ndarray:
    """Per-recording normalisation with sigma floor."""
    mu    = np.nanmean(raw, axis=0)
    sigma = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    normed = (raw - mu) / sigma_eff
    return np.where(np.isnan(normed), 0.0, normed).astype(np.float32)


def add_envelope_channels(norm12: np.ndarray) -> np.ndarray:
    """Append F_env and B_env as channels 13 and 14.

    F_env(t) = max |norm[t, F_IDX]|   — peak front-ring activation at time t
    B_env(t) = max |norm[t, B_IDX]|   — peak back-ring activation at time t

    Shape: (N, 12) -> (N, 14)
    """
    f_env = np.abs(norm12[:, F_IDX]).max(axis=1, keepdims=True)  # (N, 1)
    b_env = np.abs(norm12[:, B_IDX]).max(axis=1, keepdims=True)  # (N, 1)
    return np.concatenate([norm12, f_env, b_env], axis=1)         # (N, 14)


# ── Recording cache ───────────────────────────────────────────────────────────
class RecordingCache:
    def __init__(self):
        self._norm = {}   # key → (N, 14) float32
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
        norm12 = normalize(raw)
        self._norm[key] = add_envelope_channels(norm12)  # (N, 14)
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


# ── Negative candidate builders ───────────────────────────────────────────────
def gap_candidates(rec_key: str, events: list, cache: RecordingCache) -> list:
    n = cache.n_rows(rec_key)
    occupied = sorted(
        (max(0, s - GUARD), min(n, e + GUARD))
        for s, e in events
    )
    free_starts = [0]
    free_ends   = []
    for s, e in occupied:
        free_ends.append(max(0, s - 1))
        free_starts.append(min(n, e + 1))
    free_ends.append(n)
    candidates = []
    for fs, fe in zip(free_starts, free_ends):
        max_start = fe - WINDOW_SIZE
        for pos in range(fs, max_start, STRIDE):
            candidates.append((rec_key, pos))
    return candidates


def full_candidates(rec_key: str, cache: RecordingCache) -> list:
    n = cache.n_rows(rec_key)
    return [(rec_key, pos) for pos in range(0, max(0, n - WINDOW_SIZE + 1), STRIDE)]


# ── Build one split ───────────────────────────────────────────────────────────
def build_split(
    name: str,
    events: list,
    gt0_keys: list,
    cache: RecordingCache,
    rng: np.random.Generator,
) -> tuple:
    pos_X = []
    for rec_key, s, e, mid, _ in events:
        half  = WINDOW_SIZE // 2
        start = max(0, mid - half)
        pos_X.append(cache.window(rec_key, start))

    n_pos         = len(pos_X)
    n_neg_total   = int(n_pos / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_gt0     = int(n_neg_total * GT0_NEG_FRAC)
    n_neg_gap     = n_neg_total - n_neg_gt0

    gap_pool = []
    occupied_by_rec = defaultdict(list)
    for rec_key, s, e, mid, _ in events:
        occupied_by_rec[rec_key].append((s, e))
    for rec_key, evts in occupied_by_rec.items():
        gap_pool.extend(gap_candidates(rec_key, evts, cache))

    gt0_pool = []
    for rec_key in gt0_keys:
        gt0_pool.extend(full_candidates(rec_key, cache))

    def sample_pool(pool, n, rng):
        if not pool:
            return []
        while len(pool) < n:
            pool.append(pool[int(rng.integers(len(pool)))])
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    neg_gap = sample_pool(gap_pool, n_neg_gap, rng)
    neg_gt0 = sample_pool(gt0_pool, n_neg_gt0, rng)
    neg_all = neg_gap + neg_gt0

    neg_X = [cache.window(rk, pos) for rk, pos in neg_all]

    X  = np.stack(pos_X + neg_X).astype(np.float32)
    yp = np.array([1] * n_pos + [0] * len(neg_X), dtype=np.int64)
    perm = rng.permutation(len(X))

    print(f"  {name:5s}: {len(X):5d} samples  "
          f"({n_pos} pos / {len(neg_X)} neg  "
          f"[{len(neg_gap)} gap + {len(neg_gt0)} GT0-file])")
    return X[perm], yp[perm]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("Building carp presence v4 dataset")
    print(f"  SIGMA_FLOOR    = {SIGMA_FLOOR}")
    print(f"  GT0_NEG_FRAC   = {GT0_NEG_FRAC}")
    print(f"  PRESENCE_RATIO = {PRESENCE_RATIO}")
    print(f"  Window channels: {N_CH_RAW} raw + 2 envelope = {N_CH}")
    print("=" * 65)

    all_events  = []
    split_of    = {}
    cache       = RecordingCache()
    gt0_by_split = defaultdict(list)

    for report_path, rec_dir, source in SOURCES:
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping")
            continue

        rep   = pd.read_csv(report_path)
        valid = rep[rep["is_valid"] == True] if "is_valid" in rep.columns else pd.DataFrame()
        gt_map = valid.groupby("raw_data_file_name").size().to_dict() if len(valid) else {}

        for fpath in sorted(rec_dir.glob("*.csv")):
            if any(x in fpath.name.lower() for x in SKIP):
                continue

            key   = (source, fpath.name)
            split = ("test" if key in TEST_FILES else
                     "val"  if key in VAL_FILES  else "train")

            rec_key = str(fpath)
            if not cache.load(fpath):
                print(f"  SKIP (unreadable): {fpath.name}")
                continue

            split_of[rec_key] = split
            gt_count = gt_map.get(fpath.name, 0)

            if gt_count == 0:
                gt0_by_split[split].append(rec_key)
            else:
                file_events = valid[valid["raw_data_file_name"] == fpath.name]
                for _, row in file_events.iterrows():
                    s   = int(row["start_index"])
                    e   = int(row["end_index"])
                    mid = (s + e) // 2
                    all_events.append((rec_key, s, e, mid, source))

        print(f"  [{source}] loaded")

    events_by_split = defaultdict(list)
    for rec_key, s, e, mid, src in all_events:
        events_by_split[split_of[rec_key]].append((rec_key, s, e, mid, src))

    print()
    for sp in ("train", "val", "test"):
        n_ev  = len(events_by_split[sp])
        n_gt0 = len(gt0_by_split[sp])
        print(f"  {sp:5s}: {n_ev:4d} positive events  {n_gt0:3d} GT=0 files")

    total = sum(len(v) for v in events_by_split.values())
    print(f"  total pos events: {total}")
    print()

    for split in ("train", "val", "test"):
        X, yp = build_split(
            split,
            events_by_split[split],
            gt0_by_split[split],
            cache,
            rng,
        )
        out = OUT_DIR / f"{split}_dataset.npz"
        np.savez(
            out,
            X          = X,
            y_presence = yp,
            y_length   = np.zeros(len(X), np.float64),
            y_weight   = np.zeros(len(X), np.float64),
        )
        print(f"    saved -> {out}")

    lines = [
        "carp_presence_v4 build report",
        f"SIGMA_FLOOR    = {SIGMA_FLOOR}",
        f"GT0_NEG_FRAC   = {GT0_NEG_FRAC}",
        f"PRESENCE_RATIO = {PRESENCE_RATIO}",
        f"RANDOM_SEED    = {RANDOM_SEED}",
        f"Window shape   = (39, {N_CH})  [12 raw + 2 envelope]",
        "",
        "Envelope channels:",
        "  ch 13 — F_env(t) = max |z| across F15 F37 F18 F32 F45 F67 at time t",
        "  ch 14 — B_env(t) = max |z| across B15 B37 B18 B32 B45 B67 at time t",
        "",
        "Split positive events:",
    ]
    for sp in ("train", "val", "test"):
        n = len(events_by_split[sp])
        pct = n / sum(len(v) for v in events_by_split.values()) * 100
        lines.append(f"  {sp:5s}: {n:4d} ({pct:.1f}%)")
    lines += [
        "",
        "Test recordings (locked, never used in training):",
    ]
    for src, fname in sorted(TEST_FILES):
        lines.append(f"  [{src}] {fname}")
    lines += [
        "",
        "Val recordings:",
    ]
    for src, fname in sorted(VAL_FILES):
        lines.append(f"  [{src}] {fname}")

    (OUT_DIR / "build_report.txt").write_text("\n".join(lines))
    print(f"\nDone.  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
