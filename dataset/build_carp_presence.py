"""Build carp-only presence dataset (no bream).

Sources:
  - data/raw/carp/carp_salted/salted_water/output_report.csv   (7 recordings)
  - data/raw/carp/carp_salted/salted_water/test/output_report.csv (2 recordings)

Only is_valid==True events used.

Recording-level split (to prevent leakage across recordings):
  TEST recordings (held out completely):
    carp_1580gr_41.5cm_S400.csv   (57 events)
    carp_920gr_35cm_S400.csv      (66 events)
  TRAIN+VAL recordings: everything else

Within train+val: 70/20 split at event level stratified by weight class.
Negatives from gap regions of the TRAIN+VAL recordings only.

Output:
  data/carp_presence/
    train_dataset.npz   keys: X (N,39,12), y_presence, y_length, y_weight
    val_dataset.npz
    test_dataset.npz
    build_report.txt
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WINDOW_SIZE    = 39
GUARD          = WINDOW_SIZE
PRESENCE_RATIO = 0.20
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.20
RANDOM_SEED    = 42

ACOUSTIC_CHANNELS = [
    "F15","F37","F18","F32","F45","F67",
    "B15","B37","B18","B32","B45","B67",
]
N_CH = len(ACOUSTIC_CHANNELS)

SOURCES = [
    (
        Path("data/raw/carp/carp_salted/salted_water/output_report.csv"),
        Path("data/raw/carp/carp_salted/salted_water"),
    ),
    (
        Path("data/raw/carp/carp_salted/salted_water/test/output_report.csv"),
        Path("data/raw/carp/carp_salted/salted_water/test"),
    ),
]

# These recordings are held out as the test set (completely unseen in training)
TEST_RECORDINGS = {
    "carp_1580gr_41.5cm_S400.csv",
    "carp_920gr_35cm_S400.csv",
}

OUT_DIR = Path("data/carp_presence")


class RecordingCache:
    def __init__(self):
        self._raw  = {}
        self._mean = {}
        self._std  = {}

    def load(self, csv_path: Path) -> bool:
        key = str(csv_path)
        if key in self._raw:
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
        raw = np.where(np.isnan(raw), m[np.newaxis, :], raw)
        self._raw[key] = raw; self._mean[key] = m; self._std[key] = s
        return True

    def get_centered_window(self, key: str, mid: int) -> np.ndarray | None:
        if key not in self._raw:
            return None
        raw = self._raw[key]; n = len(raw)
        half = WINDOW_SIZE // 2
        s = mid - half; e = s + WINDOW_SIZE
        pad_l = max(0, -s); pad_r = max(0, e - n)
        seg = raw[max(0, s): min(n, e)]
        if pad_l: seg = np.concatenate([np.zeros((pad_l, N_CH), np.float32), seg])
        if pad_r: seg = np.concatenate([seg, np.zeros((pad_r, N_CH), np.float32)])
        return ((seg[:WINDOW_SIZE] - self._mean[key]) / self._std[key]).astype(np.float32)

    def get_segment(self, key: str, start: int) -> np.ndarray | None:
        if key not in self._raw:
            return None
        raw = self._raw[key]
        seg = raw[start: start + WINDOW_SIZE]
        if len(seg) < WINDOW_SIZE:
            seg = np.concatenate([seg, np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)])
        return ((seg[:WINDOW_SIZE] - self._mean[key]) / self._std[key]).astype(np.float32)

    def n_rows(self, key: str) -> int:
        return len(self._raw.get(key, []))

    def all_keys(self) -> list[str]:
        return list(self._raw.keys())


def load_events():
    """Load all is_valid events, split into train/val and test by recording."""
    trainval_events = []   # (rec_path_key, mid, weight_class)
    test_events     = []
    rec_paths       = {}   # fname -> Path

    for report_path, rec_dir in SOURCES:
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping")
            continue
        df = pd.read_csv(report_path)
        valid = df[df["is_valid"] == True]
        print(f"  {report_path.parent.name}/{report_path.name}: "
              f"{len(valid)}/{len(df)} valid events")
        for _, row in valid.iterrows():
            fname = row["raw_data_file_name"]
            rec   = rec_dir / fname
            if not rec.exists():
                continue
            rec_paths[fname] = rec
            mid = int((row["start_index"] + row["end_index"]) // 2)
            # weight class from filename (e.g. 920gr -> "920")
            import re
            m = re.search(r"(\d+)gr", fname)
            wclass = m.group(1) if m else "unknown"
            entry = (str(rec), mid, wclass)
            if fname in TEST_RECORDINGS:
                test_events.append(entry)
            else:
                trainval_events.append(entry)

    return trainval_events, test_events, rec_paths


def build_positives(events: list, cache: RecordingCache) -> tuple[np.ndarray, np.ndarray]:
    """Load windows for a list of (rec_key, mid, wclass) events."""
    waves, classes = [], []
    for rec_key, mid, wclass in events:
        cache.load(Path(rec_key))
        win = cache.get_centered_window(rec_key, mid)
        if win is None:
            continue
        waves.append(win)
        classes.append(wclass)
    return np.stack(waves), np.array(classes)


def build_negatives(events: list, cache: RecordingCache, n_total: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Sample negatives from gap regions of the given recordings."""
    occupied = defaultdict(list)
    for rec_key, mid, _ in events:
        half = WINDOW_SIZE // 2
        s = max(0, mid - half - GUARD)
        e = mid + half + GUARD
        occupied[rec_key].append((s, e))

    candidates = []
    for rec_key in set(k for k, _, _ in events):
        n_rows = cache.n_rows(rec_key)
        occ = sorted(occupied[rec_key])
        free_starts = [0]; free_ends = []
        for s, e in occ:
            free_ends.append(max(0, s - 1))
            free_starts.append(min(n_rows, e + 1))
        free_ends.append(n_rows)
        for fs, fe in zip(free_starts, free_ends):
            max_start = fe - WINDOW_SIZE
            if max_start > fs:
                step = max(1, (max_start - fs) // max(1, n_total // max(1, len(set(k for k,_,_ in events)))))
                for pos in range(fs, max_start, step):
                    candidates.append((rec_key, pos))

    # Top up if not enough
    while len(candidates) < n_total:
        rec_key, pos = candidates[int(rng.integers(len(candidates)))]
        jitter  = int(rng.integers(-5, 6))
        new_pos = max(0, min(pos + jitter, cache.n_rows(rec_key) - WINDOW_SIZE))
        candidates.append((rec_key, new_pos))

    idx = rng.choice(len(candidates), size=n_total, replace=len(candidates) < n_total)
    waves = [cache.get_segment(rec_key, pos) for rec_key, pos in [candidates[i] for i in idx]]
    return np.stack([w for w in waves if w is not None])


def save_split(name: str, pos_waves: np.ndarray, neg_waves: np.ndarray,
               rng: np.random.Generator):
    X  = np.concatenate([pos_waves, neg_waves])
    yp = np.concatenate([np.ones(len(pos_waves), np.int64),
                         np.zeros(len(neg_waves), np.int64)])
    perm = rng.permutation(len(X))
    path = OUT_DIR / f"{name}_dataset.npz"
    np.savez(path, X=X[perm], y_presence=yp[perm],
             y_length=np.zeros(len(X)), y_weight=np.zeros(len(X)))
    print(f"  {name:5s}: {len(X):5d} samples  "
          f"({int(yp.sum())} pos / {int((yp==0).sum())} neg)  -> {path}")
    return len(X)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("Building carp-only presence dataset (is_valid=True only)")
    print(f"Test recordings (held out): {sorted(TEST_RECORDINGS)}")
    print("=" * 65 + "\n")

    trainval_events, test_events, rec_paths = load_events()

    print(f"\nTrain+val events: {len(trainval_events)}")
    print(f"Test events:      {len(test_events)}")
    from collections import Counter
    print(f"Weight classes in train+val: {dict(Counter(w for _,_,w in trainval_events))}")
    print(f"Weight classes in test:      {dict(Counter(w for _,_,w in test_events))}")

    cache = RecordingCache()
    for rec_key in set(k for k, _, _ in trainval_events + test_events):
        cache.load(Path(rec_key))

    # --- Train+val split (stratified by weight class) ---
    tv_waves, tv_classes = build_positives(trainval_events, cache)
    sss = StratifiedShuffleSplit(
        1, test_size=round(VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), 4), random_state=RANDOM_SEED
    )
    train_rel, val_rel = next(sss.split(np.zeros(len(tv_waves)), tv_classes))
    train_events = [trainval_events[i] for i in train_rel]
    val_events   = [trainval_events[i] for i in val_rel]

    print(f"\nSplit: train={len(train_events)}  val={len(val_events)}  test={len(test_events)}")

    n_neg_train = int(len(train_events) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_val   = int(len(val_events)   / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    n_neg_test  = int(len(test_events)  / PRESENCE_RATIO * (1 - PRESENCE_RATIO))

    train_pos, _ = build_positives(train_events, cache)
    val_pos,   _ = build_positives(val_events,   cache)
    test_pos,  _ = build_positives(test_events,  cache)

    # Negatives: train and val share negatives from train+val recordings only
    # Test negatives come from test recordings only
    train_neg = build_negatives(train_events + val_events, cache, n_neg_train, rng)
    val_neg   = build_negatives(train_events + val_events, cache, n_neg_val,   rng)
    test_neg  = build_negatives(test_events,               cache, n_neg_test,  rng)

    print()
    save_split("train", train_pos, train_neg, rng)
    save_split("val",   val_pos,   val_neg,   rng)
    save_split("test",  test_pos,  test_neg,  rng)

    report = (
        "Carp-only presence dataset build report\n"
        f"Test recordings (held out): {sorted(TEST_RECORDINGS)}\n"
        f"Train+val events: {len(trainval_events)}\n"
        f"Test events:      {len(test_events)}\n"
        f"Split: train={len(train_events)}  val={len(val_events)}  test={len(test_events)}\n"
    )
    (OUT_DIR / "build_report.txt").write_text(report)
    print("\nDone.")


if __name__ == "__main__":
    main()
