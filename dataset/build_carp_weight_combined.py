"""Build combined carp weight-regression dataset from ALL carp recordings.

Sources:
  - data/raw/carp/output_report.csv          (85 events: 280-1100g)
  - data/raw/carp/carp_salted/salted_water/output_report.csv    (731 events: 420/720/920/1580g)
  - data/raw/carp/carp_salted/salted_water/test/output_report.csv (89 events: mostly 1580g)

Weight is always parsed from the recording filename (e.g. 880gr -> 880g).
Events whose filename contains no weight pattern are skipped.

Logic (same as build_carp_weight.py):
  - One centered window per event -> positive sample
  - Per-recording z-normalisation
  - Stratified 70 / 20 / 10 split by weight class

Output:
  data/carp/weight_combined/
    train_dataset.npz   keys: X (N,39,12), y_presence (all 1),
                               y_length (weight in g), y_weight
    val_dataset.npz
    test_dataset.npz
    build_report.txt

Run from project root:
    python dataset/build_carp_weight_combined.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
WINDOW_SIZE = 39
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.20
RANDOM_SEED = 42

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH = len(ACOUSTIC_CHANNELS)

# Each source: (report_csv, [recording search dirs])
SOURCES = [
    (
        Path("data/raw/carp/output_report.csv"),
        [Path("data/raw/carp")],
    ),
    (
        Path("data/raw/carp/carp_salted/salted_water/output_report.csv"),
        [Path("data/raw/carp/carp_salted/salted_water")],
    ),
    (
        Path("data/raw/carp/carp_salted/salted_water/test/output_report.csv"),
        [Path("data/raw/carp/carp_salted/salted_water/test"),
         Path("data/raw/carp/carp_salted/salted_water")],
    ),
]

OUT_DIR = Path("data/carp/weight_combined")
# ---------------------------------------------------------------------------


def _parse_weight_g(fname: str) -> float | None:
    m = re.search(r"(\d+)gr", fname)
    return float(m.group(1)) if m else None


def _parse_length_cm(fname: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    return float(m.group(1)) if m else None


def _find_recording(fname: str, search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        p = d / fname
        if p.exists():
            return p
    return None


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
        mask = np.isnan(raw)
        if mask.any():
            raw = np.where(mask, m[np.newaxis, :], raw)
        self._raw[key]  = raw
        self._mean[key] = m
        self._std[key]  = s
        return True

    def get_centered_window(self, key: str, mid: int) -> np.ndarray | None:
        if key not in self._raw:
            return None
        raw  = self._raw[key]
        half = WINDOW_SIZE // 2
        start = mid - half; end = start + WINDOW_SIZE; n = len(raw)
        pad_l = max(0, -start); pad_r = max(0, end - n)
        seg   = raw[max(0, start): min(n, end)]
        if pad_l: seg = np.concatenate([np.zeros((pad_l, N_CH), np.float32), seg])
        if pad_r: seg = np.concatenate([seg, np.zeros((pad_r, N_CH), np.float32)])
        return ((seg[:WINDOW_SIZE] - self._mean[key]) / self._std[key]).astype(np.float32)


def build_samples(cache: RecordingCache) -> dict:
    waves, weights, lengths, classes = [], [], [], []
    skipped_rec = 0
    skipped_weight = 0

    for report_path, search_dirs in SOURCES:
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping")
            continue
        df = pd.read_csv(report_path)
        n_source = 0
        for _, row in df.iterrows():
            fname = row["raw_data_file_name"]
            weight_g = _parse_weight_g(fname)
            if weight_g is None:
                skipped_weight += 1
                continue
            rec = _find_recording(fname, search_dirs)
            if rec is None:
                skipped_rec += 1
                continue
            cache.load(rec)
            key = str(rec)
            mid = int((row["start_index"] + row["end_index"]) // 2)
            win = cache.get_centered_window(key, mid)
            if win is None:
                skipped_rec += 1
                continue
            waves.append(win)
            weights.append(weight_g)
            lengths.append(_parse_length_cm(fname) or 0.0)
            classes.append(str(int(weight_g)))
            n_source += 1
        print(f"  {report_path.name:<35s}  +{n_source} events")

    if skipped_rec:    print(f"  Skipped {skipped_rec} events (recording not found)")
    if skipped_weight: print(f"  Skipped {skipped_weight} events (no weight in filename)")

    return {
        "wave":    np.stack(waves),
        "weight":  np.array(weights, dtype=np.float64),
        "length":  np.array(lengths, dtype=np.float64),
        "classes": np.array(classes),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Building carp_weight_combined dataset")
    print("Sources: data/raw/carp  +  data/raw/carp/carp_salted/salted_water")
    print("=" * 65)

    cache = RecordingCache()
    print("\nExtracting event windows ...")
    pos = build_samples(cache)
    n = len(pos["weight"])
    print(f"\nTotal samples: {n}  (from {len(cache._raw)} recordings)")

    print("\nWeight class distribution:")
    for wc in sorted(np.unique(pos["classes"]), key=lambda x: int(x)):
        cnt = (pos["classes"] == wc).sum()
        print(f"  {int(wc):>5}g : {cnt:>4} events")

    # Merge very rare classes (<3 samples)
    classes = pos["classes"].copy()
    from collections import Counter
    counts  = Counter(classes)
    all_vals = sorted(set(int(c) for c in classes))
    for cls, cnt in list(counts.items()):
        if cnt < 3:
            idx = all_vals.index(int(cls))
            nbr = all_vals[idx - 1] if idx > 0 else all_vals[idx + 1]
            classes[classes == cls] = str(nbr)
            print(f"  Merged rare class {cls}g ({cnt} samples) -> {nbr}g")
    classes = np.array(classes)

    # Stratified split
    sss1 = StratifiedShuffleSplit(1, test_size=0.10, random_state=42)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), classes))
    sss2 = StratifiedShuffleSplit(
        1, test_size=round(VAL_FRAC / (TRAIN_FRAC + VAL_FRAC), 4), random_state=43
    )
    train_rel, val_rel = next(
        sss2.split(np.zeros(len(trainval_idx)), classes[trainval_idx])
    )
    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]

    print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    report_lines = [
        "carp_weight_combined build report",
        f"Total samples: {n}",
        f"Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}",
        "",
        "Weight class distribution:",
    ]
    for wc in sorted(np.unique(pos["classes"]), key=lambda x: int(x)):
        cnt = (pos["classes"] == wc).sum()
        report_lines.append(f"  {int(wc):>5}g : {cnt} events")

    print()
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        X  = pos["wave"][idx]
        yp = np.ones(len(idx), dtype=np.int64)
        yl = pos["weight"][idx]   # weight (g) stored as y_length
        yw = pos["weight"][idx]

        path = OUT_DIR / f"{name}_dataset.npz"
        np.savez(path, X=X, y_presence=yp, y_length=yl, y_weight=yw)
        print(f"  {name:5s}: {len(idx):4d} samples  "
              f"weights {yl.min():.0f}-{yl.max():.0f}g  mean={yl.mean():.0f}g  -> {path}")
        report_lines.append(
            f"{name:5s}: {len(idx)} samples  "
            f"weights {yl.min():.0f}-{yl.max():.0f}g  mean={yl.mean():.0f}g"
        )

    (OUT_DIR / "build_report.txt").write_text("\n".join(report_lines))
    print("\nDone.")


if __name__ == "__main__":
    main()
