"""Build carp weight dataset with salinity as an extra physics feature.

Identical to build_carp_weight_combined.py except:
  - Salinity is parsed from the recording filename (_S400 -> 400, else 0)
  - Salinity is z-normalised across the full dataset and tiled as a
    13th channel (constant over time) in X -> shape (N, 39, 13)
  - The FishModel physics branch (physics_dim=1) processes this channel.

Sources: same three as build_carp_weight_combined.py
  - data/raw/carp/carp_old/output_report.csv          (85 events)
  - data/raw/carp/carp_salted/salted_water/output_report.csv    (731 events)
  - data/raw/carp/carp_salted/salted_water/test/output_report.csv (89 events)

Output:
  data/carp/weight_salinity/
    train_dataset.npz   X: (N,39,13)  y_length: weight_g  y_presence: all 1
    val_dataset.npz
    test_dataset.npz
    build_report.txt

Run from project root:
    python dataset/build_carp_weight_salinity.py
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

SOURCES = [
    (
        Path("data/raw/carp/carp_old/output_report.csv"),
        [Path("data/raw/carp/carp_old")],
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

OUT_DIR = Path("data/carp/weight_salinity")
# ---------------------------------------------------------------------------


def _parse_weight_g(fname: str) -> float | None:
    m = re.search(r"(\d+)gr", fname)
    return float(m.group(1)) if m else None


def _parse_length_cm(fname: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    return float(m.group(1)) if m else None


def _parse_salinity(fname: str) -> float:
    m = re.search(r"_S(\d+)", fname)
    return float(m.group(1)) if m else 0.0


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
    waves, weights, salinities, classes = [], [], [], []
    skipped_rec = skipped_weight = 0

    for report_path, search_dirs in SOURCES:
        if not report_path.exists():
            print(f"  WARNING: {report_path} not found, skipping")
            continue
        df = pd.read_csv(report_path)
        n_source = 0
        for _, row in df.iterrows():
            fname    = row["raw_data_file_name"]
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
            salinities.append(_parse_salinity(fname))
            classes.append(str(int(weight_g)))
            n_source += 1
        print(f"  {report_path.name:<35s}  +{n_source} events")

    if skipped_rec:    print(f"  Skipped {skipped_rec} (recording not found)")
    if skipped_weight: print(f"  Skipped {skipped_weight} (no weight in filename)")

    return {
        "wave":      np.stack(waves),
        "weight":    np.array(weights,    dtype=np.float64),
        "salinity":  np.array(salinities, dtype=np.float32),
        "classes":   np.array(classes),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Building carp_weight_salinity dataset  (acoustic + salinity)")
    print("=" * 65)

    cache = RecordingCache()
    print("\nExtracting event windows ...")
    pos = build_samples(cache)
    n = len(pos["weight"])
    print(f"\nTotal samples: {n}")

    # Salinity stats
    sal_vals = np.unique(pos["salinity"])
    print(f"Salinity values: {sal_vals.tolist()}")
    sal_mean = pos["salinity"].mean()
    sal_std  = pos["salinity"].std()
    sal_std  = sal_std if sal_std > 1e-8 else 1.0
    print(f"Salinity mean={sal_mean:.1f}  std={sal_std:.1f}")

    # Normalise salinity and tile across time axis -> (N, 39, 1)
    sal_norm = ((pos["salinity"] - sal_mean) / sal_std).astype(np.float32)
    sal_tiled = np.tile(sal_norm[:, np.newaxis, np.newaxis], (1, WINDOW_SIZE, 1))  # (N,39,1)

    # Concatenate: acoustic (N,39,12) + salinity (N,39,1) -> (N,39,13)
    X_full = np.concatenate([pos["wave"], sal_tiled], axis=2)
    print(f"X shape: {X_full.shape}  (acoustic 12ch + salinity 1ch)")

    # Weight class distribution
    print("\nWeight class distribution:")
    for wc in sorted(np.unique(pos["classes"]), key=lambda x: int(x)):
        cnt = (pos["classes"] == wc).sum()
        print(f"  {int(wc):>5}g : {cnt:>4} events")

    # Merge rare classes (<3)
    classes = pos["classes"].copy()
    from collections import Counter
    counts   = Counter(classes)
    all_vals = sorted(set(int(c) for c in classes))
    for cls, cnt in list(counts.items()):
        if cnt < 3:
            idx = all_vals.index(int(cls))
            nbr = all_vals[idx - 1] if idx > 0 else all_vals[idx + 1]
            classes[classes == cls] = str(nbr)
            print(f"  Merged rare class {cls}g ({cnt}) -> {nbr}g")
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

    # Save normalisation constants so inference can reproduce them
    np.save(OUT_DIR / "salinity_norm.npy",
            np.array([sal_mean, sal_std], dtype=np.float32))

    print()
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        X  = X_full[idx]
        yp = np.ones(len(idx), dtype=np.int64)
        yl = pos["weight"][idx]
        path = OUT_DIR / f"{name}_dataset.npz"
        np.savez(path, X=X, y_presence=yp, y_length=yl, y_weight=yl)
        print(f"  {name:5s}: {len(idx):4d} samples  "
              f"weights {yl.min():.0f}-{yl.max():.0f}g  -> {path}")

    (OUT_DIR / "build_report.txt").write_text(
        f"carp_weight_salinity dataset\n"
        f"Total: {n}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n"
        f"Salinity mean={sal_mean:.2f}  std={sal_std:.2f}\n"
        f"X shape: (N, {WINDOW_SIZE}, {N_CH+1})  last channel = normalised salinity\n"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
