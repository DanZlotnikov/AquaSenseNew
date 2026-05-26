"""Build carp weight dataset using recording baseline amplitude as a salinity proxy.

Key idea:
  Z-normalisation removes all absolute amplitude information, which is exactly
  what encodes water conductivity (salinity).  Instead of reading salinity from
  the filename, we compute the recording's baseline RMS *before* normalisation:

      baseline_rms = mean( sqrt( mean(raw_channel^2) ) )  across 12 channels

  This single scalar is large for high-salinity (high conductivity) recordings
  and small for fresh-water recordings.  It is then z-normalised *across the
  whole dataset* and passed as a physics feature (physics_dim=1), so the model
  can see it at inference time without any metadata — just the raw CSV.

  The acoustic windows are still per-recording z-normalised (removes DC offset),
  so the two pieces of information are complementary:
    - Temporal CNN: shape of the fish pass (species/size discriminant)
    - Physics MLP: baseline amplitude (salinity/conductivity context)

Sources: same three as build_carp_weight_combined.py
Output:  data/carp/weight_baseline/  X shape (N, 39, 13)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

OUT_DIR = Path("data/carp/weight_baseline")


def _parse_weight_g(fname):
    m = re.search(r"(\d+)gr", fname)
    return float(m.group(1)) if m else None

def _find_recording(fname, search_dirs):
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
        self._baseline_rms = {}   # mean channel RMS before normalisation

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
        # Baseline RMS: per-channel RMS, then averaged across channels.
        # Reflects absolute signal magnitude (salinity proxy).
        channel_rms = np.sqrt(np.mean(raw ** 2, axis=0))   # (12,)
        self._baseline_rms[key] = float(np.mean(channel_rms))
        self._raw[key]  = raw
        self._mean[key] = m
        self._std[key]  = s
        return True

    def get_centered_window(self, key, mid):
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

    def get_baseline_rms(self, key):
        return self._baseline_rms.get(key, 0.0)


def build_samples(cache):
    waves, weights, baselines, classes = [], [], [], []
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
            baselines.append(cache.get_baseline_rms(key))
            classes.append(str(int(weight_g)))
            n_source += 1
        print(f"  {report_path.name:<35s}  +{n_source} events")

    if skipped_rec:    print(f"  Skipped {skipped_rec} (recording not found)")
    if skipped_weight: print(f"  Skipped {skipped_weight} (no weight in filename)")

    return {
        "wave":     np.stack(waves),
        "weight":   np.array(weights,   dtype=np.float64),
        "baseline": np.array(baselines, dtype=np.float32),
        "classes":  np.array(classes),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Building carp_weight_baseline dataset")
    print("Physics feature: recording baseline RMS (salinity proxy)")
    print("=" * 65)

    cache = RecordingCache()
    print("\nExtracting event windows ...")
    pos = build_samples(cache)
    n = len(pos["weight"])
    print(f"\nTotal samples: {n}")

    # Show baseline RMS per unique recording
    print("\nBaseline RMS by recording (first occurrence per name):")
    seen = {}
    for report_path, search_dirs in SOURCES:
        if not report_path.exists(): continue
        df = pd.read_csv(report_path)
        for fname in df.raw_data_file_name.unique():
            rec = _find_recording(fname, search_dirs)
            if rec is None: continue
            key = str(rec)
            if fname not in seen:
                seen[fname] = cache.get_baseline_rms(key)
    for fname, rms in sorted(seen.items(), key=lambda x: x[1]):
        sal_m = re.search(r"_S(\d+)", fname)
        sal = int(sal_m.group(1)) if sal_m else 0
        print(f"  {fname:<42}  RMS={rms:>8.2f}  (S={sal})")

    # Z-normalise baseline RMS across the dataset
    bl_mean = pos["baseline"].mean()
    bl_std  = pos["baseline"].std()
    bl_std  = bl_std if bl_std > 1e-8 else 1.0
    print(f"\nBaseline RMS  mean={bl_mean:.3f}  std={bl_std:.3f}")

    bl_norm   = ((pos["baseline"] - bl_mean) / bl_std).astype(np.float32)
    bl_tiled  = np.tile(bl_norm[:, np.newaxis, np.newaxis], (1, WINDOW_SIZE, 1))
    X_full    = np.concatenate([pos["wave"], bl_tiled], axis=2)
    print(f"X shape: {X_full.shape}  (acoustic 12ch + baseline_rms 1ch)")

    # Save normalisation constants for inference
    np.save(OUT_DIR / "baseline_norm.npy",
            np.array([bl_mean, bl_std], dtype=np.float32))

    # Weight class distribution
    print("\nWeight class distribution:")
    for wc in sorted(np.unique(pos["classes"]), key=lambda x: int(x)):
        cnt  = (pos["classes"] == wc).sum()
        # show mean baseline RMS for this class
        mask = pos["classes"] == wc
        bl   = pos["baseline"][mask].mean()
        print(f"  {int(wc):>5}g : {cnt:>4} events  baseline_rms={bl:.2f}")

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

    print()
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        X  = X_full[idx]
        yp = np.ones(len(idx), dtype=np.int64)
        yl = pos["weight"][idx]
        path = OUT_DIR / f"{name}_dataset.npz"
        np.savez(path, X=X, y_presence=yp, y_length=yl, y_weight=yl)
        print(f"  {name:5s}: {len(idx):4d} samples  -> {path}")

    (OUT_DIR / "build_report.txt").write_text(
        f"carp_weight_baseline dataset\n"
        f"Physics feature: recording baseline RMS (data-derived salinity proxy)\n"
        f"Total: {n}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n"
        f"Baseline RMS mean={bl_mean:.4f}  std={bl_std:.4f}\n"
        f"X shape: (N, {WINDOW_SIZE}, {N_CH+1})\n"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
