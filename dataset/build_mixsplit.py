"""Build train / val / test NPZ splits for both model types using a
per-recording 70 / 15 / 15 window split.

All 7 salted-water recordings contribute to every split, so all fish sizes
(420 g, 720 g, 920 g, 1580 g) appear in train, val, and test simultaneously.

Two output formats are produced from the same windows:
  - FishSaltModel format  (X: N×39×12, X_phys: N×3)   -> *_salt.npz
  - FishModel format      (X: N×39×13, salt column 12) -> *_fish.npz

Run from the project root:
    python dataset/build_mixsplit.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_salt_model import compute_physics_features

# ---------------------------------------------------------------------------
WINDOW_SIZE = 39
STRIDE      = 10
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_ACOUSTIC = len(ACOUSTIC_CHANNELS)

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test = 1 - TRAIN_FRAC - VAL_FRAC

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")
OUT_DIR        = Path("data/carp/salted_water_mixsplit")
SALT_MAX       = 400.0

RNG_SEED = 42
# ---------------------------------------------------------------------------


def _parse_weight(fname: str) -> float:
    m = re.search(r"(\d+)gr", fname)
    return float(m.group(1)) if m else 0.0


def _build_event_index(report: pd.DataFrame) -> dict:
    idx: dict[str, list] = {}
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        idx.setdefault(fname, []).append(
            (int(row["start_index"]), int(row["end_index"]), float(row["_weight"]))
        )
    for fname in idx:
        idx[fname].sort(key=lambda x: x[0])
    return idx


def _best_overlap_weight(events: list, pos: int) -> tuple:
    win_end = pos + WINDOW_SIZE
    best_w, best_ov = 0.0, 0
    for start, end, wt in events:
        if start >= win_end:
            break
        ov = max(0, min(win_end, end) - max(pos, start))
        if ov > best_ov:
            best_ov = ov
            best_w  = wt
    return best_w, best_ov


def _process_recording(csv_path: Path, events: list, rng: np.random.Generator):
    """Return per-split dicts for one recording."""
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)

    mean = np.nanmean(raw, axis=0).astype(np.float32)
    std  = np.nanstd(raw,  axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    raw_norm = (raw - mean) / std

    x_phys   = compute_physics_features(raw).astype(np.float32)   # (3,)
    salt_norm = (1.0 if "_S400" in csv_path.name else 0.0)

    n_rows   = len(raw)
    max_pos  = n_rows - WINDOW_SIZE
    if max_pos < 0:
        return {}

    positions = list(range(0, max_pos + 1, STRIDE))
    n_windows = len(positions)

    X_wave   = np.empty((n_windows, WINDOW_SIZE, N_ACOUSTIC), np.float32)
    X_phys_a = np.tile(x_phys, (n_windows, 1))                   # (N, 3)
    y_pres   = np.zeros(n_windows, np.int64)
    y_weight = np.zeros(n_windows, np.float32)
    salt_col = np.full((n_windows, WINDOW_SIZE, 1), salt_norm, np.float32)

    for i, pos in enumerate(positions):
        X_wave[i] = raw_norm[pos: pos + WINDOW_SIZE]
        if events:
            wt, ov = _best_overlap_weight(events, pos)
            if ov > 0:
                y_pres[i]   = 1
                y_weight[i] = wt

    # Per-recording 70 / 15 / 15 random split
    perm       = rng.permutation(n_windows)
    n_train    = int(n_windows * TRAIN_FRAC)
    n_val      = int(n_windows * VAL_FRAC)
    idx_train  = perm[:n_train]
    idx_val    = perm[n_train: n_train + n_val]
    idx_test   = perm[n_train + n_val:]

    X_fish = np.concatenate([X_wave, salt_col], axis=2)  # (N, 39, 13)

    result = {}
    for split_name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        result[split_name] = {
            "X_wave":   X_wave[idx],
            "X_phys":   X_phys_a[idx],
            "X_fish":   X_fish[idx],
            "y_pres":   y_pres[idx],
            "y_weight": y_weight[idx],
        }
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 65)
    print("Building mixed-split dataset (per-recording 70/15/15)")
    print("All 7 recordings -> all fish sizes in every split")
    print("=" * 65)

    report = pd.read_csv(REPORT_PATH)
    report["_weight"] = report["raw_data_file_name"].apply(_parse_weight)
    report = report[report["_weight"] > 0].reset_index(drop=True)
    print(f"\nEvents with valid weight: {len(report)}")

    event_idx = _build_event_index(report)

    # Accumulate per-split arrays
    salt_data: dict[str, dict] = {"train": {}, "val": {}, "test": {}}

    csv_files = sorted([
        p for p in RECORDINGS_DIR.glob("*.csv")
        if "output" not in p.name.lower()
    ])

    for csv_path in csv_files:
        fname  = csv_path.name
        events = event_idx.get(fname, [])
        rec    = _process_recording(csv_path, events, rng)
        if not rec:
            print(f"  SKIP {fname}  (too short)")
            continue

        for split in ("train", "val", "test"):
            sp = rec[split]
            n_w  = len(sp["y_pres"])
            n_p  = int(sp["y_pres"].sum())
            for key in sp:
                salt_data[split].setdefault(key, []).append(sp[key])

        n_tot = sum(len(rec[s]["y_pres"]) for s in ("train","val","test"))
        n_pos = sum(int(rec[s]["y_pres"].sum()) for s in ("train","val","test"))
        print(f"  {fname:<40s}  windows={n_tot:6d}  pos={n_pos:4d} ({100*n_pos/n_tot:.1f}%)")

    # Concatenate, shuffle, save
    print()
    for split in ("train", "val", "test"):
        d = salt_data[split]
        X_wave  = np.concatenate(d["X_wave"])
        X_phys  = np.concatenate(d["X_phys"])
        X_fish  = np.concatenate(d["X_fish"])
        y_pres  = np.concatenate(d["y_pres"])
        y_wt    = np.concatenate(d["y_weight"])

        perm   = rng.permutation(len(X_wave))
        X_wave = X_wave[perm]; X_phys = X_phys[perm]
        X_fish = X_fish[perm]; y_pres = y_pres[perm]; y_wt = y_wt[perm]

        n, np_ = len(y_pres), int(y_pres.sum())
        print(f"  {split:5s}: {n:,} windows  ({np_:,} pos / {n-np_:,} neg = {100*np_/n:.2f}%)")

        # FishSaltModel format
        salt_path = OUT_DIR / f"{split}_salt.npz"
        np.savez(salt_path, X=X_wave, X_phys=X_phys,
                 y_presence=y_pres, y_weight=y_wt.astype(np.float64))
        print(f"    -> {salt_path}")

        # FishModel format (13 channels, y_weight key)
        fish_path = OUT_DIR / f"{split}_fish.npz"
        np.savez(fish_path, X=X_fish,
                 y_presence=y_pres, y_weight=y_wt.astype(np.float64))
        print(f"    -> {fish_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
