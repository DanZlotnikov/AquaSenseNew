"""Build leave-one-recording-out (LOOCV) dataset for the 12-channel salt-independent model.

7 recordings → 7 folds. Each fold:
  test  : one held-out recording (never seen during training)
  train : windows from the remaining 6 recordings  (85% random)
  val   : windows from the remaining 6 recordings  (15% random), for early stopping

Also saves all_train.npz (all 7 recordings) for the final deployment model.

Output:
  data/carp/loocv/
    fold_0/  (test = carp_1580gr_41.5cm.csv)
      train.npz, val.npz, test.npz, meta.json
    ...
    fold_6/
    all_train.npz

Format: X (N, 39, 12), y_presence (N,), y_weight (N,)
All acoustic channels are per-recording z-normalised.

Run from project root:
    python dataset/build_loocv.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
WINDOW_SIZE = 39
STRIDE      = 10
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_ACOUSTIC = len(ACOUSTIC_CHANNELS)

VAL_FRAC   = 0.15     # fraction of train windows held out for early stopping
RNG_SEED   = 42

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")
OUT_DIR        = Path("data/carp/loocv")
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


def _extract_windows(csv_path: Path, events: list) -> dict:
    """Return X (N,39,12), y_presence (N,), y_weight (N,) for one recording."""
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)

    # Per-recording z-normalisation — removes absolute amplitude (salinity effect)
    mean = np.nanmean(raw, axis=0).astype(np.float32)
    std  = np.nanstd(raw,  axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    raw_norm = (raw - mean) / std

    n_rows  = len(raw_norm)
    max_pos = n_rows - WINDOW_SIZE
    if max_pos < 0:
        return {}

    positions = list(range(0, max_pos + 1, STRIDE))
    n_windows = len(positions)

    X        = np.empty((n_windows, WINDOW_SIZE, N_ACOUSTIC), np.float32)
    y_pres   = np.zeros(n_windows, np.int64)
    y_weight = np.zeros(n_windows, np.float32)

    for i, pos in enumerate(positions):
        X[i] = raw_norm[pos: pos + WINDOW_SIZE]
        if events:
            wt, ov = _best_overlap_weight(events, pos)
            if ov > 0:
                y_pres[i]   = 1
                y_weight[i] = wt

    return {"X": X, "y_presence": y_pres, "y_weight": y_weight}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    report = pd.read_csv(REPORT_PATH)
    report["_weight"] = report["raw_data_file_name"].apply(_parse_weight)
    report = report[report["_weight"] > 0].reset_index(drop=True)
    event_idx = _build_event_index(report)

    csv_files = sorted([
        p for p in RECORDINGS_DIR.glob("*.csv")
        if "output" not in p.name.lower()
    ])

    print("=" * 65)
    print(f"Processing {len(csv_files)} recordings")
    print("=" * 65)

    recordings = []   # list of (fname, windows_dict)
    for csv_path in csv_files:
        events = event_idx.get(csv_path.name, [])
        data   = _extract_windows(csv_path, events)
        if not data:
            print(f"  SKIP {csv_path.name} (too short)")
            continue
        n   = len(data["y_presence"])
        pos = int(data["y_presence"].sum())
        print(f"  {csv_path.name:<45s}  windows={n:6d}  pos={pos:4d} ({100*pos/n:.1f}%)")
        recordings.append((csv_path.name, data))

    # ── LOOCV folds ────────────────────────────────────────────────────────
    print(f"\nBuilding {len(recordings)} LOOCV folds ...")
    all_X, all_yp, all_yw = [], [], []

    for fold_idx, (test_fname, test_data) in enumerate(recordings):
        fold_dir = OUT_DIR / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Test set: entire held-out recording
        np.savez(
            fold_dir / "test.npz",
            X=test_data["X"],
            y_presence=test_data["y_presence"],
            y_weight=test_data["y_weight"].astype(np.float64),
        )

        # Train pool: all other recordings concatenated
        pool_X, pool_yp, pool_yw = [], [], []
        for fname, data in recordings:
            if fname == test_fname:
                continue
            pool_X.append(data["X"])
            pool_yp.append(data["y_presence"])
            pool_yw.append(data["y_weight"])

        pool_X  = np.concatenate(pool_X)
        pool_yp = np.concatenate(pool_yp)
        pool_yw = np.concatenate(pool_yw)

        # Shuffle, then split 85% train / 15% val
        perm    = rng.permutation(len(pool_X))
        pool_X  = pool_X[perm]; pool_yp = pool_yp[perm]; pool_yw = pool_yw[perm]
        n_val   = max(1, int(len(pool_X) * VAL_FRAC))
        val_X, train_X = pool_X[:n_val],  pool_X[n_val:]
        val_yp, train_yp = pool_yp[:n_val], pool_yp[n_val:]
        val_yw, train_yw = pool_yw[:n_val], pool_yw[n_val:]

        np.savez(fold_dir / "train.npz",
                 X=train_X, y_presence=train_yp, y_weight=train_yw.astype(np.float64))
        np.savez(fold_dir / "val.npz",
                 X=val_X,   y_presence=val_yp,   y_weight=val_yw.astype(np.float64))

        meta = {
            "fold": fold_idx,
            "test_recording": test_fname,
            "test_fish_weight_g": _parse_weight(test_fname),
            "test_windows": len(test_data["y_presence"]),
            "test_pos": int(test_data["y_presence"].sum()),
            "train_windows": len(train_X),
            "val_windows": len(val_X),
        }
        (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  fold_{fold_idx}  test={test_fname}  "
              f"train={len(train_X)}  val={len(val_X)}  test_windows={len(test_data['X'])}")

        all_X.append(test_data["X"])
        all_yp.append(test_data["y_presence"])
        all_yw.append(test_data["y_weight"])

    # ── All-data training set ──────────────────────────────────────────────
    all_X  = np.concatenate(all_X)
    all_yp = np.concatenate(all_yp)
    all_yw = np.concatenate(all_yw)
    perm   = rng.permutation(len(all_X))
    all_X  = all_X[perm]; all_yp = all_yp[perm]; all_yw = all_yw[perm]

    n_val_all  = max(1, int(len(all_X) * VAL_FRAC))
    val_X_all, train_X_all   = all_X[:n_val_all],  all_X[n_val_all:]
    val_yp_all, train_yp_all = all_yp[:n_val_all], all_yp[n_val_all:]
    val_yw_all, train_yw_all = all_yw[:n_val_all], all_yw[n_val_all:]

    np.savez(OUT_DIR / "all_train.npz",
             X=train_X_all, y_presence=train_yp_all,
             y_weight=train_yw_all.astype(np.float64))
    np.savez(OUT_DIR / "all_val.npz",
             X=val_X_all, y_presence=val_yp_all,
             y_weight=val_yw_all.astype(np.float64))

    n, pos = len(all_yp), int(all_yp.sum())
    print(f"\nAll-data: {n} windows  ({pos} pos / {n-pos} neg = {100*pos/n:.1f}%)")
    print(f"  all_train.npz: {len(train_X_all)}")
    print(f"  all_val.npz:   {len(val_X_all)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
