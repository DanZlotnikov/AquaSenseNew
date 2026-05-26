"""Build train / val / test NPZ splits for FishSaltModel at natural positive rate.

Key differences from build_salt_dataset.py:
  - ALL sliding windows are kept — no subsampling of negatives.
  - Positive rate is the natural one (~2.8%) rather than a hard-coded 10%.
  - Split is by RECORDING, not by window, eliminating physics-feature leakage.
  - No guard zone: every window is labelled positive when it overlaps with an
    event from output_report.csv, otherwise negative.

Recording assignment
--------------------
  Train (5): carp_1580gr_41.5cm.csv, carp_1580gr_41.5cm_S400.csv,
             carp_420gr_27.5cm_S400.csv, carp_720gr_32cm.csv,
             carp_920gr_35cm_S400.csv
  Val   (1): carp_420gr_27.5cm.csv       (fresh, small fish — unseen condition)
  Test  (1): carp_920gr_35cm.csv         (fresh, larger fish — unseen condition)

Output files (saved alongside existing split):
    data/carp/salted_water_split/
        train_salt_natural.npz   keys: X, X_phys, y_presence, y_weight
        val_salt_natural.npz
        test_salt_natural.npz
        build_salt_natural_report.txt

Run from the project root:
    python dataset/build_salt_dataset_natural.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_salt_model import compute_physics_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE = 39
STRIDE      = 10
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_ACOUSTIC = len(ACOUSTIC_CHANNELS)

REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
RECORDINGS_DIR = Path("data/carp/salted_water")
OUT_DIR        = Path("data/carp/salted_water_split")

# Explicit recording → split assignment (prevents any leakage across splits)
SPLIT_ASSIGNMENT = {
    "carp_1580gr_41.5cm.csv":       "train",
    "carp_1580gr_41.5cm_S400.csv":  "train",
    "carp_420gr_27.5cm_S400.csv":   "train",
    "carp_720gr_32cm.csv":          "train",
    "carp_920gr_35cm_S400.csv":     "train",
    "carp_420gr_27.5cm.csv":        "val",   # fresh, 420 g — unseen in training
    "carp_920gr_35cm.csv":          "test",  # fresh, 920 g — unseen in training
}


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _parse_weight(fname: str) -> float:
    m = re.search(r"(\d+)gr", fname)
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------
# Event index for fast overlap lookups
# ---------------------------------------------------------------------------

def _build_event_index(report: pd.DataFrame) -> dict:
    """Return {fname: [(start, end, weight), ...]} sorted by start_index."""
    idx: dict[str, list] = {}
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        start = int(row["start_index"])
        end   = int(row["end_index"])
        wt    = float(row["_weight"])
        idx.setdefault(fname, []).append((start, end, wt))
    for fname in idx:
        idx[fname].sort(key=lambda x: x[0])
    return idx


def _best_overlap_weight(events: list, pos: int) -> tuple:
    """Return (weight, overlap_samples) for the event most overlapping [pos, pos+WINDOW_SIZE).

    Returns (0.0, 0) if no event overlaps.
    """
    win_end = pos + WINDOW_SIZE
    best_w, best_ov = 0.0, 0
    for start, end, wt in events:
        if start >= win_end:
            break  # events are sorted; no more can overlap
        ov = max(0, min(win_end, end) - max(pos, start))
        if ov > best_ov:
            best_ov = ov
            best_w  = wt
    return best_w, best_ov


# ---------------------------------------------------------------------------
# Per-recording window extraction
# ---------------------------------------------------------------------------

def _process_recording(csv_path: Path, events: list) -> dict:
    """Slide windows over one recording; return dict of arrays."""
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)

    # Per-file z-score normalisation (same as inference pipeline)
    mean = np.nanmean(raw, axis=0).astype(np.float32)
    std  = np.nanstd(raw,  axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    raw_norm = (raw - mean) / std

    # Physics features from raw ADC (before normalisation — matches inference)
    x_phys = compute_physics_features(raw).astype(np.float32)  # (3,)

    n_rows = len(raw)
    max_pos = n_rows - WINDOW_SIZE
    if max_pos < 0:
        return {}

    positions  = range(0, max_pos + 1, STRIDE)
    n_windows  = len(positions)

    X          = np.empty((n_windows, WINDOW_SIZE, N_ACOUSTIC), np.float32)
    X_phys_arr = np.tile(x_phys, (n_windows, 1))               # (N, 3)
    y_pres     = np.zeros(n_windows, np.int64)
    y_weight   = np.zeros(n_windows, np.float32)

    for i, pos in enumerate(positions):
        X[i] = raw_norm[pos: pos + WINDOW_SIZE]
        if events:
            wt, ov = _best_overlap_weight(events, pos)
            if ov > 0:
                y_pres[i]   = 1
                y_weight[i] = wt

    return {
        "X":         X,
        "X_phys":    X_phys_arr,
        "y_presence": y_pres,
        "y_weight":  y_weight,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Building natural-rate recording-split dataset for FishSaltModel")
    print("=" * 65)

    # --- Load report ---
    report = pd.read_csv(REPORT_PATH)
    report["_weight"] = report["raw_data_file_name"].apply(_parse_weight)
    report = report[report["_weight"] > 0].reset_index(drop=True)
    print(f"\nEvents with valid weight: {len(report)}")

    event_idx = _build_event_index(report)

    # --- Process each recording and assign to split ---
    split_data: dict[str, list] = {"train": [], "val": [], "test": []}

    csv_files = sorted([
        p for p in RECORDINGS_DIR.glob("*.csv")
        if "output" not in p.name.lower()
    ])

    for csv_path in csv_files:
        fname  = csv_path.name
        split  = SPLIT_ASSIGNMENT.get(fname)
        if split is None:
            print(f"  SKIP {fname}  (not in SPLIT_ASSIGNMENT)")
            continue

        events = event_idx.get(fname, [])
        rec    = _process_recording(csv_path, events)
        if not rec:
            print(f"  SKIP {fname}  (too short)")
            continue

        n_total = len(rec["y_presence"])
        n_pos   = int(rec["y_presence"].sum())
        x_phys0 = rec["X_phys"][0]
        print(f"  {fname:40s}  split={split:5s}  "
              f"windows={n_total:6d}  pos={n_pos:4d} ({100*n_pos/n_total:.1f}%)  "
              f"phys=[{x_phys0[0]:.3f}, {x_phys0[1]:.3f}, salt={x_phys0[2]:.3f}]")

        split_data[split].append(rec)

    # --- Concatenate and save each split ---
    rng = np.random.default_rng(42)
    summary = {}

    for split_name, recs in split_data.items():
        if not recs:
            print(f"  WARNING: {split_name} split is empty!")
            continue

        X         = np.concatenate([r["X"]          for r in recs])
        X_phys    = np.concatenate([r["X_phys"]     for r in recs])
        y_pres    = np.concatenate([r["y_presence"] for r in recs])
        y_weight  = np.concatenate([r["y_weight"]   for r in recs])

        # Shuffle within split (important for training)
        perm = rng.permutation(len(X))
        X, X_phys, y_pres, y_weight = X[perm], X_phys[perm], y_pres[perm], y_weight[perm]

        out_path = OUT_DIR / f"{split_name}_salt_natural.npz"
        np.savez(out_path,
                 X=X, X_phys=X_phys,
                 y_presence=y_pres,
                 y_weight=y_weight.astype(np.float64))

        n_pos = int(y_pres.sum())
        summary[split_name] = (len(y_pres), n_pos)
        print(f"\n  Saved {split_name}: {len(y_pres):,} windows  "
              f"({n_pos:,} pos / {len(y_pres)-n_pos:,} neg  "
              f"= {100*n_pos/len(y_pres):.2f}% positive)  -> {out_path}")

    # --- Print totals ---
    print("\n" + "=" * 65)
    total_w = sum(n for n, _ in summary.values())
    total_p = sum(p for _, p in summary.values())
    print(f"Grand total: {total_w:,} windows  ({total_p:,} pos = {100*total_p/total_w:.2f}%)")
    print("=" * 65)

    # --- Write text report ---
    lines = [
        "Natural-rate recording-split dataset",
        "=" * 65,
        f"Window size : {WINDOW_SIZE}  Stride : {STRIDE}",
        "Positive labelling: window overlaps any event in output_report.csv",
        "Split type: by recording (no window-level leakage)",
        "",
        "Split assignment:",
    ]
    for fname, sp in SPLIT_ASSIGNMENT.items():
        lines.append(f"  {sp:5s}  {fname}")
    lines.append("")
    lines.append("Sizes:")
    for sp, (n, p) in summary.items():
        lines.append(f"  {sp:5s}: {n:,} total  ({p:,} pos / {n-p:,} neg = {100*p/n:.2f}%)")
    lines.append("")
    lines.append(f"Grand total: {total_w:,} windows  "
                 f"({total_p:,} pos = {100*total_p/total_w:.2f}%)")
    (OUT_DIR / "build_salt_natural_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport -> {OUT_DIR / 'build_salt_natural_report.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
