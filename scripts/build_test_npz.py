"""Build a window-level test NPZ from any output_report.csv + recordings directory.

Positives : one centered 39-sample window per is_valid==True event.
Negatives : sampled from gap regions (guard = 39 samples around every event),
            at 4:1 neg:pos ratio (matching training PRESENCE_RATIO=0.20).

Usage:
    python scripts/build_test_npz.py \\
        --report   data/raw/carp/ACQ_5_3_2026/output_report.csv \\
        --rec_dir  data/raw/carp/ACQ_5_3_2026 \\
        --out      data/test_npz/ACQ_5_3_2026_test.npz

    # Multiple recording dirs searched in order (fallback):
    python scripts/build_test_npz.py \\
        --report   data/raw/carp/carp_salted/salted_water/test/output_report.csv \\
        --rec_dir  data/raw/carp/carp_salted/salted_water/test \\
                   data/raw/carp/carp_salted/salted_water \\
        --out      data/test_npz/salted_test.npz
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE    = 39
GUARD          = WINDOW_SIZE
PRESENCE_RATIO = 0.20
SIGNAL_CHANNELS = [
    "F15","F37","F18","F32","F45","F67",
    "B15","B37","B18","B32","B45","B67",
]
N_CH = len(SIGNAL_CHANNELS)


def load_norm(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[SIGNAL_CHANNELS].values.astype(np.float32)
    mu = np.nanmean(raw, axis=0)
    s  = np.nanstd(raw,  axis=0)
    s[(s < 1e-8) | np.isnan(s)] = 1.0
    mu = np.where(np.isnan(mu), 0.0, mu)
    return np.where(np.isnan((raw - mu) / s), 0.0, (raw - mu) / s).astype(np.float32)


def find_recording(fname: str, rec_dirs: list[Path]) -> Path | None:
    for d in rec_dirs:
        p = d / fname
        if p.exists():
            return p
    return None


def build_npz(report_path: Path, rec_dirs: list[Path], out_path: Path, seed: int = 42):
    rng = np.random.default_rng(seed)

    df    = pd.read_csv(report_path)
    valid = df[df["is_valid"] == True] if "is_valid" in df.columns else df
    print(f"Report: {report_path}")
    print(f"  Total events : {len(df)}")
    print(f"  Valid events : {len(valid)}")

    # Group by recording file
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for _, row in valid.iterrows():
        by_file[row["raw_data_file_name"]].append(
            (int(row["start_index"]), int(row["end_index"]))
        )

    X_list, y_list = [], []
    skipped_files = []

    for fname, events in sorted(by_file.items()):
        rec_path = find_recording(fname, rec_dirs)
        if rec_path is None:
            skipped_files.append(fname)
            continue

        norm = load_norm(rec_path)
        n    = len(norm)

        # Positives
        occupied = []
        for start_idx, end_idx in events:
            mid   = (start_idx + end_idx) // 2
            half  = WINDOW_SIZE // 2
            s     = mid - half
            seg   = norm[max(0, s): max(0, s) + WINDOW_SIZE]
            if len(seg) < WINDOW_SIZE:
                seg = np.concatenate(
                    [seg, np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)]
                )
            X_list.append(seg[:WINDOW_SIZE])
            y_list.append(1)
            occupied.append((max(0, start_idx - GUARD), end_idx + GUARD))

        # Negatives from gap regions
        n_neg = int(len(events) / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
        occ   = sorted(occupied)
        free_starts = [0]; free_ends = []
        for s, e in occ:
            free_ends.append(max(0, s - 1))
            free_starts.append(min(n, e + 1))
        free_ends.append(n)

        candidates = []
        for fs, fe in zip(free_starts, free_ends):
            max_start = fe - WINDOW_SIZE
            if max_start > fs:
                step = max(1, (max_start - fs) // max(1, n_neg // max(1, len(by_file))))
                for pos in range(fs, max_start, step):
                    candidates.append(pos)

        if candidates:
            while len(candidates) < n_neg:
                jitter = int(rng.integers(-5, 6))
                new_pos = max(0, min(candidates[int(rng.integers(len(candidates)))] + jitter,
                                     n - WINDOW_SIZE))
                candidates.append(new_pos)
            idx = rng.choice(len(candidates), size=n_neg, replace=False)
            for i in idx:
                pos = candidates[i]
                seg = norm[pos: pos + WINDOW_SIZE]
                if len(seg) < WINDOW_SIZE:
                    seg = np.concatenate(
                        [seg, np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)]
                    )
                X_list.append(seg[:WINDOW_SIZE])
                y_list.append(0)

        print(f"  {fname:40s}  +{len(events)} pos  +{n_neg} neg")

    if skipped_files:
        print(f"  WARNING: recording not found for: {skipped_files}")

    if not X_list:
        print("ERROR: no samples built — check paths.")
        return

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y_presence=y,
             y_length=np.zeros(len(X)), y_weight=np.zeros(len(X)))
    print(f"\nSaved: {out_path}")
    print(f"  {len(X)} samples  ({int(y.sum())} pos / {int((y==0).sum())} neg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report",  required=True,       help="Path to output_report.csv")
    parser.add_argument("--rec_dir", required=True, nargs="+", help="Directory(s) to search for recording CSVs")
    parser.add_argument("--out",     required=True,       help="Output .npz path")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    build_npz(
        report_path=Path(args.report),
        rec_dirs=[Path(d) for d in args.rec_dir],
        out_path=Path(args.out),
        seed=args.seed,
    )
