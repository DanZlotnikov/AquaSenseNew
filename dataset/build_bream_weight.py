"""Build bream-only weight regression dataset.

One centered window per GT event.  Weight is parsed from the recording filename.
The file-level test/val split matches bream_presence_v1 so the same held-out
recordings are used for both models.

Output: data/bream/weight/{train,val,test}_dataset.npz
  keys: X (N,39,12), y_presence (all 1), y_weight (g), y_length (0)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 39
SIGMA_FLOOR = 150.0
RANDOM_SEED = 42

SIGNAL_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH = len(SIGNAL_CHANNELS)

BASE = Path("data/raw/bream")
OUT  = Path("data/bream/weight")

SKIP = ("output", "report", "axis", "summary", "function", "points", "updated")

# Must match build_bream_presence_v1 splits exactly
BREAM_TEST = {
    "ARDG_AQUA bream 150gr 21 sm  281025 1132002.csv",
    "ARDG_AQUA bream 390gr 26cm  281025 1325003.csv",
    "ARDG_AQUA bream620 gr31cm  281025 1618003.csv",
    "ARDG_AQUA bream 680gr 33 sm  281025 1208003.csv",
    "ARDG_AQUA bream 800gr 36cm  281025 1453002.csv",
}
BREAM_VAL = {
    "ARDG_AQUA bream 150gr 21 sm  281025 1132004.csv",
    "ARDG_AQUA bream 280gr 24cm  281025 1425002.csv",
    "ARDG_AQUA bream500gr 30cm  281025 1500003.csv",
    "ARDG_AQUA bream 580gr 30 sm  281025 1254002.csv",
    "ARDG_AQUA bream 630gr 31cm  281025 1400002.csv",
}


def parse_weight_g(fname: str) -> float | None:
    m = re.search(r"(\d+)\s*gr", fname, re.IGNORECASE)
    return float(m.group(1)) if m else None


def normalize(raw: np.ndarray) -> np.ndarray:
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    mu = np.where(np.isnan(mu), 0.0, mu)
    return np.where(np.isnan((raw - mu) / sigma_eff), 0.0,
                    (raw - mu) / sigma_eff).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("Building bream weight dataset")
    print("=" * 65)

    report = pd.read_csv(BASE / "output_report.csv")
    valid  = report[report["is_valid"] == True]

    samples = {"train": [], "val": [], "test": []}
    skipped_weight = 0
    skipped_rec    = 0

    rec_dir = BASE / "recordings"
    for fpath in sorted(rec_dir.glob("*.csv")):
        if any(x in fpath.name.lower() for x in SKIP):
            continue

        weight_g = parse_weight_g(fpath.name)
        if weight_g is None:
            skipped_weight += 1
            continue

        split = ("test" if fpath.name in BREAM_TEST else
                 "val"  if fpath.name in BREAM_VAL  else "train")

        file_events = valid[valid["raw_data_file_name"] == fpath.name]
        if len(file_events) == 0:
            continue

        if not fpath.exists():
            skipped_rec += 1
            continue

        df  = pd.read_csv(fpath)
        for col in SIGNAL_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw  = df[SIGNAL_CHANNELS].values.astype(np.float32)
        norm = normalize(raw)

        for _, row in file_events.iterrows():
            s   = int(row["start_index"])
            e   = int(row["end_index"])
            mid = (s + e) // 2
            half  = WINDOW_SIZE // 2
            start = max(0, mid - half)
            seg   = norm[start: start + WINDOW_SIZE]
            if len(seg) < WINDOW_SIZE:
                pad = np.zeros((WINDOW_SIZE - len(seg), N_CH), np.float32)
                seg = np.concatenate([seg, pad])
            samples[split].append((seg[:WINDOW_SIZE], weight_g))

    if skipped_weight:
        print(f"  Skipped {skipped_weight} files (no weight in filename)")
    if skipped_rec:
        print(f"  Skipped {skipped_rec} files (not found)")

    report_lines = ["bream weight dataset build report", ""]
    print()
    for split in ("train", "val", "test"):
        sp = samples[split]
        if not sp:
            print(f"  {split}: 0 samples — skipping")
            continue
        X       = np.stack([s[0] for s in sp]).astype(np.float32)
        weights = np.array([s[1] for s in sp], dtype=np.float64)
        yp      = np.ones(len(X), dtype=np.int64)

        # Shuffle
        perm = rng.permutation(len(X))
        X, weights = X[perm], weights[perm]

        path = OUT / f"{split}_dataset.npz"
        np.savez(path,
                 X          = X,
                 y_presence = yp,
                 y_weight   = weights,
                 y_length   = np.zeros(len(X), np.float64))

        print(f"  {split:5s}: {len(X):4d} samples  "
              f"weight range {weights.min():.0f}-{weights.max():.0f}g  "
              f"mean={weights.mean():.0f}g")
        report_lines.append(
            f"{split:5s}: {len(X)} samples  "
            f"weight {weights.min():.0f}-{weights.max():.0f}g  mean={weights.mean():.0f}g")

        # Weight class breakdown
        for wc in sorted(np.unique(weights)):
            n = int((weights == wc).sum())
            print(f"          {int(wc):>4}g: {n:>3} events")
        print()

    (OUT / "build_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Done.  Output: {OUT}")


if __name__ == "__main__":
    main()
