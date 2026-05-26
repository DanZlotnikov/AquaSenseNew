"""Hard negative mining for FishSaltModel.

Runs one or more FishSaltModel checkpoints over the full training recordings,
collects windows the model fires on that are outside fish-present guard zones
(false positives), and appends them as hard negatives to the salt NPZ splits.

Output arrays match the FishSaltModel dataset format:
    X        (N, 39, 12)  — per-file z-normalised acoustic waveform (no salt col)
    X_phys   (N, 3)       — physics features, constant per file
    y_presence (N,)       — 0 for all hard negatives
    y_weight   (N,)       — 0 for all hard negatives

Run from project root:
    python dataset/mine_hard_negatives_salt.py \\
        --checkpoint checkpoints/salted_water_salt_model/<run>/best_model.pt

    # Ensemble (averages logits before thresholding)
    python dataset/mine_hard_negatives_salt.py \\
        --checkpoint ckpt1.pt ckpt2.pt ckpt3.pt \\
        --threshold 0.6 --max_hard_neg 2000
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fish_salt_model import FishSaltModel, compute_physics_features

WINDOW_SIZE       = 39
STRIDE            = 10
GUARD_SAMPLES     = 117
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]

RECORDINGS_DIR = Path("data/carp/salted_water")
REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
OUT_DIR        = Path("data/carp/salted_water_split")


def _occupied_ranges(report, fname):
    rows   = report[report["raw_data_file_name"] == fname]
    ranges = []
    for _, r in rows.iterrows():
        ranges.append((max(0, int(r["start_index"]) - GUARD_SAMPLES),
                       int(r["end_index"]) + GUARD_SAMPLES))
    return sorted(ranges)


def _is_occupied(pos, occupied):
    for s, e in occupied:
        if s <= pos + WINDOW_SIZE and e >= pos:
            return True
    return False


def mine(checkpoint_paths: list, max_hard_neg: int, threshold: float = 0.5,
         val_frac: float = 0.10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for path in checkpoint_paths:
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = FishSaltModel(ckpt["cfg_model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
        print(f"Loaded: {path}")
    print(f"Ensemble size: {len(models)}  threshold: {threshold}")

    report = pd.read_csv(REPORT_PATH)

    hard_X     = []   # (39, 12) windows
    hard_Xphys = []   # (3,) per window (constant for the file)

    for p in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in p.name.lower():
            continue

        df = pd.read_csv(p)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)

        # Physics features from raw ADC (before z-normalisation)
        x_phys = compute_physics_features(raw)

        # Per-file z-normalisation
        mean = np.nanmean(raw, axis=0).astype(np.float32)
        std  = np.nanstd(raw,  axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0
        raw_norm = (raw - mean) / std

        occupied   = _occupied_ranges(report, p.name)
        candidates = [
            pos for pos in range(0, len(raw) - WINDOW_SIZE + 1, STRIDE)
            if not _is_occupied(pos, occupied)
        ]
        if not candidates:
            continue

        # Build all candidate windows
        windows = np.stack([
            raw_norm[pos: pos + WINDOW_SIZE].astype(np.float32)
            for pos in candidates
        ])  # (N_cand, 39, 12)
        x_phys_t = np.tile(x_phys, (len(windows), 1))  # (N_cand, 3)

        # Ensemble-average logits
        batch_size = 256
        file_fp_idx = []
        for i in range(0, len(windows), batch_size):
            X_w = torch.tensor(windows[i: i + batch_size]).to(device)
            X_p = torch.tensor(x_phys_t[i: i + batch_size]).to(device)
            avg_logits = None
            with torch.no_grad():
                for m in models:
                    logit, _, _ = m(X_w, X_p)
                    lv = logit.cpu().numpy()
                    avg_logits = lv if avg_logits is None else avg_logits + lv
            avg_logits /= len(models)
            probs = 1.0 / (1.0 + np.exp(-avg_logits))
            for idx in np.where(probs >= threshold)[0]:
                file_fp_idx.append(i + idx)

        for idx in file_fp_idx:
            hard_X.append(windows[idx])
            hard_Xphys.append(x_phys)

        print(f"  {p.name:40s}  {len(candidates):6d} candidates  "
              f"{len(file_fp_idx):4d} FPs")

    print(f"\nTotal hard negatives found: {len(hard_X)}")
    if not hard_X:
        print("No hard negatives — model is already clean, nothing to add.")
        return

    rng      = np.random.default_rng(42)
    hard_X   = np.stack(hard_X).astype(np.float32)
    hard_Xp  = np.stack(hard_Xphys).astype(np.float32)
    perm     = rng.permutation(len(hard_X))
    hard_X   = hard_X[perm][:max_hard_neg]
    hard_Xp  = hard_Xp[perm][:max_hard_neg]
    print(f"Using {len(hard_X)} hard negatives (capped at {max_hard_neg})")

    n_val   = int(len(hard_X) * val_frac)
    n_train = len(hard_X) - n_val

    zeros_float = lambda n: np.zeros(n, dtype=np.float32)

    for split, sl in [("train", slice(None, n_train)), ("val", slice(n_train, None))]:
        path = OUT_DIR / f"{split}_salt.npz"
        d    = np.load(path)

        X_new   = np.concatenate([d["X"],          hard_X[sl]])
        Xp_new  = np.concatenate([d["X_phys"],     hard_Xp[sl]])
        yp_new  = np.concatenate([d["y_presence"], zeros_float(len(hard_X[sl]))])
        yw_new  = np.concatenate([d["y_weight"],   zeros_float(len(hard_X[sl]))])

        perm2   = rng.permutation(len(X_new))
        np.savez(path,
                 X=X_new[perm2].astype(np.float32),
                 X_phys=Xp_new[perm2].astype(np.float32),
                 y_presence=yp_new[perm2].astype(np.float32),
                 y_weight=yw_new[perm2].astype(np.float32))

        pos_n = int(yp_new.sum())
        neg_n = len(X_new) - pos_n
        print(f"  {split}_salt.npz: {len(X_new)} total  "
              f"({pos_n} pos / {neg_n} neg, added {len(hard_X[sl])} HNs)")

    print("Done.  Re-run training on the augmented splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True, nargs="+")
    parser.add_argument("--max_hard_neg", type=int,   default=2000)
    parser.add_argument("--threshold",    type=float, default=0.5)
    args = parser.parse_args()
    mine(args.checkpoint, args.max_hard_neg, args.threshold)
