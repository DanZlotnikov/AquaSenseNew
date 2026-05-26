"""Hard negative mining pass.

Runs a trained presence model (or ensemble of models) over the full training
recordings (sequential, not the pre-built dataset), collects windows the model
fires on that are NOT fish (false positives), and appends them to the
training/val datasets as explicit hard negatives.

Ensemble mode: pass multiple --checkpoint paths; only windows where the
*average* ensemble probability >= threshold are collected as hard negatives.
This produces genuinely difficult examples that all ensemble members agree
look like fish.

Run from project root:
    # Single model
    python dataset/mine_hard_negatives.py \
        --checkpoint checkpoints/salted_water/<run>/best_model.pt

    # Ensemble (preferred — mines harder negatives)
    python dataset/mine_hard_negatives.py \
        --checkpoint checkpoints/salted_water/run1/best_model.pt \
                    checkpoints/salted_water/run2/best_model.pt \
                    checkpoints/salted_water/run3/best_model.pt \
        --max_hard_neg 2000
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_model import FishModel

WINDOW_SIZE       = 39
STRIDE            = 10          # step between candidate windows (samples)
GUARD_SAMPLES     = 117         # same as build script — exclude near true detections
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
SALT_MAX     = 400.0
RECORDINGS_DIR = Path("data/carp/salted_water")
REPORT_PATH    = Path("data/carp/salted_water/output_report.csv")
OUT_DIR        = Path("data/carp/salted_water_split")


def _parse_salt(fname):
    return 400.0 if "_S400" in fname else 0.0


def _parse_meta(fname):
    mw = re.search(r"(\d+)gr",           fname)
    ml = re.search(r"(\d+(?:\.\d+)?)cm", fname)
    return (float(ml.group(1)) if ml else None,
            float(mw.group(1)) if mw else None,
            _parse_salt(fname))


def _occupied_ranges(report, fname):
    rows = report[report["raw_data_file_name"] == fname]
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


def _make_window(raw, mean, std, salt_norm, start):
    seg = raw[start: start + WINDOW_SIZE]
    if len(seg) < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - len(seg), len(ACOUSTIC_CHANNELS)), np.float32)
        seg = np.concatenate([seg, pad])
    seg = ((seg - mean) / std).astype(np.float32)
    salt_col = np.full((WINDOW_SIZE, 1), salt_norm, np.float32)
    return np.concatenate([seg, salt_col], axis=1)  # (39, 13)


def mine(checkpoint_paths: list, max_hard_neg: int, threshold: float = 0.5,
         val_frac: float = 0.10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in checkpoint_paths:
        ckpt  = torch.load(path, map_location=device)
        model = FishModel(ckpt["cfg_model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
        print(f"Loaded: {path}")
    print(f"Ensemble size: {len(models)}  threshold: {threshold}")

    report = pd.read_csv(REPORT_PATH)

    hard_windows = []
    for p in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in p.name.lower():
            continue
        _, _, salt = _parse_meta(p.name)
        salt_norm  = salt / SALT_MAX

        df  = pd.read_csv(p)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw  = df[ACOUSTIC_CHANNELS].values.astype(np.float32)
        mean = np.nanmean(raw, axis=0).astype(np.float32)
        std  = np.nanstd(raw,  axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0

        occupied = _occupied_ranges(report, p.name)

        # Slide over the recording, skip occupied regions
        candidates = []
        pos = 0
        while pos + WINDOW_SIZE <= len(raw):
            if not _is_occupied(pos, occupied):
                candidates.append(pos)
            pos += STRIDE

        if not candidates:
            continue

        # Batch inference (ensemble average)
        batch_size = 256
        file_hard  = []
        for i in range(0, len(candidates), batch_size):
            batch_pos = candidates[i: i + batch_size]
            windows   = np.stack([_make_window(raw, mean, std, salt_norm, s)
                                  for s in batch_pos])
            X = torch.tensor(windows).to(device)
            avg_logits = None
            with torch.no_grad():
                for m in models:
                    logits, _ = m(X)
                    lv = logits.cpu().numpy().flatten()
                    avg_logits = lv if avg_logits is None else avg_logits + lv
            avg_logits /= len(models)
            probs = 1.0 / (1.0 + np.exp(-avg_logits))
            # False positives: ensemble says fish but region is guaranteed non-fish
            fp_idx = np.where(probs >= threshold)[0]
            for idx in fp_idx:
                file_hard.append(_make_window(raw, mean, std, salt_norm,
                                              batch_pos[idx]))

        print(f"  {p.name:40s}  {len(candidates):6d} candidates  "
              f"{len(file_hard):4d} FPs")
        hard_windows.extend(file_hard)

    print(f"\nTotal hard negatives found: {len(hard_windows)}")
    if not hard_windows:
        print("No hard negatives — nothing to add.")
        return

    rng = np.random.default_rng(42)
    hard_arr = np.stack(hard_windows).astype(np.float32)
    rng.shuffle(hard_arr)
    hard_arr = hard_arr[:max_hard_neg]
    print(f"Using {len(hard_arr)} hard negatives (capped at {max_hard_neg})")

    # Split hard negatives into train / val proportions
    n_val   = int(len(hard_arr) * val_frac)
    n_train = len(hard_arr) - n_val
    hard_train = hard_arr[:n_train]
    hard_val   = hard_arr[n_train:]

    zeros_lbl = lambda n: np.zeros(n, dtype=np.int64)
    zeros_reg = lambda n: np.zeros(n, dtype=np.float64)

    for split, hard in [("train", hard_train), ("val", hard_val)]:
        path = OUT_DIR / f"{split}_dataset.npz"
        d    = np.load(path)
        X_new   = np.concatenate([d["X"],          hard])
        yp_new  = np.concatenate([d["y_presence"], zeros_lbl(len(hard))])
        yl_new  = np.concatenate([d["y_length"],   zeros_reg(len(hard))])
        yw_new  = np.concatenate([d["y_weight"],   zeros_reg(len(hard))])
        perm    = rng.permutation(len(X_new))
        np.savez(path,
                 X=X_new[perm].astype(np.float32),
                 y_presence=yp_new[perm],
                 y_length=yl_new[perm],
                 y_weight=yw_new[perm])
        pos_n = int(yp_new.sum())
        print(f"  {split}: {len(X_new)} total  ({pos_n} pos / {len(X_new)-pos_n} neg)")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True, nargs="+",
                        help="One or more checkpoint paths (ensemble)")
    parser.add_argument("--max_hard_neg", type=int, default=2000)
    parser.add_argument("--threshold",    type=float, default=0.5)
    args = parser.parse_args()
    mine(args.checkpoint, args.max_hard_neg, args.threshold)
