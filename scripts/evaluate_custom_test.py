#!/usr/bin/env python3
"""
Evaluate or scan bream recordings with trained length and weight models.

The script runs in one of two modes, selected automatically:

GT evaluation mode  (output_report.csv present in folder)
----------------------------------------------------------
  Phase 1 – Build NPZ
      Reads recording CSVs and output_report.csv.  Extracts 39-sample acoustic
      windows for each labelled detection (positives) and background windows
      (negatives).  Only the 12 raw acoustic channels go into X.
      Saves test_dataset.npz + provenance.csv to the run output folder.

  Phase 2 – Evaluate
      Runs both models on the NPZ and saves:
          predictions.csv       — one row per sample, GT and predicted values
          true_positives.csv    — GT fish present AND model predicted present
          model_detections.csv  — all model-positive predictions
          metrics.json          — MAE, RMSE, MAPE, F1, accuracy
          evaluation.png        — 2x3 diagnostic figure

Inference mode  (no output_report.csv)
--------------------------------------
  Slides a window over every recording CSV, detects fish-present events via
  peak detection, and saves:
      detections.csv   — one row per detected fish with length, weight, time
      summary.txt      — total fish count and biomass

Usage
-----
    python scripts/evaluate_custom_test.py  path/to/folder

    python scripts/evaluate_custom_test.py  path/to/folder \\
        --checkpoint        checkpoints/v4/20260318_171244/best_model.pt \\
        --weight_checkpoint checkpoints/v4_weight/20260318_171628/best_model.pt \\
        --normalizer        data/v4/normalizer.npz \\
        --threshold         0.5 \\
        --output            results
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# model package may live at project root or inside haim_export/
for _candidate in [PROJECT_ROOT, PROJECT_ROOT / "haim_export"]:
    if (_candidate / "model" / "fish_model.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from model.fish_model import FishModel

# ---------------------------------------------------------------------------
# Constants — must match v4 training setup exactly
# ---------------------------------------------------------------------------

WINDOW_SIZE    = 39
PRESENCE_RATIO = 0.20   # 20 % positives in the final dataset
RANDOM_SEED    = 42
GUARD          = WINDOW_SIZE  # samples excluded around each detection for negatives
STEP           = 2            # inference mode: slide every 2 samples
MIN_GAP        = WINDOW_SIZE // 2  # minimum samples between two distinct detections

# Only acoustic channels — IMP13/14/15/16, Distance_cm, Manual_button excluded
RAW_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CHANNELS = len(RAW_CHANNELS)  # 12

DEFAULT_CHECKPOINT        = PROJECT_ROOT / "checkpoints/v4/20260318_171244/best_model.pt"
DEFAULT_WEIGHT_CHECKPOINT = PROJECT_ROOT / "checkpoints/v4_weight/20260318_171628/best_model.pt"
DEFAULT_NORMALIZER        = PROJECT_ROOT / "data/v4/normalizer.npz"
DEFAULT_OUTPUT_DIR        = PROJECT_ROOT / "results"


# ===========================================================================
# PHASE 1 — Build NPZ
# ===========================================================================

# ---------------------------------------------------------------------------
# Report loading (indexing + labels only — no physics / derived features)
# ---------------------------------------------------------------------------

def load_report(folder: Path) -> pd.DataFrame:
    """
    Load output_report.csv from the folder.

    Only these columns are kept — everything else is discarded to prevent
    any possibility of derived-feature leakage into X:
        raw_data_file_name      – which recording file
        start_index             – detection start (sample index)
        end_index               – detection end   (sample index)
        length_from_filename    – GT length label (cm)
        weight_from_filename    – GT weight label (g)
    """
    report_path = folder / "output_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"output_report.csv not found in {folder}")

    df = pd.read_csv(report_path)

    required = ["raw_data_file_name", "start_index", "end_index",
                "length_from_filename", "weight_from_filename"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"output_report.csv is missing column: {col}")

    df = df[required].copy()
    df["length_from_filename"] = pd.to_numeric(df["length_from_filename"], errors="coerce")
    df["weight_from_filename"] = pd.to_numeric(df["weight_from_filename"], errors="coerce")
    df = df.dropna(subset=["length_from_filename", "weight_from_filename"])
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Recording cache — loads only RAW_CHANNELS
# ---------------------------------------------------------------------------

def load_recordings(folder: Path, report_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Load every recording CSV referenced in the report.

    Returns fname → (n_rows, 12) float32 array of acoustic channels only.
    Distance_cm, IMP columns, Date, Time, Manual_button are never read.
    """
    search_dirs = [folder, folder / "recordings"]
    cache: dict[str, np.ndarray] = {}
    missing = []

    for fname in report_df["raw_data_file_name"].unique():
        path = next((d / fname for d in search_dirs if (d / fname).exists()), None)
        if path is None:
            missing.append(fname)
            continue

        raw_df = pd.read_csv(path)
        for col in RAW_CHANNELS:
            if col not in raw_df.columns:
                raw_df[col] = 0.0

        # Read ONLY the 12 acoustic columns — nothing else
        arr = raw_df[RAW_CHANNELS].values.astype(np.float32)

        # Replace NaNs with column mean
        col_mean = np.nanmean(arr, axis=0)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr = np.where(nan_mask, col_mean[np.newaxis, :], arr)

        cache[fname] = arr

    if missing:
        print(f"  Warning: {len(missing)} recording file(s) not found, skipped:")
        for m in missing:
            print(f"    {m}")

    return cache


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_window(signal: np.ndarray, mid: int) -> np.ndarray:
    """39-sample window centred on mid; zero-padded at signal edges."""
    half      = WINDOW_SIZE // 2
    start     = mid - half
    end       = start + WINDOW_SIZE
    n         = len(signal)
    pad_left  = max(0, -start)
    pad_right = max(0, end - n)
    s_start   = max(0, start)
    s_end     = min(n, end)
    window    = signal[s_start:s_end]
    if pad_left:
        window = np.concatenate(
            [np.zeros((pad_left, N_CHANNELS), dtype=np.float32), window]
        )
    if pad_right:
        window = np.concatenate(
            [window, np.zeros((pad_right, N_CHANNELS), dtype=np.float32)]
        )
    return window[:WINDOW_SIZE]


# ---------------------------------------------------------------------------
# Build positive samples
# ---------------------------------------------------------------------------

def build_positives(
    report_df: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    One positive sample per detection row.

    X  : (N, 39, 12)  — raw acoustic window (no derived features)
    y_length : (N,)   — from length_from_filename
    y_weight : (N,)   — from weight_from_filename
    provenance : list of dicts {recording_file, sample_mid, is_positive}
    """
    waves, lengths, weights, provenance = [], [], [], []
    skipped = 0

    for _, row in report_df.iterrows():
        sig = cache.get(row["raw_data_file_name"])
        if sig is None:
            skipped += 1
            continue

        mid    = int((row["start_index"] + row["end_index"]) // 2)
        window = extract_window(sig, mid)
        waves.append(window)
        lengths.append(float(row["length_from_filename"]))
        weights.append(float(row["weight_from_filename"]))
        provenance.append({
            "recording_file": row["raw_data_file_name"],
            "sample_mid":     mid,
            "is_positive":    1,
        })

    if skipped:
        print(f"  Skipped {skipped} detections (recording not loaded)")

    return (
        np.stack(waves),
        np.array(lengths, dtype=np.float64),
        np.array(weights, dtype=np.float64),
        provenance,
    )


# ---------------------------------------------------------------------------
# Build negative samples
# ---------------------------------------------------------------------------

def build_negatives(
    report_df: pd.DataFrame,
    cache: dict[str, np.ndarray],
    n_total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """
    Sample n_total background windows from regions at least GUARD samples
    away from every detection.  Labels are all zero.
    """
    # Build occupied intervals per file
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for _, row in report_df.iterrows():
        fname = row["raw_data_file_name"]
        s     = max(0, int(row["start_index"]) - GUARD)
        e     = int(row["end_index"]) + GUARD
        occupied[fname].append((s, e))

    # Enumerate candidate start positions outside occupied regions
    candidates: list[tuple[str, int]] = []
    for fname, sig in cache.items():
        n_rows = len(sig)
        occ    = sorted(occupied.get(fname, []))

        free_starts = [0]
        free_ends   = []
        for s, e in occ:
            free_ends.append(max(0, s - 1))
            free_starts.append(min(n_rows, e + 1))
        free_ends.append(n_rows)

        for fs, fe in zip(free_starts, free_ends):
            max_start = fe - WINDOW_SIZE
            if max_start > fs:
                # Step to avoid excessive candidates from long files
                step = max(1, (max_start - fs) // max(1, n_total // len(cache)))
                for pos in range(fs, max_start, step):
                    candidates.append((fname, pos))

    if not candidates:
        raise RuntimeError("No background regions found for negative sampling.")

    replace = len(candidates) < n_total
    idx      = rng.choice(len(candidates), size=n_total, replace=replace)
    selected = [candidates[i] for i in idx]

    waves, provenance = [], []
    for fname, start in selected:
        sig    = cache[fname]
        window = sig[start : start + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            pad    = np.zeros((WINDOW_SIZE - len(window), N_CHANNELS), dtype=np.float32)
            window = np.concatenate([window, pad])
        waves.append(window[:WINDOW_SIZE])
        provenance.append({
            "recording_file": fname,
            "sample_mid":     start + WINDOW_SIZE // 2,
            "is_positive":    0,
        })

    return np.stack(waves), provenance


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def load_or_compute_normalizer(
    X: np.ndarray,
    normalizer_path: Path,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Load the training normalizer if available; otherwise compute from X.
    Returns (mean, std, loaded_from_file).
    """
    if normalizer_path.exists():
        nrm  = np.load(normalizer_path)
        mean = nrm["mean"].astype(np.float32)
        std  = nrm["std"].astype(np.float32)
        std[std < 1e-8] = 1.0
        return mean, std, True

    print(f"  Warning: normalizer not found at {normalizer_path}.")
    print("           Computing from test data — results may differ from training eval.")
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std, False


# ---------------------------------------------------------------------------
# Build and save the NPZ
# ---------------------------------------------------------------------------

def build_npz(
    folder: Path,
    normalizer_path: Path,
    run_dir: Path,
    rng: np.random.Generator,
) -> tuple[Path, Path]:
    """
    Full Phase-1 pipeline.

    Returns (npz_path, provenance_csv_path).
    """
    print("\n--- Phase 1: Build NPZ ---")

    print("Loading output_report.csv ...")
    report_df = load_report(folder)
    print(f"  Detections with valid GT labels: {len(report_df)}")

    print("Loading recording files (acoustic channels only) ...")
    cache = load_recordings(folder, report_df)
    print(f"  Loaded {len(cache)} recording file(s)")

    print("Extracting positive samples ...")
    pos_X, pos_lengths, pos_weights, pos_prov = build_positives(report_df, cache)
    n_pos = len(pos_X)
    print(f"  Positive samples: {n_pos}")

    n_neg = int(n_pos / PRESENCE_RATIO * (1 - PRESENCE_RATIO))
    print(f"Sampling {n_neg} negative (background) windows ...")
    neg_X, neg_prov = build_negatives(report_df, cache, n_neg, rng)

    # Combine
    X          = np.concatenate([pos_X, neg_X], axis=0)           # (N, 39, 12)
    y_presence = np.concatenate([
        np.ones(n_pos,  dtype=np.int64),
        np.zeros(n_neg, dtype=np.int64),
    ])
    y_length   = np.concatenate([pos_lengths, np.zeros(n_neg, dtype=np.float64)])
    y_weight   = np.concatenate([pos_weights, np.zeros(n_neg, dtype=np.float64)])
    provenance = pos_prov + neg_prov

    # Shuffle (keep provenance in sync)
    perm = rng.permutation(len(y_presence))
    X          = X[perm]
    y_presence = y_presence[perm]
    y_length   = y_length[perm]
    y_weight   = y_weight[perm]
    provenance = [provenance[i] for i in perm]

    # Normalise
    mean, std, from_file = load_or_compute_normalizer(X, normalizer_path)
    X_norm = ((X - mean) / std).astype(np.float32)
    print(f"  Normalizer: {'loaded from ' + str(normalizer_path) if from_file else 'computed from test data'}")

    # Save NPZ — X contains only normalised acoustic windows, nothing else
    npz_path = run_dir / "test_dataset.npz"
    np.savez(
        npz_path,
        X=X_norm,
        y_presence=y_presence,
        y_length=y_length,
        y_weight=y_weight,
    )
    print(f"  Saved NPZ -> {npz_path}  ({len(y_presence)} samples, {n_pos} pos / {n_neg} neg)")

    # Save provenance so we can join predictions to recording file + sample
    prov_df   = pd.DataFrame(provenance)
    prov_path = run_dir / "provenance.csv"
    prov_df.to_csv(prov_path, index=False)

    # Leakage audit in build report
    audit_lines = [
        "NPZ Build Report — leakage audit",
        "=" * 50,
        f"Source folder    : {folder}",
        f"Detections       : {len(report_df)}",
        f"Positive samples : {n_pos}",
        f"Negative samples : {n_neg}",
        f"Total samples    : {len(y_presence)}",
        "",
        "Input features (X) — ONLY these 12 columns from recording CSVs:",
    ]
    for i, ch in enumerate(RAW_CHANNELS):
        audit_lines.append(f"  [{i+1:2d}] {ch}")
    audit_lines += [
        "",
        "Explicitly EXCLUDED from X:",
        "  IMP13, IMP14, IMP15, IMP16   (all-zero columns)",
        "  Distance_cm                  (acoustic range to fish — leakage risk)",
        "  Manual_button                (all-zero column)",
        "  Date, Time                   (metadata)",
        "",
        "From output_report.csv — used ONLY for:",
        "  raw_data_file_name   → select which recording file",
        "  start_index / end_index → window centre location",
        "  length_from_filename → y_length label",
        "  weight_from_filename → y_weight label",
        "  (all other report columns discarded)",
        "",
        f"Normalizer source: {'file ' + str(normalizer_path) if from_file else 'computed from test data'}",
    ]
    (run_dir / "build_report.txt").write_text("\n".join(audit_lines), encoding="utf-8")

    return npz_path, prov_path


# ===========================================================================
# PHASE 2 — Evaluate
# ===========================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> FishModel:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = FishModel(ckpt["cfg_model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_inference(
    npz_path: Path,
    length_model: FishModel,
    weight_model: FishModel,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        logits_pres, pred_length, pred_weight,
        gt_presence, gt_length, gt_weight
    """
    data       = np.load(npz_path)
    X_all      = data["X"].astype(np.float32)
    gt_pres    = data["y_presence"]
    gt_length  = data["y_length"]
    gt_weight  = data["y_weight"]
    n          = len(X_all)

    all_logits, all_pred_len, all_pred_wt = [], [], []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            X = torch.tensor(X_all[start : start + batch_size]).to(device)
            logit_pres, pred_len = length_model(X)
            _,          pred_wt  = weight_model(X)
            all_logits.append(logit_pres.cpu().numpy())
            all_pred_len.append(pred_len.cpu().numpy())
            all_pred_wt.append(pred_wt.cpu().numpy())

    return (
        np.concatenate(all_logits).flatten(),
        np.concatenate(all_pred_len).flatten(),
        np.concatenate(all_pred_wt).flatten(),
        gt_pres,
        gt_length,
        gt_weight,
    )


def compute_metrics(
    logits: np.ndarray,
    pred_length: np.ndarray,
    pred_weight: np.ndarray,
    gt_presence: np.ndarray,
    gt_length: np.ndarray,
    gt_weight: np.ndarray,
    threshold: float,
) -> dict:
    probs     = 1.0 / (1.0 + np.exp(-logits))
    preds_bin = (probs >= threshold).astype(int)
    targets   = gt_presence.astype(int)

    accuracy = float((preds_bin == targets).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds_bin, average="binary", zero_division=0
    )

    mask = gt_presence.astype(bool)
    metrics = {
        "accuracy":  accuracy,
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }

    if mask.sum() > 0:
        err_len = pred_length[mask] - gt_length[mask]
        err_wt  = pred_weight[mask] - gt_weight[mask]
        metrics.update({
            "length_mae":  float(np.abs(err_len).mean()),
            "length_rmse": float(np.sqrt((err_len ** 2).mean())),
            "length_mape": float((np.abs(err_len) / gt_length[mask]).mean() * 100),
            "weight_mae":  float(np.abs(err_wt).mean()),
            "weight_rmse": float(np.sqrt((err_wt ** 2).mean())),
            "weight_mape": float((np.abs(err_wt) / gt_weight[mask]).mean() * 100),
        })
    else:
        metrics.update({k: 0.0 for k in [
            "length_mae", "length_rmse", "length_mape",
            "weight_mae", "weight_rmse", "weight_mape",
        ]})

    return metrics


def save_predictions_csv(
    logits: np.ndarray,
    pred_length: np.ndarray,
    pred_weight: np.ndarray,
    gt_presence: np.ndarray,
    gt_length: np.ndarray,
    gt_weight: np.ndarray,
    prov_path: Path,
    threshold: float,
    run_dir: Path,
) -> Path:
    """
    Save one row per sample with GT and predicted values.

    Columns:
        recording_file, sample_mid, is_positive,
        gt_length_cm, pred_length_cm,
        gt_weight_g,  pred_weight_g,
        presence_prob, pred_present
    """
    probs     = 1.0 / (1.0 + np.exp(-logits))
    preds_bin = (probs >= threshold).astype(int)

    prov_df = pd.read_csv(prov_path)

    out_df = prov_df.copy()
    out_df["gt_length_cm"]  = gt_length
    out_df["pred_length_cm"] = np.round(pred_length, 2)
    out_df["gt_weight_g"]   = gt_weight
    out_df["pred_weight_g"] = np.round(pred_weight, 1)
    out_df["presence_prob"] = np.round(probs, 4)
    out_df["pred_present"]  = preds_bin

    out_path = run_dir / "predictions.csv"
    out_df.to_csv(out_path, index=False)

    # All model-positive predictions (true positives + false positives)
    detected_path = run_dir / "model_detections.csv"
    out_df[out_df["pred_present"] == 1].to_csv(detected_path, index=False)

    # Strict true positives: GT fish AND model predicted present
    tp_path = run_dir / "true_positives.csv"
    out_df[(out_df["is_positive"] == 1) & (out_df["pred_present"] == 1)].to_csv(tp_path, index=False)

    return out_path


def save_figure(
    logits: np.ndarray,
    pred_length: np.ndarray,
    pred_weight: np.ndarray,
    gt_presence: np.ndarray,
    gt_length: np.ndarray,
    gt_weight: np.ndarray,
    metrics: dict,
    threshold: float,
    run_dir: Path,
) -> Path:
    mask        = gt_presence.astype(bool)
    p_len       = pred_length[mask]
    t_len       = gt_length[mask]
    p_wt        = pred_weight[mask]
    t_wt        = gt_weight[mask]
    err_len     = p_len - t_len
    err_wt      = p_wt - t_wt
    ape_len     = np.abs(err_len) / t_len * 100
    probs       = 1.0 / (1.0 + np.exp(-logits))
    preds_bin   = (probs >= threshold).astype(int)
    cm          = confusion_matrix(gt_presence.astype(int), preds_bin)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Evaluation Results", fontsize=15, fontweight="bold", y=0.99)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # [0,0] Length scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(t_len, p_len, alpha=0.45, s=18, color="#2196F3", edgecolors="none")
    lims = [min(t_len.min(), p_len.min()) - 1, max(t_len.max(), p_len.max()) + 1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title("Length: Predicted vs True", fontsize=10, fontweight="bold")
    ax.set_xlabel("True Length (cm)", fontsize=8); ax.set_ylabel("Pred Length (cm)", fontsize=8)
    ax.text(0.05, 0.93, f"MAE={metrics['length_mae']:.2f} cm  MAPE={metrics['length_mape']:.2f}%",
            transform=ax.transAxes, fontsize=7)
    ax.grid(True, alpha=0.3)

    # [0,1] Weight scatter
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(t_wt, p_wt, alpha=0.45, s=18, color="#FF5722", edgecolors="none")
    lims = [min(t_wt.min(), p_wt.min()) - 10, max(t_wt.max(), p_wt.max()) + 10]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title("Weight: Predicted vs True", fontsize=10, fontweight="bold")
    ax.set_xlabel("True Weight (g)", fontsize=8); ax.set_ylabel("Pred Weight (g)", fontsize=8)
    ax.text(0.05, 0.93, f"MAE={metrics['weight_mae']:.1f} g  MAPE={metrics['weight_mape']:.2f}%",
            transform=ax.transAxes, fontsize=7)
    ax.grid(True, alpha=0.3)

    # [0,2] Length cumulative APE
    ax = fig.add_subplot(gs[0, 2])
    sorted_ape = np.sort(ape_len)
    cumulative  = np.arange(1, len(sorted_ape) + 1) / len(sorted_ape) * 100
    ax.plot(sorted_ape, cumulative, color="#9C27B0", lw=1.5)
    ax.axvline(5.0, color="#4CAF50", lw=1.2, ls="--", label="5% target")
    pct5 = float((ape_len <= 5.0).mean() * 100)
    ax.axhline(pct5, color="#FF9800", lw=1.0, ls=":", label=f"{pct5:.1f}% within 5%")
    ax.set_title("Length Cumul. % Error", fontsize=10, fontweight="bold")
    ax.set_xlabel("Abs % Error", fontsize=8); ax.set_ylabel("Cumulative Samples (%)", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # [1,0] Confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Absent", "Present"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Presence Confusion Matrix", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [1,1] Presence prob distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(probs[gt_presence == 0], bins=30, alpha=0.65, color="#2196F3",
            label="True Absent",  edgecolor="white", lw=0.4)
    ax.hist(probs[gt_presence == 1], bins=30, alpha=0.65, color="#FF5722",
            label="True Present", edgecolor="white", lw=0.4)
    ax.axvline(threshold, color="black", lw=1.2, ls="--", label=f"Threshold={threshold}")
    ax.set_title("Presence Prob by True Class", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Probability", fontsize=8); ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # [1,2] Metrics bar chart
    ax = fig.add_subplot(gs[1, 2])
    names  = ["Accuracy", "Precision", "Recall", "F1",
              "Len MAPE (%)", "Len MAE (cm)", "Wt MAPE (%)", "Wt MAE (g)"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
              metrics["length_mape"], metrics["length_mae"],
              metrics["weight_mape"], metrics["weight_mae"]]
    colors = ["#4CAF50", "#2196F3", "#2196F3", "#2196F3",
              "#FF5722", "#FF9800", "#9C27B0", "#E91E63"]
    bars = ax.barh(names, values, color=colors, alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7)
    ax.set_title("Summary Metrics", fontsize=10, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8); ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(5.0, color="#4CAF50", lw=1.0, ls="--", alpha=0.5)

    out_path = run_dir / "evaluation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Main
# ===========================================================================

# ===========================================================================
# INFERENCE MODE (no GT)
# ===========================================================================

def _simple_peaks(probs: np.ndarray, threshold: float, min_gap: int) -> np.ndarray:
    peaks = []
    i = 0
    while i < len(probs):
        if probs[i] >= threshold:
            end   = min(i + min_gap, len(probs))
            local = i + int(probs[i:end].argmax())
            peaks.append(local)
            i = local + min_gap
        else:
            i += 1
    return np.array(peaks, dtype=int)


def scan_recording(
    recording_path: Path,
    length_model: FishModel,
    weight_model: FishModel,
    device: torch.device,
    threshold: float,
    batch_size: int,
) -> list[dict]:
    """Slide a window over one recording and return a list of detection dicts."""
    df = pd.read_csv(recording_path)
    for col in RAW_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0

    raw    = df[RAW_CHANNELS].values.astype(np.float32)
    n_rows = len(raw)
    if n_rows < WINDOW_SIZE:
        return []

    col_mean = np.nanmean(raw, axis=0)
    nan_mask = np.isnan(raw)
    if nan_mask.any():
        raw = np.where(nan_mask, col_mean[np.newaxis, :], raw)

    # Per-file normalisation (no training stats required)
    f_mean = raw.mean(axis=0).astype(np.float32)
    f_std  = raw.std(axis=0).astype(np.float32)
    f_std[f_std < 1e-8] = 1.0

    has_time   = "Time" in df.columns and "Date" in df.columns
    timestamps = (df["Date"].astype(str) + " " + df["Time"].astype(str)).values if has_time else None

    half      = WINDOW_SIZE // 2
    positions = list(range(0, n_rows - WINDOW_SIZE + 1, STEP))

    all_probs   = np.zeros(n_rows, dtype=np.float32)
    all_lengths = np.zeros(n_rows, dtype=np.float32)
    all_weights = np.zeros(n_rows, dtype=np.float32)

    with torch.no_grad():
        for batch_start in range(0, len(positions), batch_size):
            batch_pos = positions[batch_start : batch_start + batch_size]
            windows   = [(raw[s : s + WINDOW_SIZE] - f_mean) / f_std for s in batch_pos]
            X         = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)

            logit_pres, pred_len = length_model(X)
            _,          pred_wt  = weight_model(X)

            probs   = torch.sigmoid(logit_pres).cpu().numpy().flatten()
            lengths = pred_len.cpu().numpy().flatten()
            weights = pred_wt.cpu().numpy().flatten()

            for i, start in enumerate(batch_pos):
                centre = start + half
                if probs[i] > all_probs[centre]:
                    all_probs[centre]   = probs[i]
                    all_lengths[centre] = lengths[i]
                    all_weights[centre] = weights[i]

    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(all_probs, height=threshold, distance=MIN_GAP)
    except ImportError:
        peaks = _simple_peaks(all_probs, threshold, MIN_GAP)

    detections = []
    for p in peaks:
        length = float(all_lengths[p])
        weight = float(all_weights[p])
        if length <= 0 or weight <= 0:
            continue

        win_start = max(0, int(p) - half)
        win_end   = min(n_rows - 1, int(p) + half)
        det = {
            "recording_file": recording_path.name,
            "peak_sample":    int(p),
            "window_start":   win_start,
            "window_end":     win_end,
            "presence_prob":  round(float(all_probs[p]), 4),
            "length_cm":      round(length, 2),
            "weight_g":       round(weight, 1),
        }
        if timestamps is not None:
            det["peak_timestamp"]    = str(timestamps[int(p)])
            det["window_start_time"] = str(timestamps[win_start])
            det["window_end_time"]   = str(timestamps[win_end])
        detections.append(det)

    return detections


def run_inference_mode(
    folder: Path,
    length_model: FishModel,
    weight_model: FishModel,
    device: torch.device,
    threshold: float,
    batch_size: int,
    run_dir: Path,
) -> None:
    """Scan all recordings in folder and save detections.csv + summary.txt."""
    search_dirs = [folder, folder / "recordings"]
    candidate_files: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            candidate_files.extend(sorted(d.glob("*.csv")))

    seen: set[str] = set()
    recording_files = []
    for f in candidate_files:
        if f.name in seen:
            continue
        seen.add(f.name)
        try:
            header = pd.read_csv(f, nrows=0).columns.tolist()
            if any(ch in header for ch in RAW_CHANNELS):
                recording_files.append(f)
        except Exception:
            pass

    if not recording_files:
        raise RuntimeError(f"No valid recording CSV files found in {folder}")

    print(f"Found {len(recording_files)} recording file(s)")

    all_detections = []
    for rec_path in recording_files:
        dets = scan_recording(rec_path, length_model, weight_model,
                              device, threshold, batch_size)
        label = f"{len(dets):3d} fish detected" if dets else "no fish detected"
        print(f"  {rec_path.name:60s}  {label}")
        all_detections.extend(dets)

    if not all_detections:
        print("\nNo fish detected across all recordings.")
        return

    df_out = pd.DataFrame(all_detections)
    fixed_cols = ["recording_file", "peak_sample", "window_start", "window_end"]
    if "peak_timestamp" in df_out.columns:
        fixed_cols += ["peak_timestamp", "window_start_time", "window_end_time"]
    fixed_cols += ["presence_prob", "length_cm", "weight_g"]
    df_out = df_out[fixed_cols]

    total_fish    = len(df_out)
    total_biomass = df_out["weight_g"].sum()
    mean_length   = df_out["length_cm"].mean()
    mean_weight   = df_out["weight_g"].mean()

    csv_path = run_dir / "detections.csv"
    df_out.to_csv(csv_path, index=False)

    length_bins = df_out["length_cm"].apply(lambda x: int(x)).value_counts().sort_index()
    lines = [
        "AquaSense Bream Detection Report  [inference mode — no GT]",
        f"Recordings    : {len(recording_files)}",
        f"Threshold     : {threshold}",
        "",
        "Results",
        "-------",
        f"Total fish detected : {total_fish}",
        f"Total biomass       : {total_biomass:.1f} g  ({total_biomass / 1000:.3f} kg)",
        f"Mean length         : {mean_length:.2f} cm",
        f"Mean weight         : {mean_weight:.1f} g",
        "",
        "Length distribution (1 cm bins):",
    ]
    for bin_val, count in length_bins.items():
        lines.append(f"  {bin_val:3d} cm  {count:5d}  {'#' * min(count, 40)}")
    lines += ["", "Per-recording summary:"]
    for rec_name, grp in df_out.groupby("recording_file"):
        lines.append(f"  {rec_name:60s}  {len(grp):3d} fish  {grp['weight_g'].sum():.1f} g")

    summary_text = "\n".join(lines)
    summary_path = run_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print("=" * 65)
    print(summary_text)
    print("=" * 65)
    print(f"\nDetections -> {csv_path}")
    print(f"Summary    -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NPZ from labelled recordings, then evaluate length and weight models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder", type=Path,
        help="Folder containing output_report.csv and recording CSVs (or recordings/ subfolder)",
    )
    parser.add_argument("--checkpoint",        type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weight_checkpoint", type=Path, default=DEFAULT_WEIGHT_CHECKPOINT)
    parser.add_argument("--normalizer",        type=Path, default=DEFAULT_NORMALIZER,
                        help="Training-set normalizer.npz (mean/std); computed from test data if absent")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--output",     type=Path,  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int,   default=512)
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")
    if not args.checkpoint.exists():
        parser.error(f"Length checkpoint not found: {args.checkpoint}")
    if not args.weight_checkpoint.exists():
        parser.error(f"Weight checkpoint not found: {args.weight_checkpoint}")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    length_model = load_model(args.checkpoint, device)
    weight_model = load_model(args.weight_checkpoint, device)

    # ------------------------------------------------------------------
    # Mode selection: GT evaluation if output_report.csv present,
    # otherwise sliding-window inference with no ground truth.
    # ------------------------------------------------------------------
    has_report = (args.folder / "output_report.csv").exists()
    if not has_report:
        print("\nNo output_report.csv found — running in inference mode (no GT).")
        run_inference_mode(
            args.folder, length_model, weight_model,
            device, args.threshold, args.batch_size, run_dir,
        )
        return

    print("\noutput_report.csv found — running in GT evaluation mode.")
    rng = np.random.default_rng(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Phase 1: Build NPZ
    # ------------------------------------------------------------------
    npz_path, prov_path = build_npz(args.folder, args.normalizer, run_dir, rng)

    # ------------------------------------------------------------------
    # Phase 2: Evaluate
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Evaluate ---")
    print("Running inference ...")
    logits, pred_len, pred_wt, gt_pres, gt_len, gt_wt = run_inference(
        npz_path, length_model, weight_model, device, args.batch_size
    )

    metrics = compute_metrics(
        logits, pred_len, pred_wt, gt_pres, gt_len, gt_wt, args.threshold
    )

    # Predictions CSV
    pred_csv = save_predictions_csv(
        logits, pred_len, pred_wt, gt_pres, gt_len, gt_wt,
        prov_path, args.threshold, run_dir,
    )
    print(f"  Predictions CSV -> {pred_csv}")

    # Metrics JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "run_ts":     run_ts,
            "folder":     str(args.folder),
            "checkpoint": str(args.checkpoint),
            "weight_checkpoint": str(args.weight_checkpoint),
            "threshold":  args.threshold,
            "metrics":    {k: round(float(v), 6) for k, v in metrics.items()},
        }, f, indent=2)
    print(f"  Metrics JSON    -> {metrics_path}")

    # Figure
    fig_path = save_figure(
        logits, pred_len, pred_wt, gt_pres, gt_len, gt_wt,
        metrics, args.threshold, run_dir,
    )
    print(f"  Figure          -> {fig_path}")

    # Console summary
    print()
    print("=" * 60)
    print(f"  Length  |  MAE: {metrics['length_mae']:.3f} cm  "
          f"RMSE: {metrics['length_rmse']:.3f} cm  "
          f"MAPE: {metrics['length_mape']:.2f}%")
    print(f"  Weight  |  MAE: {metrics['weight_mae']:.1f} g   "
          f"RMSE: {metrics['weight_rmse']:.1f} g   "
          f"MAPE: {metrics['weight_mape']:.2f}%")
    print(f"  Presence|  F1: {metrics['f1']:.4f}  "
          f"Acc: {metrics['accuracy']:.4f}  "
          f"Prec: {metrics['precision']:.4f}  "
          f"Rec: {metrics['recall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
