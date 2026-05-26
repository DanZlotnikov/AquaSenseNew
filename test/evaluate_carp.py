"""Evaluation script for new carp recordings with ground truth labels.

Loads raw sensor windows from data/raw/carp/ recordings, runs both length
and weight models, then computes MAPE/MAE/RMSE against the ground truth
encoded in the filenames (length_from_filename, weight_from_filename).

NO physics features from the detection report are used as model input —
predictions come entirely from raw acoustic windows.

Usage (from project root):
    python test/evaluate_carp.py

Outputs to results/carp_inference/<timestamp>/:
    predictions.csv          — per-detection predictions + ground truth
    summary.txt              — human-readable summary with metrics
    evaluation_length.png    — 6-panel figure (same format as evaluate.py)
    evaluation_weight.png    — 6-panel figure (same format as evaluate_weight.py)
    inference_plots.png      — per-detection bar charts
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fish_model import FishModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPORT_PATH     = Path("data/raw/carp/output_report.csv")
RECORDINGS_DIR  = Path("data/raw/carp")
NORMALIZER_PATH = Path("data/v4/normalizer.npz")

LENGTH_CHECKPOINT = Path("checkpoints/v4/20260318_171244/best_model.pt")
WEIGHT_CHECKPOINT = Path("checkpoints/v4_weight/20260318_171628/best_model.pt")

RESULTS_DIR = Path("results/carp_inference")

WINDOW_SIZE = 39
THRESHOLD   = 0.5

USE_PER_FILE_NORM = True

RAW_CHANNELS = [
    "F15",  "F37",  "F18",  "F32",  "F45",  "F67",
    "B15",  "B37",  "B18",  "B32",  "B45",  "B67",
]
INPUT_CHANNELS = len(RAW_CHANNELS)   # 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_recording(path: Path) -> np.ndarray:
    """Load a recording CSV and return a (n_rows, 12) float32 array.

    Only the 12 acoustic channels are used (IMP13/14/15/16, Distance_cm excluded).
    Any missing acoustic channels are filled with zeros.
    """
    df = pd.read_csv(path)
    for col in RAW_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    return df[RAW_CHANNELS].values.astype(np.float32)


def _extract_window(signal: np.ndarray, mid: int) -> np.ndarray:
    half      = WINDOW_SIZE // 2
    start     = mid - half
    end       = start + WINDOW_SIZE
    n         = len(signal)
    pad_left  = max(0, -start)
    pad_right = max(0, end - n)
    s_start   = max(0, start)
    s_end     = min(n, end)

    window = signal[s_start:s_end]
    if pad_left > 0:
        window = np.concatenate(
            [np.zeros((pad_left, INPUT_CHANNELS), dtype=np.float32), window]
        )
    if pad_right > 0:
        window = np.concatenate(
            [window, np.zeros((pad_right, INPUT_CHANNELS), dtype=np.float32)]
        )
    return window[:WINDOW_SIZE]


def _load_model(checkpoint_path: Path, device: torch.device) -> FishModel:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = FishModel(ckpt["cfg_model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation figures  (same 6-panel format as evaluate.py / evaluate_weight.py)
# ---------------------------------------------------------------------------

def _save_length_eval_figure(
    logit_presence: np.ndarray,
    pred_length: np.ndarray,
    y_presence: np.ndarray,
    y_length: np.ndarray,
    out_dir: Path,
) -> None:
    mask        = y_presence.astype(bool)
    p_len       = pred_length[mask]
    t_len       = y_length[mask]
    errors      = p_len - t_len
    abs_pct_err = np.abs(errors) / t_len * 100.0
    probs       = 1.0 / (1.0 + np.exp(-logit_presence))
    preds_bin   = (probs >= THRESHOLD).astype(int)
    cm          = confusion_matrix(y_presence.astype(int), preds_bin,
                                   labels=[0, 1])

    mae  = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    mape = float(abs_pct_err.mean())
    acc  = float((preds_bin == y_presence.astype(int)).mean())
    tp   = int(cm[1, 1]) if cm.shape == (2, 2) else int((preds_bin[mask] == 1).sum())
    fn   = int(cm[1, 0]) if cm.shape == (2, 2) else int((preds_bin[mask] == 0).sum())
    prec = tp / max(tp + int(cm[0, 1] if cm.shape == (2, 2) else 0), 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Carp Test Set Evaluation — Length Model", fontsize=15,
                 fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # [0,0] Predicted vs True scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(t_len, p_len, alpha=0.45, s=18, color="#2196F3", edgecolors="none")
    lims = [min(t_len.min(), p_len.min()) - 1, max(t_len.max(), p_len.max()) + 1]
    ax.plot(lims, lims, "k--", linewidth=1.0, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title("Predicted vs True Length", fontsize=10, fontweight="bold")
    ax.set_xlabel("True Length (cm)", fontsize=8)
    ax.set_ylabel("Predicted Length (cm)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.93, f"MAE={mae:.2f}cm  MAPE={mape:.2f}%",
            transform=ax.transAxes, fontsize=7, color="#333333")

    # [0,1] Residual histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(errors, bins=30, color="#FF5722", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(errors.mean(), color="#4CAF50", linewidth=1.2, linestyle="-",
               label=f"Mean={errors.mean():.2f}cm")
    ax.set_title("Residuals (Pred - True)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Error (cm)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # [0,2] Cumulative % error
    ax = fig.add_subplot(gs[0, 2])
    sorted_ape = np.sort(abs_pct_err)
    cumulative = np.arange(1, len(sorted_ape) + 1) / len(sorted_ape) * 100
    ax.plot(sorted_ape, cumulative, color="#9C27B0", linewidth=1.5)
    ax.axvline(5.0, color="#4CAF50", linewidth=1.2, linestyle="--", label="5% target")
    pct_within_5 = float((abs_pct_err <= 5.0).mean() * 100)
    ax.axhline(pct_within_5, color="#FF9800", linewidth=1.0, linestyle=":",
               label=f"{pct_within_5:.1f}% within 5%")
    ax.set_title("Cumulative % Error Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Absolute Percentage Error (%)", fontsize=8)
    ax.set_ylabel("Cumulative Samples (%)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # [1,0] Confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="Blues")
    labels_cm = ["Absent", "Present"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_cm, fontsize=8)
    ax.set_yticklabels(labels_cm, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Presence Confusion Matrix", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [1,1] Presence probability distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(probs[y_presence == 0], bins=30, alpha=0.65, color="#2196F3",
            label="True Absent", edgecolor="white", linewidth=0.4)
    ax.hist(probs[y_presence == 1], bins=30, alpha=0.65, color="#FF5722",
            label="True Present", edgecolor="white", linewidth=0.4)
    ax.axvline(THRESHOLD, color="black", linewidth=1.2, linestyle="--",
               label=f"Threshold={THRESHOLD}")
    ax.set_title("Presence Probability by True Class", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Probability", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # [1,2] Summary metrics bar chart
    ax = fig.add_subplot(gs[1, 2])
    metric_names  = ["Accuracy", "Precision", "Recall", "F1", "MAPE (%)", "MAE (cm)", "RMSE (cm)"]
    metric_values = [acc, prec, rec, f1, mape, mae, rmse]
    colors = ["#4CAF50", "#2196F3", "#2196F3", "#2196F3", "#FF5722", "#FF9800", "#FF9800"]
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
    ax.set_title("Summary Metrics", fontsize=10, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(5.0, color="#4CAF50", linewidth=1.0, linestyle="--", alpha=0.6)

    fig.savefig(out_dir / "evaluation_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"mae": mae, "rmse": rmse, "mape": mape, "acc": acc,
            "prec": prec, "rec": rec, "f1": f1, "pct_within_5": pct_within_5}


def _save_weight_eval_figure(
    logit_presence: np.ndarray,
    pred_weight: np.ndarray,
    y_presence: np.ndarray,
    y_weight: np.ndarray,
    out_dir: Path,
) -> None:
    mask        = y_presence.astype(bool)
    p_wt        = pred_weight[mask]
    t_wt        = y_weight[mask]
    errors      = p_wt - t_wt
    abs_pct_err = np.abs(errors) / t_wt * 100.0
    probs       = 1.0 / (1.0 + np.exp(-logit_presence))
    preds_bin   = (probs >= THRESHOLD).astype(int)
    cm          = confusion_matrix(y_presence.astype(int), preds_bin,
                                   labels=[0, 1])

    mae  = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    mape = float(abs_pct_err.mean())
    acc  = float((preds_bin == y_presence.astype(int)).mean())
    tp   = int(cm[1, 1]) if cm.shape == (2, 2) else int((preds_bin[mask] == 1).sum())
    fn   = int(cm[1, 0]) if cm.shape == (2, 2) else int((preds_bin[mask] == 0).sum())
    prec = tp / max(tp + int(cm[0, 1] if cm.shape == (2, 2) else 0), 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    sum_true = float(t_wt.sum())
    sum_pred = float(p_wt.sum())
    sum_diff_pct = (sum_pred - sum_true) / sum_true * 100.0

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Carp Test Set Evaluation — Weight Model", fontsize=15,
                 fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # [0,0] Predicted vs True scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(t_wt, p_wt, alpha=0.45, s=18, color="#2196F3", edgecolors="none")
    lims = [min(t_wt.min(), p_wt.min()) - 10, max(t_wt.max(), p_wt.max()) + 10]
    ax.plot(lims, lims, "k--", linewidth=1.0, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title("Predicted vs True Weight", fontsize=10, fontweight="bold")
    ax.set_xlabel("True Weight (g)", fontsize=8)
    ax.set_ylabel("Predicted Weight (g)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.93,
            f"MAE={mae:.1f}g  MAPE={mape:.2f}%\n"
            f"Sum true={sum_true:.0f}g  Sum pred={sum_pred:.0f}g  ({sum_diff_pct:+.2f}%)",
            transform=ax.transAxes, fontsize=7, color="#333333")

    # [0,1] Residual histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(errors, bins=30, color="#FF5722", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(errors.mean(), color="#4CAF50", linewidth=1.2, linestyle="-",
               label=f"Mean={errors.mean():.1f}g")
    ax.set_title("Residuals (Pred - True)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Error (g)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # [0,2] Cumulative % error
    ax = fig.add_subplot(gs[0, 2])
    sorted_ape = np.sort(abs_pct_err)
    cumulative = np.arange(1, len(sorted_ape) + 1) / len(sorted_ape) * 100
    ax.plot(sorted_ape, cumulative, color="#9C27B0", linewidth=1.5)
    ax.axvline(5.0, color="#4CAF50", linewidth=1.2, linestyle="--", label="5% target")
    pct_within_5 = float((abs_pct_err <= 5.0).mean() * 100)
    ax.axhline(pct_within_5, color="#FF9800", linewidth=1.0, linestyle=":",
               label=f"{pct_within_5:.1f}% within 5%")
    ax.set_title("Cumulative % Error Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Absolute Percentage Error (%)", fontsize=8)
    ax.set_ylabel("Cumulative Samples (%)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # [1,0] Confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="Blues")
    labels_cm = ["Absent", "Present"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_cm, fontsize=8)
    ax.set_yticklabels(labels_cm, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Presence Confusion Matrix", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [1,1] Presence probability distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(probs[y_presence == 0], bins=30, alpha=0.65, color="#2196F3",
            label="True Absent", edgecolor="white", linewidth=0.4)
    ax.hist(probs[y_presence == 1], bins=30, alpha=0.65, color="#FF5722",
            label="True Present", edgecolor="white", linewidth=0.4)
    ax.axvline(THRESHOLD, color="black", linewidth=1.2, linestyle="--",
               label=f"Threshold={THRESHOLD}")
    ax.set_title("Presence Probability by True Class", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Probability", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # [1,2] Summary metrics bar chart
    ax = fig.add_subplot(gs[1, 2])
    metric_names  = ["Accuracy", "Precision", "Recall", "F1", "MAPE (%)",
                     "MAE (g)", "RMSE (g)", "Sum Diff (%)"]
    metric_values = [acc, prec, rec, f1, mape, mae, rmse, abs(sum_diff_pct)]
    colors = ["#4CAF50", "#2196F3", "#2196F3", "#2196F3",
              "#FF5722", "#FF9800", "#FF9800", "#9C27B0"]
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8, edgecolor="white")
    for bar, val, name in zip(bars, metric_values, metric_names):
        label = f"{sum_diff_pct:+.2f}%" if name == "Sum Diff (%)" else f"{val:.3f}"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=7)
    ax.set_title("Summary Metrics", fontsize=10, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(5.0, color="#4CAF50", linewidth=1.0, linestyle="--", alpha=0.6)

    fig.savefig(out_dir / "evaluation_weight.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"mae": mae, "rmse": rmse, "mape": mape, "acc": acc,
            "prec": prec, "rec": rec, "f1": f1, "pct_within_5": pct_within_5,
            "sum_true": sum_true, "sum_pred": sum_pred, "sum_diff_pct": sum_diff_pct}


# ---------------------------------------------------------------------------
# Inference plots (per-detection bar charts)
# ---------------------------------------------------------------------------

def _save_inference_plots(df: pd.DataFrame, out_dir: Path) -> None:
    n = len(df)
    if n == 0:
        return

    labels = [f"{r['file'].replace('.csv','')}\n{r['start_index']}-{r['end_index']}"
              for _, r in df.iterrows()]
    x = np.arange(n)

    fig = plt.figure(figsize=(max(14, n * 0.6), 10))
    fig.suptitle("Carp Inference — Raw Acoustic Window Predictions", fontsize=13,
                 fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x - 0.2, df["presence_prob_length"], 0.4, label="Length model",
           color="#2196F3", alpha=0.8)
    ax.bar(x + 0.2, df["presence_prob_weight"], 0.4, label="Weight model",
           color="#FF5722", alpha=0.8)
    ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--", label="Threshold 0.5")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Presence Probability per Detection", fontsize=10, fontweight="bold")
    ax.set_ylabel("P(fish present)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    len_vals = df["predicted_length_cm"].fillna(0).values
    true_len = df["true_length_cm"].values if "true_length_cm" in df.columns else None
    colors_l = ["#2196F3" if v > 0 else "#BDBDBD" for v in len_vals]
    bars = ax.bar(x, len_vals, color=colors_l, alpha=0.85, edgecolor="white")
    if true_len is not None:
        ax.scatter(x, true_len, color="#FF5722", s=20, zorder=5, label="True", marker="_",
                   linewidths=2)
        ax.legend(fontsize=7)
    for bar, val in zip(bars, len_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_title("Predicted Length (cm) — grey = not detected", fontsize=10, fontweight="bold")
    ax.set_ylabel("Length (cm)", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 0])
    wt_vals = df["predicted_weight_g"].fillna(0).values
    true_wt = df["true_weight_g"].values if "true_weight_g" in df.columns else None
    colors_w = ["#FF5722" if v > 0 else "#BDBDBD" for v in wt_vals]
    bars = ax.bar(x, wt_vals, color=colors_w, alpha=0.85, edgecolor="white")
    if true_wt is not None:
        ax.scatter(x, true_wt, color="#2196F3", s=20, zorder=5, label="True", marker="_",
                   linewidths=2)
        ax.legend(fontsize=7)
    for bar, val in zip(bars, wt_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}g", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_title("Predicted Weight (g) — grey = not detected", fontsize=10, fontweight="bold")
    ax.set_ylabel("Weight (g)", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    detected = df[df["fish_present_length"] & df["fish_present_weight"]]
    if len(detected) > 0:
        ax.scatter(detected["predicted_length_cm"], detected["predicted_weight_g"],
                   c=range(len(detected)), cmap="viridis", s=80, zorder=3,
                   edgecolors="black", linewidths=0.5, label="Predicted")
        if "true_length_cm" in detected.columns and "true_weight_g" in detected.columns:
            ax.scatter(detected["true_length_cm"], detected["true_weight_g"],
                       marker="x", color="red", s=60, zorder=4, label="True")
            ax.legend(fontsize=7)
        ax.set_xlabel("Length (cm)", fontsize=8)
        ax.set_ylabel("Weight (g)", fontsize=8)
        ax.set_title("Length vs Weight (detected fish only)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No fish detected by both models",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Length vs Weight", fontsize=10, fontweight="bold")

    fig.savefig(out_dir / "inference_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    norm      = np.load(NORMALIZER_PATH)
    mean_     = norm["mean"].astype(np.float32)
    std_      = norm["std"].astype(np.float32)
    norm_mode = "per-file (recording-specific)" if USE_PER_FILE_NORM else "training global"
    print(f"Normalisation     : {norm_mode}")

    length_model = _load_model(LENGTH_CHECKPOINT, device)
    weight_model = _load_model(WEIGHT_CHECKPOINT, device)
    print(f"Length model : {LENGTH_CHECKPOINT}")
    print(f"Weight model : {WEIGHT_CHECKPOINT}")

    # Load report — keep ground truth columns when present
    report_raw = pd.read_csv(REPORT_PATH)
    has_gt = ("length_from_filename" in report_raw.columns and
              report_raw["length_from_filename"].notna().any())

    keep_cols = ["raw_data_file_name", "start_index", "end_index", "is_valid"]
    if has_gt:
        keep_cols += ["length_from_filename", "weight_from_filename"]
    report = report_raw[keep_cols].copy()

    print(f"\nDetections in report : {len(report)}")
    print(f"  Valid   : {report['is_valid'].sum()}")
    print(f"  Invalid : {(~report['is_valid']).sum()}")
    print(f"  Ground truth available : {has_gt}")
    print(f"  Files   : {sorted(report['raw_data_file_name'].unique())}")

    # Cache recordings
    rec_cache:      dict[str, np.ndarray] = {}
    rec_norm_cache: dict[str, tuple] = {}
    for fname in report["raw_data_file_name"].unique():
        fpath = RECORDINGS_DIR / fname
        if fpath.exists():
            raw = _load_recording(fpath)
            rec_cache[fname] = raw
            f_mean = raw.mean(axis=0).astype(np.float32)
            f_std  = raw.std(axis=0).astype(np.float32)
            f_std[f_std < 1e-8] = 1.0
            rec_norm_cache[fname] = (f_mean, f_std)
            missing = sum(c not in pd.read_csv(fpath).columns for c in RAW_CHANNELS)
            print(f"  Loaded {fname}  ({len(raw)} rows, {missing} channels zero-filled)")
        else:
            print(f"  WARNING: {fpath} not found — skipping")

    # Extract windows
    windows_list, meta_list = [], []
    for _, det in report.iterrows():
        fname = det["raw_data_file_name"]
        if fname not in rec_cache:
            continue
        sig = rec_cache[fname]
        mid = int((det["start_index"] + det["end_index"]) // 2)
        win = _extract_window(sig, mid)
        if USE_PER_FILE_NORM:
            f_mean, f_std = rec_norm_cache[fname]
            win_norm = (win - f_mean) / f_std
        else:
            win_norm = (win - mean_) / std_

        windows_list.append(win_norm)
        meta = {
            "file":        fname,
            "start_index": int(det["start_index"]),
            "end_index":   int(det["end_index"]),
            "is_valid_detection": bool(det["is_valid"]),
        }
        if has_gt:
            meta["true_length_cm"] = float(det["length_from_filename"])
            meta["true_weight_g"]  = float(det["weight_from_filename"])
        meta_list.append(meta)

    if not windows_list:
        print("ERROR: No windows extracted.")
        return

    X_np = np.stack(windows_list, axis=0)
    X_t  = torch.tensor(X_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        logit_pres_len, pred_len   = length_model(X_t)
        logit_pres_wt,  pred_wt    = weight_model(X_t)

    prob_len    = torch.sigmoid(logit_pres_len).cpu().numpy().flatten()
    pred_len_np = pred_len.cpu().numpy().flatten()
    prob_wt     = torch.sigmoid(logit_pres_wt).cpu().numpy().flatten()
    pred_wt_np  = pred_wt.cpu().numpy().flatten()

    # Compile results
    rows = []
    for i, meta in enumerate(meta_list):
        present_len = bool(prob_len[i] >= THRESHOLD)
        present_wt  = bool(prob_wt[i]  >= THRESHOLD)
        row = {
            "file":                 meta["file"],
            "start_index":          meta["start_index"],
            "end_index":            meta["end_index"],
            "is_valid_detection":   meta["is_valid_detection"],
            "presence_prob_length": round(float(prob_len[i]),    4),
            "fish_present_length":  present_len,
            "predicted_length_cm":  round(float(pred_len_np[i]), 2),
            "presence_prob_weight": round(float(prob_wt[i]),     4),
            "fish_present_weight":  present_wt,
            "predicted_weight_g":   round(float(pred_wt_np[i]),  2),
        }
        if has_gt:
            row["true_length_cm"] = meta["true_length_cm"]
            row["true_weight_g"]  = meta["true_weight_g"]
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Output directory
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"\nPredictions saved : {out_dir / 'predictions.csv'}")

    # Print table
    print("\n" + "=" * 110)
    hdr = (f"{'File':<28} {'Start':>7} {'End':>7} {'Valid':>6}  "
           f"{'P(len)':>7} {'Pred L':>7}")
    if has_gt:
        hdr += f" {'True L':>7} {'Err L':>7}"
    hdr += f"  {'P(wt)':>7} {'Pred W':>8}"
    if has_gt:
        hdr += f" {'True W':>8} {'Err W':>8}"
    print(hdr)
    print("-" * 110)
    for _, r in results_df.iterrows():
        line = (f"{r['file']:<28} {r['start_index']:>7} {r['end_index']:>7} "
                f"{'Yes' if r['is_valid_detection'] else 'No':>6}  "
                f"{r['presence_prob_length']:>7.4f} {r['predicted_length_cm']:>7.2f}")
        if has_gt:
            err_l = r['predicted_length_cm'] - r['true_length_cm']
            line += f" {r['true_length_cm']:>7.1f} {err_l:>+7.2f}"
        line += f"  {r['presence_prob_weight']:>7.4f} {r['predicted_weight_g']:>8.1f}"
        if has_gt:
            err_w = r['predicted_weight_g'] - r['true_weight_g']
            line += f" {r['true_weight_g']:>8.1f} {err_w:>+8.1f}"
        print(line)
    print("=" * 110)

    # Evaluation figures + metrics (only when ground truth is available)
    len_metrics = wt_metrics = None
    if has_gt:
        # All detections are real fish — y_presence = 1 for all
        y_pres = np.ones(len(results_df), dtype=np.float32)
        y_len  = results_df["true_length_cm"].values.astype(np.float32)
        y_wt   = results_df["true_weight_g"].values.astype(np.float32)

        logits_len_np = np.log(prob_len / (1 - np.clip(prob_len, 1e-7, 1 - 1e-7)))
        logits_wt_np  = np.log(prob_wt  / (1 - np.clip(prob_wt,  1e-7, 1 - 1e-7)))

        len_metrics = _save_length_eval_figure(
            logits_len_np, pred_len_np, y_pres, y_len, out_dir)
        wt_metrics = _save_weight_eval_figure(
            logits_wt_np, pred_wt_np, y_pres, y_wt, out_dir)

        print(f"\nLength eval figure : {out_dir / 'evaluation_length.png'}")
        print(f"Weight eval figure : {out_dir / 'evaluation_weight.png'}")

    _save_inference_plots(results_df, out_dir)
    print(f"Inference plots    : {out_dir / 'inference_plots.png'}")

    # Summary
    summary_lines = [
        "Carp Evaluation Summary",
        "=" * 60,
        f"Report           : {REPORT_PATH}",
        f"Length model     : {LENGTH_CHECKPOINT}",
        f"Weight model     : {WEIGHT_CHECKPOINT}",
        f"Normalizer       : {NORMALIZER_PATH}",
        f"Normalisation    : {norm_mode}",
        f"Run time         : {datetime.now().isoformat(timespec='seconds')}",
        f"Threshold        : {THRESHOLD}",
        f"Ground truth     : {'yes' if has_gt else 'no'}",
        "",
        f"Total detections : {len(results_df)}",
        f"  valid by detector : {results_df['is_valid_detection'].sum()}",
    ]

    if has_gt and len_metrics:
        summary_lines += [
            "",
            "Length model results (all detections vs ground truth):",
            f"  MAE          : {len_metrics['mae']:.4f} cm",
            f"  RMSE         : {len_metrics['rmse']:.4f} cm",
            f"  MAPE         : {len_metrics['mape']:.2f}%",
            f"  Within 5%    : {len_metrics['pct_within_5']:.1f}%",
            f"  F1           : {len_metrics['f1']:.4f}",
            f"  Recall       : {len_metrics['rec']:.4f}",
            f"  Precision    : {len_metrics['prec']:.4f}",
            f"  Accuracy     : {len_metrics['acc']:.4f}",
        ]
    if has_gt and wt_metrics:
        summary_lines += [
            "",
            "Weight model results (all detections vs ground truth):",
            f"  MAE          : {wt_metrics['mae']:.2f} g",
            f"  RMSE         : {wt_metrics['rmse']:.2f} g",
            f"  MAPE         : {wt_metrics['mape']:.2f}%",
            f"  Within 5%    : {wt_metrics['pct_within_5']:.1f}%",
            f"  F1           : {wt_metrics['f1']:.4f}",
            f"  Recall       : {wt_metrics['rec']:.4f}",
            f"  Precision    : {wt_metrics['prec']:.4f}",
            f"  Accuracy     : {wt_metrics['acc']:.4f}",
            f"  Total true   : {wt_metrics['sum_true']:.1f} g",
            f"  Total pred   : {wt_metrics['sum_pred']:.1f} g",
            f"  Total diff   : {wt_metrics['sum_diff_pct']:+.2f}%",
        ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (out_dir / "summary.txt").write_text(summary_text)
    print(f"\nSummary saved     : {out_dir / 'summary.txt'}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_inference()
