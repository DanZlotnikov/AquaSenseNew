"""Evaluation script for the fish weight prediction model.

Called via:
    python main.py --mode test_weight --checkpoint checkpoints/v3_weight/<run_id>/best_model.pt
or directly (for debugging):
    python test/evaluate_weight.py --checkpoint checkpoints/v3_weight/<run_id>/best_model.pt
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

# Ensure project root is on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fish_dataset_weight import make_weight_dataloader
from model.fish_model import FishModel
from utils.logger import setup_logger
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Results figure
# ---------------------------------------------------------------------------

def _save_evaluation_figures(
    logit_presence: np.ndarray,
    pred_weight: np.ndarray,
    y_presence: np.ndarray,
    y_weight: np.ndarray,
    metrics: dict,
    agg: dict,
    threshold: float,
    out_dir: Path,
) -> Path:
    """
    Save a 2x3 evaluation figure with six panels:

        [0,0] Predicted vs true weight scatter  (fish-present samples only)
        [0,1] Residual distribution histogram    (fish-present samples only)
        [0,2] Per-sample absolute percentage error (sorted, fish-present)
        [1,0] Confusion matrix for fish presence
        [1,1] Presence probability distribution   (by true class)
        [1,2] Summary metrics bar chart           (MAE, RMSE, MAPE, F1, Acc)
    """
    mask        = y_presence.astype(bool)
    p_wt        = pred_weight[mask]
    t_wt        = y_weight[mask]
    errors      = p_wt - t_wt
    abs_pct_err = np.abs(errors) / t_wt * 100.0
    probs       = 1.0 / (1.0 + np.exp(-logit_presence))    # sigmoid
    preds_bin   = (probs >= threshold).astype(int)
    cm          = confusion_matrix(y_presence.astype(int), preds_bin)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Test Set Evaluation (Weight Model)", fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ---- [0,0] Predicted vs True scatter ----
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
            f"MAE={metrics['mae']:.1f}g  MAPE={metrics['mape']:.2f}%\n"
            f"Sum true={agg['sum_true_weight_g']:.0f}g  "
            f"Sum pred={agg['sum_pred_weight_g']:.0f}g  "
            f"({agg['sum_diff_pct']:+.2f}%)",
            transform=ax.transAxes, fontsize=7, color="#333333")

    # ---- [0,1] Residual histogram ----
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

    # ---- [0,2] Sorted absolute percentage error ----
    ax = fig.add_subplot(gs[0, 2])
    sorted_ape = np.sort(abs_pct_err)
    cumulative  = np.arange(1, len(sorted_ape) + 1) / len(sorted_ape) * 100
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

    # ---- [1,0] Confusion matrix ----
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Absent", "Present"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Presence Confusion Matrix", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- [1,1] Presence probability distribution by class ----
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(probs[y_presence == 0], bins=30, alpha=0.65, color="#2196F3",
            label="True Absent",  edgecolor="white", linewidth=0.4)
    ax.hist(probs[y_presence == 1], bins=30, alpha=0.65, color="#FF5722",
            label="True Present", edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color="black", linewidth=1.2, linestyle="--",
               label=f"Threshold={threshold}")
    ax.set_title("Presence Probability by True Class", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Probability", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # ---- [1,2] Summary metrics bar chart ----
    ax = fig.add_subplot(gs[1, 2])
    metric_names  = ["Accuracy", "Precision", "Recall", "F1", "MAPE (%)", "MAE (g)", "RMSE (g)",
                     "Sum Diff (%)"]
    metric_values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["mape"],
        metrics["mae"],
        metrics["rmse"],
        abs(agg["sum_diff_pct"]),
    ]
    colors = ["#4CAF50", "#2196F3", "#2196F3", "#2196F3", "#FF5722", "#FF9800", "#FF9800",
              "#9C27B0"]
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8, edgecolor="white")
    for bar, val, name in zip(bars, metric_values, metric_names):
        label = f"{val:+.2f}%" if name == "Sum Diff (%)" else f"{val:.2f}"
        # for Sum Diff show the signed version
        if name == "Sum Diff (%)":
            label = f"{agg['sum_diff_pct']:+.2f}%"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=7)
    ax.set_title("Summary Metrics", fontsize=10, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    # Draw 5% MAPE target reference
    ax.axvline(5.0, color="#4CAF50", linewidth=1.0, linestyle="--", alpha=0.6)

    out_path = out_dir / "evaluation_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, cfg_test: dict) -> dict:
    """
    Load a weight-model checkpoint and evaluate on the test split.

    The model architecture is reconstructed from cfg_model stored inside the
    checkpoint, so no separate model config file is required at inference time.

    Saves to results/weight/<run_id>/:
        metrics.json           — metrics and metadata
        evaluation_results.png — six-panel visual evaluation summary

    Args:
        checkpoint_path : Path to a .pt checkpoint produced by training/train_weight.py.
        cfg_test        : Test configuration dict (from test/test_config_weight.yaml).

    Returns:
        Metrics dict with keys: accuracy, precision, recall, f1,
        mae (g), rmse (g), mape (%).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("evaluate_weight")

    # --- Load checkpoint ---
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = FishModel(ckpt["cfg_model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(
        f"Loaded checkpoint : {checkpoint_path} "
        f"(trained for {ckpt['epoch'] + 1} epochs, val_loss={ckpt['val_loss']:.4f})"
    )

    # --- Data ---
    data_dir   = cfg_test["data_dir"]
    batch_size = cfg_test["batch_size"]
    threshold  = cfg_test.get("threshold", 0.5)

    test_loader = make_weight_dataloader(
        cfg_test.get("test_file") or str(Path(data_dir) / "test_dataset.npz"),
        batch_size, shuffle=False
    )
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # --- Collect predictions ---
    all_logits, all_pred_weight, all_y_pres, all_y_weight = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            logit_pres, pred_weight = model(X)
            all_logits.append(logit_pres.cpu().numpy())
            all_pred_weight.append(pred_weight.cpu().numpy())
            all_y_pres.append(batch["presence"].numpy())
            all_y_weight.append(batch["weight"].numpy())

    all_logits      = np.concatenate(all_logits)
    all_pred_weight = np.concatenate(all_pred_weight)
    all_y_pres      = np.concatenate(all_y_pres)
    all_y_weight    = np.concatenate(all_y_weight)

    # --- Compute per-sample metrics ---
    metrics = compute_all_metrics(
        all_logits, all_pred_weight, all_y_pres, all_y_weight, threshold
    )

    # --- Aggregate (total biomass) metrics ---
    # True total: sum of ground-truth weights for all actual fish.
    # Predicted total: sum of model weight outputs for all samples the model
    # calls present — this is what deployment would actually compute.
    probs_all    = 1.0 / (1.0 + np.exp(-all_logits))
    pred_mask    = probs_all >= threshold
    gt_mask      = all_y_pres.astype(bool)
    sum_true_weight  = float(all_y_weight[gt_mask].sum())
    sum_pred_weight  = float(all_pred_weight[pred_mask].sum())
    sum_abs_diff     = abs(sum_pred_weight - sum_true_weight)
    sum_diff_pct     = (sum_pred_weight - sum_true_weight) / sum_true_weight * 100.0 \
                       if sum_true_weight > 0 else 0.0
    agg = {
        "sum_true_weight_g": sum_true_weight,
        "sum_pred_weight_g": sum_pred_weight,
        "sum_abs_diff_g":    sum_abs_diff,
        "sum_diff_pct":      sum_diff_pct,
    }

    # --- Log results ---
    logger.info("=" * 60)
    logger.info("Test Results (Weight Model)")
    logger.info("=" * 60)
    logger.info(
        f"  Presence  |  Accuracy: {metrics['accuracy']:.4f}  "
        f"Precision: {metrics['precision']:.4f}  "
        f"Recall: {metrics['recall']:.4f}  "
        f"F1: {metrics['f1']:.4f}"
    )
    logger.info(
        f"  Weight    |  MAE: {metrics['mae']:.2f} g  "
        f"RMSE: {metrics['rmse']:.2f} g  "
        f"MAPE: {metrics['mape']:.2f}%"
    )
    logger.info(
        f"  Totals    |  True sum: {sum_true_weight:.1f} g  "
        f"Pred sum: {sum_pred_weight:.1f} g  "
        f"Diff: {sum_abs_diff:.1f} g ({sum_diff_pct:+.2f}%)"
    )
    logger.info("=" * 60)

    # --- Save results ---
    results_dir = Path(cfg_test["results_dir"])
    tag      = cfg_test.get("run_tag", "")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    run_dir  = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # JSON metrics
    result_file = run_dir / "metrics.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "checkpoint":  checkpoint_path,
                "eval_time":   datetime.now().isoformat(timespec="seconds"),
                "target":      "weight",
                "metrics":     {k: round(float(v), 6) for k, v in metrics.items()},
                "aggregate":   {k: round(float(v), 4) for k, v in agg.items()},
            },
            f,
            indent=2,
        )
    logger.info(f"Metrics saved   : {result_file}")

    # Evaluation figure
    fig_path = _save_evaluation_figures(
        all_logits, all_pred_weight, all_y_pres, all_y_weight,
        metrics, agg, threshold, run_dir,
    )
    logger.info(f"Figure saved    : {fig_path}")

    return metrics


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate a fish weight model checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--test_config", default="test/test_config_weight.yaml")
    args = parser.parse_args()

    cfg_test = yaml.safe_load(open(args.test_config))
    evaluate(args.checkpoint, cfg_test)
