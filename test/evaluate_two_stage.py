"""Two-stage evaluation: length model for presence detection, weight model for regression.

Stage 1: salted length model  -> presence logits (threshold-tunable)
Stage 2: weight-only regressor -> weight prediction on all windows

Biomass = sum of stage-2 weight predictions for windows where stage-1 >= threshold.

Usage:
    # Tune threshold on val set, then evaluate on test set:
    python test/evaluate_two_stage.py \
        --presence_ckpt checkpoints/salted_water/20260423_143759/best_model.pt \
        --weight_ckpt   checkpoints/salted_water_weight_stage2/<run>/best_model.pt \
        --val_file      data/carp/salted_water_split/val_dataset.npz \
        --test_file     data/carp/salted_water_split/test_dataset.npz \
        --results_dir   results/two_stage
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fish_dataset_weight import make_weight_dataloader
from model.fish_model import FishModel
from utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference(model: FishModel, npz_path: str, device: torch.device,
                   batch_size: int = 64, log_scale: bool = False):
    """Return (logits, pred_weight, y_presence, y_weight) arrays — always in grams."""
    loader = make_weight_dataloader(npz_path, batch_size, shuffle=False)
    all_logits, all_weight, all_y_pres, all_y_weight = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            logit, pw = model(X)
            pw_np = pw.cpu().numpy()
            if log_scale:
                pw_np = np.exp(pw_np)
            all_logits.append(logit.cpu().numpy())
            all_weight.append(pw_np)
            all_y_pres.append(batch["presence"].numpy())
            all_y_weight.append(batch["weight"].numpy())
    return (np.concatenate(all_logits),
            np.concatenate(all_weight),
            np.concatenate(all_y_pres),
            np.concatenate(all_y_weight))


def _biomass_at_threshold(logits_pres, pred_weight, y_weight, y_pres, threshold):
    probs     = 1.0 / (1.0 + np.exp(-logits_pres))
    pred_mask = probs >= threshold
    gt_mask   = y_pres.astype(bool)
    sum_true  = float(y_weight[gt_mask].sum())
    sum_pred  = float(pred_weight[pred_mask].sum())
    diff_pct  = (sum_pred - sum_true) / sum_true * 100.0 if sum_true > 0 else 0.0
    return sum_true, sum_pred, diff_pct


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(logits_pres, pred_weight, y_weight, y_pres,
                   thresholds=None, logger=None):
    """Sweep thresholds on val set; return the one minimising |biomass diff|."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.5
    best_abs_diff  = float("inf")
    rows = []
    for t in thresholds:
        probs    = 1.0 / (1.0 + np.exp(-logits_pres))
        pred_bin = (probs >= t).astype(int)
        gt_bin   = y_pres.astype(int)
        recall   = float((pred_bin[gt_bin == 1] == 1).mean()) if gt_bin.sum() > 0 else 0.0
        prec     = float((gt_bin[pred_bin == 1] == 1).mean()) if pred_bin.sum() > 0 else 0.0
        _, _, diff_pct = _biomass_at_threshold(
            logits_pres, pred_weight, y_weight, y_pres, t
        )
        rows.append((t, recall, prec, diff_pct))
        if abs(diff_pct) < best_abs_diff:
            best_abs_diff  = abs(diff_pct)
            best_threshold = t

    if logger:
        logger.info(f"Threshold sweep — best: {best_threshold:.2f}  "
                    f"|biomass diff|={best_abs_diff:.2f}%")
    return best_threshold, rows


# ---------------------------------------------------------------------------
# Result figure
# ---------------------------------------------------------------------------

def _save_figure(logits_pres, pred_weight, y_pres, y_weight,
                 threshold, sweep_rows, out_dir: Path) -> Path:
    probs    = 1.0 / (1.0 + np.exp(-logits_pres))
    pred_bin = (probs >= threshold).astype(int)
    gt_bin   = y_pres.astype(int)
    cm       = confusion_matrix(gt_bin, pred_bin)

    gt_mask   = y_pres.astype(bool)
    pred_mask = probs >= threshold
    p_wt = pred_weight[gt_mask]     # weight predictions on true fish (for regression panel)
    t_wt = y_weight[gt_mask]
    errors = p_wt - t_wt

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Two-Stage Evaluation (Presence: length model | Weight: weight model)",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # [0,0] Pred vs True weight scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(t_wt, p_wt, alpha=0.45, s=18, color="#2196F3", edgecolors="none")
    lims = [min(t_wt.min(), p_wt.min()) - 10, max(t_wt.max(), p_wt.max()) + 10]
    ax.plot(lims, lims, "k--", linewidth=1.0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title("Predicted vs True Weight (true fish)", fontsize=10, fontweight="bold")
    ax.set_xlabel("True Weight (g)", fontsize=8); ax.set_ylabel("Predicted Weight (g)", fontsize=8)
    mape = float((np.abs(errors) / t_wt).mean() * 100)
    mae  = float(np.abs(errors).mean())
    ax.text(0.05, 0.93, f"MAE={mae:.1f}g  MAPE={mape:.2f}%",
            transform=ax.transAxes, fontsize=7)
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)

    # [0,1] Threshold sweep
    ax = fig.add_subplot(gs[0, 1])
    ts     = [r[0] for r in sweep_rows]
    diffs  = [abs(r[3]) for r in sweep_rows]
    recalls= [r[1] for r in sweep_rows]
    ax.plot(ts, diffs,   color="#FF5722", linewidth=1.5, label="|Biomass diff %|")
    ax2 = ax.twinx()
    ax2.plot(ts, recalls, color="#2196F3", linewidth=1.5, linestyle="--", label="Recall")
    ax.axvline(threshold, color="black", linewidth=1.2, linestyle="--",
               label=f"Chosen={threshold:.2f}")
    ax.set_title("Threshold Sweep (val set)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Threshold", fontsize=8)
    ax.set_ylabel("|Biomass diff %|", fontsize=8, color="#FF5722")
    ax2.set_ylabel("Recall", fontsize=8, color="#2196F3")
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # [0,2] Cumulative % error (true fish)
    ax = fig.add_subplot(gs[0, 2])
    ape = np.sort(np.abs(errors) / t_wt * 100)
    cum = np.arange(1, len(ape) + 1) / len(ape) * 100
    ax.plot(ape, cum, color="#9C27B0", linewidth=1.5)
    ax.axvline(5.0, color="#4CAF50", linewidth=1.2, linestyle="--", label="5% target")
    pct5 = float((ape <= 5.0).mean() * 100)
    ax.axhline(pct5, color="#FF9800", linewidth=1.0, linestyle=":",
               label=f"{pct5:.1f}% within 5%")
    ax.set_title("Cumulative % Error (true fish)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Absolute % Error", fontsize=8); ax.set_ylabel("Cumulative %", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)

    # [1,0] Confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Absent", "Present"], fontsize=8)
    ax.set_yticklabels(["Absent", "Present"], fontsize=8)
    ax.set_title(f"Presence Confusion Matrix (t={threshold:.2f})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [1,1] Presence probability by class
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(probs[gt_bin == 0], bins=30, alpha=0.65, color="#2196F3", label="True Absent")
    ax.hist(probs[gt_bin == 1], bins=30, alpha=0.65, color="#FF5722", label="True Present")
    ax.axvline(threshold, color="black", linewidth=1.2, linestyle="--",
               label=f"Threshold={threshold:.2f}")
    ax.set_title("Presence Probability by True Class", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Probability", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)

    # [1,2] Summary metrics
    ax = fig.add_subplot(gs[1, 2])
    recall_v = float((pred_bin[gt_bin == 1] == 1).mean()) if gt_bin.sum() > 0 else 0.0
    prec_v   = float((gt_bin[pred_bin == 1] == 1).mean()) if pred_bin.sum() > 0 else 0.0
    f1_v     = 2 * prec_v * recall_v / (prec_v + recall_v) if (prec_v + recall_v) > 0 else 0.0
    _, _, diff_pct = _biomass_at_threshold(logits_pres, pred_weight, y_weight, y_pres, threshold)
    metric_names  = ["Precision", "Recall", "F1", "MAPE (%)", "MAE (g)", "|Biomass diff %|"]
    metric_values = [prec_v, recall_v, f1_v, mape, mae, abs(diff_pct)]
    colors = ["#2196F3", "#2196F3", "#2196F3", "#FF5722", "#FF9800", "#9C27B0"]
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8, edgecolor="white")
    for bar, val, name in zip(bars, metric_values, metric_names):
        lbl = f"{diff_pct:+.2f}%" if name == "|Biomass diff %|" else f"{val:.2f}"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=7)
    ax.set_title("Summary Metrics", fontsize=10, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8); ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="x")

    out_path = out_dir / "evaluation_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ensemble_logits(ckpt_paths: list[str], npz_path: str,
                     device: torch.device, batch_size: int):
    """Average logits across multiple presence model checkpoints."""
    all_logits, y_pres_out, y_wt_out = None, None, None
    for path in ckpt_paths:
        ckpt  = torch.load(path, map_location=device)
        model = FishModel(ckpt["cfg_model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        logits, _, y_pres, y_wt = _run_inference(model, npz_path, device, batch_size)
        all_logits = logits if all_logits is None else all_logits + logits
        y_pres_out = y_pres
        y_wt_out   = y_wt
    return all_logits / len(ckpt_paths), y_pres_out, y_wt_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presence_ckpt", required=True, nargs="+",
                        help="One or more presence model checkpoints (averaged for ensemble)")
    parser.add_argument("--weight_ckpt",   required=True)
    parser.add_argument("--val_file",      required=True)
    parser.add_argument("--test_file",     required=True)
    parser.add_argument("--results_dir",   default="results/two_stage")
    parser.add_argument("--run_tag",       default="")
    parser.add_argument("--batch_size",    type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("evaluate_two_stage")

    logger.info(f"Presence models ({len(args.presence_ckpt)}): {args.presence_ckpt}")

    ckpt_wt = torch.load(args.weight_ckpt, map_location=device)
    model_wt = FishModel(ckpt_wt["cfg_model"]).to(device)
    model_wt.load_state_dict(ckpt_wt["model_state"])
    model_wt.eval()
    log_scale_wt = ckpt_wt.get("log_scale_weight", False)
    logger.info(f"Weight model   : {args.weight_ckpt}  (log_scale={log_scale_wt})")

    # Val inference (ensemble average logits)
    logger.info(f"Val file  : {args.val_file}")
    val_logits, val_y_pres, val_y_wt = _ensemble_logits(
        args.presence_ckpt, args.val_file, device, args.batch_size
    )
    _, val_pw2, _, _ = _run_inference(
        model_wt, args.val_file, device, args.batch_size, log_scale=log_scale_wt
    )
    logger.info(f"Val samples: {len(val_y_pres)}  positives: {int(val_y_pres.sum())}")

    # Threshold tuning on val
    best_t, sweep = tune_threshold(val_logits, val_pw2, val_y_wt, val_y_pres, logger=logger)

    # Test inference (ensemble)
    logger.info(f"Test file : {args.test_file}")
    test_logits, test_y_pres, test_y_wt = _ensemble_logits(
        args.presence_ckpt, args.test_file, device, args.batch_size
    )
    _, test_pw2, _, _ = _run_inference(
        model_wt, args.test_file, device, args.batch_size, log_scale=log_scale_wt
    )
    logger.info(f"Test samples: {len(test_y_pres)}  positives: {int(test_y_pres.sum())}")

    # Compute test metrics
    probs    = 1.0 / (1.0 + np.exp(-test_logits))
    pred_bin = (probs >= best_t).astype(int)
    gt_bin   = test_y_pres.astype(int)
    gt_mask  = test_y_pres.astype(bool)

    recall = float((pred_bin[gt_bin == 1] == 1).mean()) if gt_bin.sum() > 0 else 0.0
    prec   = float((gt_bin[pred_bin == 1] == 1).mean()) if pred_bin.sum() > 0 else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    acc    = float((pred_bin == gt_bin).mean())

    errors  = test_pw2[gt_mask] - test_y_wt[gt_mask]
    mae     = float(np.abs(errors).mean())
    rmse    = float(np.sqrt((errors ** 2).mean()))
    mape    = float((np.abs(errors) / test_y_wt[gt_mask]).mean() * 100)
    sum_true, sum_pred, diff_pct = _biomass_at_threshold(
        test_logits, test_pw2, test_y_wt, test_y_pres, best_t
    )

    # Soft biomass: weight × presence_prob for ALL windows (no threshold)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    soft_pred  = float((test_pw2 * test_probs).sum())
    soft_diff  = (soft_pred - sum_true) / sum_true * 100.0 if sum_true > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Test Results (Two-Stage)")
    logger.info("=" * 60)
    logger.info(f"  Threshold : {best_t:.2f} (tuned on val)")
    logger.info(f"  Presence  | Acc={acc:.4f}  Prec={prec:.4f}  Rec={recall:.4f}  F1={f1:.4f}")
    logger.info(f"  Weight    | MAE={mae:.2f}g  RMSE={rmse:.2f}g  MAPE={mape:.2f}%")
    logger.info(f"  Biomass   | True={sum_true:.0f}g  Pred={sum_pred:.0f}g  Diff={diff_pct:+.2f}%")
    logger.info(f"  Soft Bio  | Pred={soft_pred:.0f}g  Diff={soft_diff:+.2f}%  "
                f"(weight×prob, no threshold)")
    logger.info("=" * 60)

    # Save results
    tag      = args.run_tag
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    out_dir  = Path(args.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "presence_ckpt": args.presence_ckpt,
            "weight_ckpt":   args.weight_ckpt,
            "threshold":     float(best_t),
            "metrics": {
                "accuracy": round(acc, 6), "precision": round(prec, 6),
                "recall": round(recall, 6), "f1": round(f1, 6),
                "mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4),
            },
            "biomass_hard": {
                "sum_true_g": round(sum_true, 2),
                "sum_pred_g": round(sum_pred, 2),
                "diff_pct":   round(diff_pct, 4),
            },
            "biomass_soft": {
                "sum_pred_g": round(soft_pred, 2),
                "diff_pct":   round(soft_diff, 4),
            },
        }, f, indent=2)
    logger.info(f"Metrics saved : {json_path}")

    fig_path = _save_figure(test_logits, test_pw2, test_y_pres, test_y_wt,
                            best_t, sweep, out_dir)
    logger.info(f"Figure saved  : {fig_path}")


if __name__ == "__main__":
    main()
