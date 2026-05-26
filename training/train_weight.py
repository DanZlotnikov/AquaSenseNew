"""Training script for the fish presence and weight prediction model.

Called via:
    python main.py --mode train_weight
or directly (for debugging):
    python training/train_weight.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fish_dataset_weight import make_weight_dataloader
from model.fish_model import FishModel
from utils.logger import setup_logger
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _compute_loss(
    logit_pres: torch.Tensor,
    pred_weight: torch.Tensor,
    y_pres: torch.Tensor,
    y_weight: torch.Tensor,
    bce_fn: nn.Module,
    huber_fn: nn.Module,
    w_pres: float,
    w_weight: float,
) -> tuple:
    """
    Weighted sum of presence BCE loss and masked weight Huber loss.

    The Huber loss is only computed for samples where y_pres == 1,
    preventing the model from being penalised for weight predictions
    on fish-absent samples.

    Returns:
        (total_loss, presence_loss_scalar, weight_loss_scalar)
    """
    presence_loss = bce_fn(logit_pres, y_pres)

    mask = y_pres.bool()
    if mask.sum() > 0:
        weight_loss = huber_fn(pred_weight[mask], y_weight[mask])
    else:
        weight_loss = torch.tensor(0.0, device=pred_weight.device)

    total = w_pres * presence_loss + w_weight * weight_loss
    return total, presence_loss.item(), weight_loss.item()


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(
    model: FishModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    bce_fn: nn.Module,
    huber_fn: nn.Module,
    cfg: dict,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    """
    Run one full pass over the dataloader.

    Pass optimizer=None for validation / evaluation (no gradient updates).

    Returns:
        Metrics dict including loss, mae, rmse, mape, accuracy, precision, recall, f1.
        Note: mae/rmse/mape here are computed on weight (g), not length (cm).
    """
    training = optimizer is not None
    model.train(training)

    w_pres   = cfg["loss_weight_presence"]
    w_weight = cfg["loss_weight_weight"]

    total_loss_sum = 0.0
    all_logits, all_pred_weight, all_y_pres, all_y_weight = [], [], [], []

    noise_std = cfg.get("weight_noise_std", 0.0)

    with torch.set_grad_enabled(training):
        for batch in loader:
            X        = batch["X"].to(device)
            y_pres   = batch["presence"].to(device)
            y_weight = batch["weight"].to(device)

            log_scale = cfg.get("log_scale_weight", False)

            # Convert targets to log space if requested
            if log_scale:
                y_weight = torch.log(y_weight.clamp(min=1.0))

            # Label noise augmentation
            if training and noise_std > 0.0:
                mask = y_pres.bool()
                noise = torch.randn_like(y_weight) * noise_std
                y_weight = y_weight.clone()
                y_weight[mask] = y_weight[mask] + noise[mask]

            logit_pres, pred_weight = model(X)
            loss, _, _ = _compute_loss(
                logit_pres, pred_weight, y_pres, y_weight,
                bce_fn, huber_fn, w_pres, w_weight,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += loss.item()

            # Collect for metric computation (always in original gram space)
            pw_np = pred_weight.detach().cpu().numpy()
            yw_np = y_weight.cpu().numpy()
            if log_scale:
                pw_np = np.exp(pw_np)
                yw_np = np.exp(yw_np)
            all_logits.append(logit_pres.detach().cpu().numpy())
            all_pred_weight.append(pw_np)
            all_y_pres.append(y_pres.cpu().numpy())
            all_y_weight.append(yw_np)

    metrics = compute_all_metrics(
        np.concatenate(all_logits),
        np.concatenate(all_pred_weight),
        np.concatenate(all_y_pres),
        np.concatenate(all_y_weight),
    )
    metrics["loss"] = total_loss_sum / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Training summary figure
# ---------------------------------------------------------------------------

def _save_training_figures(history: dict, run_dir: Path) -> None:
    """
    Save a 2x3 summary figure of training and validation metrics over epochs.

    Panels:
        [0,0] Total loss (train vs val)
        [0,1] Presence F1 (train vs val)
        [0,2] Presence accuracy (train vs val)
        [1,0] Weight MAE in g (train vs val)
        [1,1] Weight RMSE in g (train vs val)
        [1,2] Weight MAPE in % (train vs val) — with 5% target line
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Training Summary (Weight Model)", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    panels = [
        (gs[0, 0], "Total Loss",          "train_loss",     "val_loss",     None,  "Loss"),
        (gs[0, 1], "Presence F1",         "train_f1",       "val_f1",       None,  "F1"),
        (gs[0, 2], "Presence Accuracy",   "train_accuracy", "val_accuracy", None,  "Accuracy"),
        (gs[1, 0], "Weight MAE (g)",      "train_mae",      "val_mae",      None,  "MAE (g)"),
        (gs[1, 1], "Weight RMSE (g)",     "train_rmse",     "val_rmse",     None,  "RMSE (g)"),
        (gs[1, 2], "Weight MAPE (%)",     "train_mape",     "val_mape",     5.0,   "MAPE (%)"),
    ]

    for spec, title, train_key, val_key, target, ylabel in panels:
        ax = fig.add_subplot(spec)
        ax.plot(epochs, history[train_key], label="Train", linewidth=1.5, color="#2196F3")
        ax.plot(epochs, history[val_key],   label="Val",   linewidth=1.5, color="#FF5722")
        if target is not None:
            ax.axhline(target, color="#4CAF50", linestyle="--", linewidth=1.2,
                       label=f"Target ({target}%)")
        # Mark best val epoch
        best_epoch = int(np.argmin(history[val_key])) if "loss" in val_key or "mae" in val_key \
                     or "rmse" in val_key or "mape" in val_key \
                     else int(np.argmax(history[val_key]))
        ax.axvline(best_epoch + 1, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    out_path = run_dir / "training_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg_model: dict, cfg_train: dict) -> str:
    """
    Train the FishModel for weight prediction and save per-run checkpoints.

    Each call creates a timestamped subdirectory under cfg_train["checkpoint_dir"]
    containing:
        best_model.pt        — checkpoint saved whenever validation loss improves
        train.log            — full training log
        training_summary.png — loss and metric curves over all epochs

    The checkpoint stores cfg_model alongside the model weights so that
    evaluate_weight.py can reconstruct the exact architecture without needing
    the YAML config file at inference time.

    Args:
        cfg_model : Model architecture config (from model/model_config.yaml).
        cfg_train : Training config (from training/train_config_weight.yaml).

    Returns:
        Absolute path to the best checkpoint file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Run directory ---
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg_train["checkpoint_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_weight", log_file=run_dir / "train.log")
    logger.info(f"Run ID  : {run_id}")
    logger.info(f"Device  : {device}")
    logger.info("Target  : weight (g)")

    # --- Data ---
    data_dir   = cfg_train["data_dir"]
    batch_size = cfg_train["batch_size"]

    train_loader = make_weight_dataloader(
        cfg_train.get("train_file") or os.path.join(data_dir, "train_dataset.npz"),
        batch_size, shuffle=True
    )
    val_file   = cfg_train.get("val_file") or str(Path(data_dir) / "val_dataset.npz")
    val_loader = (
        make_weight_dataloader(val_file, batch_size, shuffle=False)
        if Path(val_file).exists() else None
    )
    logger.info(
        f"Train samples: {len(train_loader.dataset)} | "
        + (f"Val samples: {len(val_loader.dataset)}" if val_loader else "No val set — training for fixed epochs")
    )

    # --- Model ---
    model = FishModel(cfg_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # --- Loss functions ---
    pos_weight = torch.tensor([cfg_train["pos_weight"]], device=device)
    bce_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber_fn = nn.HuberLoss(delta=cfg_train["huber_delta"])

    # --- Optimiser and scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_train["lr"],
        weight_decay=cfg_train["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg_train["lr_patience"],
        factor=0.5,
        min_lr=1e-6,
    )

    # --- Training loop ---
    use_val        = val_loader is not None
    best_loss      = float("inf")
    no_improve     = 0
    best_ckpt_path = run_dir / "best_model.pt"

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_mae",  "val_mae",
        "train_rmse", "val_rmse",
        "train_mape", "val_mape",
        "train_f1",   "val_f1",
        "train_accuracy", "val_accuracy",
    ]}

    early_stop_patience = cfg_train.get("early_stopping_patience", 999999)
    logger.info(f"Starting training for up to {cfg_train['max_epochs']} epochs")
    logger.info("-" * 80)

    for epoch in range(cfg_train["max_epochs"]):
        train_m = _run_epoch(
            model, train_loader, device, bce_fn, huber_fn, cfg_train, optimizer
        )
        val_m = (
            _run_epoch(model, val_loader, device, bce_fn, huber_fn, cfg_train)
            if use_val else train_m
        )
        scheduler.step(val_m["loss"])

        for key in ["loss", "mae", "rmse", "mape", "f1", "accuracy"]:
            history[f"train_{key}"].append(train_m[key])
            history[f"val_{key}"].append(val_m[key])

        monitor_loss = val_m["loss"]
        is_best = monitor_loss < best_loss
        if is_best:
            best_loss  = monitor_loss
            no_improve = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state":      model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "val_loss":         best_loss,
                    "cfg_model":        cfg_model,
                    "target":           "weight",
                    "log_scale_weight": cfg_train.get("log_scale_weight", False),
                },
                best_ckpt_path,
            )
        else:
            no_improve += 1

        if use_val:
            logger.info(
                f"Epoch {epoch + 1:3d}/{cfg_train['max_epochs']} | "
                f"train_loss={train_m['loss']:.4f} | val_loss={val_m['loss']:.4f} | "
                f"val_mae={val_m['mae']:.2f}g | val_mape={val_m['mape']:.2f}% | "
                f"val_f1={val_m['f1']:.3f}"
                + ("  [best]" if is_best else "")
            )
        else:
            logger.info(
                f"Epoch {epoch + 1:3d}/{cfg_train['max_epochs']} | "
                f"train_loss={train_m['loss']:.4f} | "
                f"train_mae={train_m['mae']:.2f}g | train_mape={train_m['mape']:.2f}% | "
                f"train_f1={train_m['f1']:.3f}"
                + ("  [best]" if is_best else "")
            )

        if no_improve >= early_stop_patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch + 1} "
                f"(no improvement for {no_improve} epochs)"
            )
            break

    # --- Save training summary figure ---
    fig_path = _save_training_figures(history, run_dir)
    logger.info("-" * 80)
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Training figure  : {fig_path}")
    logger.info(f"Best checkpoint  : {best_ckpt_path}")

    return str(best_ckpt_path)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/train_config_weight.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--model_config", default="model/model_config.yaml",
                        help="Path to model config YAML")
    args = parser.parse_args()

    cfg_model = yaml.safe_load(open(args.model_config))
    cfg_train = yaml.safe_load(open(args.config))
    train(cfg_model, cfg_train)
