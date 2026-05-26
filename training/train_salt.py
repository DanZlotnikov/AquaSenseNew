"""Training script for FishSaltModel.

Trains jointly on:
    1. Fish presence  (BCEWithLogitsLoss)
    2. Fish weight    (HuberLoss in log-space)

The model's internal water-condition head (see fish_salt_model.py) is
self-supervised using the physics feature already present in X_phys — no
external labels of any kind are required beyond y_presence and y_weight.

Run from the project root:
    python training/train_salt.py
    python training/train_salt.py --config training/train_config_salt.yaml \
                                  --model_config model/model_config_salt.yaml
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.fish_dataset_salt import make_salt_dataloader
from model.fish_salt_model import FishSaltModel
from utils.logger import setup_logger
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced presence detection.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard value from Lin et al. (RetinaNet).  No alpha is
    used — the focal weight alone handles moderate class imbalance; avoid
    adding another dataset-specific hyperparameter.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p      = torch.sigmoid(logits)
        p_t    = p * targets + (1.0 - p) * (1.0 - targets)
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weight = (1.0 - p_t) ** self.gamma
        return (weight * bce).mean()


def _compute_loss(
    logit_pres:    torch.Tensor,
    pred_weight:   torch.Tensor,
    pred_aux:      torch.Tensor,   # internal water-condition estimate
    x_phys:        torch.Tensor,   # physics features — col 2 is the self-supervision target
    y_pres:        torch.Tensor,
    y_weight:      torch.Tensor,
    bce_fn:        nn.Module,      # BCEWithLogitsLoss or FocalLoss
    huber_fn:      nn.Module,
    aux_bce_fn:    nn.Module,
    w_pres:        float,
    w_weight:      float,
    w_aux:         float,
) -> tuple:
    """
    Multi-task loss.

    Presence:  BCEWithLogitsLoss on all samples.
    Weight:    HuberLoss masked to fish-present samples only.
    Auxiliary: BCELoss for the internal water-condition head, self-supervised
               using the physics-formula estimate stored in x_phys[:,2].
               This requires no external labels — x_phys[:,2] is computed
               directly from the raw electrode baseline in build_salt_dataset.py.

    Returns:
        (total, presence_scalar, weight_scalar, aux_scalar)
    """
    presence_loss = bce_fn(logit_pres, y_pres)

    mask = y_pres.bool()
    weight_loss = (
        huber_fn(pred_weight[mask], y_weight[mask])
        if mask.sum() > 0
        else torch.tensor(0.0, device=pred_weight.device)
    )

    # Self-supervised: the physics formula column is the target (no external label)
    aux_target = x_phys[:, 2].detach()
    aux_loss   = aux_bce_fn(pred_aux, aux_target)

    total = w_pres * presence_loss + w_weight * weight_loss + w_aux * aux_loss
    return total, presence_loss.item(), weight_loss.item(), aux_loss.item()


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(
    model:      FishSaltModel,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    bce_fn:     nn.Module,
    huber_fn:   nn.Module,
    aux_bce_fn: nn.Module,
    cfg:        dict,
    optimizer:  torch.optim.Optimizer = None,
) -> dict:
    """One full pass.  optimizer=None → evaluation mode."""
    training  = optimizer is not None
    model.train(training)

    w_pres    = cfg["loss_weight_presence"]
    w_weight  = cfg["loss_weight_weight"]
    w_aux     = cfg.get("loss_weight_aux", 0.1)
    log_scale = cfg.get("log_scale_weight", False)
    noise_std = cfg.get("weight_noise_std", 0.0)

    total_loss_sum = 0.0
    all_logits, all_pred_w, all_y_pres, all_y_weight = [], [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            X        = batch["X"].to(device)
            X_phys   = batch["X_phys"].to(device)
            y_pres   = batch["presence"].to(device)
            y_weight = batch["weight"].to(device)

            if log_scale:
                y_weight = torch.log(y_weight.clamp(min=1.0))

            if training and noise_std > 0.0:
                mask = y_pres.bool()
                noise = torch.randn_like(y_weight) * noise_std
                y_weight = y_weight.clone()
                y_weight[mask] = y_weight[mask] + noise[mask]

            logit_pres, pred_weight, pred_aux = model(X, X_phys)

            loss, _, _, _ = _compute_loss(
                logit_pres, pred_weight, pred_aux, X_phys,
                y_pres, y_weight,
                bce_fn, huber_fn, aux_bce_fn,
                w_pres, w_weight, w_aux,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += loss.item()

            pw_np = pred_weight.detach().cpu().numpy()
            yw_np = y_weight.cpu().numpy()
            if log_scale:
                pw_np = np.exp(pw_np)
                yw_np = np.exp(yw_np)

            all_logits.append(logit_pres.detach().cpu().numpy())
            all_pred_w.append(pw_np)
            all_y_pres.append(y_pres.cpu().numpy())
            all_y_weight.append(yw_np)

    metrics = compute_all_metrics(
        np.concatenate(all_logits),
        np.concatenate(all_pred_w),
        np.concatenate(all_y_pres),
        np.concatenate(all_y_weight),
    )
    metrics["loss"] = total_loss_sum / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Training summary figure
# ---------------------------------------------------------------------------

def _save_figures(history: dict, run_dir: Path) -> Path:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Training Summary — FishSaltModel", fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    panels = [
        (gs[0, 0], "Total Loss",        "train_loss",     "val_loss",     None),
        (gs[0, 1], "Presence F1",       "train_f1",       "val_f1",       None),
        (gs[0, 2], "Presence Accuracy", "train_accuracy", "val_accuracy", None),
        (gs[1, 0], "Weight MAE (g)",    "train_mae",      "val_mae",      None),
        (gs[1, 1], "Weight RMSE (g)",   "train_rmse",     "val_rmse",     None),
        (gs[1, 2], "Weight MAPE (%)",   "train_mape",     "val_mape",     5.0),
    ]

    for spec, title, tk_, vk, target in panels:
        ax = fig.add_subplot(spec)
        ax.plot(epochs, history[tk_], label="Train", linewidth=1.5, color="#2196F3")
        ax.plot(epochs, history[vk],  label="Val",   linewidth=1.5, color="#FF5722")
        if target is not None:
            ax.axhline(target, color="#4CAF50", linestyle="--", linewidth=1.2,
                       label=f"Target ({target}%)")
        lower = any(k in vk for k in ["loss", "mae", "rmse", "mape"])
        best  = int(np.argmin(history[vk])) if lower else int(np.argmax(history[vk]))
        ax.axvline(best + 1, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    out = run_dir / "training_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg_model: dict, cfg_train: dict) -> str:
    """Train FishSaltModel and return path to the best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg_train["checkpoint_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_salt", log_file=run_dir / "train.log")
    logger.info(f"Run ID : {run_id}")
    logger.info(f"Device : {device}")
    logger.info("Targets: presence + weight (log-scale)")

    # --- Data ---
    data_dir   = cfg_train["data_dir"]
    batch_size = cfg_train["batch_size"]

    train_loader = make_salt_dataloader(
        cfg_train.get("train_file") or os.path.join(data_dir, "train_salt.npz"),
        batch_size, shuffle=True,
    )
    val_file   = cfg_train.get("val_file") or str(Path(data_dir) / "val_salt.npz")
    val_loader = (
        make_salt_dataloader(val_file, batch_size, shuffle=False)
        if Path(val_file).exists() else None
    )
    logger.info(
        f"Train: {len(train_loader.dataset)} | "
        + (f"Val: {len(val_loader.dataset)}" if val_loader else "No val set")
    )

    # --- Model ---
    model    = FishSaltModel(cfg_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # --- Loss functions ---
    if cfg_train.get("use_focal_loss", False):
        gamma  = cfg_train.get("focal_gamma", 2.0)
        bce_fn = FocalLoss(gamma=gamma)
        logger.info(f"Loss: FocalLoss(gamma={gamma})")
    else:
        pos_weight = torch.tensor([cfg_train.get("pos_weight", 1.0)], device=device)
        bce_fn     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Loss: BCEWithLogitsLoss(pos_weight={cfg_train.get('pos_weight', 1.0)})")
    huber_fn   = nn.HuberLoss(delta=cfg_train["huber_delta"])
    aux_bce_fn = nn.BCELoss()   # for the self-supervised internal head

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"]
    )
    sched_mode = "max" if cfg_train.get("use_focal_loss", False) else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode, patience=cfg_train["lr_patience"], factor=0.5, min_lr=1e-6
    )

    # --- Training loop ---
    # Criterion: best val F1 when use_focal_loss=True (focal loss magnitude
    # drops sharply in early epochs before the model is useful, so val_loss
    # picks an under-trained checkpoint; F1 tracks actual detection quality).
    use_val             = val_loader is not None
    use_focal           = cfg_train.get("use_focal_loss", False)
    best_metric         = -float("inf") if use_focal else float("inf")
    no_improve          = 0
    best_ckpt           = run_dir / "best_model.pt"
    early_stop_patience = cfg_train.get("early_stopping_patience", 999999)

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_mae",  "val_mae",
        "train_rmse", "val_rmse",
        "train_mape", "val_mape",
        "train_f1",   "val_f1",
        "train_accuracy", "val_accuracy",
    ]}

    logger.info(f"Training up to {cfg_train['max_epochs']} epochs")
    logger.info("-" * 80)

    for epoch in range(cfg_train["max_epochs"]):
        train_m = _run_epoch(
            model, train_loader, device, bce_fn, huber_fn, aux_bce_fn, cfg_train, optimizer
        )
        val_m = (
            _run_epoch(model, val_loader, device, bce_fn, huber_fn, aux_bce_fn, cfg_train)
            if use_val else train_m
        )
        scheduler.step(val_m["f1"] if use_focal else val_m["loss"])

        for key in ["loss", "mae", "rmse", "mape", "f1", "accuracy"]:
            history[f"train_{key}"].append(train_m[key])
            history[f"val_{key}"].append(val_m[key])

        monitor_val = val_m["f1"] if use_focal else val_m["loss"]
        is_best     = (monitor_val > best_metric) if use_focal else (monitor_val < best_metric)
        if is_best:
            best_metric = monitor_val
            no_improve  = 0
            torch.save({
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_loss":         val_m["loss"],
                "val_f1":           val_m["f1"],
                "cfg_model":        cfg_model,
                "target":           "weight_salt",
                "log_scale_weight": cfg_train.get("log_scale_weight", False),
            }, best_ckpt)
        else:
            no_improve += 1

        if use_val:
            crit_tag = f"val_f1={val_m['f1']:.3f}" if use_focal else f"val_loss={val_m['loss']:.4f}"
            logger.info(
                f"Epoch {epoch+1:3d}/{cfg_train['max_epochs']} | "
                f"train_loss={train_m['loss']:.4f} | val_loss={val_m['loss']:.4f} | "
                f"val_mae={val_m['mae']:.1f}g | val_mape={val_m['mape']:.1f}% | "
                f"{crit_tag}"
                + ("  [best]" if is_best else "")
            )

        if no_improve >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    fig_path = _save_figures(history, run_dir)
    logger.info("-" * 80)
    crit_name = "val F1" if use_focal else "val loss"
    logger.info(f"Best {crit_name} : {best_metric:.4f}")
    logger.info(f"Figure        : {fig_path}")
    logger.info(f"Checkpoint    : {best_ckpt}")
    return str(best_ckpt)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="training/train_config_salt.yaml")
    parser.add_argument("--model_config", default="model/model_config_salt.yaml")
    args = parser.parse_args()

    cfg_model = yaml.safe_load(open(args.model_config))
    cfg_train = yaml.safe_load(open(args.config))
    train(cfg_model, cfg_train)
