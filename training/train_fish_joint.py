"""Training script for FishModel with joint presence + weight regression.

Mirrors train_salt.py but uses FishModel (13-channel, filename-based salt)
instead of FishSaltModel.  No physics branch, no aux loss.

Run from the project root:
    python training/train_fish_joint.py \\
        --config       training/train_config_fish_mixsplit.yaml \\
        --model_config model/model_config_fish_joint.yaml
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

from model.fish_model import FishModel
from utils.logger import setup_logger
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p      = torch.sigmoid(logits)
        p_t    = p * targets + (1.0 - p) * (1.0 - targets)
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return ((1.0 - p_t) ** self.gamma * bce).mean()


class FishWeightDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.X        = torch.from_numpy(d["X"].astype(np.float32))
        self.presence = torch.from_numpy(d["y_presence"].astype(np.float32))
        self.weight   = torch.from_numpy(d["y_weight"].astype(np.float32))

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return {"X": self.X[i], "presence": self.presence[i], "weight": self.weight[i]}


def make_loader(path, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        FishWeightDataset(path), batch_size=batch_size,
        shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
def _run_epoch(model, loader, device, bce_fn, huber_fn, cfg, optimizer=None):
    training  = optimizer is not None
    model.train(training)

    w_pres    = cfg["loss_weight_presence"]
    w_weight  = cfg["loss_weight_weight"]
    log_scale = cfg.get("log_scale_weight", False)
    noise_std = cfg.get("weight_noise_std", 0.0)

    total_loss = 0.0
    all_logits, all_pw, all_yp, all_yw = [], [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            X        = batch["X"].to(device)
            y_pres   = batch["presence"].to(device)
            y_weight = batch["weight"].to(device)

            if log_scale:
                y_weight = torch.log(y_weight.clamp(min=1.0))
            if training and noise_std > 0.0:
                mask = y_pres.bool()
                noise = torch.randn_like(y_weight) * noise_std
                y_weight = y_weight.clone()
                y_weight[mask] = y_weight[mask] + noise[mask]

            logit_pres, pred_weight = model(X)

            pres_loss   = bce_fn(logit_pres, y_pres)
            mask        = y_pres.bool()
            weight_loss = (
                huber_fn(pred_weight[mask], y_weight[mask])
                if mask.sum() > 0
                else torch.tensor(0.0, device=device)
            )
            loss = w_pres * pres_loss + w_weight * weight_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            pw_np = pred_weight.detach().cpu().numpy()
            yw_np = y_weight.cpu().numpy()
            if log_scale:
                pw_np = np.exp(pw_np)
                yw_np = np.exp(yw_np)

            all_logits.append(logit_pres.detach().cpu().numpy())
            all_pw.append(pw_np)
            all_yp.append(y_pres.cpu().numpy())
            all_yw.append(yw_np)

    metrics = compute_all_metrics(
        np.concatenate(all_logits), np.concatenate(all_pw),
        np.concatenate(all_yp),    np.concatenate(all_yw),
    )
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
def _save_figures(history, run_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Training Summary — FishModel (joint)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.35)
    panels = [
        (gs[0, 0], "Total Loss",    "train_loss", "val_loss"),
        (gs[0, 1], "Presence F1",   "train_f1",   "val_f1"),
        (gs[0, 2], "Weight MAPE %", "train_mape", "val_mape"),
    ]
    for spec, title, tk, vk in panels:
        ax = fig.add_subplot(spec)
        ax.plot(epochs, history[tk], label="Train", linewidth=1.5, color="#2196F3")
        ax.plot(epochs, history[vk], label="Val",   linewidth=1.5, color="#FF5722")
        lower = any(k in vk for k in ["loss","mae","rmse","mape"])
        best  = int(np.argmin(history[vk])) if lower else int(np.argmax(history[vk]))
        ax.axvline(best+1, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    out = run_dir / "training_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def train(cfg_model: dict, cfg_train: dict) -> str:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg_train["checkpoint_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_fish_joint", log_file=run_dir / "train.log")
    logger.info(f"Run ID : {run_id} | Device : {device}")

    train_loader = make_loader(
        cfg_train.get("train_file") or
        os.path.join(cfg_train["data_dir"], "train_fish.npz"),
        cfg_train["batch_size"], shuffle=True,
    )
    val_file   = cfg_train.get("val_file") or str(Path(cfg_train["data_dir"]) / "val_fish.npz")
    val_loader = make_loader(val_file, cfg_train["batch_size"], shuffle=False) if Path(val_file).exists() else None
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset) if val_loader else 'none'}")

    model    = FishModel(cfg_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"FishModel parameters: {n_params:,}")

    use_focal = cfg_train.get("use_focal_loss", False)
    if use_focal:
        gamma  = cfg_train.get("focal_gamma", 2.0)
        bce_fn = FocalLoss(gamma=gamma)
        logger.info(f"Loss: FocalLoss(gamma={gamma})")
    else:
        pos_weight = torch.tensor([cfg_train.get("pos_weight", 1.0)], device=device)
        bce_fn     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Loss: BCEWithLogitsLoss(pos_weight={cfg_train.get('pos_weight', 1.0)})")
    huber_fn = nn.HuberLoss(delta=cfg_train["huber_delta"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"]
    )
    sched_mode = "max" if use_focal else "min"
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode, patience=cfg_train["lr_patience"], factor=0.5, min_lr=1e-6
    )

    use_val    = val_loader is not None
    best_metric         = -float("inf") if use_focal else float("inf")
    no_improve          = 0
    best_ckpt           = run_dir / "best_model.pt"
    early_stop_patience = cfg_train.get("early_stopping_patience", 999999)

    history = {k: [] for k in [
        "train_loss","val_loss","train_mae","val_mae","train_rmse","val_rmse",
        "train_mape","val_mape","train_f1","val_f1","train_accuracy","val_accuracy",
    ]}

    for epoch in range(cfg_train["max_epochs"]):
        train_m = _run_epoch(model, train_loader, device, bce_fn, huber_fn, cfg_train, optimizer)
        val_m   = _run_epoch(model, val_loader,   device, bce_fn, huber_fn, cfg_train) if use_val else train_m
        scheduler.step(val_m["f1"] if use_focal else val_m["loss"])

        for key in ["loss","mae","rmse","mape","f1","accuracy"]:
            history[f"train_{key}"].append(train_m[key])
            history[f"val_{key}"].append(val_m[key])

        monitor_val = val_m["f1"] if use_focal else val_m["loss"]
        is_best     = (monitor_val > best_metric) if use_focal else (monitor_val < best_metric)
        if is_best:
            best_metric = monitor_val; no_improve = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_m["loss"], "val_f1": val_m["f1"],
                "cfg_model": cfg_model,
                "target": "weight_fish",
                "log_scale_weight": cfg_train.get("log_scale_weight", False),
            }, best_ckpt)
        else:
            no_improve += 1

        crit_tag = f"val_f1={val_m['f1']:.3f}" if use_focal else f"val_loss={val_m['loss']:.4f}"
        if use_val:
            logger.info(
                f"Epoch {epoch+1:3d}/{cfg_train['max_epochs']} | "
                f"train_loss={train_m['loss']:.4f} | val_loss={val_m['loss']:.4f} | "
                f"val_mae={val_m['mae']:.1f}g | val_mape={val_m['mape']:.1f}% | {crit_tag}"
                + ("  [best]" if is_best else "")
            )

        if no_improve >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    fig_path = _save_figures(history, run_dir)
    crit_name = "val F1" if use_focal else "val loss"
    logger.info(f"Best {crit_name} : {best_metric:.4f}")
    logger.info(f"Checkpoint : {best_ckpt}")
    return str(best_ckpt)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="training/train_config_fish_mixsplit.yaml")
    parser.add_argument("--model_config", default="model/model_config_fish_joint.yaml")
    args = parser.parse_args()
    cfg_model = yaml.safe_load(open(args.model_config))
    cfg_train = yaml.safe_load(open(args.config))
    train(cfg_model, cfg_train)
