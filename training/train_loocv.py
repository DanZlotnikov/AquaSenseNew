"""LOOCV training for the 12-channel salt-independent FishModel.

Runs 7 leave-one-recording-out folds, then trains a final model on all data.
Each fold: train on 6 recordings (85% windows), val on 15% of those windows
for early stopping, test on the held-out recording (never seen during training).

Run from the project root:
    python training/train_loocv.py \\
        --config       training/train_config_loocv.yaml \\
        --model_config model/model_config_12ch.yaml
"""

import argparse
import json
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

    def forward(self, logits, targets):
        p   = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return ((1.0 - p_t) ** self.gamma * bce).mean()


class FishDataset(torch.utils.data.Dataset):
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
        FishDataset(path), batch_size=batch_size,
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
        np.concatenate(all_yp),     np.concatenate(all_yw),
    )
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
def _train_one(
    cfg_model: dict,
    cfg_train: dict,
    train_path: str,
    val_path: str,
    ckpt_path: Path,
    device: torch.device,
    logger,
    label: str = "",
) -> dict:
    """Train one model instance; return best val metrics."""
    train_loader = make_loader(train_path, cfg_train["batch_size"], shuffle=True)
    val_loader   = make_loader(val_path,   cfg_train["batch_size"], shuffle=False)

    model    = FishModel(cfg_model).to(device)
    use_focal = cfg_train.get("use_focal_loss", False)
    gamma     = cfg_train.get("focal_gamma", 2.0)
    bce_fn    = FocalLoss(gamma=gamma) if use_focal else nn.BCEWithLogitsLoss()
    huber_fn  = nn.HuberLoss(delta=cfg_train["huber_delta"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg_train["lr_patience"], factor=0.5, min_lr=1e-6
    )

    best_f1         = -float("inf")
    no_improve      = 0
    early_patience  = cfg_train.get("early_stopping_patience", 999999)

    for epoch in range(cfg_train["max_epochs"]):
        train_m = _run_epoch(model, train_loader, device, bce_fn, huber_fn, cfg_train, optimizer)
        val_m   = _run_epoch(model, val_loader,   device, bce_fn, huber_fn, cfg_train)
        scheduler.step(val_m["f1"])

        is_best = val_m["f1"] > best_f1
        if is_best:
            best_f1    = val_m["f1"]
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": val_m["f1"],
                "val_loss": val_m["loss"],
                "cfg_model": cfg_model,
                "log_scale_weight": cfg_train.get("log_scale_weight", False),
            }, ckpt_path)
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or is_best:
            tag = "  [best]" if is_best else ""
            logger.info(
                f"{label} epoch {epoch+1:3d} | "
                f"train_loss={train_m['loss']:.4f} | "
                f"val_f1={val_m['f1']:.3f} | val_mape={val_m['mape']:.1f}%{tag}"
            )

        if no_improve >= early_patience:
            logger.info(f"{label} early stop at epoch {epoch+1}")
            break

    return {"best_val_f1": best_f1}


# ---------------------------------------------------------------------------
def _eval_on_test(
    cfg_model: dict,
    cfg_train: dict,
    ckpt_path: Path,
    test_path: str,
    device: torch.device,
) -> dict:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FishModel(cfg_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bce_fn   = FocalLoss(cfg_train.get("focal_gamma", 2.0))
    huber_fn = nn.HuberLoss(delta=cfg_train["huber_delta"])
    loader   = make_loader(test_path, cfg_train["batch_size"], shuffle=False)
    return _run_epoch(model, loader, device, bce_fn, huber_fn, cfg_train)


# ---------------------------------------------------------------------------
def _save_summary(fold_results: list, run_dir: Path):
    lines = ["fold | test_recording                            | f1    | mape%  | mae_g | acc"]
    for r in fold_results:
        lines.append(
            f"  {r['fold']:d}  | {r['test_recording']:<42s} | "
            f"{r['f1']:.3f} | {r['mape']:6.1f} | {r['mae']:5.1f} | {r['accuracy']:.3f}"
        )
    f1s   = [r["f1"]   for r in fold_results]
    mapes = [r["mape"] for r in fold_results]
    maes  = [r["mae"]  for r in fold_results]
    accs  = [r["accuracy"] for r in fold_results]
    lines.append("-" * 80)
    lines.append(
        f"  avg  |                                            | "
        f"{np.mean(f1s):.3f} | {np.mean(mapes):6.1f} | {np.mean(maes):5.1f} | {np.mean(accs):.3f}"
    )
    lines.append(
        f"  std  |                                            | "
        f"{np.std(f1s):.3f} | {np.std(mapes):6.1f} | {np.std(maes):5.1f} | {np.std(accs):.3f}"
    )
    summary = "\n".join(lines)
    (run_dir / "loocv_summary.txt").write_text(summary)
    return summary


# ---------------------------------------------------------------------------
def train(cfg_model: dict, cfg_train: dict):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg_train["checkpoint_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_loocv", log_file=run_dir / "train.log")
    logger.info(f"Run ID: {run_id} | Device: {device}")

    data_dir   = Path(cfg_train["data_dir"])
    fold_dirs  = sorted(data_dir.glob("fold_*"))
    n_folds    = len(fold_dirs)
    logger.info(f"Found {n_folds} folds in {data_dir}")

    fold_results = []

    # ── LOOCV folds ────────────────────────────────────────────────────────
    for fold_dir in fold_dirs:
        meta      = json.loads((fold_dir / "meta.json").read_text())
        fold_idx  = meta["fold"]
        test_rec  = meta["test_recording"]
        label     = f"[fold {fold_idx}]"

        logger.info(f"\n{'='*60}")
        logger.info(f"{label} test recording : {test_rec}")
        logger.info(f"{label} train/val/test windows : "
                    f"{meta['train_windows']} / {meta['val_windows']} / {meta['test_windows']}")

        ckpt_path = run_dir / f"fold_{fold_idx}_best.pt"
        _train_one(
            cfg_model, cfg_train,
            str(fold_dir / "train.npz"),
            str(fold_dir / "val.npz"),
            ckpt_path, device, logger, label,
        )

        test_m = _eval_on_test(
            cfg_model, cfg_train, ckpt_path,
            str(fold_dir / "test.npz"), device,
        )
        logger.info(
            f"{label} TEST -> f1={test_m['f1']:.3f}  "
            f"precision={test_m['precision']:.3f}  recall={test_m['recall']:.3f}  "
            f"mape={test_m['mape']:.1f}%  mae={test_m['mae']:.1f}g  "
            f"acc={test_m['accuracy']:.3f}"
        )

        fold_results.append({
            "fold": fold_idx,
            "test_recording": test_rec,
            "f1":       test_m["f1"],
            "mape":     test_m["mape"],
            "mae":      test_m["mae"],
            "accuracy": test_m["accuracy"],
            "precision": test_m["precision"],
            "recall":   test_m["recall"],
        })

    # ── LOOCV summary ──────────────────────────────────────────────────────
    summary = _save_summary(fold_results, run_dir)
    logger.info(f"\n{'='*60}")
    logger.info("LOOCV SUMMARY\n" + summary)

    # ── Final model on all data ────────────────────────────────────────────
    all_train = data_dir / "all_train.npz"
    all_val   = data_dir / "all_val.npz"
    if all_train.exists():
        logger.info(f"\n{'='*60}")
        logger.info("Training final model on all data ...")
        final_ckpt = run_dir / "final_model.pt"
        _train_one(
            cfg_model, cfg_train,
            str(all_train), str(all_val),
            final_ckpt, device, logger, "[final]",
        )
        logger.info(f"Final model saved: {final_ckpt}")
    else:
        logger.warning("all_train.npz not found — skipping final model training")
        final_ckpt = None

    logger.info(f"\nRun dir: {run_dir}")
    return run_dir, final_ckpt


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="training/train_config_loocv.yaml")
    parser.add_argument("--model_config", default="model/model_config_12ch.yaml")
    args = parser.parse_args()
    cfg_model = yaml.safe_load(open(args.model_config))
    cfg_train = yaml.safe_load(open(args.config))
    train(cfg_model, cfg_train)
