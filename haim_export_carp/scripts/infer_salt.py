"""Core inference pipeline for FishSaltModel biomass estimation.

Processes raw CSV acoustic recordings — no ground truth or filename
salt convention required.  Salinity is detected automatically from the
raw electrode baseline using the Kohlrausch conductivity formula.

Model: 3-member FishSaltModel ensemble trained with hard negative mining.
       Logits are averaged across all members before thresholding.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.fish_salt_model import FishSaltModel, compute_physics_features

WINDOW_SIZE = 39
STRIDE = 10
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]

CHECKPOINTS_DIR = ROOT / "checkpoints"
SALT_CHECKPOINTS = [
    str(CHECKPOINTS_DIR / "salt_model_1.pt"),
    str(CHECKPOINTS_DIR / "salt_model_2.pt"),
    str(CHECKPOINTS_DIR / "salt_model_3.pt"),
]


def _load_models(ckpt_paths: list, device: torch.device):
    models, log_scales = [], []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = FishSaltModel(ckpt["cfg_model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
        log_scales.append(ckpt.get("log_scale_weight", False))
    return models, log_scales


def _dedup_events(positions: list, probs: np.ndarray, weights: np.ndarray,
                  gap: int = WINDOW_SIZE) -> tuple:
    """Merge overlapping/consecutive detection windows into discrete events.

    Windows whose start positions are within `gap` samples of the previous
    window's start are grouped into one event.  The event's representative
    weight is taken from the highest-probability window in the group.

    Returns:
        event_weights  — (E,) weight for each distinct event
        event_probs    — (E,) peak probability for each event
        n_events       — number of distinct events
    """
    if len(positions) == 0:
        return np.array([]), np.array([]), 0

    pos_arr = np.array(positions)
    order   = np.argsort(pos_arr)
    pos_s   = pos_arr[order]
    prob_s  = probs[order]
    w_s     = weights[order]

    event_weights, event_probs = [], []
    grp_start = pos_s[0]
    grp_probs, grp_weights = [prob_s[0]], [w_s[0]]

    for i in range(1, len(pos_s)):
        if pos_s[i] - grp_start <= gap:
            grp_probs.append(prob_s[i])
            grp_weights.append(w_s[i])
        else:
            best = int(np.argmax(grp_probs))
            event_weights.append(grp_weights[best])
            event_probs.append(grp_probs[best])
            grp_start   = pos_s[i]
            grp_probs   = [prob_s[i]]
            grp_weights = [w_s[i]]

    best = int(np.argmax(grp_probs))
    event_weights.append(grp_weights[best])
    event_probs.append(grp_probs[best])

    return np.array(event_weights), np.array(event_probs), len(event_weights)


def _build_windows(raw_norm: np.ndarray) -> tuple:
    """Slide a 39-sample window over the z-normalised recording."""
    positions = list(range(0, max(0, len(raw_norm) - WINDOW_SIZE + 1), STRIDE))
    if not positions:
        return np.empty((0, WINDOW_SIZE, 12), np.float32), positions
    windows = np.stack([
        raw_norm[pos: pos + WINDOW_SIZE].astype(np.float32)
        for pos in positions
    ])  # (N, 39, 12)
    return windows, positions


def run_salt_inference(
    data_dir,
    salt_ckpts: list = None,
    mode: str = "soft",
    threshold: float = 0.55,
    batch_size: int = 256,
    progress_callback=None,
) -> dict:
    """
    Estimate total fish biomass from all CSV recordings in data_dir.

    Salinity is inferred automatically from the raw electrode baseline —
    no filename convention or external annotation is required.

    Uses a 3-member ensemble; logits are averaged before thresholding.

    Args:
        data_dir:          Path to folder containing recording CSV files.
        salt_ckpts:        List of FishSaltModel checkpoint paths.
                           Defaults to the bundled 3-member ensemble.
        mode:              "soft"  -> sum(weight * presence_prob) for all windows
                           "hard"  -> sum(weight) for windows where prob >= threshold
        threshold:         Decision threshold (hard mode only; also used for
                           detection-count display in soft mode).
        batch_size:        Inference batch size.
        progress_callback: Optional callable(current_pct, 100, message).

    Returns:
        dict with keys:
            biomass_g, n_detections, n_windows_total, n_files,
            mode, threshold, per_file
    """
    if salt_ckpts is None:
        salt_ckpts = SALT_CHECKPOINTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if progress_callback:
        progress_callback(0, 100, "Loading models…")

    models, log_scales = _load_models(salt_ckpts, device)

    data_dir = Path(data_dir)
    csv_files = sorted([
        p for p in data_dir.glob("*.csv")
        if "output" not in p.name.lower()
    ])

    if not csv_files:
        raise ValueError(f"No CSV recording files found in: {data_dir}")

    total_biomass    = 0.0
    total_detections = 0
    total_windows    = 0
    per_file         = []

    for i, csv_path in enumerate(csv_files):
        pct = int(5 + i / len(csv_files) * 90)
        if progress_callback:
            progress_callback(pct, 100, f"Processing {csv_path.name}…")

        df = pd.read_csv(csv_path)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)  # (N, 12) raw ADC

        # Physics features from raw ADC (before z-normalisation)
        x_phys = compute_physics_features(raw)   # (3,)

        # Per-file z-score normalisation (matches training)
        mean = np.nanmean(raw, axis=0).astype(np.float32)
        std  = np.nanstd(raw,  axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0
        raw_norm = (raw - mean) / std

        windows, positions = _build_windows(raw_norm)

        if len(positions) == 0:
            per_file.append({"file": csv_path.name, "n_windows": 0,
                             "n_detections": 0, "biomass_g": 0.0})
            continue

        # Tile physics features to match window count
        x_phys_tiled = np.tile(x_phys, (len(windows), 1))  # (N, 3)

        sum_logits  = None
        sum_weights = None

        for b in range(0, len(windows), batch_size):
            X_w = torch.tensor(windows[b: b + batch_size]).to(device)
            X_p = torch.tensor(x_phys_tiled[b: b + batch_size]).to(device)

            batch_logit_sum  = None
            batch_weight_sum = None

            with torch.no_grad():
                for model, log_scale in zip(models, log_scales):
                    logit, pw, _ = model(X_w, X_p)
                    lv = logit.cpu().numpy()
                    pw = pw.cpu().numpy()
                    if log_scale:
                        pw = np.exp(pw)
                    batch_logit_sum  = lv if batch_logit_sum  is None else batch_logit_sum  + lv
                    batch_weight_sum = pw if batch_weight_sum is None else batch_weight_sum + pw

            batch_logit_sum  /= len(models)
            batch_weight_sum /= len(models)

            sum_logits  = (batch_logit_sum  if sum_logits  is None
                           else np.concatenate([sum_logits,  batch_logit_sum]))
            sum_weights = (batch_weight_sum if sum_weights is None
                           else np.concatenate([sum_weights, batch_weight_sum]))

        all_logits  = sum_logits
        all_weights = sum_weights
        probs       = 1.0 / (1.0 + np.exp(-all_logits))

        # Deduplicate: collapse overlapping windows into discrete fish events
        det_mask = probs >= threshold
        ev_w, ev_p, n_events = _dedup_events(
            [positions[i] for i in range(len(positions)) if det_mask[i]],
            probs[det_mask], all_weights[det_mask],
        )

        if mode == "hard":
            file_biomass = float(ev_w.sum())
            n_detected   = n_events
        else:
            ev_w_all, ev_p_all, _ = _dedup_events(
                positions, probs, all_weights)
            file_biomass = float((ev_w_all * ev_p_all).sum())
            n_detected   = n_events

        total_biomass    += file_biomass
        total_detections += n_detected
        total_windows    += len(positions)

        per_file.append({
            "file":         csv_path.name,
            "n_windows":    len(positions),
            "n_detections": n_detected,
            "biomass_g":    file_biomass,
        })

    if progress_callback:
        progress_callback(100, 100, "Done.")

    return {
        "biomass_g":       total_biomass,
        "n_detections":    total_detections,
        "n_windows_total": total_windows,
        "n_files":         len(csv_files),
        "mode":            mode,
        "threshold":       threshold,
        "per_file":        per_file,
    }
