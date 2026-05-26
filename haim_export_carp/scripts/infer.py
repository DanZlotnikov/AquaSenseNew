"""Core inference pipeline for carp biomass estimation.

Processes raw CSV acoustic recordings — no ground truth labels required.
Uses a 6-model ensemble presence detector + log-scale weight regressor.

Salt concentration is inferred from the filename convention:
    *_S400* → 400 g/L salt   (all other files → 0 g/L / fresh water)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.fish_model import FishModel

WINDOW_SIZE = 39
STRIDE = 10
ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
SALT_MAX = 400.0

CHECKPOINTS_DIR = ROOT / "checkpoints"

# 6 presence models: 3 pass-2 (single-model HN) + 3 pass-3 (ensemble HN)
PRESENCE_CHECKPOINTS = [
    str(CHECKPOINTS_DIR / "presence_1.pt"),
    str(CHECKPOINTS_DIR / "presence_2.pt"),
    str(CHECKPOINTS_DIR / "presence_3.pt"),
    str(CHECKPOINTS_DIR / "presence_4.pt"),
    str(CHECKPOINTS_DIR / "presence_5.pt"),
    str(CHECKPOINTS_DIR / "presence_6.pt"),
]
WEIGHT_CHECKPOINT = str(CHECKPOINTS_DIR / "weight.pt")


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
            grp_start  = pos_s[i]
            grp_probs  = [prob_s[i]]
            grp_weights = [w_s[i]]

    best = int(np.argmax(grp_probs))
    event_weights.append(grp_weights[best])
    event_probs.append(grp_probs[best])

    return np.array(event_weights), np.array(event_probs), len(event_weights)


def _parse_salt(fname: str) -> float:
    return 400.0 if "_S400" in fname else 0.0


def _load_models(presence_ckpts: list, weight_ckpt: str, device: torch.device):
    presence_models = []
    for path in presence_ckpts:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = FishModel(ckpt["cfg_model"]).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        presence_models.append(m)

    ckpt_wt = torch.load(weight_ckpt, map_location=device, weights_only=False)
    weight_model = FishModel(ckpt_wt["cfg_model"]).to(device)
    weight_model.load_state_dict(ckpt_wt["model_state"])
    weight_model.eval()
    log_scale = ckpt_wt.get("log_scale_weight", False)

    return presence_models, weight_model, log_scale


def _build_windows(raw: np.ndarray, mean: np.ndarray, std: np.ndarray,
                   salt_norm: float) -> tuple:
    """Slide a 39-sample window over the recording, return array and position list."""
    positions = list(range(0, max(0, len(raw) - WINDOW_SIZE + 1), STRIDE))
    if not positions:
        return np.empty((0, WINDOW_SIZE, 13), np.float32), positions

    salt_col = np.full((WINDOW_SIZE, 1), salt_norm, np.float32)
    windows = np.stack([
        np.concatenate([
            ((raw[pos: pos + WINDOW_SIZE] - mean) / std).astype(np.float32),
            salt_col,
        ], axis=1)
        for pos in positions
    ])  # (N, 39, 13)
    return windows, positions


def run_inference(
    data_dir,
    presence_ckpts=None,
    weight_ckpt=None,
    mode: str = "soft",
    threshold: float = 0.54,
    batch_size: int = 256,
    progress_callback=None,
) -> dict:
    """
    Estimate total fish biomass from all CSV recordings in data_dir.

    Args:
        data_dir:          Path to folder containing recording CSV files.
        presence_ckpts:    List of presence model checkpoint paths.
                           Defaults to the bundled 6-model ensemble.
        weight_ckpt:       Weight model checkpoint path.
                           Defaults to the bundled log-scale weight model.
        mode:              "soft"  → Σ(weight × presence_prob)  for all windows
                           "hard"  → Σ(weight) for windows where prob ≥ threshold
        threshold:         Decision threshold (hard mode only; also used for
                           detection-count display in soft mode).
        batch_size:        Inference batch size.
        progress_callback: Optional callable(current_pct: int, 100, message: str).

    Returns:
        dict with keys:
            biomass_g        — total estimated biomass in grams
            n_detections     — number of windows classified as fish
            n_windows_total  — total windows evaluated
            n_files          — number of CSV files processed
            mode             — "soft" or "hard"
            threshold        — threshold used
            per_file         — list of per-file result dicts
    """
    if presence_ckpts is None:
        presence_ckpts = PRESENCE_CHECKPOINTS
    if weight_ckpt is None:
        weight_ckpt = WEIGHT_CHECKPOINT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if progress_callback:
        progress_callback(0, 100, "Loading models…")

    presence_models, weight_model, log_scale = _load_models(
        presence_ckpts, weight_ckpt, device
    )

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

        salt_norm = _parse_salt(csv_path.name) / SALT_MAX

        df = pd.read_csv(csv_path)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)

        # Per-file z-score normalisation (matches training procedure)
        mean = np.nanmean(raw, axis=0).astype(np.float32)
        std  = np.nanstd(raw,  axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0

        windows, positions = _build_windows(raw, mean, std, salt_norm)

        if len(positions) == 0:
            per_file.append({"file": csv_path.name, "n_windows": 0,
                             "n_detections": 0, "biomass_g": 0.0})
            continue

        all_logits  = None
        all_weights = []

        for b in range(0, len(windows), batch_size):
            X = torch.tensor(windows[b: b + batch_size]).to(device)
            with torch.no_grad():
                # Average logits across all presence models (ensemble)
                batch_logits = None
                for m in presence_models:
                    lv = m(X)[0].cpu().numpy()
                    batch_logits = lv if batch_logits is None else batch_logits + lv
                batch_logits /= len(presence_models)

                pw = weight_model(X)[1].cpu().numpy()
                if log_scale:
                    pw = np.exp(pw)

            all_logits = (batch_logits if all_logits is None
                         else np.concatenate([all_logits, batch_logits]))
            all_weights.append(pw)

        all_weights = np.concatenate(all_weights)
        probs = 1.0 / (1.0 + np.exp(-all_logits))

        # Deduplicate: collapse overlapping windows into discrete fish events
        det_mask = probs >= threshold
        ev_w, ev_p, n_events = _dedup_events(
            [positions[i] for i in range(len(positions)) if det_mask[i]],
            probs[det_mask], all_weights[det_mask],
        )

        if mode == "hard":
            file_biomass = float(ev_w.sum())
            n_detected   = n_events
        else:  # soft
            # For soft mode deduplicate all windows (not just above threshold)
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
