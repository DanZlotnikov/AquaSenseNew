"""Two-stage inference pipeline: bream_carp_presence + carp_weight model.

Stage 1: sliding-window presence detection with NMS merging.
Stage 2: weight regression on each detected event.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent   # haim_export_carp/
sys.path.insert(0, str(ROOT))

from model.fish_model import FishModel

WINDOW_SIZE = 39
STRIDE      = 5
MERGE_GAP   = 39

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]

_CKPTS     = ROOT / "checkpoints"
_MODEL_DIR = ROOT / "model"

PRESENCE_CKPT_DIR  = _CKPTS / "bream_carp_presence"
PRESENCE_CFG_PATH  = _MODEL_DIR / "model_config_12ch.yaml"
PLATT_PATH         = _CKPTS / "bream_carp_presence" / "platt.npy"

WEIGHT_CKPT_DIRS = {
    "no_sal":      _CKPTS / "carp_weight_combined_model",
    "filename_sal": _CKPTS / "carp_weight_salinity_model",
}
WEIGHT_CFG_PATHS = {
    "no_sal":      _MODEL_DIR / "model_config_12ch.yaml",
    "filename_sal": _MODEL_DIR / "model_config_12ch_salinity.yaml",
}
SAL_NORM_PATH = ROOT / "data" / "carp" / "weight_salinity" / "salinity_norm.npy"


def _load_ensemble(ckpt_dir: Path, cfg_path: Path, device: torch.device) -> list:
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(ckpt_dir.glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(device)
        ck = torch.load(pt, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
    return models


def _nms_peaks(starts: list, probs: np.ndarray, threshold: float, merge_gap: int) -> list:
    """Return list of window indices (one per NMS group) above threshold."""
    pos_idx = [i for i, p in enumerate(probs) if p >= threshold]
    if not pos_idx:
        return []
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= merge_gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [g[int(np.argmax(probs[g]))] for g in groups]


def _load_platt() -> tuple:
    """Load Platt scaling params (a, b). Returns (1.0, 0.0) if file missing."""
    if PLATT_PATH.exists():
        ab = np.load(str(PLATT_PATH))
        return float(ab[0]), float(ab[1])
    return 1.0, 0.0


def _score_presence(wins: np.ndarray, models: list, device: torch.device,
                    platt_a: float, platt_b: float,
                    batch_size: int = 256) -> np.ndarray:
    """Return Platt-calibrated probability array of shape (N,)."""
    logits = np.zeros(len(wins), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), batch_size):
            X = torch.from_numpy(wins[b: b + batch_size]).to(device)
            batch_sum = None
            for m in models:
                lv, _ = m(X)
                lv = lv.cpu().numpy()
                batch_sum = lv if batch_sum is None else batch_sum + lv
            batch_sum /= len(models)
            logits[b: b + len(batch_sum)] = batch_sum
    # Apply Platt transform then sigmoid
    return (1.0 / (1.0 + np.exp(-(platt_a * logits + platt_b)))).astype(np.float32)


def _predict_weight(win: np.ndarray, models: list, device: torch.device) -> float:
    """Return ensemble-averaged weight prediction (g) for a single window."""
    total = 0.0
    t = torch.from_numpy(win[None]).to(device)
    with torch.no_grad():
        for m in models:
            _, p = m(t)
            total += p.item()
    return max(100.0, total / len(models))


THRESHOLD = 0.5   # fixed; calibrated via Platt scaling on val set


def run_twostage_inference(
    data_dir,
    weight_variant: str = "no_sal",
    batch_size: int = 256,
    progress_callback=None,
) -> dict:
    """
    Estimate fish count, average weight and total biomass from CSV recordings.

    Args:
        data_dir:          Folder containing *.csv acoustic recordings.
        weight_variant:    "no_sal" or "filename_sal".
        batch_size:        Batch size for presence scoring.
        progress_callback: Optional callable(pct: int, 100, msg: str).

    Returns:
        dict with keys:
            n_fish, avg_weight_g, biomass_g, n_files, weight_variant,
            per_file (list of dicts: file, n_fish, avg_weight_g, biomass_g)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if progress_callback:
        progress_callback(0, 100, "Loading models…")

    platt_a, platt_b = _load_platt()
    presence_models = _load_ensemble(PRESENCE_CKPT_DIR,   PRESENCE_CFG_PATH,             device)
    weight_models   = _load_ensemble(WEIGHT_CKPT_DIRS[weight_variant],
                                     WEIGHT_CFG_PATHS[weight_variant], device)

    sal_norm = None
    if weight_variant == "filename_sal":
        sal_norm = np.load(str(SAL_NORM_PATH))   # [mean, std]

    data_dir  = Path(data_dir)
    _skip = ("output", "report", "axis", "summary", "function", "points", "updated")
    csv_files = sorted([
        p for p in data_dir.glob("*.csv")
        if not any(x in p.name.lower() for x in _skip)
    ])
    if not csv_files:
        raise ValueError(f"No CSV recording files found in: {data_dir}")

    total_fish    = 0
    total_biomass = 0.0
    per_file      = []

    for i, csv_path in enumerate(csv_files):
        pct = int(5 + i / len(csv_files) * 90)
        if progress_callback:
            progress_callback(pct, 100, f"Processing {csv_path.name}…")

        df = pd.read_csv(csv_path)
        for col in ACOUSTIC_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)

        mu = np.nanmean(raw, axis=0).astype(np.float32)
        s  = np.nanstd(raw,  axis=0).astype(np.float32)
        s[(s < 1e-8) | np.isnan(s)] = 1.0          # also replace NaN std (all-NaN channel)
        mu = np.where(np.isnan(mu), 0.0, mu)        # replace NaN mean with 0
        norm = ((raw - mu) / s).astype(np.float32)
        norm = np.where(np.isnan(norm), 0.0, norm)  # replace any residual NaN with 0

        starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
        if not starts:
            per_file.append({"file": csv_path.name, "n_fish": 0,
                             "avg_weight_g": 0.0, "biomass_g": 0.0})
            continue

        # Stage 1 — presence scoring (Platt-calibrated probabilities)
        wins  = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])   # (N,39,12)
        probs = _score_presence(wins, presence_models, device, platt_a, platt_b, batch_size)

        # NMS merge
        peak_indices = _nms_peaks(starts, probs, THRESHOLD, MERGE_GAP)

        # Stage 2 — weight regression on each detected event
        extra_scalar = None
        if weight_variant == "filename_sal" and sal_norm is not None:
            sal_val      = 400.0 if "_S400" in csv_path.name else 0.0
            extra_scalar = float((sal_val - sal_norm[0]) / sal_norm[1])

        fish_weights = []
        for pi in peak_indices:
            ps  = starts[pi]
            win = norm[ps: ps + WINDOW_SIZE]
            if len(win) < WINDOW_SIZE:
                win = np.concatenate(
                    [win, np.zeros((WINDOW_SIZE - len(win), 12), np.float32)])

            if extra_scalar is not None:
                extra_col = np.full((WINDOW_SIZE, 1), extra_scalar, np.float32)
                win = np.concatenate([win, extra_col], axis=1)  # (39,13)

            fish_weights.append(_predict_weight(win, weight_models, device))

        n_fish   = len(fish_weights)
        biomass  = float(sum(fish_weights))
        avg_w    = biomass / n_fish if n_fish > 0 else 0.0

        total_fish    += n_fish
        total_biomass += biomass

        per_file.append({
            "file":         csv_path.name,
            "n_fish":       n_fish,
            "avg_weight_g": avg_w,
            "biomass_g":    biomass,
        })

    if progress_callback:
        progress_callback(100, 100, "Done.")

    overall_avg = total_biomass / total_fish if total_fish > 0 else 0.0
    return {
        "n_fish":         total_fish,
        "avg_weight_g":   overall_avg,
        "biomass_g":      total_biomass,
        "n_files":        len(csv_files),
        "weight_variant": weight_variant,
        "per_file":       per_file,
    }
