"""Core two-stage inference: presence detection → weight regression."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_HERE = Path(__file__).resolve().parent   # inference/
_ROOT = _HERE.parent                      # haim_local_app/

WINDOW_SIZE = 39
STRIDE      = 5
SIGMA_FLOOR = 150.0

MERGE_GAP = {
    "carp":  79,
    "bream": 20,
}

SIGNAL_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]

_CKPTS = _ROOT / "checkpoints"

PRESENCE_CKPT = {
    "carp":  _CKPTS / "carp_presence_v5_knockdown",
    "bream": _CKPTS / "bream_presence_v1",
}

_W = "carp_weight"
WEIGHT_CKPT = {
    "carp_no_sal":       _CKPTS / "carp_weight_combined_model",
    "carp_filename_sal": _CKPTS / "carp_weight_salinity_model",
    "bream":             _CKPTS / "bream_weight_model",
}
WEIGHT_CFG = {
    "carp_no_sal":       _ROOT / "model" / "model_config_12ch.yaml",
    "carp_filename_sal": _ROOT / "model" / "model_config_12ch_salinity.yaml",
    "bream":             _ROOT / "model" / "model_config_12ch.yaml",
}

SAL_NORM_PATH = _ROOT / "data" / "salinity_norm.npy"

_model_cache: dict = {}


def _load_ensemble(ckpt_dir: Path, cfg_path: Path, device) -> list:
    sys.path.insert(0, str(_ROOT))
    from model.fish_model import FishModel

    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("physics_dim", 0)
    cfg.setdefault("coact_gate", False)
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
    if not models:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    return models


def _load_platt(presence_dir: Path) -> tuple:
    p = presence_dir / "platt.npy"
    if p.exists():
        ab = np.load(str(p))
        return float(ab[0]), float(ab[1])
    return 1.0, 0.0


def _get_models(species: str, weight_variant: str, device):
    key = (species, weight_variant, str(device))
    if key not in _model_cache:
        pres_dir = PRESENCE_CKPT[species]
        platt_a, platt_b = _load_platt(pres_dir)
        pres  = _load_ensemble(pres_dir, _ROOT / "model" / "model_config_12ch.yaml", device)
        w_key = "bream" if species == "bream" else f"carp_{weight_variant}"
        wt    = _load_ensemble(WEIGHT_CKPT[w_key], WEIGHT_CFG[w_key], device)
        sal_norm = None
        if species == "carp" and weight_variant == "filename_sal" and SAL_NORM_PATH.exists():
            sal_norm = np.load(str(SAL_NORM_PATH))
        _model_cache[key] = (pres, wt, platt_a, platt_b, sal_norm)
    return _model_cache[key]


def _score_presence(wins: np.ndarray, models: list, device,
                    platt_a: float, platt_b: float, batch_size: int) -> np.ndarray:
    logits = np.zeros(len(wins), np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), batch_size):
            X = torch.from_numpy(wins[b: b + batch_size]).to(device)
            s = None
            for m in models:
                lv, _ = m(X)
                lv = lv.cpu().numpy()
                s = lv if s is None else s + lv
            s /= len(models)
            logits[b: b + len(s)] = s
    return (1.0 / (1.0 + np.exp(-(platt_a * logits + platt_b)))).astype(np.float32)


def _nms(starts: list, probs: np.ndarray, threshold: float, merge_gap: int) -> list:
    pos_idx = [i for i, p in enumerate(probs) if p >= threshold]
    if not pos_idx:
        return []
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= merge_gap:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    dets = []
    for g in groups:
        pk = g[int(np.argmax(probs[g]))]
        dets.append({
            "peak_start":    int(starts[pk]),
            "center_sample": int(starts[pk]) + WINDOW_SIZE // 2,
            "peak_prob":     float(probs[pk]),
        })
    return dets


def _predict_weight(win: np.ndarray, models: list, device) -> float:
    t = torch.from_numpy(win[None]).to(device)
    total = 0.0
    with torch.no_grad():
        for m in models:
            _, p = m(t)
            total += p.item()
    return max(100.0, total / len(models))


def run_inference(
    csv_path,
    species: str = "bream",
    weight_variant: str = "no_sal",
    threshold: float = 0.5,
    batch_size: int = 256,
) -> dict:
    """
    Run two-stage inference on a single CSV recording.

    Returns a dict with keys: metadata, summary, detections.
    """
    csv_path = Path(csv_path)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pres_models, wt_models, platt_a, platt_b, sal_norm = \
        _get_models(species, weight_variant, device)

    merge_gap = MERGE_GAP[species]

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0

    has_date = "Date" in df.columns
    has_time = "Time" in df.columns

    # Unix timestamp from the first row of the CSV
    unix_timestamp = _csv_start_unix(df)

    raw = df[SIGNAL_CHANNELS].values.astype(np.float32)

    # ── Per-recording z-score normalisation ───────────────────────────────────
    mu        = np.nanmean(raw, axis=0)
    sigma_eff = np.maximum(np.nanstd(raw, axis=0), SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu   = np.where(np.isnan(mu), 0.0, mu)
    norm = np.where(np.isnan((raw - mu) / sigma_eff), 0.0,
                    (raw - mu) / sigma_eff).astype(np.float32)

    # ── Stage 1: presence detection ───────────────────────────────────────────
    starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
    if not starts:
        return _empty_result(csv_path, species, weight_variant)

    wins  = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
    probs = _score_presence(wins, pres_models, device, platt_a, platt_b, batch_size)
    dets  = _nms(starts, probs, threshold, merge_gap)

    # ── Stage 2: weight regression ────────────────────────────────────────────
    sal_scalar = None
    if sal_norm is not None:
        sal_val    = 400.0 if "_S400" in csv_path.name else 0.0
        sal_scalar = float((sal_val - sal_norm[0]) / sal_norm[1])

    result_dets  = []
    fish_weights = []

    for fish_id, det in enumerate(dets, start=1):
        ps  = det["peak_start"]
        win = norm[ps: ps + WINDOW_SIZE]
        if len(win) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(win), len(SIGNAL_CHANNELS)), np.float32)
            win = np.concatenate([win, pad])

        if sal_scalar is not None:
            col = np.full((WINDOW_SIZE, 1), sal_scalar, np.float32)
            win = np.concatenate([win, col], axis=1)

        w = _predict_weight(win, wt_models, device)
        fish_weights.append(w)

        # Detection timestamp — use the row at the detection centre
        row_idx = min(det["center_sample"], len(df) - 1)
        det_date = str(df["Date"].iloc[row_idx]) if has_date else None
        det_time = str(df["Time"].iloc[row_idx]) if has_time else None

        result_dets.append({
            "detection_date":  det_date,
            "detection_time":  det_time,
            "number_of_fish":  fish_id,
            "is_valid":        True,
            "length ":         None,
            "weight ":         round(w, 2),
            "ml_confidence":   round(det["peak_prob"], 4),
        })

    return {
        "unix_timestamp":    unix_timestamp,
        "authentication": {
            "authentication_code": "",
            "device_id":           "",
        },
        "Water temperature": None,
        "Water resistance":  None,
        "detections":        result_dets,
    }


def _csv_start_unix(df: "pd.DataFrame") -> int:
    """Return the Unix timestamp of the first row in the CSV, or 0 if unparseable."""
    try:
        date_str = str(df["Date"].iloc[0])
        time_str = str(df["Time"].iloc[0])
        # Time may be "HH:MM:SS.fff" or "HH:MM:SS"
        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in time_str else "%Y-%m-%d %H:%M:%S"
        dt  = datetime.strptime(f"{date_str} {time_str}", fmt)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return 0


def _empty_result(csv_path, species, weight_variant) -> dict:
    return {
        "unix_timestamp":    0,
        "authentication": {
            "authentication_code": "",
            "device_id":           "",
        },
        "Water temperature": None,
        "Water resistance":  None,
        "detections":        [],
    }
