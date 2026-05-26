"""Two-stage fish detection + weight estimation pipeline.

Stage 1 — fish_presence_model (3-ensemble):
    Slides a 39-sample window over a raw recording CSV.
    Windows above the presence threshold are candidate detections.

Stage 2 — carp_weight_model (3-ensemble):
    For each confirmed detection (after NMS merging of overlapping windows),
    predicts the carp weight in grams.

Non-maximum suppression (NMS):
    Consecutive positive windows within MERGE_GAP samples of each other
    are merged into a single detection. The peak-probability window within
    each group is selected; its center is used for weight prediction.

Usage (from project root):
    python pipeline/two_stage_inference.py path/to/recording.csv [--threshold 0.5]

Output: table of detected events + predicted weights, total biomass estimate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fish_model import FishModel

# ---------------------------------------------------------------------------
WINDOW_SIZE  = 39
STRIDE       = 5       # samples between consecutive windows (~8x overlap)
MERGE_GAP    = WINDOW_SIZE   # merge detections closer than this many samples

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH = len(ACOUSTIC_CHANNELS)

PRESENCE_CFG_DIR  = Path("checkpoints/fish_presence_model")
WEIGHT_CFG_DIR    = Path("checkpoints/carp_weight_model")
MODEL_CFG_PATH    = Path("model/model_config_12ch.yaml")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------


def _load_models(ckpt_dir: Path, cfg: dict) -> list:
    models = []
    for run_dir in sorted(ckpt_dir.glob("2*")):
        pt = run_dir / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(DEVICE)
        ckpt = torch.load(pt, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        m.eval()
        models.append(m)
    return models


def load_all_models():
    with open(MODEL_CFG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["physics_dim"] = 0

    presence_models = _load_models(PRESENCE_CFG_DIR, base_cfg)
    weight_models   = _load_models(WEIGHT_CFG_DIR,   base_cfg)

    print(f"  Presence ensemble: {len(presence_models)} models")
    print(f"  Weight ensemble  : {len(weight_models)} models")
    return presence_models, weight_models


def load_recording(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV, z-normalise, return (raw_norm, mean, std)."""
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)
    m   = np.nanmean(raw, axis=0).astype(np.float32)
    s   = np.nanstd(raw,  axis=0).astype(np.float32)
    s[s < 1e-8] = 1.0
    mask = np.isnan(raw)
    if mask.any():
        raw = np.where(mask, m[np.newaxis, :], raw)
    norm = ((raw - m) / s).astype(np.float32)
    return norm, m, s


def build_windows(norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return windows tensor (N, WINDOW_SIZE, N_CH) and start positions."""
    n = len(norm)
    starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
    wins   = np.stack([norm[s: s + WINDOW_SIZE] for s in starts])
    return wins, np.array(starts)


def infer_presence(models: list, wins: np.ndarray) -> np.ndarray:
    """Return ensemble-averaged presence probabilities (N,)."""
    probs_list = []
    with torch.no_grad():
        for m in models:
            t = torch.from_numpy(wins).to(DEVICE)
            logits, _ = m(t)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
    return np.mean(probs_list, axis=0)


def infer_weight(models: list, win: np.ndarray) -> float:
    """Return ensemble-averaged weight prediction for a single window."""
    t = torch.from_numpy(win[np.newaxis]).to(DEVICE)
    preds = []
    with torch.no_grad():
        for m in models:
            _, pred = m(t)
            preds.append(pred.item())
    return float(np.mean(preds))


def nms_merge(starts: np.ndarray, probs: np.ndarray, threshold: float) -> list[dict]:
    """
    Merge overlapping positive windows into single detections.

    Returns list of dicts: {center_sample, peak_prob, start_sample, end_sample}
    """
    positive_idx = np.where(probs >= threshold)[0]
    if len(positive_idx) == 0:
        return []

    groups = []
    current = [positive_idx[0]]
    for i in positive_idx[1:]:
        if starts[i] - starts[current[-1]] <= MERGE_GAP:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    detections = []
    for group in groups:
        group_probs  = probs[group]
        peak_local   = int(np.argmax(group_probs))
        peak_idx     = group[peak_local]
        peak_start   = starts[peak_idx]
        center       = peak_start + WINDOW_SIZE // 2
        detections.append({
            "center_sample": center,
            "peak_start":    peak_start,
            "peak_prob":     float(group_probs[peak_local]),
            "span_start":    int(starts[group[0]]),
            "span_end":      int(starts[group[-1]]) + WINDOW_SIZE,
        })
    return detections


def run_pipeline(csv_path: Path, threshold: float, sample_rate_hz: float = 40.0):
    print(f"\nRecording : {csv_path.name}")

    norm, _, _ = load_recording(csv_path)
    n_samples  = len(norm)
    print(f"Samples   : {n_samples}  ({n_samples / sample_rate_hz:.1f} s @ {sample_rate_hz:.0f} Hz)")

    wins, starts = build_windows(norm)
    print(f"Windows   : {len(wins)} (stride={STRIDE})")

    print(f"\n--- Stage 1: presence detection (thr={threshold}) ---")
    probs = infer_presence(presence_models, wins)
    n_positive = int((probs >= threshold).sum())
    print(f"Positive windows: {n_positive} / {len(wins)}")

    detections = nms_merge(starts, probs, threshold)
    print(f"Merged detections: {len(detections)}")

    if not detections:
        print("No fish detected.")
        return

    print(f"\n--- Stage 2: weight estimation ---")
    print(f"  {'#':>3}  {'Sample':>8}  {'Time(s)':>8}  {'Prob':>6}  {'Weight':>8}")
    print("  " + "-" * 46)

    total_weight = 0.0
    for i, det in enumerate(detections, 1):
        peak_start = det["peak_start"]
        win_data   = norm[peak_start: peak_start + WINDOW_SIZE]
        if len(win_data) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(win_data), N_CH), np.float32)
            win_data = np.concatenate([win_data, pad])
        w_pred       = infer_weight(weight_models, win_data)
        w_pred_clamp = max(100.0, w_pred)   # floor at 100g
        time_s       = det["center_sample"] / sample_rate_hz
        det["weight_pred_g"] = w_pred_clamp
        total_weight += w_pred_clamp
        print(f"  {i:>3}  {det['center_sample']:>8}  {time_s:>8.2f}s  "
              f"{det['peak_prob']:>6.3f}  {w_pred_clamp:>7.0f}g")

    print(f"\n  Detected fish: {len(detections)}")
    print(f"  Total biomass: {total_weight:.0f} g  ({total_weight / 1000:.3f} kg)")
    return detections


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage fish detection + weight pipeline")
    parser.add_argument("csv", nargs="+", help="Recording CSV file(s) to analyse")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Presence detection threshold (default: 0.5)")
    parser.add_argument("--sample_rate", type=float, default=40.0,
                        help="Recording sample rate in Hz (default: 40)")
    args = parser.parse_args()

    print("=" * 60)
    print("Two-stage fish detection + weight estimation pipeline")
    print("=" * 60)
    print(f"\nLoading models ...")
    presence_models, weight_models = load_all_models()

    all_results = []
    for csv_file in args.csv:
        path = Path(csv_file)
        if not path.exists():
            print(f"\nWARNING: {path} not found, skipping.")
            continue
        dets = run_pipeline(path, args.threshold, args.sample_rate)
        if dets:
            all_results.extend(dets)

    if len(args.csv) > 1:
        total = sum(d["weight_pred_g"] for d in all_results)
        print(f"\n{'=' * 40}")
        print(f"Grand total — {len(all_results)} fish detected")
        print(f"Grand total biomass: {total:.0f} g  ({total / 1000:.3f} kg)")
