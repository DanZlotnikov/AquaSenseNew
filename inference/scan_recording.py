"""Sliding-window fish detector for a single raw recording CSV.

No detection report or start/end indices required — the model scans the
entire recording and identifies fish-present regions autonomously.

Usage (standalone):
    python inference/scan_recording.py path/to/recording.csv

Returns a list of FishEvent namedtuples, one per detected fish pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_model import FishModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 39
STEP        = 2          # slide every 2 samples (50 ms at 40 Hz) — good speed/sensitivity tradeoff
BATCH_SIZE  = 512        # windows processed per model forward pass
MIN_GAP     = WINDOW_SIZE // 2   # minimum samples between two distinct fish events

RAW_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]


class FishEvent(NamedTuple):
    peak_sample:  int    # sample index of the presence-probability peak
    presence_prob: float
    length_cm:    float
    weight_g:     float  # derived via weight = length * 3


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_file(
    recording_path: Path,
    model: FishModel,
    device: torch.device,
    threshold: float = 0.5,
    weight_formula=None,     # callable(length_cm) -> weight_g; default = length * 3
) -> list[FishEvent]:
    """Scan a single recording CSV and return detected fish events."""

    if weight_formula is None:
        weight_formula = lambda length: length * 3

    # --- Load & per-file normalise ---
    df = pd.read_csv(recording_path)
    for col in RAW_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0

    raw = df[RAW_CHANNELS].values.astype(np.float32)

    # Handle NaN (corrupted rows) — fill with column mean
    col_mean = np.nanmean(raw, axis=0)
    nan_mask = np.isnan(raw)
    if nan_mask.any():
        raw = np.where(nan_mask, col_mean[np.newaxis, :], raw)

    f_mean = raw.mean(axis=0).astype(np.float32)
    f_std  = raw.std(axis=0).astype(np.float32)
    f_std[f_std < 1e-8] = 1.0

    n_rows = len(raw)
    if n_rows < WINDOW_SIZE:
        return []   # recording too short

    # --- Extract all sliding windows ---
    half  = WINDOW_SIZE // 2
    positions = list(range(0, n_rows - WINDOW_SIZE + 1, STEP))

    all_probs   = np.zeros(n_rows, dtype=np.float32)
    all_lengths = np.zeros(n_rows, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(positions), BATCH_SIZE):
            batch_pos = positions[batch_start : batch_start + BATCH_SIZE]
            windows   = []
            for start in batch_pos:
                win = raw[start : start + WINDOW_SIZE]
                win = (win - f_mean) / f_std
                windows.append(win)

            X = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)
            logit_pres, pred_len = model(X)

            probs   = torch.sigmoid(logit_pres).cpu().numpy().flatten()
            lengths = pred_len.cpu().numpy().flatten()

            # Map each window's result to its centre sample
            for i, start in enumerate(batch_pos):
                centre = start + half
                if probs[i] > all_probs[centre]:
                    all_probs[centre]   = probs[i]
                    all_lengths[centre] = lengths[i]

    # --- Peak detection (non-maximum suppression) ---
    # Find all samples above threshold, then pick local maxima
    try:
        from scipy.signal import find_peaks
        above  = all_probs >= threshold
        cands  = np.where(above)[0]
        if len(cands) == 0:
            return []
        # find_peaks on the full probability array with minimum separation
        peaks, _ = find_peaks(all_probs, height=threshold, distance=MIN_GAP)
    except ImportError:
        # scipy not available — simple argmax-based approach
        peaks = _simple_peaks(all_probs, threshold, MIN_GAP)

    events = []
    for p in peaks:
        length = float(all_lengths[p])
        if length <= 0:
            continue
        events.append(FishEvent(
            peak_sample   = int(p),
            presence_prob = float(all_probs[p]),
            length_cm     = round(length, 2),
            weight_g      = round(weight_formula(length), 1),
        ))

    return events


def _simple_peaks(probs: np.ndarray, threshold: float, min_gap: int) -> np.ndarray:
    """Fallback peak detection without scipy."""
    peaks = []
    i = 0
    n = len(probs)
    while i < n:
        if probs[i] >= threshold:
            # Find the maximum in the next min_gap samples
            end   = min(i + min_gap, n)
            local = i + int(probs[i:end].argmax())
            peaks.append(local)
            i = local + min_gap
        else:
            i += 1
    return np.array(peaks, dtype=int)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> FishModel:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = FishModel(ckpt["cfg_model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    DEFAULT_CHECKPOINT = Path("checkpoints/v5/20260318_183559/best_model.pt")

    parser = argparse.ArgumentParser(description="Scan a single recording CSV for fish")
    parser.add_argument("recording", type=Path, help="Path to raw recording CSV")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)

    events = scan_file(args.recording, model, device, threshold=args.threshold)

    print(f"\nFile : {args.recording.name}")
    print(f"Fish detected : {len(events)}")
    if events:
        print(f"\n{'Sample':>8}  {'P(present)':>11}  {'Length(cm)':>10}  {'Weight(g)':>10}")
        print("-" * 48)
        for e in events:
            print(f"{e.peak_sample:>8}  {e.presence_prob:>11.4f}  "
                  f"{e.length_cm:>10.2f}  {e.weight_g:>10.1f}")
        total_w = sum(e.weight_g for e in events)
        print(f"\nTotal biomass : {total_w:.1f} g")
