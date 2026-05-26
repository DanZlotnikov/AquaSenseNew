"""Run carp presence inference on a single recording and produce a scrollable
interactive HTML plot showing signal vs time with red detection spans.

The most important electrode is selected automatically via gradient saliency:
average |d(logit)/d(input)| over high-confidence windows, summed over time,
per channel.

Usage (from project root):
    python scripts/infer_plot_recording.py data/raw/carp/ACQ_5_3_2026/AQUA1474.csv
    python scripts/infer_plot_recording.py path/to/recording.csv --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_model import FishModel

# ---------------------------------------------------------------------------
WINDOW_SIZE = 39
STRIDE      = 5
MERGE_GAP   = WINDOW_SIZE
SAMPLE_RATE = 40.0   # Hz

ACOUSTIC_CHANNELS = [
    "F15", "F37", "F18", "F32", "F45", "F67",
    "B15", "B37", "B18", "B32", "B45", "B67",
]
N_CH = len(ACOUSTIC_CHANNELS)

CKPT_DIR       = Path("checkpoints/bream_carp_presence")
MODEL_CFG_PATH = Path("model/model_config_12ch.yaml")
PLATT_PATH     = CKPT_DIR / "platt.npy"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------


def load_ensemble():
    with open(MODEL_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(CKPT_DIR.glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(DEVICE)
        ck = torch.load(pt, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
    return models


def load_recording(csv_path: Path):
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype(np.float32)
    m = np.nanmean(raw, axis=0).astype(np.float32)
    s = np.nanstd(raw,  axis=0).astype(np.float32)
    s[s < 1e-8] = 1.0
    mask = np.isnan(raw)
    if mask.any():
        raw = np.where(mask, m[np.newaxis, :], raw)
    norm = ((raw - m) / s).astype(np.float32)
    return raw, norm, m, s


def build_windows(norm: np.ndarray):
    n      = len(norm)
    starts = np.arange(0, n - WINDOW_SIZE + 1, STRIDE)
    wins   = np.stack([norm[s: s + WINDOW_SIZE] for s in starts])
    return wins, starts


def infer_presence(models, wins: np.ndarray):
    """Ensemble-averaged presence logits and probabilities."""
    sum_logits = None
    with torch.no_grad():
        for m in models:
            t = torch.from_numpy(wins).to(DEVICE)
            logits, _ = m(t)
            lv = logits.cpu().numpy()
            sum_logits = lv if sum_logits is None else sum_logits + lv
    avg_logits = (sum_logits / len(models)).ravel()
    return avg_logits


def nms_merge(starts: np.ndarray, probs: np.ndarray, threshold: float):
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
        gp   = probs[group]
        peak = group[int(np.argmax(gp))]
        detections.append({
            "center_sample": int(starts[peak]) + WINDOW_SIZE // 2,
            "peak_prob":     float(gp.max()),
            "span_start":    int(starts[group[0]]),
            "span_end":      int(starts[group[-1]]) + WINDOW_SIZE,
        })
    return detections


def most_important_channel(models, wins: np.ndarray, probs: np.ndarray, top_k=50):
    """
    Gradient saliency: average |d(mean_ensemble_logit)/d(input)| over the
    top_k highest-probability windows, summed across the time axis.
    Returns (channel_index, channel_name, per_channel_scores).
    """
    # Pick high-confidence windows
    high_idx = np.argsort(probs)[-top_k:]
    if len(high_idx) == 0:
        return 0, ACOUSTIC_CHANNELS[0], np.zeros(N_CH)

    X = torch.from_numpy(wins[high_idx]).to(DEVICE).requires_grad_(True)
    sum_scores = np.zeros(N_CH)

    for m in models:
        logits, _ = m(X)
        grad = torch.autograd.grad(logits.mean(), X)[0]   # (k, T, C)
        sum_scores += grad.detach().abs().sum(dim=(0, 1)).cpu().numpy()

    avg_scores = sum_scores / len(models)
    best_ch    = int(np.argmax(avg_scores))
    return best_ch, ACOUSTIC_CHANNELS[best_ch], avg_scores


def build_html_plot(
    raw: np.ndarray,
    norm: np.ndarray,
    detections: list,
    probs_on_samples: np.ndarray,
    ch_idx: int,
    ch_name: str,
    ch_scores: np.ndarray,
    csv_path: Path,
    threshold: float,
    platt_probs: np.ndarray,
) -> go.Figure:
    n = len(raw)
    t = np.arange(n) / SAMPLE_RATE   # seconds

    # Channels ranked best → worst by saliency
    ranked_indices = np.argsort(ch_scores)[::-1]

    # 12 visually distinct colours (ColorBrewer Set3 + extras)
    PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]

    fig = go.Figure()

    # ── Electrode traces (z-normalised, ranked by saliency) ──────────
    for rank, ci in enumerate(ranked_indices):
        name  = ACOUSTIC_CHANNELS[ci]
        score = ch_scores[ci]
        color = PALETTE[rank % len(PALETTE)]
        is_best = (ci == ch_idx)

        DEFAULT_VISIBLE = {"F32", "B32"}
        if name in DEFAULT_VISIBLE:
            width   = 2.0 if is_best else 1.5
            opacity = 1.0
            visible = True
        else:
            width   = 0.9
            opacity = 0.75
            visible = "legendonly"

        label = f"{'★ ' if is_best else ''}{name}  (sal={score:.2f})"

        fig.add_trace(go.Scatter(
            x=t, y=norm[:, ci],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            visible=visible,
            name=label,
            legendrank=rank,
            yaxis="y1",
            hovertemplate=f"<b>{name}</b>  t=%{{x:.3f}}s  z=%{{y:.3f}}<extra></extra>",
        ))

    # ── Detection probability trace (secondary y-axis) ───────────────
    fig.add_trace(go.Scatter(
        x=t, y=platt_probs,
        mode="lines",
        line=dict(color="orange", width=1.0),
        name="Presence prob",
        yaxis="y2",
        opacity=0.75,
        hovertemplate="t=%{x:.3f}s<br>prob=%{y:.3f}<extra></extra>",
    ))

    # ── Threshold line ────────────────────────────────────────────────
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="orange",
        annotation_text=f"thr={threshold}", annotation_position="bottom right",
        yref="y2",
    )

    # ── Red shaded detection spans ────────────────────────────────────
    for det in detections:
        t0 = det["span_start"] / SAMPLE_RATE
        t1 = det["span_end"]   / SAMPLE_RATE
        fig.add_vrect(
            x0=t0, x1=t1,
            fillcolor="red", opacity=0.20,
            layer="below", line_width=0,
            annotation_text=f"p={det['peak_prob']:.2f}",
            annotation_position="top left",
            annotation_font_size=9,
        )

    # ── Layout ───────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Carp Presence Inference — {csv_path.name}</b><br>"
                f"<sup>★ Most important electrode: <b>{ch_name}</b> (gradient saliency) "
                f"| {len(detections)} detection(s) | thr={threshold} "
                f"| top-6 shown, rest toggle-able in legend</sup>"
            ),
            font_size=14,
        ),
        xaxis=dict(
            title="Time (s)",
            rangeslider=dict(visible=True, thickness=0.05),
            type="linear",
        ),
        yaxis=dict(title="Signal (z-normalised)", side="left"),
        yaxis2=dict(
            title="Presence probability",
            side="right",
            overlaying="y",
            range=[0, 1.05],
            showgrid=False,
        ),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
        dragmode="pan",
        hovermode="x unified",
        height=520,
        margin=dict(t=110, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", selector=dict(side="left"))

    return fig


def main(csv_path: Path, threshold: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading ensemble from {CKPT_DIR} ...")
    models = load_ensemble()
    print(f"  {len(models)} models loaded")

    platt_ab = np.load(PLATT_PATH)
    a_p, b_p = float(platt_ab[0]), float(platt_ab[1])
    print(f"  Platt params: a={a_p:.4f}  b={b_p:.4f}")

    print(f"\nLoading {csv_path.name} ...")
    raw, norm, _, _ = load_recording(csv_path)
    n = len(raw)
    print(f"  {n} samples  ({n / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE:.0f} Hz)")

    print("Building windows ...")
    wins, starts = build_windows(norm)
    print(f"  {len(wins)} windows (stride={STRIDE})")

    print("Running inference ...")
    avg_logits = infer_presence(models, wins)
    raw_probs  = expit(avg_logits)
    cal_probs  = expit(a_p * avg_logits + b_p)

    # Map per-window probs back to a per-sample array (for plotting)
    probs_on_samples = np.zeros(n)
    counts           = np.zeros(n)
    for i, s in enumerate(starts):
        probs_on_samples[s: s + WINDOW_SIZE] += cal_probs[i]
        counts[s: s + WINDOW_SIZE]           += 1
    counts = np.maximum(counts, 1)
    probs_on_samples /= counts

    # NMS detections
    detections = nms_merge(starts, cal_probs, threshold)
    n_pos_win  = int((cal_probs >= threshold).sum())
    print(f"  Positive windows: {n_pos_win}/{len(wins)}")
    print(f"  Detections (after NMS): {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"    {i:3d}  t={d['center_sample']/SAMPLE_RATE:.2f}s  "
              f"span=[{d['span_start']/SAMPLE_RATE:.2f}s, {d['span_end']/SAMPLE_RATE:.2f}s]  "
              f"prob={d['peak_prob']:.3f}")

    print("\nComputing gradient saliency to select most informative electrode ...")
    ch_idx, ch_name, ch_scores = most_important_channel(models, wins, cal_probs)
    print(f"  Most important channel: {ch_name} (index {ch_idx})")
    ranked = sorted(zip(ch_scores, ACOUSTIC_CHANNELS), reverse=True)
    for score, name in ranked:
        marker = " <-- selected" if name == ch_name else ""
        print(f"    {name}: {score:.2f}{marker}")

    print("\nBuilding interactive plot ...")
    fig = build_html_plot(
        raw, norm, detections, probs_on_samples,
        ch_idx, ch_name, ch_scores,
        csv_path, threshold, probs_on_samples,
    )

    stem = csv_path.stem
    html_path = out_dir / f"{stem}_inference.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"\nInteractive plot saved to: {html_path}")
    print("Open in any browser to scroll left/right and zoom.")

    # ── Text summary ─────────────────────────────────────────────────
    summary_path = out_dir / f"{stem}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Inference summary: {csv_path.name}\n")
        f.write(f"Model: bream_carp_presence ensemble (Platt a={a_p:.4f}, b={b_p:.4f})\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Samples: {n}  ({n/SAMPLE_RATE:.1f}s)\n")
        f.write(f"Windows: {len(wins)}\n")
        f.write(f"Positive windows: {n_pos_win}\n")
        f.write(f"Detections (NMS): {len(detections)}\n\n")
        f.write(f"Most important electrode: {ch_name}\n\n")
        f.write(f"{'#':>3}  {'Sample':>7}  {'Time(s)':>8}  {'Peak prob':>9}  "
                f"{'SpanStart':>10}  {'SpanEnd':>8}\n")
        f.write("-" * 60 + "\n")
        for i, d in enumerate(detections, 1):
            f.write(f"{i:>3}  {d['center_sample']:>7}  "
                    f"{d['center_sample']/SAMPLE_RATE:>8.2f}  "
                    f"{d['peak_prob']:>9.4f}  "
                    f"{d['span_start']/SAMPLE_RATE:>10.2f}  "
                    f"{d['span_end']/SAMPLE_RATE:>8.2f}\n")
    print(f"Summary saved to:    {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer + plot a single recording")
    parser.add_argument("csv", help="Path to recording CSV")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir",   default="results/inference_plots")
    args = parser.parse_args()

    main(Path(args.csv), args.threshold, Path(args.out_dir))
