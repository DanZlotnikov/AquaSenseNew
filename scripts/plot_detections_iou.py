"""
Window-wise detection analysis: signal timeline + GT spans + ML spans + IoU.

Outputs interactive HTML files (one per recording) viewable in any browser.
An index.html summary table links to all per-file charts.

Usage:
  python scripts/plot_detections_iou.py              # all ACQ files with GT>0 or ML>0
  python scripts/plot_detections_iou.py AQUA1418     # specific file(s)

Outputs: results/<timestamp>_detections_iou_v3/
  index.html           — summary table with links
  AQUA1234_iou.html    — per-file interactive Plotly chart
  iou_per_event.csv    — per-match IoU data
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly not installed.  Run: pip install plotly")
    sys.exit(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from model.fish_model import FishModel

# ── config ────────────────────────────────────────────────────────────────────
CKPT_DIR   = _ROOT / "checkpoints" / "carp_presence_v3"
CFG_PATH   = _ROOT / "model" / "model_config_12ch.yaml"
DATA_DIR   = _ROOT / "data" / "raw" / "carp" / "ACQ_5_3_2026"
REPORT_CSV = DATA_DIR / "output_report.csv"
_TS        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = _ROOT / "results" / f"{_TS}_detections_iou_v3"

SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]
SKIP        = ("output","report","axis","summary","function","points","updated")
WINDOW_SIZE = 39
STRIDE      = 5
SIGMA_FLOOR = 150.0
THRESHOLD   = 0.5
MERGE_GAP   = 79
BATCH_SIZE  = 512
SAMPLE_RATE = 40.0

# Paper-space y domains for the two stacked panels (gap 0.06 between them)
_SIG_Y0,  _SIG_Y1  = 0.38, 1.00   # signal panel
_PROB_Y0, _PROB_Y1 = 0.00, 0.32   # probability panel


# ── model ─────────────────────────────────────────────────────────────────────
def load_ensemble(device):
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(CKPT_DIR.glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(device)
        ck = torch.load(pt, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
    return models


# ── normalization ─────────────────────────────────────────────────────────────
def normalize(raw):
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    out = (raw - mu) / sigma_eff
    return np.where(np.isnan(out), 0.0, out).astype(np.float32)


# ── inference ──────────────────────────────────────────────────────────────────
def infer(csv_path, models, device):
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw  = df[SIGNAL_CHANNELS].values.astype(np.float32)
    sig  = raw.mean(axis=1)
    norm = normalize(raw)
    starts = np.array(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE), dtype=np.int32)
    if len(starts) == 0:
        return sig, starts, np.array([], dtype=np.float32), []
    wins = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
    logits = np.zeros(len(wins), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), BATCH_SIZE):
            X = torch.from_numpy(wins[b: b + BATCH_SIZE]).to(device)
            s = None
            for m in models:
                lv, _ = m(X)
                lv = lv.cpu().numpy()
                s  = lv if s is None else s + lv
            s /= len(models)
            logits[b: b + len(s)] = s
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    pos_idx = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    dets = []
    if pos_idx:
        groups, cur = [], [pos_idx[0]]
        for i in pos_idx[1:]:
            if starts[i] - starts[cur[-1]] <= MERGE_GAP:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        for g in groups:
            pk = g[int(np.argmax(probs[g]))]
            dets.append({
                "span_start": int(starts[g[0]]),
                "span_end":   int(starts[g[-1]]) + WINDOW_SIZE,
                "peak_prob":  float(probs[pk]),
            })
    return sig, starts, probs, dets


# ── IoU & matching ─────────────────────────────────────────────────────────────
def iou(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_events, dets):
    if not gt_events or not dets:
        return [], set(range(len(gt_events))), set(range(len(dets)))
    mat = np.zeros((len(gt_events), len(dets)))
    for i, ev in enumerate(gt_events):
        for j, d in enumerate(dets):
            mat[i, j] = iou(ev["start_index"], ev["end_index"],
                            d["span_start"], d["span_end"])
    mg, md, matches = set(), set(), []
    pairs = sorted(
        ((mat[i, j], i, j) for i in range(len(gt_events)) for j in range(len(dets))),
        reverse=True,
    )
    for val, i, j in pairs:
        if val == 0:
            break
        if i in mg or j in md:
            continue
        matches.append((i, j, val))
        mg.add(i)
        md.add(j)
    return matches, set(range(len(gt_events))) - mg, set(range(len(dets))) - md


# ── HTML chart ────────────────────────────────────────────────────────────────
def make_html(fname, sig, starts, probs, gt_events, dets,
              matches, unmatched_gt, unmatched_det) -> str:
    sr      = SAMPLE_RATE
    n       = len(sig)
    t       = [i / sr for i in range(n)]
    prob_xs = [(int(s) + WINDOW_SIZE // 2) / sr for s in starts]

    match_by_gt  = {i: (j, v) for i, j, v in matches}
    match_by_det = {j: (i, v) for i, j, v in matches}
    iou_vals     = [v for _, _, v in matches]
    mean_iou     = float(np.mean(iou_vals)) if iou_vals else 0.0
    tp, fp, fn   = len(matches), len(unmatched_det), len(unmatched_gt)

    fig = go.Figure()

    # Signal trace (top panel — yaxis)
    fig.add_trace(go.Scatter(
        x=t,
        y=sig.tolist(),
        mode="lines",
        line=dict(color="#555", width=0.8),
        name="Signal (mean 12ch, ADC)",
        yaxis="y",
        hovertemplate="t=%{x:.2f}s  amp=%{y:.0f}<extra></extra>",
    ))

    # Probability trace (bottom panel — yaxis2)
    fig.add_trace(go.Scatter(
        x=prob_xs,
        y=probs.tolist() if len(probs) else [],
        mode="lines",
        line=dict(color="#2980b9", width=1.2),
        name="P(fish)",
        yaxis="y2",
        hovertemplate="t=%{x:.2f}s  p=%{y:.3f}<extra></extra>",
    ))

    # Legend swatches for shape colors (invisible traces, legend only)
    _legend_items = [
        ("GT matched",        "rgba(39,174,96,0.6)",  "#1e8449"),
        ("GT unmatched (FN)", "rgba(230,126,34,0.6)", "#ca6f1e"),
        ("ML detection TP",   "rgba(52,152,219,0.5)", "#2471a3"),
        ("ML detection FP",   "rgba(231,76,60,0.5)",  "#c0392b"),
    ]
    for label, fill, border in _legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", size=13, color=fill,
                        line=dict(color=border, width=2)),
            name=label,
            showlegend=True,
        ))

    shapes      = []
    annotations = []

    # ── GT event spans (signal panel only) ────────────────────────────────────
    # Matched GT = green, unmatched GT (FN) = orange
    for i, ev in enumerate(gt_events):
        x0      = ev["start_index"] / sr
        x1      = ev["end_index"]   / sr
        matched = i in match_by_gt
        fill    = "rgba(39,174,96,0.20)"  if matched else "rgba(230,126,34,0.20)"
        border  = "#1e8449"               if matched else "#ca6f1e"

        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=x0, x1=x1,
            y0=_SIG_Y0, y1=_SIG_Y1,
            fillcolor=fill,
            line=dict(color=border, width=1.5),
            layer="below",
        ))
        mid = (x0 + x1) / 2
        # GT label pinned to the top of the signal panel
        annotations.append(dict(
            x=mid, xref="x",
            y=_SIG_Y1 - 0.01, yref="paper",
            text=f"GT{i+1}",
            showarrow=False,
            font=dict(size=9, color=border, family="monospace"),
            yanchor="top", xanchor="center",
        ))
        # IoU value shown inside matched GT span
        if matched:
            _, val = match_by_gt[i]
            annotations.append(dict(
                x=mid, xref="x",
                y=_SIG_Y0 + (_SIG_Y1 - _SIG_Y0) * 0.85, yref="paper",
                text=f"IoU={val:.2f}",
                showarrow=False,
                font=dict(size=9, color="#1a5276"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#1a5276",
                borderwidth=1,
                borderpad=3,
                xanchor="center", yanchor="middle",
            ))

    # ── ML detection spans (both panels) ──────────────────────────────────────
    # TP = blue, FP = red  — same color throughout, clear dashed border per detection
    for j, det in enumerate(dets):
        x0      = det["span_start"] / sr
        x1      = det["span_end"]   / sr
        matched = j in match_by_det
        fill    = "rgba(52,152,219,0.14)"  if matched else "rgba(231,76,60,0.14)"
        border  = "#2471a3"                if matched else "#c0392b"
        tp_fp   = "TP" if matched else "FP"

        # Rectangle spans both panels so you can visually trace detection to its probability burst
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=x0, x1=x1,
            y0=_PROB_Y0, y1=_SIG_Y1,
            fillcolor=fill,
            line=dict(color=border, width=2),
            layer="above",
        ))
        mid = (x0 + x1) / 2
        # Detection label at top of the probability panel
        annotations.append(dict(
            x=mid, xref="x",
            y=_PROB_Y1 - 0.005, yref="paper",
            text=f"M{j+1} {tp_fp}  p={det['peak_prob']:.2f}",
            showarrow=False,
            font=dict(size=8, color=border, family="monospace"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=border,
            borderwidth=1,
            borderpad=3,
            xanchor="center", yanchor="top",
        ))

    # Threshold line in probability panel
    if n > 0:
        shapes.append(dict(
            type="line", xref="x", yref="y2",
            x0=t[0], x1=t[-1],
            y0=THRESHOLD, y1=THRESHOLD,
            line=dict(color="#c0392b", width=1, dash="dash"),
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{fname}</b>  |  "
                f"GT={len(gt_events)}  ML={len(dets)}  "
                f"TP={tp}  FP={fp}  FN={fn}  "
                f"mean IoU={mean_iou:.3f}"
            ),
            font=dict(size=13),
            x=0.01, xanchor="left",
        ),
        shapes=shapes,
        annotations=annotations,
        height=580,
        hovermode="x",
        dragmode="pan",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(family="Inter, sans-serif", size=11),
        # Shared x-axis — rangeslider controls both panels simultaneously
        xaxis=dict(
            title="Time (s)",
            domain=[0, 1],
            rangeslider=dict(visible=True, thickness=0.04),
        ),
        yaxis=dict(
            title="Signal (ADC)",
            domain=[_SIG_Y0, _SIG_Y1],
        ),
        yaxis2=dict(
            title="P(fish)",
            domain=[_PROB_Y0, _PROB_Y1],
            anchor="x",
            range=[0, 1.05],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(l=65, r=20, t=55, b=90),
    )

    return fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )


# ── index page ────────────────────────────────────────────────────────────────
def make_index_html(rows: list) -> str:
    total_gt   = sum(r["gt"] for r in rows)
    total_ml   = sum(r["ml"] for r in rows)
    total_tp   = sum(r["tp"] for r in rows)
    total_fp   = sum(r["fp"] for r in rows)
    total_fn   = sum(r["fn"] for r in rows)
    all_ious   = [v for r in rows for v in r["iou_vals"]]
    global_iou = f"{np.mean(all_ious):.3f}" if all_ious else "—"

    def row_html(r):
        mi  = f"{r['mean_iou']:.3f}" if r["gt"] > 0 and r["iou_vals"] else "—"
        bg  = (
            "background:#fde8e8;" if r["fp"] > 3 else
            "background:#fff8ec;" if r["fp"] > 0 else ""
        )
        return (
            f'<tr style="{bg}">'
            f'<td><a href="{r["html_file"]}">{r["file"]}</a></td>'
            f'<td>{r["gt"]}</td>'
            f'<td>{r["ml"]}</td>'
            f'<td style="color:#1e8449">{r["tp"]}</td>'
            f'<td style="color:#c0392b">{r["fp"]}</td>'
            f'<td style="color:#ca6f1e">{r["fn"]}</td>'
            f'<td>{mi}</td>'
            f'</tr>'
        )

    rows_html = "\n".join(row_html(r) for r in rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Detection IoU — {_TS}</title>
<style>
  body  {{ font-family: Inter, system-ui, sans-serif; font-size: 13px;
           margin: 28px 36px; color: #1a1a2e; }}
  h2   {{ font-size: 15px; font-weight: 600; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; min-width: 560px; }}
  th, td {{ border: 1px solid #dde3ea; padding: 5px 14px; }}
  th   {{ background: #f5f7fa; font-weight: 600; text-align: center; }}
  td   {{ text-align: right; }}
  td:first-child {{ text-align: left; white-space: nowrap; }}
  tr:hover {{ background: #eef3fb !important; }}
  tfoot td {{ font-weight: 700; background: #eaf2ff; }}
  a    {{ color: #2471a3; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .legend {{ margin-top: 14px; padding: 10px 14px; background: #f5f7fa;
             border: 1px solid #dde3ea; border-radius: 4px;
             font-size: 11px; color: #555; line-height: 1.8; }}
  .dot  {{ display:inline-block; width:10px; height:10px;
           border-radius:2px; margin-right:4px; vertical-align:middle; }}
</style>
</head>
<body>
<h2>Detection IoU — carp_presence_v3 — {_TS}</h2>
<table>
  <thead>
    <tr>
      <th style="text-align:left">File</th>
      <th>GT</th><th>ML</th>
      <th>TP</th><th>FP</th><th>FN</th>
      <th>mean IoU</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
  <tfoot>
    <tr>
      <td>TOTAL ({len(rows)} files)</td>
      <td>{total_gt}</td><td>{total_ml}</td>
      <td style="color:#1e8449">{total_tp}</td>
      <td style="color:#c0392b">{total_fp}</td>
      <td style="color:#ca6f1e">{total_fn}</td>
      <td>{global_iou}</td>
    </tr>
  </tfoot>
</table>
<div class="legend">
  <span class="dot" style="background:#1e8449"></span> GT matched (green) &nbsp;
  <span class="dot" style="background:#ca6f1e"></span> GT unmatched / FN (orange) &nbsp;
  <span class="dot" style="background:#2471a3"></span> ML detection TP (blue) &nbsp;
  <span class="dot" style="background:#c0392b"></span> ML detection FP (red) &nbsp;&nbsp;
  IoU = intersection-over-union of matched GT&nbsp;↔&nbsp;ML span pairs
</div>
</body>
</html>"""


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(device)
    print(f"Loaded {len(models)} models. Device: {device}")

    report = pd.read_csv(REPORT_CSV)
    valid  = report[report["is_valid"] == True] if "is_valid" in report.columns else pd.DataFrame()
    gt_by_file = {}
    for fname, grp in valid.groupby("raw_data_file_name"):
        gt_by_file[fname] = [
            {"start_index": int(r["start_index"]), "end_index": int(r["end_index"])}
            for _, r in grp.iterrows()
        ]

    cli_filters = [a for a in sys.argv[1:] if not a.startswith("-")]
    all_files   = sorted(f for f in DATA_DIR.glob("*.csv")
                         if not any(x in f.name.lower() for x in SKIP))

    summary_rows = []
    index_rows   = []
    all_iou_vals = []

    for csv_path in all_files:
        fname     = csv_path.name
        if cli_filters and not any(kw in fname for kw in cli_filters):
            continue

        gt_events              = gt_by_file.get(fname, [])
        sig, starts, probs, dets = infer(csv_path, models, device)

        if not cli_filters and not gt_events and not dets:
            continue

        matches, unmatched_gt, unmatched_det = greedy_match(gt_events, dets)

        iou_vals = [v for _, _, v in matches]
        mean_iou = float(np.mean(iou_vals)) if iou_vals else 0.0
        all_iou_vals.extend(iou_vals)
        tp, fp, fn = len(matches), len(unmatched_det), len(unmatched_gt)

        print(f"  {fname:25s}  GT={len(gt_events):3d}  ML={len(dets):3d}  "
              f"TP={tp:3d}  FP={fp:3d}  FN={fn:3d}  mean_IoU={mean_iou:.3f}")

        html_str  = make_html(fname, sig, starts, probs, gt_events, dets,
                              matches, unmatched_gt, unmatched_det)
        html_name = f"{csv_path.stem}_iou.html"
        (OUT_DIR / html_name).write_text(html_str, encoding="utf-8")

        index_rows.append({
            "file": fname, "html_file": html_name,
            "gt": len(gt_events), "ml": len(dets),
            "tp": tp, "fp": fp, "fn": fn,
            "mean_iou": mean_iou, "iou_vals": iou_vals,
        })

        # Per-event rows for CSV
        match_map       = {i: (j, v) for i, j, v in matches}
        matched_det_set = {j for _, j, _ in matches}
        for i, ev in enumerate(gt_events):
            row = {"file": fname, "gt_start": ev["start_index"],
                   "gt_end": ev["end_index"],
                   "gt_duration": ev["end_index"] - ev["start_index"]}
            if i in match_map:
                j, val = match_map[i]
                det = dets[j]
                row.update({"ml_start": det["span_start"], "ml_end": det["span_end"],
                            "peak_prob": det["peak_prob"],
                            "iou": round(val, 4), "matched": True})
            else:
                row.update({"ml_start": None, "ml_end": None,
                            "peak_prob": None, "iou": 0.0, "matched": False})
            summary_rows.append(row)
        for j, det in enumerate(dets):
            if j not in matched_det_set:
                summary_rows.append({
                    "file": fname,
                    "gt_start": None, "gt_end": None, "gt_duration": None,
                    "ml_start": det["span_start"], "ml_end": det["span_end"],
                    "peak_prob": det["peak_prob"], "iou": 0.0, "matched": False,
                })

    # Write index and CSV
    (OUT_DIR / "index.html").write_text(make_index_html(index_rows), encoding="utf-8")
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "iou_per_event.csv", index=False)

    # Summary stats
    print()
    if all_iou_vals:
        print(f"Matched-pair IoU ({len(all_iou_vals)} pairs):")
        print(f"  mean={np.mean(all_iou_vals):.4f}  "
              f"median={np.median(all_iou_vals):.4f}  "
              f"p25={np.percentile(all_iou_vals,25):.4f}  "
              f"p75={np.percentile(all_iou_vals,75):.4f}")

    print(f"\nOpen: {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
