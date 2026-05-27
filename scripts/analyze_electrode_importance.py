"""
Electrode importance analysis — two independent methods on all ACQ GT events.

Method 1 — Empirical peak amplitude:
    For each GT event span, compute max |z-score| per channel inside the span.
    Average across all events. Higher = channel responds more strongly to
    real fish crossings in the training distribution.

Method 2 — Model gradient saliency:
    For each GT event, take the 39-sample window centered on (start+end)//2
    (same window the training pipeline calls a positive), run a forward pass
    with requires_grad=True, backprop the logit, average |grad| over the time
    axis → per-channel sensitivity of the model output to that channel.
    Averaged across all events and all 3 ensemble members.

Outputs:
    results/electrode_importance.txt   — saved report
    Printed table + top-3 summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from model.fish_model import FishModel

CKPT_DIR   = _ROOT / "checkpoints" / "carp_presence_v3"
CFG_PATH   = _ROOT / "model" / "model_config_12ch.yaml"
DATA_DIR   = _ROOT / "data" / "raw" / "carp" / "ACQ_5_3_2026"
REPORT_CSV = DATA_DIR / "output_report.csv"
OUT_FILE   = _ROOT / "results" / "electrode_importance.txt"

SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67",
                   "B15","B37","B18","B32","B45","B67"]
WINDOW_SIZE = 39
SIGMA_FLOOR = 150.0


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


def normalize(raw):
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    out = (raw - mu) / sigma_eff
    return np.where(np.isnan(out), 0.0, out).astype(np.float32)


def load_norm(csv_path):
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw  = df[SIGNAL_CHANNELS].values.astype(np.float32)
    return normalize(raw)


def centered_window(norm, mid):
    half  = WINDOW_SIZE // 2
    start = max(0, mid - half)
    win   = norm[start: start + WINDOW_SIZE]
    if len(win) < WINDOW_SIZE:
        win = np.concatenate(
            [win, np.zeros((WINDOW_SIZE - len(win), len(SIGNAL_CHANNELS)), np.float32)]
        )
    return win  # (39, 12)


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = load_ensemble(device)
    print(f"Loaded {len(models)} ensemble model(s)\n")

    # ── Load GT events ────────────────────────────────────────────────────────
    report = pd.read_csv(REPORT_CSV)
    valid  = report[report["is_valid"] == True] if "is_valid" in report.columns else pd.DataFrame()
    events = []   # list of (file, start, end, mid)
    for _, row in valid.iterrows():
        fname = row["raw_data_file_name"]
        s, e  = int(row["start_index"]), int(row["end_index"])
        csv   = DATA_DIR / fname
        if csv.exists():
            events.append((fname, s, e, (s + e) // 2))

    print(f"Total GT events with existing CSVs: {len(events)}\n")

    # ── Method 1: empirical peak amplitude ───────────────────────────────────
    # Cache normalized signals per file to avoid re-reading
    norm_cache: dict[str, np.ndarray] = {}
    emp_sum  = np.zeros(len(SIGNAL_CHANNELS), dtype=np.float64)
    emp_cnt  = 0

    print("Method 1 — Empirical peak |z-score| inside each GT span:")
    for fname, s, e, _ in events:
        if fname not in norm_cache:
            norm_cache[fname] = load_norm(DATA_DIR / fname)
        norm = norm_cache[fname]
        span = norm[s: e + 1]   # (duration, 12)
        if len(span) == 0:
            continue
        peak = np.abs(span).max(axis=0)  # (12,) — max per channel over span
        emp_sum += peak
        emp_cnt += 1

    emp_scores = emp_sum / emp_cnt if emp_cnt > 0 else emp_sum
    emp_rank   = np.argsort(-emp_scores)

    print(f"  {'Channel':>7}  {'Peak |z|':>9}  {'Rank':>5}")
    for rank, idx in enumerate(emp_rank):
        print(f"  {SIGNAL_CHANNELS[idx]:>7}  {emp_scores[idx]:>9.4f}  {rank+1:>5}")

    # ── Method 2: model gradient saliency ────────────────────────────────────
    grad_sum = np.zeros(len(SIGNAL_CHANNELS), dtype=np.float64)
    grad_cnt = 0

    print(f"\nMethod 2 — Gradient saliency |d(logit)/d(input)| per channel:")
    for fname, s, e, mid in events:
        if fname not in norm_cache:
            norm_cache[fname] = load_norm(DATA_DIR / fname)
        norm = norm_cache[fname]
        win  = centered_window(norm, mid)              # (39, 12)

        x = torch.from_numpy(win[None]).to(device)    # (1, 39, 12)
        x.requires_grad_(True)

        model_grads = np.zeros(len(SIGNAL_CHANNELS), dtype=np.float64)
        for m in models:
            if x.grad is not None:
                x.grad.zero_()
            logit, _ = m(x)
            logit.backward()
            # |grad| averaged over time axis → (12,)
            ch_grad = x.grad.abs().squeeze(0).mean(dim=0).detach().cpu().numpy()
            model_grads += ch_grad

        grad_sum += model_grads / len(models)
        grad_cnt += 1

    grad_scores = grad_sum / grad_cnt if grad_cnt > 0 else grad_sum
    grad_rank   = np.argsort(-grad_scores)

    print(f"  {'Channel':>7}  {'|grad|':>9}  {'Rank':>5}")
    for rank, idx in enumerate(grad_rank):
        print(f"  {SIGNAL_CHANNELS[idx]:>7}  {grad_scores[idx]:>9.6f}  {rank+1:>5}")

    # ── Combined ranking ──────────────────────────────────────────────────────
    # Normalise each score to [0,1] then average
    emp_norm  = (emp_scores  - emp_scores.min())  / (emp_scores.max()  - emp_scores.min()  + 1e-9)
    grad_norm = (grad_scores - grad_scores.min()) / (grad_scores.max() - grad_scores.min() + 1e-9)
    combined  = (emp_norm + grad_norm) / 2.0
    comb_rank = np.argsort(-combined)

    summary = []
    summary.append("=" * 60)
    summary.append("ELECTRODE IMPORTANCE — carp_presence_v3")
    summary.append("=" * 60)
    summary.append(f"GT events analysed : {emp_cnt}")
    summary.append(f"Ensemble models    : {len(models)}")
    summary.append("")
    summary.append("Method 1 — Empirical peak |z-score| inside GT span")
    summary.append(f"  Top 3: {', '.join(SIGNAL_CHANNELS[i] for i in emp_rank[:3])}")
    summary.append("")
    summary.append("Method 2 — Model gradient saliency |d(logit)/d(input)|")
    summary.append(f"  Top 3: {', '.join(SIGNAL_CHANNELS[i] for i in grad_rank[:3])}")
    summary.append("")
    summary.append("Combined ranking (normalised average of both methods)")
    summary.append(f"  {'Rank':>5}  {'Channel':>7}  {'Emp score':>10}  {'Grad score':>11}  {'Combined':>9}")
    for rank, idx in enumerate(comb_rank):
        summary.append(
            f"  {rank+1:>5}  {SIGNAL_CHANNELS[idx]:>7}  "
            f"{emp_norm[idx]:>10.4f}  {grad_norm[idx]:>11.4f}  {combined[idx]:>9.4f}"
        )
    summary.append("")
    summary.append(f"TOP 3 OVERALL: {', '.join(SIGNAL_CHANNELS[i] for i in comb_rank[:3])}")
    summary.append("=" * 60)

    report_str = "\n".join(summary)
    print("\n" + report_str)
    OUT_FILE.write_text(report_str, encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
