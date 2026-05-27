"""
F-B ring co-activation analysis on ACQ GT events.

Tests the hypothesis: a real fish crossing produces synchronized activation
in at least one front (F) electrode AND one back (B) electrode simultaneously.

Measures (computed inside each GT event span):
  1. f_peak / b_peak      — max |z-score| in front ring / back ring
  2. joint_min            — min(f_peak, b_peak): proxy for "both rings activated"
  3. fb_corr              — Pearson r between mean-F and mean-B signals over time
  4. best_fb_pair_corr    — highest |r| across all 36 (F_i, B_j) pairs

All measures also computed for random non-event windows (same length as median
GT event) to establish a baseline.

Outputs: results/fb_correlation.txt + printed tables
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

DATA_DIR   = _ROOT / "data" / "raw" / "carp" / "ACQ_5_3_2026"
REPORT_CSV = DATA_DIR / "output_report.csv"
OUT_FILE   = _ROOT / "results" / "fb_correlation.txt"

F_CHANNELS = ["F15","F37","F18","F32","F45","F67"]
B_CHANNELS = ["B15","B37","B18","B32","B45","B67"]
ALL_CHANNELS = F_CHANNELS + B_CHANNELS

SIGMA_FLOOR = 150.0
SKIP = ("output","report","axis","summary","function","points","updated")

RNG = np.random.default_rng(42)


def normalize(raw):
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu  = np.where(np.isnan(mu), 0.0, mu)
    out = (raw - mu) / sigma_eff
    return np.where(np.isnan(out), 0.0, out).astype(np.float32)


def safe_corr(a, b):
    """Pearson r; returns 0 if either array has zero variance."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def span_metrics(norm, s, e):
    """Compute all F-B metrics for a single span [s, e) in a normalized recording."""
    span = norm[s: e]
    if len(span) < 2:
        return None

    f_idx = [ALL_CHANNELS.index(c) for c in F_CHANNELS]
    b_idx = [ALL_CHANNELS.index(c) for c in B_CHANNELS]

    f_peak = float(np.abs(span[:, f_idx]).max())
    b_peak = float(np.abs(span[:, b_idx]).max())
    joint_min = min(f_peak, b_peak)

    f_mean = span[:, f_idx].mean(axis=1)
    b_mean = span[:, b_idx].mean(axis=1)
    fb_corr = safe_corr(f_mean, b_mean)

    # Best F-B pair
    best_pair_r = 0.0
    best_pair   = ("", "")
    for fi, bi in product(f_idx, b_idx):
        r = safe_corr(span[:, fi], span[:, bi])
        if abs(r) > abs(best_pair_r):
            best_pair_r = r
            best_pair   = (ALL_CHANNELS[fi], ALL_CHANNELS[bi])

    return dict(
        f_peak=f_peak, b_peak=b_peak, joint_min=joint_min,
        fb_corr=fb_corr, best_pair_r=best_pair_r,
        best_f=best_pair[0], best_b=best_pair[1],
    )


def pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def summarize(rows, label):
    if not rows:
        print(f"  {label}: no data")
        return
    fp   = [r["f_peak"]     for r in rows]
    bp   = [r["b_peak"]     for r in rows]
    jm   = [r["joint_min"]  for r in rows]
    fc   = [r["fb_corr"]    for r in rows]
    bpr  = [r["best_pair_r"]for r in rows]

    # Fraction where both rings show strong activation (|z| > 1.0)
    both_strong = sum(1 for r in rows if r["f_peak"] > 1.0 and r["b_peak"] > 1.0)

    lines = [
        f"\n  {label}  (n={len(rows)})",
        f"  {'Metric':<22}  {'mean':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}",
        f"  {'-'*58}",
        f"  {'F-ring peak |z|':<22}  {np.mean(fp):>7.3f}  {pct(fp,25):>7.3f}  {pct(fp,50):>7.3f}  {pct(fp,75):>7.3f}",
        f"  {'B-ring peak |z|':<22}  {np.mean(bp):>7.3f}  {pct(bp,25):>7.3f}  {pct(bp,50):>7.3f}  {pct(bp,75):>7.3f}",
        f"  {'joint min(F,B)':<22}  {np.mean(jm):>7.3f}  {pct(jm,25):>7.3f}  {pct(jm,50):>7.3f}  {pct(jm,75):>7.3f}",
        f"  {'F-mean / B-mean corr':<22}  {np.mean(fc):>7.3f}  {pct(fc,25):>7.3f}  {pct(fc,50):>7.3f}  {pct(fc,75):>7.3f}",
        f"  {'best F-B pair |r|':<22}  {np.mean(np.abs(bpr)):>7.3f}  {pct(np.abs(bpr),25):>7.3f}  {pct(np.abs(bpr),50):>7.3f}  {pct(np.abs(bpr),75):>7.3f}",
        f"  Both rings >1.0 |z| : {both_strong}/{len(rows)} = {100*both_strong/len(rows):.1f}%",
    ]
    for l in lines:
        print(l)
    return lines


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Load GT events ────────────────────────────────────────────────────────
    report = pd.read_csv(REPORT_CSV)
    valid  = report[report["is_valid"] == True] if "is_valid" in report.columns else pd.DataFrame()

    events_by_file: dict[str, list] = {}
    for _, row in valid.iterrows():
        fname = row["raw_data_file_name"]
        if any(x in fname.lower() for x in SKIP):
            continue
        s, e = int(row["start_index"]), int(row["end_index"])
        events_by_file.setdefault(fname, []).append((s, e))

    gt_durations = [e - s for evs in events_by_file.values() for s, e in evs]
    median_dur   = max(2, int(np.median(gt_durations))) if gt_durations else 20
    print(f"GT events: {sum(len(v) for v in events_by_file.values())}  "
          f"median duration: {median_dur} samples\n")

    # ── Compute metrics on GT events and random non-event windows ─────────────
    event_rows   = []
    random_rows  = []

    norm_cache: dict[str, np.ndarray] = {}

    for fname, evs in events_by_file.items():
        csv_path = DATA_DIR / fname
        if not csv_path.exists():
            continue
        if fname not in norm_cache:
            df = pd.read_csv(csv_path)
            for col in ALL_CHANNELS:
                if col not in df.columns:
                    df[col] = 0.0
            norm_cache[fname] = normalize(df[ALL_CHANNELS].values.astype(np.float32))
        norm = norm_cache[fname]
        n    = len(norm)

        # Build a set of GT-occupied sample indices for this file
        occupied = set()
        for s, e in evs:
            occupied.update(range(max(0, s - median_dur), min(n, e + median_dur)))

        # GT event metrics
        for s, e in evs:
            m = span_metrics(norm, s, max(s + 2, e))
            if m:
                event_rows.append(m)

        # Random non-event windows (same number as GT events in this file)
        n_rand = len(evs)
        candidates = [i for i in range(0, n - median_dur) if i not in occupied]
        if candidates:
            chosen = RNG.choice(candidates,
                                size=min(n_rand, len(candidates)),
                                replace=False)
            for start in chosen:
                m = span_metrics(norm, int(start), int(start) + median_dur)
                if m:
                    random_rows.append(m)

    # ── Per-pair F-B correlation ranking ──────────────────────────────────────
    pair_corr: dict[tuple, list] = {(f, b): [] for f in F_CHANNELS for b in B_CHANNELS}
    for fname, evs in events_by_file.items():
        norm = norm_cache.get(fname)
        if norm is None:
            continue
        f_idx = [ALL_CHANNELS.index(c) for c in F_CHANNELS]
        b_idx = [ALL_CHANNELS.index(c) for c in B_CHANNELS]
        for s, e in evs:
            span = norm[s: max(s + 2, e)]
            if len(span) < 2:
                continue
            for fi, bi in product(range(len(F_CHANNELS)), range(len(B_CHANNELS))):
                r = safe_corr(span[:, f_idx[fi]], span[:, b_idx[bi]])
                pair_corr[(F_CHANNELS[fi], B_CHANNELS[bi])].append(abs(r))

    pair_means = {k: float(np.mean(v)) for k, v in pair_corr.items() if v}
    top_pairs  = sorted(pair_means.items(), key=lambda x: -x[1])[:6]

    # ── Print results ─────────────────────────────────────────────────────────
    output_lines = []

    header = [
        "=" * 62,
        "F-B RING CO-ACTIVATION ANALYSIS — ACQ GT events",
        "=" * 62,
    ]
    for l in header:
        print(l)
        output_lines.append(l)

    ev_lines  = summarize(event_rows,  "GT events   ") or []
    ran_lines = summarize(random_rows, "Random windows (non-event)") or []
    output_lines.extend(ev_lines + ran_lines)

    print("\n  Top 6 F-B electrode pairs by mean |corr| during GT events:")
    output_lines.append("\n  Top 6 F-B electrode pairs by mean |corr| during GT events:")
    for (f, b), mean_r in top_pairs:
        line = f"    {f} — {b}   mean |r| = {mean_r:.4f}"
        print(line)
        output_lines.append(line)

    # ── Conclusion ────────────────────────────────────────────────────────────
    ev_joint   = np.mean([r["joint_min"]  for r in event_rows])
    ran_joint  = np.mean([r["joint_min"]  for r in random_rows]) if random_rows else 0
    ev_corr    = np.mean([r["fb_corr"]    for r in event_rows])
    ran_corr   = np.mean([r["fb_corr"]    for r in random_rows]) if random_rows else 0
    ev_both    = sum(1 for r in event_rows  if r["f_peak"] > 1.0 and r["b_peak"] > 1.0)
    ran_both   = sum(1 for r in random_rows if r["f_peak"] > 1.0 and r["b_peak"] > 1.0)

    conclusion = [
        "",
        "=" * 62,
        "CONCLUSION",
        "=" * 62,
        f"joint min(F,B) — events: {ev_joint:.3f}  vs  random: {ran_joint:.3f}  "
        f"(ratio {ev_joint/(ran_joint+1e-9):.1f}x)",
        f"F-mean/B-mean corr — events: {ev_corr:.3f}  vs  random: {ran_corr:.3f}",
        f"Both rings >1.0|z| — events: {ev_both}/{len(event_rows)} "
        f"({100*ev_both/max(1,len(event_rows)):.0f}%)  "
        f"vs  random: {ran_both}/{len(random_rows)} "
        f"({100*ran_both/max(1,len(random_rows)):.0f}%)",
        f"Best F-B pair: {top_pairs[0][0][0]} — {top_pairs[0][0][1]}  "
        f"(mean |r| = {top_pairs[0][1]:.4f})",
    ]
    for l in conclusion:
        print(l)
        output_lines.append(l)

    OUT_FILE.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
