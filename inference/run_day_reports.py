"""Batch day-folder biomass reporter with growth graph.

Scans all recording CSVs inside each provided day-folder, detects fish using
the sliding-window model, applies weight = length * 3, and produces:

    results/day_reports/<run_timestamp>/
        day_YYYY-MM-DD.csv       — per-detection results for each day
        summary.csv              — one row per day: fish count, total biomass, mean length
        growth_graph.png         — biomass over time plot

Usage:
    python inference/run_day_reports.py  folder1  folder2  folder3 ...

    # The folder name is used as the day label. Recommended naming: YYYY-MM-DD
    # e.g.:  python inference/run_day_reports.py  data/days/2026-03-01  data/days/2026-03-08

Options:
    --checkpoint   Path to model checkpoint  (default: checkpoints/v5 best)
    --threshold    Presence probability threshold  (default: 0.5)
    --output       Output directory  (default: results/day_reports/<timestamp>)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.scan_recording import load_model, scan_file, FishEvent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = Path("checkpoints/v5/20260318_183559/best_model.pt")
RESULTS_ROOT       = Path("results/day_reports")
WEIGHT_FORMULA     = lambda length: length * 3   # weight (g) = length (cm) * 3


# ---------------------------------------------------------------------------
# Per-day processing
# ---------------------------------------------------------------------------

def process_day(
    day_folder: Path,
    model,
    device: torch.device,
    threshold: float,
) -> pd.DataFrame:
    """Scan all CSV files in day_folder. Return DataFrame of all detections."""

    csv_files = sorted(f for f in day_folder.glob("*.csv")
                       if "output" not in f.name.lower()
                       and "report" not in f.name.lower())

    if not csv_files:
        print(f"  [WARNING] No CSV files found in {day_folder}")
        return pd.DataFrame()

    rows = []
    for csv_path in csv_files:
        try:
            events = scan_file(csv_path, model, device,
                               threshold=threshold,
                               weight_formula=WEIGHT_FORMULA)
            for e in events:
                rows.append({
                    "file":         csv_path.name,
                    "peak_sample":  e.peak_sample,
                    "presence_prob": e.presence_prob,
                    "length_cm":    e.length_cm,
                    "weight_g":     e.weight_g,
                })
            status = f"{len(events)} fish"
        except Exception as ex:
            status = f"ERROR: {ex}"

        print(f"    {csv_path.name:<50}  {status}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Growth graph
# ---------------------------------------------------------------------------

def _try_parse_date(label: str):
    """Try to parse a folder name as a date for the x-axis."""
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%d-%m-%Y", "%d%m%Y"):
        try:
            return datetime.strptime(label, fmt)
        except ValueError:
            pass
    return None


def save_growth_graph(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("Fish Population Growth — Daily Biomass Report",
                 fontsize=14, fontweight="bold", y=0.98)

    # Try to use real dates on x-axis
    dates = [_try_parse_date(d) for d in summary["day"]]
    use_dates = all(d is not None for d in dates)

    if use_dates:
        x = dates
        x_label = "Date"
    else:
        x = list(range(len(summary)))
        x_label = "Day index"

    days    = summary["day"].tolist()
    biomass = summary["total_biomass_g"].tolist()
    fish_n  = summary["fish_count"].tolist()
    mean_l  = summary["mean_length_cm"].tolist()
    mean_w  = summary["mean_weight_g"].tolist()

    # ---- [0] Total biomass ----
    ax = axes[0]
    bars = ax.bar(x if not use_dates else range(len(x)), biomass,
                  color="#2196F3", alpha=0.8, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, biomass):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(biomass) * 0.01,
                f"{val:.0f}g", ha="center", va="bottom", fontsize=8)
    # Trend line
    if len(biomass) >= 2:
        z   = np.polyfit(range(len(biomass)), biomass, 1)
        p   = np.poly1d(z)
        xs  = range(len(biomass))
        ax.plot(list(xs), p(list(xs)), "r--", linewidth=1.5, alpha=0.7,
                label=f"Trend ({z[0]:+.1f} g/day)")
        ax.legend(fontsize=8)
    ax.set_ylabel("Total Biomass (g)", fontsize=9)
    ax.set_title("Total Biomass per Day", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=30, ha="right", fontsize=8)

    # ---- [1] Fish count ----
    ax = axes[1]
    bars = ax.bar(range(len(fish_n)), fish_n,
                  color="#FF5722", alpha=0.8, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, fish_n):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(max(fish_n) * 0.01, 0.1),
                str(val), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Fish Detections", fontsize=9)
    ax.set_title("Fish Detections per Day", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=30, ha="right", fontsize=8)

    # ---- [2] Mean length & weight ----
    ax  = axes[2]
    ax2 = ax.twinx()
    l1, = ax.plot(range(len(mean_l)),  mean_l, "o-", color="#4CAF50",
                  linewidth=2, markersize=6, label="Mean length (cm)")
    l2, = ax2.plot(range(len(mean_w)), mean_w, "s--", color="#9C27B0",
                   linewidth=2, markersize=6, label="Mean weight (g)")
    ax.set_ylabel("Mean Length (cm)", fontsize=9, color="#4CAF50")
    ax2.set_ylabel("Mean Weight (g)", fontsize=9, color="#9C27B0")
    ax.set_title("Mean Length & Weight per Day", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelcolor="#4CAF50")
    ax2.tick_params(axis="y", labelcolor="#9C27B0")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=30, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sliding-window fish detection on day folders and produce growth report"
    )
    parser.add_argument("folders", nargs="+", type=Path,
                        help="Day folders to process (one per day, in chronological order)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Presence probability threshold (default: 0.5)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: results/day_reports/<timestamp>)")
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output or (RESULTS_ROOT / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Threshold  : {args.threshold}")
    print(f"Output     : {out_dir}")
    print(f"Weight     : length_cm × 3")
    print(f"Days       : {len(args.folders)}\n")

    model = load_model(args.checkpoint, device)

    summary_rows = []

    for day_folder in args.folders:
        day_label = day_folder.name
        print(f"{'=' * 60}")
        print(f"Day: {day_label}  ({day_folder})")

        day_df = process_day(day_folder, model, device, threshold=args.threshold)

        if day_df.empty:
            fish_count   = 0
            total_biomass = 0.0
            mean_length  = float("nan")
            mean_weight  = float("nan")
        else:
            fish_count    = len(day_df)
            total_biomass = float(day_df["weight_g"].sum())
            mean_length   = float(day_df["length_cm"].mean())
            mean_weight   = float(day_df["weight_g"].mean())

        print(f"  Fish detected   : {fish_count}")
        print(f"  Total biomass   : {total_biomass:.1f} g")
        if fish_count > 0:
            print(f"  Mean length     : {mean_length:.2f} cm")
            print(f"  Mean weight     : {mean_weight:.1f} g")

        # Save per-day CSV
        if not day_df.empty:
            day_df.insert(0, "day", day_label)
            day_csv = out_dir / f"day_{day_label}.csv"
            day_df.to_csv(day_csv, index=False)

        summary_rows.append({
            "day":             day_label,
            "fish_count":      fish_count,
            "total_biomass_g": round(total_biomass, 1),
            "mean_length_cm":  round(mean_length, 2) if not np.isnan(mean_length) else "",
            "mean_weight_g":   round(mean_weight, 1) if not np.isnan(mean_weight) else "",
        })

    print(f"\n{'=' * 60}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary table
    print("\nSummary")
    print(f"{'Day':<20}  {'Fish':>6}  {'Biomass(g)':>11}  {'MeanLen(cm)':>12}  {'MeanWt(g)':>10}")
    print("-" * 66)
    for _, r in summary_df.iterrows():
        print(f"{r['day']:<20}  {r['fish_count']:>6}  "
              f"{r['total_biomass_g']:>11.1f}  "
              f"{str(r['mean_length_cm']):>12}  "
              f"{str(r['mean_weight_g']):>10}")

    # Growth graph (only if >= 2 days)
    if len(summary_rows) >= 2:
        graph_path = out_dir / "growth_graph.png"
        # Filter to days with at least one detection for the trend lines
        plot_df = summary_df[summary_df["fish_count"] > 0].copy()
        plot_df["mean_length_cm"] = pd.to_numeric(plot_df["mean_length_cm"], errors="coerce")
        plot_df["mean_weight_g"]  = pd.to_numeric(plot_df["mean_weight_g"],  errors="coerce")
        save_growth_graph(summary_df, graph_path)
        print(f"\nGrowth graph : {graph_path}")
    else:
        print("\n(Growth graph requires >= 2 days)")

    print(f"Summary CSV  : {summary_path}")
    print(f"Per-day CSVs : {out_dir}/day_*.csv")


if __name__ == "__main__":
    main()
