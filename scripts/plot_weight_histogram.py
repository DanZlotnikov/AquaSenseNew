"""Heatmap of fish weight bucket distribution per day — GT and ML side by side."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_ROOT   = Path(__file__).resolve().parent.parent
BASE    = _ROOT / "data" / "raw" / "carp" / "daily" / "Daily carp sampling"
OUT_DIR = _ROOT / "results" / "daily_stats"
ML_PER_FISH_CSV = OUT_DIR / "daily_ml_per_fish.csv"

DATE_RE   = re.compile(r"ACQ_(\d{2})_(\d{2})_(\d{4})$")
BIN_WIDTH = 50   # grams
BIN_MIN   = 200
BIN_MAX   = 1000


def load_gt_per_fish():
    rows = []
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():
            continue
        m = DATE_RE.match(folder.name)
        if not m:
            continue
        report = folder / "output_report.csv"
        if not report.exists():
            continue
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        date = pd.Timestamp(year=year, month=month, day=day)
        df    = pd.read_csv(report, low_memory=False)
        valid = df[df["is_valid"] == True]
        for w in valid["weight"].dropna():
            rows.append({"date": date, "weight_g": w})
    return pd.DataFrame(rows)


def make_heatmap(per_fish: pd.DataFrame, title: str, ax: plt.Axes, vmax=None):
    bins  = np.arange(BIN_MIN, BIN_MAX + BIN_WIDTH, BIN_WIDTH)
    dates = sorted(per_fish["date"].unique())

    grid = np.zeros((len(bins) - 1, len(dates)))
    for j, date in enumerate(dates):
        day_w = per_fish.loc[per_fish["date"] == date, "weight_g"].values
        counts, _ = np.histogram(day_w, bins=bins)
        grid[:, j] = counts

    # Normalize each day to show relative distribution (% of fish)
    col_sums = grid.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    grid_pct = grid / col_sums * 100

    im = ax.imshow(grid_pct, aspect="auto", origin="lower",
                   cmap="YlOrRd", vmin=0, vmax=vmax or grid_pct.max())

    # Y-axis: weight buckets
    bin_labels = [f"{b}–{b+BIN_WIDTH}" for b in bins[:-1]]
    ax.set_yticks(range(len(bin_labels)))
    ax.set_yticklabels(bin_labels, fontsize=7)
    ax.set_ylabel("Weight bucket (g)")

    # X-axis: dates
    date_labels = [d.strftime("%d %b") for d in dates]
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Date")
    ax.set_title(title, fontsize=11, fontweight="bold")

    return im, grid_pct.max()


def plot(gt: pd.DataFrame, ml: pd.DataFrame | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    has_ml = ml is not None and len(ml) > 0

    ncols = 2 if has_ml else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    fig.suptitle("Fish weight distribution per day", fontsize=13, fontweight="bold")

    _, vmax_gt = make_heatmap(gt, "GT detections", axes[0])
    if has_ml:
        make_heatmap(ml, "ML inference (v5_knockdown)", axes[1], vmax=None)

    plt.colorbar(axes[0].images[0], ax=axes[0], label="% of fish that day")
    if has_ml:
        plt.colorbar(axes[1].images[0], ax=axes[1], label="% of fish that day")

    plt.tight_layout()
    out = OUT_DIR / "daily_weight_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")



if __name__ == "__main__":
    print("Loading GT per-fish weights...")
    gt = load_gt_per_fish()
    print(f"  GT: {len(gt)} fish across {gt['date'].nunique()} days")

    ml = None
    if ML_PER_FISH_CSV.exists():
        print("Loading ML per-fish weights...")
        ml = pd.read_csv(ML_PER_FISH_CSV, parse_dates=["date"])
        print(f"  ML: {len(ml)} fish across {ml['date'].nunique()} days")
    else:
        print(f"  ML per-fish CSV not found ({ML_PER_FISH_CSV.name}) — plotting GT only.")
        print("  Run plot_daily_stats_ml.py to generate it.")

    plot(gt, ml)
