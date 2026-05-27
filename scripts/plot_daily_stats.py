"""Plot daily fish count and biomass from output_report.csv files."""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = Path("data/raw/carp/daily/Daily carp sampling")
OUT_DIR = Path("results/daily_stats")

DATE_RE = re.compile(r"ACQ_(\d{2})_(\d{2})_(\d{4})$")


def load_daily():
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

        df = pd.read_csv(report, low_memory=False)
        valid = df[df["is_valid"] == True]
        n_fish = len(valid)
        biomass_kg = valid["weight"].sum() / 1000.0

        median_mass_g = valid["weight"].median() if n_fish > 0 else float("nan")
        rows.append({"date": date, "n_fish": n_fish, "biomass_kg": biomass_kg,
                     "median_mass_g": median_mass_g})

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def plot(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle("Daily carp sampling — Feb–Mar 2026", fontsize=14, fontweight="bold")

    date_fmt = mdates.DateFormatter("%d %b")

    # ── Fish count ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(df["date"], df["n_fish"], width=0.7, color="#2196F3", edgecolor="white")
    ax.set_ylabel("Fish count (valid detections)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ── Biomass ───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(df["date"], df["biomass_kg"], width=0.7, color="#4CAF50", edgecolor="white")
    ax.set_ylabel("Total biomass (kg)")
    ax.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ── Avg fish mass ─────────────────────────────────────────────────────────
    rolling = (df.set_index("date")["median_mass_g"]
                 .rolling("7D", min_periods=1).median()
                 .reset_index())
    ax = axes[2]
    ax.plot(df["date"], df["median_mass_g"], color="#FF5722", linewidth=1.2,
            marker="o", markersize=4, alpha=0.5, zorder=3, label="Daily")
    ax.plot(rolling["date"], rolling["median_mass_g"], color="#B71C1C", linewidth=2.5,
            zorder=4, label="7-day rolling median")
    ax.set_ylabel("Median fish mass (g)")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ── Shared x-axis ─────────────────────────────────────────────────────────
    axes[2].xaxis.set_major_formatter(date_fmt)
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    out = OUT_DIR / "daily_fish_count_and_biomass.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")



def save_csv(df, path):
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_daily()
    print(df.to_string(index=False))
    save_csv(df, OUT_DIR / "daily_gt_stats.csv")
    plot(df)
