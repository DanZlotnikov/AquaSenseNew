"""GT daily stats with temperature-normalised weight estimates."""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_ROOT   = Path(__file__).resolve().parent.parent
BASE    = _ROOT / "data" / "raw" / "carp" / "daily" / "Daily carp sampling"
OUT_DIR = _ROOT / "results" / "daily_stats"
TEMP_CSV = _ROOT / "data" / "daily_temperature.csv"

DATE_RE = re.compile(r"ACQ_(\d{2})_(\d{2})_(\d{4})$")
ALPHA   = 0.02


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
        df    = pd.read_csv(report, low_memory=False)
        valid = df[df["is_valid"] == True]
        n_fish     = len(valid)
        biomass_kg = valid["weight"].sum() / 1000.0
        median_mass_g = valid["weight"].median() if n_fish > 0 else float("nan")
        rows.append({"date": date, "n_fish": n_fish,
                     "biomass_kg": biomass_kg, "median_mass_g": median_mass_g})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def apply_temp_norm(df):
    temp = pd.read_csv(TEMP_CSV)
    temp["date"] = pd.to_datetime(temp["date"], dayfirst=True)
    df = df.merge(temp, on="date", how="left")
    df = df.set_index("date").sort_index()
    df["temp_c"] = df["temp_c"].interpolate(method="time")
    df = df.reset_index()
    T_ref = df["temp_c"].mean()
    correction = 1 + ALPHA * (T_ref - df["temp_c"])
    df["median_mass_g_norm"] = df["median_mass_g"] * correction
    df["biomass_kg_norm"]     = df["biomass_kg"]    * correction
    return df, T_ref


def plot(df, T_ref):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"GT detections — temperature-normalised weight  "
        f"(a=2%/degC, T_ref={T_ref:.1f}C)",
        fontsize=13, fontweight="bold")

    date_fmt = mdates.DateFormatter("%d %b")

    ax = axes[0]
    ax.bar(df["date"], df["biomass_kg"],      width=0.7, color="#4CAF50",
           alpha=0.4, label="Raw", edgecolor="none")
    ax.bar(df["date"], df["biomass_kg_norm"], width=0.7, color="#1B5E20",
           alpha=0.85, label="Temp-normalised", edgecolor="none")
    ax.set_ylabel("Total biomass (kg)")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    rolling_raw  = (df.set_index("date")["median_mass_g"]
                      .rolling("7D", min_periods=1).median().reset_index())
    rolling_norm = (df.set_index("date")["median_mass_g_norm"]
                      .rolling("7D", min_periods=1).median().reset_index())

    ax = axes[1]
    ax.plot(df["date"], df["median_mass_g"],      color="#FF5722", linewidth=1,
            marker="o", markersize=3, alpha=0.4, label="Raw daily")
    ax.plot(df["date"], df["median_mass_g_norm"], color="#B71C1C", linewidth=1,
            marker="o", markersize=3, alpha=0.4, label="Norm daily")
    ax.plot(rolling_raw["date"],  rolling_raw["median_mass_g"],
            color="#FF5722", linewidth=2, linestyle="--", label="Raw 7-day median")
    ax.plot(rolling_norm["date"], rolling_norm["median_mass_g_norm"],
            color="#B71C1C", linewidth=2.5, label="Norm 7-day median")
    ax.set_ylabel("Median fish mass (g)")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    axes[1].xaxis.set_major_formatter(date_fmt)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    out = OUT_DIR / "daily_gt_tempnorm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")



if __name__ == "__main__":
    df, T_ref = apply_temp_norm(load_daily())
    print(f"Reference temperature: {T_ref:.2f}C")
    df.to_csv(OUT_DIR / "daily_gt_stats_tempnorm.csv", index=False, float_format="%.4f")
    plot(df, T_ref)
