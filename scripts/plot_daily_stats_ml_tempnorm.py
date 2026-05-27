"""ML daily stats with temperature-normalized weight estimates.

Correction: water conductivity ≈ 2% per °C, so predicted weight scales with
conductivity. We multiply each day's weights by (σ_ref / σ_day) to bring all
days to a common reference temperature.

  correction = (1 + 0.02*(T_ref - T_day))

where T_ref is the mean temperature across all days with measurements.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_ROOT   = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "results" / "daily_stats"

TEMP_CSV = _ROOT / "data" / "daily_temperature.csv"
ML_CSV   = OUT_DIR / "daily_ml_stats.csv"
ALPHA    = 0.02   # conductivity change per °C


def load_and_normalize():
    ml = pd.read_csv(ML_CSV, parse_dates=["date"])

    temp = pd.read_csv(TEMP_CSV)
    temp["date"] = pd.to_datetime(temp["date"], dayfirst=True)

    df = ml.merge(temp, on="date", how="left")

    # Fill missing temps (Feb 1, Mar 5–11) by linear interpolation
    df = df.set_index("date").sort_index()
    df["temp_c"] = df["temp_c"].interpolate(method="time")
    df = df.reset_index()

    T_ref = df["temp_c"].mean()
    print(f"Reference temperature: {T_ref:.2f}°C")
    print(f"Temperature range: {df['temp_c'].min():.1f}–{df['temp_c'].max():.1f}°C")

    correction = 1 + ALPHA * (T_ref - df["temp_c"])
    df["avg_mass_g_norm"] = df["avg_mass_g"] * correction
    df["biomass_kg_norm"]  = df["biomass_kg"]  * correction

    print("\nCorrection factors:")
    for _, row in df.iterrows():
        print(f"  {row['date'].date()}  T={row['temp_c']:.1f}°C  "
              f"factor={1 + ALPHA*(T_ref - row['temp_c']):.3f}  "
              f"avg: {row['avg_mass_g']:.0f} -> {row['avg_mass_g_norm']:.0f} g")

    return df, T_ref


def plot(df, T_ref):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"ML inference — temperature-normalised weight  "
        f"(α=2%/°C, T_ref={T_ref:.1f}°C)",
        fontsize=13, fontweight="bold")

    date_fmt = mdates.DateFormatter("%d %b")

    # ── Biomass ───────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(df["date"], df["biomass_kg"],      width=0.7, color="#4CAF50",
           alpha=0.4, label="Raw", edgecolor="none")
    ax.bar(df["date"], df["biomass_kg_norm"], width=0.7, color="#1B5E20",
           alpha=0.85, label="Temp-normalised", edgecolor="none")
    ax.set_ylabel("Total biomass (kg)")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # ── Avg fish mass ─────────────────────────────────────────────────────────
    rolling_raw  = (df.set_index("date")["avg_mass_g"]
                      .rolling("7D", min_periods=1).mean().reset_index())
    rolling_norm = (df.set_index("date")["avg_mass_g_norm"]
                      .rolling("7D", min_periods=1).mean().reset_index())

    ax = axes[1]
    ax.plot(df["date"], df["avg_mass_g"],      color="#FF5722", linewidth=1,
            marker="o", markersize=3, alpha=0.4, label="Raw daily")
    ax.plot(df["date"], df["avg_mass_g_norm"], color="#B71C1C", linewidth=1,
            marker="o", markersize=3, alpha=0.4, label="Norm daily")
    ax.plot(rolling_raw["date"],  rolling_raw["avg_mass_g"],
            color="#FF5722", linewidth=2, linestyle="--", label="Raw 7-day avg")
    ax.plot(rolling_norm["date"], rolling_norm["avg_mass_g_norm"],
            color="#B71C1C", linewidth=2.5, label="Norm 7-day avg")
    ax.set_ylabel("Avg fish mass (g)")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    axes[1].xaxis.set_major_formatter(date_fmt)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    out = OUT_DIR / "daily_ml_tempnorm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")



if __name__ == "__main__":
    df, T_ref = load_and_normalize()
    df.to_csv(OUT_DIR / "daily_ml_stats_tempnorm.csv", index=False, float_format="%.4f")
    plot(df, T_ref)
