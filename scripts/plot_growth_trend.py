"""Quantile regression on temperature-normalised per-fish ML weights.

Fits quantile regression at q=0.25, 0.50, 0.75 across all individual fish
detections (not daily aggregates) to estimate population growth rate.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

_ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR     = _ROOT / "results" / "daily_stats"
PER_FISH    = OUT_DIR / "daily_ml_per_fish.csv"
TEMP_CSV    = _ROOT / "data" / "daily_temperature.csv"
ALPHA_COND  = 0.02   # conductivity correction per degC
QUANTILES   = [0.25, 0.50, 0.75]
COLORS      = ["#2196F3", "#B71C1C", "#4CAF50"]
LABELS      = ["25th pct", "50th (median)", "75th pct"]


def load_data():
    fish = pd.read_csv(PER_FISH, parse_dates=["date"])

    # Build a per-date temperature table covering all sampling days, then interpolate
    temp = pd.read_csv(TEMP_CSV)
    temp["date"] = pd.to_datetime(temp["date"], dayfirst=True)
    all_dates = pd.DataFrame({"date": sorted(fish["date"].unique())})
    temp_full = (all_dates.merge(temp, on="date", how="left")
                          .set_index("date")
                          .sort_index())
    temp_full["temp_c"] = temp_full["temp_c"].interpolate(method="index")
    temp_full = temp_full.reset_index()

    T_ref = temp_full["temp_c"].mean()
    temp_full["correction"] = 1 + ALPHA_COND * (T_ref - temp_full["temp_c"])

    fish = fish.merge(temp_full[["date", "correction"]], on="date", how="left")
    fish["weight_norm"] = fish["weight_g"] * fish["correction"]
    fish = fish.dropna(subset=["weight_norm"])
    fish["day_num"] = (fish["date"] - fish["date"].min()).dt.days
    print(f"  After temp-norm: {len(fish):,} fish, "
          f"correction range {fish['correction'].min():.3f}-{fish['correction'].max():.3f}")
    return fish, T_ref


def fit_qr(fish, q):
    X = sm.add_constant(fish["day_num"].values.astype(float))
    y = fish["weight_norm"].values.astype(float)
    result = QuantReg(y, X).fit(q=q, max_iter=2000)
    ci = result.conf_int(alpha=0.05)
    slope    = result.params[1]
    slope_lo = ci[1, 0]
    slope_hi = ci[1, 1]
    return slope, slope_lo, slope_hi, result.params[0]


def plot(fish, T_ref):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Daily quantiles for dots
    daily = (fish.groupby("date")["weight_norm"]
               .quantile(QUANTILES)
               .unstack()
               .sort_index())
    daily.columns = QUANTILES

    day_min = int(fish["day_num"].min())
    day_max = int(fish["day_num"].max())
    day_range = np.array([day_min, day_max])
    t0 = fish["date"].min()
    date_range = [t0 + pd.Timedelta(days=int(d)) for d in day_range]

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.suptitle(
        f"ML weight quantile growth trends — temp-normalised  (T_ref={T_ref:.1f}C)",
        fontsize=13, fontweight="bold")

    # IQR shading
    ax.fill_between(daily.index, daily[0.25], daily[0.75],
                    alpha=0.1, color="#888", label="Daily IQR")

    for q, color, label in zip(QUANTILES, COLORS, LABELS):
        slope, lo, hi, intercept = fit_qr(fish, q)

        # Daily quantile line
        ax.plot(daily.index, daily[q], "o-", color=color,
                markersize=3, linewidth=1, alpha=0.45)

        # Regression line
        y_fit = intercept + slope * day_range
        ax.plot(date_range, y_fit, "-", color=color, linewidth=2.5,
                label=f"{label}:  {slope:+.2f} g/day  [{lo:+.2f}, {hi:+.2f}]")

    ax.set_xlabel("Date")
    ax.set_ylabel("Temp-normalised weight (g)")
    ax.legend(fontsize=9, framealpha=0.85, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "ml_weight_growth_trend.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")



if __name__ == "__main__":
    fish, T_ref = load_data()
    print(f"Loaded {len(fish):,} fish across {fish['date'].nunique()} days")
    print(f"T_ref={T_ref:.2f}C  weight range: "
          f"{fish['weight_norm'].min():.0f}-{fish['weight_norm'].max():.0f} g")
    print("Fitting quantile regressions...")
    for q, label in zip(QUANTILES, LABELS):
        slope, lo, hi, _ = fit_qr(fish, q)
        print(f"  {label}: {slope:+.3f} g/day  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print()
    plot(fish, T_ref)
