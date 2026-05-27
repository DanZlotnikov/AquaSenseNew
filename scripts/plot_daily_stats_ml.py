"""Two-stage ML inference on daily carp folders → fish count / biomass / avg mass plots."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from model.fish_model import FishModel

# ── Paths ──────────────────────────────────────────────────────────────────────
DAILY_BASE        = _ROOT / "data" / "raw" / "carp" / "daily" / "Daily carp sampling"
PRESENCE_CKPT_DIR = _ROOT / "checkpoints" / "carp_presence_v5_knockdown"
WEIGHT_CKPT_DIR   = _ROOT / "checkpoints" / "carp_weight_combined_model"
MODEL_CFG_PATH    = _ROOT / "model" / "model_config_12ch.yaml"
PLATT_PATH        = PRESENCE_CKPT_DIR / "platt.npy"
OUT_DIR           = _ROOT / "results" / "daily_stats"

# ── Inference constants ────────────────────────────────────────────────────────
WINDOW_SIZE     = 39
STRIDE          = 5
MERGE_GAP       = 79
THRESHOLD       = 0.5
SIGMA_FLOOR     = 150.0
BATCH_SIZE      = 512
SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]
DATE_RE         = re.compile(r"ACQ_(\d{2})_(\d{2})_(\d{4})$")
SKIP            = ("output","report","axis","summary","function","points","updated")


def load_ensemble(ckpt_dir: Path, cfg_path: Path, device: torch.device) -> list:
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(ckpt_dir.glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(device)
        ck = torch.load(pt, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
    return models


def load_platt():
    if PLATT_PATH.exists():
        ab = np.load(str(PLATT_PATH))
        return float(ab[0]), float(ab[1])
    return 1.0, 0.0


def normalize(raw: np.ndarray) -> np.ndarray:
    mu        = np.nanmean(raw, axis=0).astype(np.float32)
    s         = np.nanstd(raw,  axis=0).astype(np.float32)
    sigma_eff = np.maximum(s, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    out = (raw - mu) / sigma_eff
    return np.where(np.isnan(out), 0.0, out).astype(np.float32)


def score_presence(wins, models, device, platt_a, platt_b):
    logits = np.zeros(len(wins), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), BATCH_SIZE):
            X = torch.from_numpy(wins[b: b + BATCH_SIZE]).to(device)
            s = None
            for m in models:
                lv, _ = m(X)
                lv = lv.cpu().numpy()
                s = lv if s is None else s + lv
            s /= len(models)
            logits[b: b + len(s)] = s
    return (1.0 / (1.0 + np.exp(-(platt_a * logits + platt_b)))).astype(np.float32)


def nms(starts, probs):
    pos_idx = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    if not pos_idx:
        return []
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= MERGE_GAP:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    detections = []
    for g in groups:
        peak_i = g[int(np.argmax(probs[g]))]
        detections.append(int(starts[peak_i]))
    return detections


def predict_weight(win, models, device):
    t = torch.from_numpy(win[None]).to(device)
    total = 0.0
    with torch.no_grad():
        for m in models:
            _, p = m(t)
            total += p.item()
    return max(100.0, total / len(models))


def infer_folder(folder: Path, presence_models, weight_models, device, platt_a, platt_b):
    csv_files = [f for f in sorted(folder.glob("*.csv"))
                 if not any(x in f.name.lower() for x in SKIP)]
    weights = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, low_memory=False)
        for col in SIGNAL_CHANNELS:
            if col not in df.columns:
                df[col] = 0.0
        raw  = df[SIGNAL_CHANNELS].values.astype(np.float32)
        norm = normalize(raw)

        starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
        if not starts:
            continue
        wins  = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
        probs = score_presence(wins, presence_models, device, platt_a, platt_b)
        peak_starts = nms(starts, probs)

        for ps in peak_starts:
            win = norm[ps: ps + WINDOW_SIZE]
            if len(win) < WINDOW_SIZE:
                win = np.concatenate([win, np.zeros((WINDOW_SIZE - len(win), 12), np.float32)])
            weights.append(predict_weight(win, weight_models, device))

    n_fish        = len(weights)
    biomass_kg    = sum(weights) / 1000.0
    median_mass_g = float(np.median(weights)) if weights else float("nan")
    return n_fish, biomass_kg, median_mass_g, weights


def load_daily(presence_models, weight_models, device, platt_a, platt_b):
    rows, per_fish_rows = [], []
    folders = sorted(
        [f for f in DAILY_BASE.iterdir() if f.is_dir() and DATE_RE.match(f.name)],
        key=lambda f: f.name
    )
    for folder in folders:
        m = DATE_RE.match(folder.name)
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        date = pd.Timestamp(year=year, month=month, day=day)
        print(f"  {folder.name} ...", end=" ", flush=True)
        n_fish, biomass_kg, median_mass_g, weights = infer_folder(
            folder, presence_models, weight_models, device, platt_a, platt_b)
        print(f"{n_fish} fish  {biomass_kg:.1f} kg  median {median_mass_g:.0f} g")
        rows.append({"date": date, "n_fish": n_fish,
                     "biomass_kg": biomass_kg, "median_mass_g": median_mass_g})
        for w in weights:
            per_fish_rows.append({"date": date, "weight_g": w})
    summary = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    per_fish = pd.DataFrame(per_fish_rows)
    return summary, per_fish


def plot(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle("Daily carp sampling — ML two-stage inference (v5_knockdown + weight model)",
                 fontsize=13, fontweight="bold")

    date_fmt = mdates.DateFormatter("%d %b")

    ax = axes[0]
    ax.bar(df["date"], df["n_fish"], width=0.7, color="#2196F3", edgecolor="white")
    ax.set_ylabel("Fish count (ML detections)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    for s in ("top","right"): ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.bar(df["date"], df["biomass_kg"], width=0.7, color="#4CAF50", edgecolor="white")
    ax.set_ylabel("Total biomass (kg)")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top","right"): ax.spines[s].set_visible(False)

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
    for s in ("top","right"): ax.spines[s].set_visible(False)

    axes[2].xaxis.set_major_formatter(date_fmt)
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    out = OUT_DIR / "daily_ml_inference.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading models...")
    platt_a, platt_b = load_platt()
    presence_models  = load_ensemble(PRESENCE_CKPT_DIR, MODEL_CFG_PATH, device)
    weight_models    = load_ensemble(WEIGHT_CKPT_DIR,   MODEL_CFG_PATH, device)
    print(f"  Presence ensemble: {len(presence_models)} models  (Platt a={platt_a}, b={platt_b})")
    print(f"  Weight ensemble:   {len(weight_models)} models")
    print()

    df, per_fish = load_daily(presence_models, weight_models, device, platt_a, platt_b)
    print()
    print(df.to_string(index=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "daily_ml_stats.csv", index=False, float_format="%.4f")
    print(f"Saved -> {OUT_DIR / 'daily_ml_stats.csv'}")
    per_fish.to_csv(OUT_DIR / "daily_ml_per_fish.csv", index=False, float_format="%.4f")
    print(f"Saved -> {OUT_DIR / 'daily_ml_per_fish.csv'}")
    plot(df)
