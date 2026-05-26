"""Compare 3 weight models on salted-water carp recordings (full two-stage pipeline).

Presence model: checkpoints/bream_carp_presence  (fixed for all runs)
Weight models:
  1. No salinity    - checkpoints/carp_weight_combined_model   (12-ch)
  2. Filename sal.  - checkpoints/carp_weight_salinity_model   (12-ch + S from filename)
  3. Baseline RMS   - checkpoints/carp_weight_baseline_model   (12-ch + recording RMS)
"""
import re, sys
import numpy as np, torch, yaml, pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_model import FishModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACOUSTIC_CHANNELS = ["F15","F37","F18","F32","F45","F67","B15","B37","B18","B32","B45","B67"]
WINDOW_SIZE = 39; STRIDE = 5; MERGE_GAP = 39; MATCH_RADIUS = 78; N_CH = 12
THRESHOLD = 0.5

# Both the main dir and the test subdir
SOURCES = [
    (Path("data/raw/carp/carp_salted/salted_water"),
     Path("data/raw/carp/carp_salted/salted_water/output_report.csv")),
    (Path("data/raw/carp/carp_salted/salted_water/test"),
     Path("data/raw/carp/carp_salted/salted_water/test/output_report.csv")),
]


def load_ensemble(ckpt_dir, cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(Path(ckpt_dir).glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(DEVICE)
        ck = torch.load(pt, map_location=DEVICE)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)
    return models


def run_presence(csv_path, presence_models):
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype("float32")
    mu = np.nanmean(raw, axis=0); s = np.nanstd(raw, axis=0); s[s < 1e-8] = 1.0
    norm = ((raw - mu) / s).astype("float32")

    starts = list(range(0, len(norm) - WINDOW_SIZE + 1, STRIDE))
    if not starts:
        return [], raw, mu, s
    wins = np.stack([norm[i:i + WINDOW_SIZE] for i in starts])

    pp = []
    with torch.no_grad():
        for m in presence_models:
            logits, _ = m(torch.from_numpy(wins).to(DEVICE))
            pp.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.mean(pp, axis=0)

    pos_idx = np.where(probs >= THRESHOLD)[0]
    if len(pos_idx) == 0:
        return [], raw, mu, s

    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= MERGE_GAP:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)

    detections = []
    for g in groups:
        pk = g[int(np.argmax(probs[g]))]
        detections.append({"start": starts[pk], "prob": float(probs[pk])})

    return detections, raw, mu, s


def predict_weights(detections, raw, mu, s, weight_models, variant, fname):
    """Add weight prediction to each detection. Mutates detections in-place."""
    if not detections:
        return

    # Build the extra channel scalar for variants 1 and 2
    extra_scalar = None
    if variant == "filename_sal":
        sal_norm = np.load("data/carp/weight_salinity/salinity_norm.npy")
        sal_mean, sal_std = float(sal_norm[0]), float(sal_norm[1])
        sal_val = float(re.search(r"_S(\d+)", fname).group(1)) if re.search(r"_S(\d+)", fname) else 0.0
        extra_scalar = (sal_val - sal_mean) / sal_std
    elif variant == "baseline_rms":
        bl_norm = np.load("data/carp/weight_baseline/baseline_norm.npy")
        bl_mean, bl_std = float(bl_norm[0]), float(bl_norm[1])
        channel_rms = np.sqrt(np.mean(raw ** 2, axis=0))
        rms_val = float(np.mean(channel_rms))
        extra_scalar = (rms_val - bl_mean) / bl_std

    norm = ((raw - mu) / s).astype("float32")

    for det in detections:
        ps = det["start"]
        win = norm[ps: ps + WINDOW_SIZE]
        if len(win) < WINDOW_SIZE:
            win = np.concatenate([win, np.zeros((WINDOW_SIZE - len(win), N_CH), "float32")])

        if extra_scalar is not None:
            extra_col = np.full((WINDOW_SIZE, 1), extra_scalar, dtype="float32")
            win = np.concatenate([win, extra_col], axis=1)  # (39, 13)

        wp = []
        with torch.no_grad():
            t = torch.from_numpy(win[None]).to(DEVICE)
            for m in weight_models:
                _, p = m(t)
                wp.append(p.item())
        det["weight"] = max(100.0, float(np.mean(wp)))


def evaluate_file(csv_path, presence_models, weight_models, variant, truth_events):
    """Returns per-file metrics dict."""
    detections, raw, mu, s = run_presence(csv_path, presence_models)
    predict_weights(detections, raw, mu, s, weight_models, variant, csv_path.name)

    used = set()
    tp_dets = []
    for det in detections:
        center = det["start"] + WINDOW_SIZE // 2
        best_d, best_t = float("inf"), -1
        for ti, (tc, _) in enumerate(truth_events):
            if ti in used:
                continue
            dist = abs(center - tc)
            if dist < best_d:
                best_d, best_t = dist, ti
        if best_t >= 0 and best_d <= MATCH_RADIUS:
            used.add(best_t)
            tp_dets.append((det["weight"], truth_events[best_t][1]))

    tp = len(tp_dets)
    fp = len(detections) - tp
    fn = len(truth_events) - tp

    true_biomass = sum(w for _, w in truth_events)
    pred_biomass = sum(d["weight"] for d in detections)

    # Weight accuracy only on TP matched events
    tp_mae  = np.mean([abs(pw - tw) for pw, tw in tp_dets]) if tp_dets else float("nan")
    tp_mape = np.mean([abs(pw - tw) / tw * 100 for pw, tw in tp_dets]) if tp_dets else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_truth": len(truth_events), "n_det": len(detections),
        "true_biomass": true_biomass, "pred_biomass": pred_biomass,
        "tp_mae": tp_mae, "tp_mape": tp_mape,
        "tp_dets": tp_dets,
    }


def collect_truth():
    """Collect ground-truth events from all sources, deduplicating recordings seen in multiple sources."""
    truth = {}  # resolved_path_str -> list of (mid, weight_g)
    seen_keys = set()
    for rec_dir, report_path in SOURCES:
        if not report_path.exists():
            continue
        df = pd.read_csv(report_path)
        # Group all events per filename first, then resolve path
        for fname, grp in df.groupby("raw_data_file_name"):
            csv_p = rec_dir / fname
            if not csv_p.exists():
                csv_p = rec_dir.parent / fname
            if not csv_p.exists():
                continue
            key = str(csv_p.resolve())
            if key in seen_keys:
                continue  # same physical file already added from another source
            seen_keys.add(key)
            wm = re.search(r"(\d+)gr", fname)
            if wm is None:
                continue
            w = float(wm.group(1))
            events = []
            for _, row in grp.iterrows():
                mid = int((row["start_index"] + row["end_index"]) // 2)
                events.append((mid, w))
            truth[key] = events
    return truth


def run_variant(label, ckpt_dir, cfg_path, variant, presence_models, truth):
    weight_models = load_ensemble(ckpt_dir, cfg_path)
    print(f"\n{'='*70}")
    print(f"  {label}  ({len(weight_models)} weight models)")
    print(f"{'='*70}")
    print(f"  {'File':<42} {'True':>5} {'Det':>4} {'TP':>3} {'FP':>3} {'FN':>3}"
          f"  {'TrueB':>7} {'PredB':>7} {'BDiff':>7}  {'MAE':>6} {'MAPE':>6}")
    print("  " + "-" * 100)

    totals = defaultdict(float)
    all_tp_dets = []

    all_csv = {}
    for rec_dir, _ in SOURCES:
        for csv in sorted(rec_dir.glob("*.csv")):
            if "output" in csv.name.lower() or "report" in csv.name.lower():
                continue
            key = str(csv.resolve())
            if key not in all_csv:
                all_csv[key] = csv

    for key, csv in sorted(all_csv.items()):
        # skip non-recording utility CSVs
        if any(x in csv.name.lower() for x in ("output", "report", "axis", "summary", "function")):
            continue
        te = truth.get(key, [])
        m = evaluate_file(csv, presence_models, weight_models, variant, te)

        bd = (m["pred_biomass"] - m["true_biomass"]) / m["true_biomass"] * 100 if m["true_biomass"] > 0 else float("nan")
        mae_s  = f"{m['tp_mae']:>5.0f}g" if m['tp_mae'] == m['tp_mae'] else "   n/a"
        mape_s = f"{m['tp_mape']:>5.1f}%"  if m['tp_mape'] == m['tp_mape'] else "   n/a"
        bd_s   = f"{bd:>+.0f}%"            if bd == bd else "  n/a"

        print(f"  {csv.name:<42} {m['n_truth']:>5} {m['n_det']:>4} {m['tp']:>3} {m['fp']:>3} {m['fn']:>3}"
              f"  {m['true_biomass']:>7.0f} {m['pred_biomass']:>7.0f} {bd_s:>7}  {mae_s} {mape_s}")

        totals["tp"]           += m["tp"]
        totals["fp"]           += m["fp"]
        totals["fn"]           += m["fn"]
        totals["n_truth"]      += m["n_truth"]
        totals["n_det"]        += m["n_det"]
        totals["true_biomass"] += m["true_biomass"]
        totals["pred_biomass"] += m["pred_biomass"]
        all_tp_dets.extend(m["tp_dets"])

    tp, fp, fn = int(totals["tp"]), int(totals["fp"]), int(totals["fn"])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    tb, pb = totals["true_biomass"], totals["pred_biomass"]
    bio_diff = (pb - tb) / tb * 100 if tb > 0 else float("nan")

    overall_mae  = np.mean([abs(pw - tw) for pw, tw in all_tp_dets]) if all_tp_dets else float("nan")
    overall_mape = np.mean([abs(pw - tw) / tw * 100 for pw, tw in all_tp_dets]) if all_tp_dets else float("nan")

    print()
    print(f"  Events:   truth={int(totals['n_truth'])}  detected={int(totals['n_det'])}  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Presence: precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(f"  Weight (TP only):  MAE={overall_mae:.0f}g  MAPE={overall_mape:.1f}%")
    print(f"  Biomass:  true={tb:.0f}g  pred={pb:.0f}g  diff={bio_diff:+.1f}%")

    return {
        "prec": prec, "rec": rec, "f1": f1,
        "mae": overall_mae, "mape": overall_mape, "bio_diff": bio_diff,
    }


def main():
    print("Loading presence ensemble ...")
    presence_models = load_ensemble("checkpoints/bream_carp_presence",
                                    "model/model_config_12ch.yaml")
    print(f"  {len(presence_models)} presence models loaded")

    truth = collect_truth()
    print(f"  {len(truth)} recordings with ground-truth events")

    variants = [
        ("No salinity",   "checkpoints/carp_weight_combined_model",
         "model/model_config_12ch.yaml",          "no_sal"),
        ("Filename sal.", "checkpoints/carp_weight_salinity_model",
         "model/model_config_12ch_salinity.yaml", "filename_sal"),
        ("Baseline RMS",  "checkpoints/carp_weight_baseline_model",
         "model/model_config_12ch_salinity.yaml", "baseline_rms"),
    ]

    summary = {}
    for label, ckpt, cfg, variant in variants:
        summary[label] = run_variant(label, ckpt, cfg, variant, presence_models, truth)

    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<18}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'MAE':>7}  {'MAPE':>6}  {'BioDiff':>8}")
    print("  " + "-" * 72)
    for label, r in summary.items():
        print(f"  {label:<18}  {r['prec']:>6.3f}  {r['rec']:>6.3f}  {r['f1']:>6.3f}"
              f"  {r['mae']:>6.0f}g  {r['mape']:>5.1f}%  {r['bio_diff']:>+7.1f}%")


if __name__ == "__main__":
    main()
