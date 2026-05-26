"""Run two-stage pipeline on salted water carp recordings and report biomass."""
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

RECORDINGS_DIR = Path("data/raw/carp/carp_salted/salted_water")
REPORT_PATH    = Path("data/raw/carp/carp_salted/salted_water/output_report.csv")

def load_ensemble(ckpt_dir):
    cfg = yaml.safe_load(open("model/model_config_12ch.yaml")); cfg["physics_dim"] = 0
    models = []
    for run in sorted(Path(ckpt_dir).glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists(): continue
        m = FishModel(cfg).to(DEVICE)
        ck = torch.load(pt, map_location=DEVICE)
        m.load_state_dict(ck["model_state"]); m.eval(); models.append(m)
    return models

def run_file(csv_path, pm, wm):
    df = pd.read_csv(csv_path)
    for col in ACOUSTIC_CHANNELS:
        if col not in df.columns: df[col] = 0.0
    raw = df[ACOUSTIC_CHANNELS].values.astype("float32")
    mu = np.nanmean(raw, axis=0); s = np.nanstd(raw, axis=0); s[s < 1e-8] = 1.0
    norm = ((raw - mu) / s).astype("float32")
    starts = list(range(0, len(norm) - WINDOW_SIZE + 1, STRIDE))
    wins = np.stack([norm[i:i + WINDOW_SIZE] for i in starts])
    pp = []
    with torch.no_grad():
        for m in pm:
            logits, _ = m(torch.from_numpy(wins).to(DEVICE))
            pp.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.mean(pp, axis=0)
    pos_idx = np.where(probs >= THRESHOLD)[0]
    if len(pos_idx) == 0:
        return [], (probs >= THRESHOLD).mean()
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= MERGE_GAP: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    dets = []
    for g in groups:
        pk = g[int(np.argmax(probs[g]))]; ps = starts[pk]
        win = norm[ps:ps + WINDOW_SIZE]
        if len(win) < WINDOW_SIZE:
            win = np.concatenate([win, np.zeros((WINDOW_SIZE - len(win), N_CH), "float32")])
        wp = []
        with torch.no_grad():
            for m in wm:
                _, p = m(torch.from_numpy(win[None]).to(DEVICE)); wp.append(p.item())
        dets.append({"center": ps + WINDOW_SIZE // 2, "prob": float(probs[pk]),
                     "weight": max(100., float(np.mean(wp)))})
    return dets, (probs >= THRESHOLD).mean()

def main():
    pm = load_ensemble("checkpoints/bream_carp_presence")
    wm = load_ensemble("checkpoints/carp_weight_combined_model")
    print(f"Presence: {len(pm)} models   Weight: {len(wm)} models")

    report = pd.read_csv(REPORT_PATH)
    truth = defaultdict(list)
    for _, row in report.iterrows():
        fname = row["raw_data_file_name"]
        mid = int((row["start_index"] + row["end_index"]) // 2)
        m = re.search(r"(\d+)gr", fname)
        w = float(m.group(1)) if m else None
        if w: truth[fname].append((mid, w))

    print(f"\n  {'File':<42} {'True':>5} {'Det':>4} {'TP':>3} {'FP':>3} {'FN':>3}"
          f" {'TrueW':>7} {'PredW':>7} {'Diff':>7} {'%>0.5':>6}")
    print("  " + "-" * 90)

    total_true_b = total_pred_b = 0
    total_true_e = total_det = tp_all = fp_all = fn_all = 0

    for csv in sorted(RECORDINGS_DIR.glob("*.csv")):
        if "output" in csv.name.lower(): continue
        dets, frac_pos = run_file(csv, pm, wm)
        true_ev = truth.get(csv.name, [])
        true_b  = sum(e[1] for e in true_ev)
        pred_b  = sum(d["weight"] for d in dets)
        used = set(); tp = fp = 0
        for d in dets:
            best_d, best_t = float("inf"), -1
            for ti, (tc, _) in enumerate(true_ev):
                if ti in used: continue
                dist = abs(d["center"] - tc)
                if dist < best_d: best_d, best_t = dist, ti
            if best_t >= 0 and best_d <= MATCH_RADIUS: tp += 1; used.add(best_t)
            else: fp += 1
        fn   = len(true_ev) - tp
        diff = (pred_b - true_b) / true_b * 100 if true_b > 0 else float("nan")
        diff_s = f"{diff:+.0f}%" if diff == diff else "n/a"
        print(f"  {csv.name:<42} {len(true_ev):>5} {len(dets):>4} {tp:>3} {fp:>3} {fn:>3}"
              f" {true_b:>7.0f} {pred_b:>7.0f} {diff_s:>7} {frac_pos*100:>5.1f}%")
        total_true_b += true_b; total_pred_b += pred_b
        total_true_e += len(true_ev); total_det += len(dets)
        tp_all += tp; fp_all += fp; fn_all += fn

    prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
    rec  = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print()
    print(f"  True events  : {total_true_e}")
    print(f"  Detections   : {total_det}  (TP={tp_all}  FP={fp_all}  FN={fn_all})")
    print(f"  Precision    : {prec:.3f}")
    print(f"  Recall       : {rec:.3f}")
    print(f"  F1           : {f1:.3f}")
    print(f"  True biomass : {total_true_b:.0f} g  ({total_true_b/1000:.2f} kg)")
    print(f"  Pred biomass : {total_pred_b:.0f} g  ({total_pred_b/1000:.2f} kg)")
    print(f"  Biomass diff : {(total_pred_b - total_true_b) / total_true_b * 100:+.1f}%")

if __name__ == "__main__":
    main()
