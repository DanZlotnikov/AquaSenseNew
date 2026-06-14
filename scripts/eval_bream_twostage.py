"""Evaluate the bream two-stage model on held-out test recordings.

Stage 1 — bream_presence_v1 ensemble (merge_gap=20)
Stage 2 — bream_weight_model ensemble

Reports:
  - Per-file fish count accuracy (GT vs ML)
  - Weight MAE / MAPE on all detected events (GT weight from filename)
  - Comparison: carp model (gap=79) vs bream model (gap=20) on same files
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from model.fish_model import FishModel

REPORT_CSV  = _ROOT / "data" / "raw" / "bream" / "output_report.csv"
REC_DIR     = _ROOT / "data" / "raw" / "bream" / "recordings"

BREAM_PRES_DIR   = _ROOT / "checkpoints" / "bream_presence_v1"
BREAM_WEIGHT_DIR = _ROOT / "checkpoints" / "bream_weight_model"
CARP_PRES_DIR    = _ROOT / "checkpoints" / "carp_presence_v5_knockdown"
CFG_PATH         = _ROOT / "model" / "model_config_12ch.yaml"

OUT_TXT = _ROOT / "data" / "raw" / "bream" / "bream_twostage_eval.txt"

SIGNAL_CHANNELS = ["F15","F37","F18","F32","F45","F67",
                   "B15","B37","B18","B32","B45","B67"]
SKIP = ("output","report","axis","summary","function","points","updated")

WINDOW_SIZE      = 39
STRIDE           = 5
SIGMA_FLOOR      = 150.0
THRESHOLD        = 0.5
MERGE_GAP_BREAM  = 20   # tuned for bream crossing density
MERGE_GAP_CARP   = 79   # original carp parameter
VALLEY_THRESHOLD = 0.25  # min prob in valley to merge two adjacent groups (large-fish crossing dip)
MAX_VALLEY_GAP   = 60    # max sample gap to even attempt valley merge (~1.5s at 40Hz)
BATCH_SIZE       = 512

# Test files — must match build_bream_presence_v1.py
TEST_FILES = {
    "ARDG_AQUA bream 150gr 21 sm  281025 1132002.csv",
    "ARDG_AQUA bream 390gr 26cm  281025 1325003.csv",
    "ARDG_AQUA bream620 gr31cm  281025 1618003.csv",
    "ARDG_AQUA bream 680gr 33 sm  281025 1208003.csv",
    "ARDG_AQUA bream 800gr 36cm  281025 1453002.csv",
}


def parse_weight_g(fname: str) -> float | None:
    m = re.search(r"(\d+)\s*gr", fname, re.IGNORECASE)
    return float(m.group(1)) if m else None


def load_ensemble(ckpt_dir: Path, device: torch.device) -> list:
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg.setdefault("physics_dim", 0)
    cfg.setdefault("coact_gate", False)
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


def normalize(raw: np.ndarray) -> np.ndarray:
    mu        = np.nanmean(raw, axis=0)
    sigma     = np.nanstd(raw,  axis=0)
    sigma_eff = np.maximum(sigma, SIGMA_FLOOR)
    sigma_eff[np.isnan(sigma_eff)] = SIGMA_FLOOR
    mu = np.where(np.isnan(mu), 0.0, mu)
    return np.where(np.isnan((raw - mu) / sigma_eff), 0.0,
                    (raw - mu) / sigma_eff).astype(np.float32)


def get_probs(csv_path: Path, pres_models: list, device: torch.device):
    df = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw    = df[SIGNAL_CHANNELS].values.astype(np.float32)
    norm   = normalize(raw)
    starts = list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE))
    if not starts:
        return np.array([]), np.array([])
    wins   = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
    logits = np.zeros(len(wins), np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), BATCH_SIZE):
            X = torch.from_numpy(wins[b: b + BATCH_SIZE]).to(device)
            s = None
            for m in pres_models:
                lv, _ = m(X); lv = lv.cpu().numpy()
                s = lv if s is None else s + lv
            s /= len(pres_models)
            logits[b: b + len(s)] = s
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return probs, np.array(starts)


def nms(probs: np.ndarray, starts: np.ndarray, merge_gap: int) -> list[dict]:
    pos_idx = [i for i, p in enumerate(probs) if p >= THRESHOLD]
    if not pos_idx:
        return []

    # Pass 1: standard gap-based grouping
    groups, cur = [], [pos_idx[0]]
    for i in pos_idx[1:]:
        if starts[i] - starts[cur[-1]] <= merge_gap:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)

    # Pass 2: valley-aware merge — if the probability in the gap between two
    # adjacent groups never fully returns to zero, it's one crossing with a
    # mid-body dip (large fish), not two separate fish.
    merged = [groups[0]]
    for g in groups[1:]:
        prev = merged[-1]
        prev_end = starts[prev[-1]]
        g_start  = starts[g[0]]
        gap      = int(g_start) - int(prev_end)
        if gap <= MAX_VALLEY_GAP:
            valley_mask = (starts > prev_end) & (starts < g_start)
            valley_probs = probs[valley_mask]
            valley_min = float(valley_probs.min()) if len(valley_probs) > 0 else 0.0
            if valley_min >= VALLEY_THRESHOLD:
                merged[-1] = prev + g
                continue
        merged.append(g)

    detections = []
    for grp in merged:
        peak = grp[int(np.argmax(probs[grp]))]
        detections.append({"peak_start": int(starts[peak]),
                           "peak_prob":  float(probs[peak])})
    return detections


def predict_weight(peak_start: int, norm: np.ndarray,
                   w_models: list, device: torch.device) -> float:
    seg = norm[peak_start: peak_start + WINDOW_SIZE]
    if len(seg) < WINDOW_SIZE:
        seg = np.concatenate([seg, np.zeros((WINDOW_SIZE - len(seg),
                                             len(SIGNAL_CHANNELS)), np.float32)])
    t = torch.from_numpy(seg[np.newaxis]).to(device)
    preds = []
    with torch.no_grad():
        for m in w_models:
            _, pw = m(t); preds.append(pw.item())
    return float(np.mean(preds))


def run_on_file(csv_path, pres_models, weight_models, device, merge_gap):
    df  = pd.read_csv(csv_path)
    for col in SIGNAL_CHANNELS:
        if col not in df.columns:
            df[col] = 0.0
    raw  = df[SIGNAL_CHANNELS].values.astype(np.float32)
    norm = normalize(raw)

    starts = np.array(list(range(0, max(0, len(norm) - WINDOW_SIZE + 1), STRIDE)))
    if len(starts) == 0:
        return [], 0.0

    wins   = np.stack([norm[i: i + WINDOW_SIZE] for i in starts])
    logits = np.zeros(len(wins), np.float32)
    with torch.no_grad():
        for b in range(0, len(wins), BATCH_SIZE):
            X = torch.from_numpy(wins[b: b + BATCH_SIZE]).to(device)
            s = None
            for m in pres_models:
                lv, _ = m(X); lv = lv.cpu().numpy()
                s = lv if s is None else s + lv
            s /= len(pres_models)
            logits[b: b + len(s)] = s

    probs      = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    detections = nms(probs, starts, merge_gap)

    weights = []
    for det in detections:
        w = predict_weight(det["peak_start"], norm, weight_models, device)
        weights.append(max(50.0, w))

    return weights, float(probs.max()) if len(probs) else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading bream presence ensemble...")
    bream_pres = load_ensemble(BREAM_PRES_DIR, device)
    print(f"  {len(bream_pres)} model(s)")

    print("Loading bream weight ensemble...")
    bream_w = load_ensemble(BREAM_WEIGHT_DIR, device)
    print(f"  {len(bream_w)} model(s)")

    print("Loading carp presence ensemble (for comparison)...")
    carp_pres = load_ensemble(CARP_PRES_DIR, device)
    print(f"  {len(carp_pres)} model(s)")

    report = pd.read_csv(REPORT_CSV)
    valid  = report[report["is_valid"] == True]
    gt_map = valid.groupby("raw_data_file_name").size().to_dict()

    test_files = sorted(f for f in REC_DIR.glob("*.csv")
                        if f.name in TEST_FILES
                        and not any(x in f.name.lower() for x in SKIP))

    if not test_files:
        print("ERROR: No test files found. Check TEST_FILES set and REC_DIR.")
        return

    print(f"\nEvaluating {len(test_files)} test files...\n")

    rows = []
    all_pred_weights = []
    all_true_weights = []

    for f in test_files:
        gt_n     = gt_map.get(f.name, 0)
        gt_w     = parse_weight_g(f.name)

        # Bream model (gap=20)
        b_weights, b_maxp = run_on_file(f, bream_pres, bream_w, device, MERGE_GAP_BREAM)
        b_n = len(b_weights)

        # Carp model count only (gap=20 and gap=79 for comparison)
        probs, starts = get_probs(f, carp_pres, device)
        c20_n = len(nms(probs, starts, 20))
        c79_n = len(nms(probs, starts, 79))

        b_diff  = b_n - gt_n
        b_pct   = (b_diff / gt_n * 100) if gt_n > 0 else (float("inf") if b_n > 0 else 0.0)

        # Weight accuracy
        w_mae, w_mape = None, None
        if gt_w and b_weights:
            errs  = [abs(w - gt_w) for w in b_weights]
            w_mae  = np.mean(errs)
            w_mape = np.mean([e / gt_w * 100 for e in errs])
            all_pred_weights.extend(b_weights)
            all_true_weights.extend([gt_w] * len(b_weights))

        rows.append((f.name, gt_n, b_n, b_diff, b_pct, c20_n, c79_n,
                     gt_w, w_mae, w_mape, b_maxp))

        pct_str = f"{b_pct:+.1f}%" if b_pct != float("inf") else "inf%"
        w_str   = f"MAE={w_mae:.0f}g MAPE={w_mape:.1f}%" if w_mae else "-"
        print(f"  {f.name}")
        print(f"    Count: GT={gt_n}  Bream-model={b_n} ({pct_str})  "
              f"Carp-gap20={c20_n}  Carp-gap79={c79_n}")
        print(f"    Weight({gt_w:.0f}g actual): {w_str}  max_p={b_maxp:.3f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_gt = sum(r[1] for r in rows)
    total_b  = sum(r[2] for r in rows)
    total_c20 = sum(r[6] for r in rows)  # c20_n is index 6, c79_n is index 5
    # wait, let me recheck: rows tuple = (fname, gt_n, b_n, b_diff, b_pct, c20_n, c79_n, ...)
    total_c79 = sum(r[6] for r in rows)
    total_c20 = sum(r[5] for r in rows)

    overall_b_pct = (total_b - total_gt) / total_gt * 100 if total_gt else 0

    global_mae  = np.mean([abs(p - t) for p, t in zip(all_pred_weights, all_true_weights)]) if all_pred_weights else float("nan")
    global_mape = np.mean([abs(p - t) / t * 100 for p, t in zip(all_pred_weights, all_true_weights)]) if all_pred_weights else float("nan")
    global_bias = np.mean([p - t for p, t in zip(all_pred_weights, all_true_weights)]) if all_pred_weights else float("nan")

    lines = [
        "=" * 72,
        "Bream two-stage model evaluation — held-out test set",
        f"Presence model : bream_presence_v1 ensemble ({len(bream_pres)} models), merge_gap={MERGE_GAP_BREAM}",
        f"Weight model   : bream_weight_model ensemble ({len(bream_w)} models)",
        f"Carp reference : carp_presence_v5_knockdown (gap=20 and gap=79)",
        "=" * 72,
        "",
        f"{'File':<50}  {'GT':>4}  {'Bream':>5}  {'Diff%':>7}  {'C-20':>5}  {'C-79':>5}  {'W-MAE':>7}  {'W-MAPE':>7}",
        "-" * 100,
    ]
    for fname, gt_n, b_n, b_diff, b_pct, c20_n, c79_n, gt_w, w_mae, w_mape, _ in rows:
        pct_str = f"{b_pct:+.1f}%" if b_pct != float("inf") else "  inf%"
        w_str   = f"{w_mae:>6.0f}g" if w_mae else "     -"
        mp_str  = f"{w_mape:>6.1f}%" if w_mape else "      -"
        lines.append(f"  {fname:<50}  {gt_n:>4}  {b_n:>5}  {pct_str:>7}  {c20_n:>5}  {c79_n:>5}  {w_str}  {mp_str}")

    lines += [
        "-" * 100,
        f"  {'TOTAL':<50}  {total_gt:>4}  {total_b:>5}  {overall_b_pct:>+6.1f}%  {total_c20:>5}  {total_c79:>5}",
        "",
        "Weight regression (all detected events vs GT weight from filename):",
        f"  Detections used : {len(all_pred_weights)}",
        f"  Global MAE      : {global_mae:.1f} g",
        f"  Global MAPE     : {global_mape:.1f} %",
        f"  Global bias     : {global_bias:+.1f} g  ({'over' if global_bias > 0 else 'under'}-predicting)",
    ]

    out = "\n".join(lines)
    OUT_TXT.write_text(out, encoding="utf-8")
    print("\n" + out)
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
