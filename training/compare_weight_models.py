"""Compare carp weight models: no salinity / filename salinity / baseline RMS."""
import sys, numpy as np, torch, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.fish_model import FishModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_model(ckpt_dir, cfg_path, test_npz):
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("physics_dim", 0)
    models = []
    for run in sorted(Path(ckpt_dir).glob("2*")):
        pt = run / "best_model.pt"
        if not pt.exists(): continue
        m = FishModel(cfg).to(DEVICE)
        ck = torch.load(pt, map_location=DEVICE)
        m.load_state_dict(ck["model_state"]); m.eval(); models.append(m)
    data = np.load(test_npz)
    X = torch.from_numpy(data["X"]).to(DEVICE)
    y = data["y_length"]
    preds = []
    with torch.no_grad():
        for m in models:
            _, p = m(X); preds.append(p.cpu().numpy())
    yp = np.mean(preds, axis=0)
    err = np.abs(yp - y)
    return {
        "n":    len(models),
        "mae":  err.mean(),
        "mape": (err / y * 100).mean(),
        "bias": (yp - y).mean(),
        "bio":  (yp.sum() - y.sum()) / y.sum() * 100,
        "y":    y,
        "yp":   yp,
    }

configs = [
    ("No salinity",    "checkpoints/carp_weight_combined_model",
     "model/model_config_12ch.yaml",          "data/carp/weight_combined/test_dataset.npz"),
    ("Filename sal.",  "checkpoints/carp_weight_salinity_model",
     "model/model_config_12ch_salinity.yaml", "data/carp/weight_salinity/test_dataset.npz"),
    ("Baseline RMS",   "checkpoints/carp_weight_baseline_model",
     "model/model_config_12ch_salinity.yaml", "data/carp/weight_baseline/test_dataset.npz"),
]

print(f"\n  {'Model':<20} {'Ens':>3}  {'MAE':>6}  {'MAPE':>6}  {'Bias':>7}  {'Biomass diff':>13}")
print("  " + "-" * 65)
results = {}
for label, ckpt, cfg, npz in configs:
    r = eval_model(ckpt, cfg, npz)
    results[label] = r
    print(f"  {label:<20} {r['n']:>3}  {r['mae']:>5.0f}g  {r['mape']:>5.1f}%  "
          f"{r['bias']:>+6.0f}g  {r['bio']:>+12.1f}%")

# Per weight-class breakdown
print("\n  Per weight-class MAPE:")
print(f"  {'Weight':>7}  {'No sal.':>10}  {'File sal.':>10}  {'Baseline':>10}")
print("  " + "-" * 46)
r0 = results["No salinity"]
r1 = results["Filename sal."]
r2 = results["Baseline RMS"]
for wc in sorted(np.unique(r0["y"]).tolist()):
    mask0 = r0["y"] == wc
    mask2 = r2["y"] == wc
    if mask0.sum() == 0: continue
    mape0 = (np.abs(r0["yp"][mask0] - wc) / wc * 100).mean()
    mape1 = (np.abs(r1["yp"][mask0] - wc) / wc * 100).mean() if mask0.sum() > 0 else float("nan")
    mape2 = (np.abs(r2["yp"][mask2] - wc) / wc * 100).mean() if mask2.sum() > 0 else float("nan")
    best = min(mape0, mape1, mape2)
    flag2 = " *" if (mape2 == best and mape2 < mape0 - 2) else ""
    print(f"  {int(wc):>6}g   {mape0:>9.1f}%  {mape1:>9.1f}%  {mape2:>9.1f}%{flag2}")

if __name__ == "__main__":
    pass
