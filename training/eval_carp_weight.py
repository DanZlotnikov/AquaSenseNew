"""Evaluate carp_weight_model on the weight test set.

Loads the 3-model ensemble from checkpoints/carp_weight_model/ and
evaluates on data/carp/weight/test_dataset.npz.

Run from project root:
    python training/eval_carp_weight.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.fish_model import FishModel

CHECKPOINT_DIRS = sorted(Path("checkpoints/carp_weight_model").glob("2*"))
MODEL_CFG_PATH  = Path("model/model_config_12ch.yaml")
TEST_NPZ        = Path("data/carp/weight/test_dataset.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ensemble():
    with open(MODEL_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["physics_dim"] = 0

    models = []
    for ckpt_dir in CHECKPOINT_DIRS:
        pt = ckpt_dir / "best_model.pt"
        if not pt.exists():
            continue
        m = FishModel(cfg).to(DEVICE)
        ckpt = torch.load(pt, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        m.eval()
        models.append(m)
        print(f"  Loaded {pt}")
    return models


def predict(models, X: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(X).to(DEVICE)
    preds = []
    with torch.no_grad():
        for m in models:
            _, pred = m(t)
            preds.append(pred.cpu().numpy())
    return np.mean(preds, axis=0)


def main():
    print("=" * 55)
    print("carp_weight_model evaluation — test set")
    print("=" * 55)

    print("\nLoading ensemble ...")
    models = load_ensemble()
    print(f"Ensemble size: {len(models)}")

    data = np.load(TEST_NPZ)
    X       = data["X"]
    y_true  = data["y_length"]   # weight in grams (stored as y_length)
    n       = len(y_true)

    print(f"\nTest samples: {n}")
    print(f"True weights: {y_true.astype(int).tolist()}")

    y_pred = predict(models, X)

    print(f"\nPredicted weights: {y_pred.astype(int).tolist()}")

    errors  = np.abs(y_pred - y_true)
    mae     = errors.mean()
    rmse    = np.sqrt((errors ** 2).mean())
    mape    = (errors / y_true * 100).mean()
    bias    = (y_pred - y_true).mean()

    print(f"\nMAE  : {mae:.1f} g")
    print(f"RMSE : {rmse:.1f} g")
    print(f"MAPE : {mape:.1f} %")
    print(f"Bias : {bias:+.1f} g  ({'over' if bias > 0 else 'under'}-predicting)")

    print("\nPer-sample breakdown:")
    print(f"  {'True':>7}  {'Pred':>7}  {'Error':>7}  {'MAPE':>6}")
    for yt, yp in zip(y_true, y_pred):
        err  = abs(yp - yt)
        pct  = err / yt * 100
        flag = " <-- large" if pct > 40 else ""
        print(f"  {int(yt):>7}  {int(yp):>7}  {err:>7.0f}g  {pct:>5.1f}%{flag}")

    print(f"\nTotal true biomass   : {y_true.sum():.0f} g")
    print(f"Total predicted biomass: {y_pred.sum():.0f} g")
    diff = (y_pred.sum() - y_true.sum()) / y_true.sum() * 100
    print(f"Biomass diff         : {diff:+.1f}%")


if __name__ == "__main__":
    main()
