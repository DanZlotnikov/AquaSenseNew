"""Train an ensemble of N weight models with different random seeds.

Run from project root:
    python training/train_ensemble_weight.py \\
        --config training/train_config_bream_weight.yaml \\
        --model_config model/model_config_12ch.yaml \\
        --n_models 3
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--n_models",     type=int, default=3)
    parser.add_argument("--seeds",        type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg_model = yaml.safe_load(open(args.model_config))
    cfg_train = yaml.safe_load(open(args.config))

    seeds = args.seeds or list(range(1, args.n_models + 1))

    from training.train_weight import train

    checkpoints = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training weight ensemble member  seed={seed}")
        print(f"{'='*60}\n")
        set_seed(seed)
        ckpt = train(cfg_model, cfg_train)
        checkpoints.append(ckpt)
        print(f"Saved: {ckpt}")

    print("\nEnsemble checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt}")
