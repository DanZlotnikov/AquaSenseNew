"""Train an ensemble of FishSaltModel instances with different random seeds.

Each member is trained independently on the same (augmented) dataset.
Ensemble inference averages logits across all members.

Run from project root:
    python training/train_ensemble_salt.py \\
        --config       training/train_config_salt.yaml \\
        --model_config model/model_config_salt.yaml \\
        --n_models 3

    # Custom seeds
    python training/train_ensemble_salt.py ... --seeds 7 42 99
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
    parser.add_argument("--config",       default="training/train_config_salt.yaml")
    parser.add_argument("--model_config", default="model/model_config_salt.yaml")
    parser.add_argument("--n_models",     type=int, default=3)
    parser.add_argument("--seeds",        type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg_model = yaml.safe_load(open(args.model_config))
    cfg_train = yaml.safe_load(open(args.config))

    seeds = args.seeds or list(range(1, args.n_models + 1))

    from training.train_salt import train

    checkpoints = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training ensemble member  seed={seed}")
        print(f"{'='*60}\n")
        set_seed(seed)
        ckpt = train(cfg_model, cfg_train)
        checkpoints.append(ckpt)
        print(f"Saved: {ckpt}")

    print("\nEnsemble checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt}")
