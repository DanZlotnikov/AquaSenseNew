"""Fish Length Prediction — CLI entry point.

Usage:
    # Train a new model
    python main.py --mode train

    # Evaluate a saved checkpoint
    python main.py --mode test --checkpoint checkpoints/<run_id>/best_model.pt

    # Override default config paths
    python main.py --mode train \\
        --model_config model/model_config.yaml \\
        --train_config training/train_config.yaml
"""

import argparse
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_train(args: argparse.Namespace) -> None:
    cfg_model = load_yaml(args.model_config)
    cfg_train = load_yaml(args.train_config)

    from training.train import train

    best_ckpt = train(cfg_model, cfg_train)

    print()
    print(f"Best checkpoint : {best_ckpt}")
    print(f"To evaluate     : python main.py --mode test --checkpoint {best_ckpt}")


def run_test(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --mode test")

    cfg_test = load_yaml(args.test_config)

    # Allow checkpoint path override from CLI without editing the config file
    from test.evaluate import evaluate

    metrics = evaluate(args.checkpoint, cfg_test)

    print()
    print("--- Final Metrics ---")
    print(f"  MAPE      : {metrics['mape']:.2f}%")
    print(f"  MAE       : {metrics['mae']:.4f} cm")
    print(f"  RMSE      : {metrics['rmse']:.4f} cm")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fish Length Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "test"], required=True,
        help="Run training or evaluation",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a .pt checkpoint file (required for --mode test)",
    )
    parser.add_argument("--model_config", default="model/model_config.yaml")
    parser.add_argument("--train_config", default="training/train_config.yaml")
    parser.add_argument("--test_config",  default="test/test_config.yaml")

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
