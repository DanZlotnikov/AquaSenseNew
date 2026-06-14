#!/usr/bin/env python
"""AquaSense local inference — run on a single CSV recording.

Usage
-----
Basic (uses input_file from config.yaml):
    python run.py

Override the input file at runtime:
    python run.py --input path/to/recording.csv

Use a different config file:
    python run.py --config my_config.yaml --input path/to/recording.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AquaSense local inference — fish detection and weight estimation"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--input", default=None,
        help="CSV recording file to analyse (overrides input_file in config)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    species        = cfg.get("species", "bream")
    weight_variant = cfg.get("weight_variant", "no_sal")
    results_dir    = Path(cfg.get("results_dir", "results"))
    threshold      = float(cfg.get("threshold", 0.5))
    batch_size     = int(cfg.get("batch_size", 256))

    # Input file: CLI flag beats config
    input_file_str = args.input or cfg.get("input_file", "")
    if not input_file_str:
        print(
            "ERROR: no input file specified.\n"
            "  Set input_file in config.yaml  or  use --input path/to/file.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_path = Path(input_file_str)
    if not csv_path.exists():
        print(f"ERROR: input file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"Species       : {species}")
    print(f"Weight model  : {weight_variant}" + (" (ignored for bream)" if species == "bream" else ""))
    print(f"Input         : {csv_path}")
    print("Loading models…")

    # Add project root to path so `model` package resolves
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference.infer import run_inference

    result = run_inference(
        csv_path=csv_path,
        species=species,
        weight_variant=weight_variant,
        threshold=threshold,
        batch_size=batch_size,
    )

    # ── Save output JSON ──────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = csv_path.stem[:50]          # truncate very long filenames
    out_path  = results_dir / f"{stem}_{timestamp}.json"

    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    s = result["summary"]
    print(f"\nFish detected : {s['n_fish']}")
    print(f"Avg weight    : {s['avg_weight_g']:.1f} g")
    print(f"Total biomass : {s['biomass_g']:.1f} g")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
