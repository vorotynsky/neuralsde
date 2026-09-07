import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .data import load_dataset
from .training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the latent Neural SDE.")
    parser.add_argument("--config", default="configs/default.train.yaml")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/latent_sde")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    dataset = load_dataset(args.dataset)
    result = train(config, dataset, device=args.device, output=args.output)
    print(json.dumps(result, indent=2))
