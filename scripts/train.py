"""Train an experiment: python scripts/train.py configs/experiments/baseline.yaml"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import Trainer
from src.utils.config import load_experiment_config


def main():
    cfg = load_experiment_config(sys.argv[1])
    out = Path("experiments") / sys.argv[2] / cfg["experiment"]["name"] if len(sys.argv) > 2 else None
    if out is None:
        raise SystemExit("usage: train.py <config.yaml> <experiments subfolder, e.g. baseline>")
    Trainer(cfg, out).fit()


if __name__ == "__main__":
    main()
