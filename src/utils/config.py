"""YAML configuration loading."""
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return cfg
