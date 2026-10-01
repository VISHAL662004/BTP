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


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment_config(path, base_path="configs/base.yaml"):
    """Experiment YAML overrides configs/base.yaml (deep merge)."""
    return deep_merge(load_config(base_path), load_config(path))
