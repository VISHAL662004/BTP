"""LUNA16 annotation / candidate CSV loading."""
from pathlib import Path

import pandas as pd

ANN_COLS = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
CAND_COLS = ["seriesuid", "coordX", "coordY", "coordZ", "class"]


def _load(path, cols):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    if list(df.columns) != cols:
        raise ValueError(f"{path}: expected columns {cols}, got {list(df.columns)}")
    return df


def load_annotations(path):
    return _load(path, ANN_COLS)


def load_candidates(path):
    return _load(path, CAND_COLS)
