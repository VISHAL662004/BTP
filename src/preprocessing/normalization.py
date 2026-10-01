"""Intensity windowing + normalization with fixed constants (no dataset statistics,
so no information flows between train/val/test)."""
import numpy as np


def window_normalize(hu: np.ndarray, vmin: float = -1000.0, vmax: float = 400.0) -> np.ndarray:
    if vmax <= vmin:
        raise ValueError(f"Invalid window [{vmin}, {vmax}]")
    out = (hu - np.float32(vmin)) / np.float32(vmax - vmin)
    return np.clip(out, 0.0, 1.0, out=out).astype(np.float32, copy=False)
