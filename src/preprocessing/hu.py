"""Hounsfield-unit conversion and clipping."""
import numpy as np


def to_hu(raw: np.ndarray, slope: float = 1.0, intercept: float = 0.0, clip=(-1024.0, 3071.0)) -> np.ndarray:
    """Raw stored values -> HU (float32), clipped. Clipping also neutralises
    out-of-scanner padding values such as -2000."""
    hu = raw.astype(np.float32) * np.float32(slope) + np.float32(intercept)
    if clip is not None:
        np.clip(hu, clip[0], clip[1], out=hu)
    return hu
