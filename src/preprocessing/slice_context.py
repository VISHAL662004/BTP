"""2.5D slice stacking: channels are adjacent axial slices (z-k ... z+k)."""
import numpy as np


def slice_indices(z: int, n_slices: int, depth: int, boundary: str = "edge") -> list:
    if n_slices < 1 or n_slices % 2 == 0:
        raise ValueError(f"input_slices must be odd and >= 1, got {n_slices}")
    half = n_slices // 2
    idx = list(range(z - half, z + half + 1))  # ascending z: preserves spatial order
    if boundary == "edge":
        return [min(max(i, 0), depth - 1) for i in idx]
    if boundary == "error":
        if idx[0] < 0 or idx[-1] >= depth:
            raise IndexError(f"Slice window {idx} outside depth {depth}")
        return idx
    raise ValueError(f"Unknown boundary mode {boundary}")


def stack_slices(volume_zyx: np.ndarray, z: int, n_slices: int = 3, boundary: str = "edge") -> np.ndarray:
    """Returns (n_slices, H, W)."""
    return volume_zyx[slice_indices(z, n_slices, volume_zyx.shape[0], boundary)]
