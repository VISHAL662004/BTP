"""Patch extraction around world-coordinate centres (candidates / nodules)."""
import numpy as np

from .io import Volume
from .slice_context import slice_indices


def extract_25d_patch(vol: Volume, center_xyz_mm, size: int = 64, n_slices: int = 3,
                      boundary: str = "edge", pad_value: float = 0.0):
    """Axial 2.5D patch (n_slices, size, size) centred on a world coordinate.

    The centre is rounded to the nearest voxel of `vol` (the resampled grid), so the
    sub-voxel offset is returned as well. Regions outside the volume are `pad_value`.
    Returns (patch, center_voxel_xyz_float).
    """
    c = vol.world_to_voxel(center_xyz_mm)
    cx, cy, cz = (int(round(v)) for v in c)
    depth, H, W = vol.array.shape
    zs = slice_indices(min(max(cz, 0), depth - 1), n_slices, depth, boundary) if 0 <= cz < depth else None
    if zs is None:
        return np.full((n_slices, size, size), pad_value, np.float32), c
    half = size // 2
    y0, x0 = cy - half, cx - half
    out = np.full((n_slices, size, size), pad_value, np.float32)
    ys, ye = max(y0, 0), min(y0 + size, H)
    xs, xe = max(x0, 0), min(x0 + size, W)
    if ys < ye and xs < xe:
        out[:, ys - y0:ye - y0, xs - x0:xe - x0] = vol.array[zs][:, ys:ye, xs:xe]
    return out, c


def extract_3d_patch(vol: Volume, center_xyz_mm, size: int = 64, pad_value: float = 0.0):
    """Cubic patch (size, size, size) in (z, y, x) order, for validation/visualisation."""
    c = vol.world_to_voxel(center_xyz_mm)
    cx, cy, cz = (int(round(v)) for v in c)
    half = size // 2
    out = np.full((size,) * 3, pad_value, np.float32)
    lo = (cz - half, cy - half, cx - half)
    shp = vol.array.shape
    src = [(max(l, 0), min(l + size, s)) for l, s in zip(lo, shp)]
    if all(a < b for a, b in src):
        dst = tuple(slice(a - l, b - l) for (a, b), l in zip(src, lo))
        out[dst] = vol.array[tuple(slice(a, b) for a, b in src)]
    return out, c
