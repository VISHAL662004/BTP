"""Resampling to a target voxel spacing, keeping world-coordinate correspondence."""
import numpy as np
from .io import Volume


def resample(vol: Volume, target_spacing=(1.0, 1.0, 1.0), order: int = 1) -> Volume:
    """Resample so voxel centres lie on a grid of `target_spacing`, aligned so that the
    first voxel keeps the original origin (world position of voxel 0 is unchanged).

    Output size n' = round(n * s / s'); the sampling coordinate of output index j is
    j * s' / s in input index space, so  world(j) = origin + j * s'  exactly.
    """
    target = np.asarray(target_spacing, float)
    src = np.asarray(vol.spacing, float)
    nx, ny, nz = vol.shape_xyz
    new_shape_xyz = np.maximum(np.round(np.array([nx, ny, nz]) * src / target).astype(int), 1)
    if np.allclose(src, target) and tuple(new_shape_xyz) == (nx, ny, nz):
        return vol
    out = vol.array
    for axis_np, ax in ((0, 2), (1, 1), (2, 0)):   # numpy axes z, y, x -> xyz index
        out = _resample_axis(out, axis_np, new_shape_xyz[ax], target[ax] / src[ax], order)
    return Volume(out.astype(vol.array.dtype, copy=False), tuple(target), vol.origin)


def _resample_axis(a: np.ndarray, axis: int, new_n: int, step: float, order: int) -> np.ndarray:
    """1-D resampling along `axis`; output index j samples input position j * step
    (clamped to the valid range). Separable => low memory vs. dense coordinate grids."""
    n = a.shape[axis]
    pos = np.clip(np.arange(new_n) * step, 0, n - 1)
    if order == 0:
        return np.take(a, np.rint(pos).astype(int), axis=axis)
    if order != 1:
        raise ValueError("Only order 0 (nearest) and 1 (linear) are supported")
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo).astype(np.float32)
    shape = [1] * a.ndim
    shape[axis] = new_n
    w = w.reshape(shape)
    return np.take(a, lo, axis=axis).astype(np.float32) * (1 - w) + np.take(a, hi, axis=axis).astype(np.float32) * w
