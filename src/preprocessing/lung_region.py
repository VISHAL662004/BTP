"""Lung region of interest from the official LUNA16 lung segmentation masks."""
import numpy as np
from scipy import ndimage

from .io import Volume


def lung_binary(mask: Volume, labels=(3, 4)) -> np.ndarray:
    return np.isin(mask.array, labels)


def lung_bbox_voxels(binary: np.ndarray, spacing_xyz, margin_mm: float = 10.0):
    """Bounding box (z0, z1, y0, y1, x0, x1) (exclusive end) of the lung plus margin."""
    if not binary.any():
        raise ValueError("Empty lung mask")
    idx = np.where(binary)
    sx, sy, sz = spacing_xyz
    margins = (int(np.ceil(margin_mm / sz)), int(np.ceil(margin_mm / sy)), int(np.ceil(margin_mm / sx)))
    box = []
    for ax, m in enumerate(margins):
        lo = max(int(idx[ax].min()) - m, 0)
        hi = min(int(idx[ax].max()) + 1 + m, binary.shape[ax])
        box += [lo, hi]
    return tuple(box)


def point_in_lung(binary: np.ndarray, voxel_xyz, dilate_vox: int = 0) -> bool:
    """Is a voxel (x, y, z) inside (optionally dilated) lung? Out-of-volume -> False."""
    x, y, z = (int(round(v)) for v in voxel_xyz)
    if not (0 <= z < binary.shape[0] and 0 <= y < binary.shape[1] and 0 <= x < binary.shape[2]):
        return False
    if dilate_vox <= 0:
        return bool(binary[z, y, x])
    r = dilate_vox
    sub = binary[max(z - r, 0):z + r + 1, max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
    return bool(sub.any())


def fill_lung_holes(binary: np.ndarray) -> np.ndarray:
    """Slice-wise hole filling (vessels / nodules inside the lung are labelled as non-lung
    in some masks)."""
    return np.stack([ndimage.binary_fill_holes(s) for s in binary])
