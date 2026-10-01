"""Load CT volumes / lung masks straight from the LUNA16 zips and canonicalize orientation.

Canonical form: numpy array indexed (z, y, x), direction matrix = identity, so that
world -> voxel is  (world - origin) / spacing  for every scan. Scans stored with flipped
x/y axes (TransformMatrix diag -1,-1,1) are flipped back on load.
"""
import zlib
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.data.luna16 import parse_mhd

_DTYPES = {"MET_SHORT": "<i2", "MET_USHORT": "<u2", "MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_FLOAT": "<f4"}
SEG_ZIP = "seg-lungs-LUNA16.zip"


@dataclass(frozen=True)
class Volume:
    array: np.ndarray        # (z, y, x)
    spacing: tuple           # (x, y, z) mm
    origin: tuple            # (x, y, z) mm, world coords of voxel (0, 0, 0)

    @property
    def shape_xyz(self):
        z, y, x = self.array.shape
        return (x, y, z)

    def world_to_voxel(self, xyz_mm):
        """World mm -> voxel (x, y, z), float. Valid for canonical volumes."""
        return (np.asarray(xyz_mm, float) - np.asarray(self.origin)) / np.asarray(self.spacing)

    def voxel_to_world(self, xyz_vox):
        return np.asarray(xyz_vox, float) * np.asarray(self.spacing) + np.asarray(self.origin)


def _canonicalize(arr_zyx, spacing, origin, transform) -> Volume:
    d = np.asarray(transform, float).reshape(3, 3)
    if not np.allclose(np.abs(d), np.eye(3), atol=1e-6):
        raise ValueError(f"Non axis-aligned orientation not supported: {transform}")
    origin = list(origin)
    nx, ny, nz = arr_zyx.shape[2], arr_zyx.shape[1], arr_zyx.shape[0]
    for axis_np, i, n in ((2, 0, nx), (1, 1, ny), (0, 2, nz)):
        sign = d[i, i]
        if sign < 0:
            arr_zyx = np.flip(arr_zyx, axis=axis_np)
            origin[i] = origin[i] + sign * (n - 1) * spacing[i]
    return Volume(np.ascontiguousarray(arr_zyx), tuple(spacing), tuple(origin))


def _parse(header_text):
    h = parse_mhd(header_text)
    dims = tuple(int(x) for x in h["DimSize"].split())
    spacing = tuple(float(x) for x in h["ElementSpacing"].split())
    origin = tuple(float(x) for x in h["Offset"].split())
    transform = tuple(float(x) for x in h.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split())
    return h, dims, spacing, origin, transform


def _array(buf, h, dims):
    arr = np.frombuffer(buf, dtype=_DTYPES[h["ElementType"]])
    if arr.size != int(np.prod(dims)):
        raise ValueError(f"Raw size {arr.size} does not match DimSize {dims}")
    return arr.reshape(dims[2], dims[1], dims[0])


def load_ct(data_dir, subset: int, seriesuid: str) -> Volume:
    data_dir = Path(data_dir)
    with zipfile.ZipFile(data_dir / f"subset{subset}.zip") as z:
        h, dims, spacing, origin, tr = _parse(z.read(f"subset{subset}/{seriesuid}.mhd").decode())
        arr = _array(z.read(f"subset{subset}/{seriesuid}.raw"), h, dims)
    return _canonicalize(arr, spacing, origin, tr)


def load_lung_mask(data_dir, seriesuid: str) -> Volume:
    with zipfile.ZipFile(Path(data_dir) / SEG_ZIP) as z:
        h, dims, spacing, origin, tr = _parse(z.read(f"seg-lungs-LUNA16/{seriesuid}.mhd").decode())
        buf = z.read(f"seg-lungs-LUNA16/{seriesuid}.zraw")
    if h.get("CompressedData", "False") == "True":
        buf = zlib.decompress(buf)
    return _canonicalize(_array(buf, h, dims), spacing, origin, tr)
