"""LUNA16 scan indexing. Reads MetaImage headers directly from the subset zips
(no extraction): the raw data is ~66 GB and must never be modified."""
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_SUBSETS = 10
RAW_BYTES_PER_VOXEL = {"MET_SHORT": 2, "MET_USHORT": 2, "MET_UCHAR": 1, "MET_CHAR": 1, "MET_FLOAT": 4}


@dataclass
class ScanInfo:
    seriesuid: str
    subset: int
    dim_size: tuple          # (x, y, z) voxels
    spacing: tuple           # (x, y, z) mm
    origin: tuple            # (x, y, z) mm, world coordinates
    transform: tuple         # 9 values, row-major direction cosines
    element_type: str
    raw_name: str
    raw_bytes_in_zip: int    # uncompressed size of the .raw member
    mhd_member: str

    @property
    def expected_raw_bytes(self) -> int:
        return int(np.prod(self.dim_size)) * RAW_BYTES_PER_VOXEL[self.element_type]

    def world_to_voxel(self, xyz_mm):
        """World (mm) -> voxel index (x, y, z) in the stored array axes.

        Uses the direction matrix: some scans have x/y axes flipped
        (TransformMatrix diag = -1, -1, 1)."""
        d = np.asarray(self.transform, float).reshape(3, 3)
        rel = np.asarray(xyz_mm, float) - np.asarray(self.origin)
        return np.linalg.solve(d, rel) / np.asarray(self.spacing)


def parse_mhd(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _floats(s):
    return tuple(float(x) for x in s.split())


def index_subset_zip(zip_path: Path, subset: int) -> list:
    """Parse every .mhd header in a subset zip and record the .raw size."""
    scans = []
    with zipfile.ZipFile(zip_path) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
        for name in sorted(sizes):
            if not name.endswith(".mhd"):
                continue
            h = parse_mhd(z.read(name).decode())
            raw_member = name[: -len(".mhd")] + ".raw"
            scans.append(ScanInfo(
                seriesuid=Path(name).stem,
                subset=subset,
                dim_size=tuple(int(x) for x in h["DimSize"].split()),
                spacing=_floats(h["ElementSpacing"]),
                origin=_floats(h["Offset"]),
                transform=_floats(h.get("TransformMatrix", "1 0 0 0 1 0 0 0 1")),
                element_type=h["ElementType"],
                raw_name=h["ElementDataFile"],
                raw_bytes_in_zip=sizes.get(raw_member, -1),
                mhd_member=name,
            ))
    return scans


def index_all(data_dir) -> list:
    data_dir = Path(data_dir)
    scans = []
    for i in range(N_SUBSETS):
        p = data_dir / f"subset{i}.zip"
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")
        scans.extend(index_subset_zip(p, i))
    return scans
