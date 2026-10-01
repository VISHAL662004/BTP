"""Candidate index: world coordinates -> split, canonical voxel coordinates, lung flag."""
import numpy as np
import pandas as pd

from .lung_region import lung_binary, point_in_lung
from .io import load_lung_mask


def canonical_origin(row) -> tuple:
    """Origin of the flip-corrected (canonical) grid; matches io._canonicalize
    (only axes with a negative direction sign are shifted)."""
    def one(origin, flip, n, sp):
        return origin + flip * (n - 1) * sp if flip < 0 else origin
    return (one(row.origin_x, row.flip_x, row.dim_x, row.sp_x),
            one(row.origin_y, row.flip_y, row.dim_y, row.sp_y),
            one(row.origin_z, row.flip_z, row.n_slices, row.sp_z))


def build_candidate_index(data_dir, cands: pd.DataFrame, scans: pd.DataFrame, split_of: dict,
                          lung_labels=(3, 4), dilate_mm: float = 5.0, log=None) -> pd.DataFrame:
    parts = []
    for n, row in enumerate(scans.itertuples()):
        c = cands[cands.seriesuid == row.seriesuid].copy()
        if c.empty:
            continue
        o = canonical_origin(row)
        sp = (row.sp_x, row.sp_y, row.sp_z)
        for i, ax in enumerate("xyz"):
            c[f"vox_{ax}"] = (c[f"coord{ax.upper()}"] - o[i]) / sp[i]
        mask = lung_binary(load_lung_mask(data_dir, row.seriesuid), lung_labels)
        r = int(np.ceil(dilate_mm / min(sp[0], sp[1])))
        c["in_lung"] = [point_in_lung(mask, v, dilate_vox=r) for v in c[["vox_x", "vox_y", "vox_z"]].to_numpy()]
        c["split"] = split_of[row.seriesuid]
        c["subset"] = row.subset
        parts.append(c)
        if log and (n + 1) % 100 == 0:
            log(f"{n + 1}/{len(scans)} scans")
    return pd.concat(parts, ignore_index=True)
