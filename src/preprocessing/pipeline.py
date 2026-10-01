"""End-to-end preprocessing of one LUNA16 scan: raw -> model-ready normalized volume."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .hu import to_hu
from .io import Volume, load_ct, load_lung_mask
from .lung_region import lung_binary
from .normalization import window_normalize
from .resampling import resample


@dataclass
class Processed:
    volume: Volume                       # normalized [0,1], resampled, canonical
    lung: Optional[np.ndarray]           # bool (z,y,x) on the same grid, or None
    hu_volume: Optional[Volume] = None   # resampled HU (only when keep_hu)


def preprocess_scan(data_dir, subset: int, seriesuid: str, cfg: dict, keep_hu: bool = False) -> Processed:
    hu_cfg, win, rs, lr = cfg["hu"], cfg["window"], cfg["resampling"], cfg["lung_region"]
    ct = load_ct(data_dir, subset, seriesuid)
    hu = Volume(to_hu(ct.array, hu_cfg["slope"], hu_cfg["intercept"], (hu_cfg["clip_min"], hu_cfg["clip_max"])),
                ct.spacing, ct.origin)
    del ct
    mask = load_lung_mask(data_dir, seriesuid) if lr["enabled"] else None
    if mask is not None and (mask.array.shape != hu.array.shape or not np.allclose(mask.origin, hu.origin)):
        raise ValueError(f"{seriesuid}: lung mask geometry differs from CT")
    if rs["enabled"]:
        hu = resample(hu, rs["target_spacing_mm"], order=rs["order"])
        if mask is not None:
            mask = resample(mask, rs["target_spacing_mm"], order=rs["mask_order"])
    norm = Volume(window_normalize(hu.array, win["min"], win["max"]), hu.spacing, hu.origin)
    lung = lung_binary(mask, lr["labels"]) if mask is not None else None
    return Processed(norm, lung, hu if keep_hu else None)
