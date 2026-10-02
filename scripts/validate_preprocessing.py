"""Phase 3 validation: does preprocessing keep CT <-> annotation correspondence?

For every annotation in a sample of scans (all 14 flipped-axis scans + random others) we
measure the mean normalized intensity in a small sphere at the annotated world position
after the full pipeline (canonicalize -> HU -> resample -> window). Nodules are denser than
lung parenchyma, so aligned annotations should be much brighter than random lung points.
For flipped scans we also test the mirrored (wrongly un-flipped) position as a negative control.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.lung_region import point_in_lung
from src.preprocessing.pipeline import preprocess_scan
from src.utils.config import load_config
from src.utils.progress import Progress

DATA = Path("data")
THR = (-500.0 + 1000.0) / 1400.0   # -500 HU in normalized units


def sphere_mean(vol, center_xyz_mm, radius_mm=1.5):
    c = vol.world_to_voxel(center_xyz_mm)
    r = np.ceil(radius_mm / np.asarray(vol.spacing)).astype(int)
    lo = np.maximum(np.round(c).astype(int) - r, 0)
    hi = np.minimum(np.round(c).astype(int) + r + 1, np.array(vol.shape_xyz))
    if (lo >= hi).any():
        return np.nan
    sub = vol.array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    zz, yy, xx = np.meshgrid(*(np.arange(l, h) for l, h in zip(lo[::-1], hi[::-1])), indexing="ij")
    d = np.sqrt(((xx - c[0]) * vol.spacing[0]) ** 2 + ((yy - c[1]) * vol.spacing[1]) ** 2 + ((zz - c[2]) * vol.spacing[2]) ** 2)
    return float(sub[d <= radius_mm].mean())   # radius < smallest nodule radius (1.6 mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config("configs/preprocessing.yaml")
    scans = pd.read_csv("data/metadata/scans.csv")
    ann = pd.read_csv("data/metadata/annotations_voxel.csv")
    with_nod = scans[scans.n_nodules > 0]
    flipped = with_nod[(with_nod.flip_x < 0) | (with_nod.flip_y < 0)]
    rest = with_nod.drop(flipped.index)
    sample = pd.concat([flipped, rest.sample(args.n_random, random_state=0)])
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for s in Progress(list(sample.itertuples()), desc="alignment check", unit="scan", step_pct=10):
        p = preprocess_scan(DATA, s.subset, s.seriesuid, cfg)
        vol = p.volume
        zl, yl, xl = np.where(p.lung)
        for a in ann[ann.seriesuid == s.seriesuid].itertuples():
            w = np.array([a.coordX, a.coordY, a.coordZ])
            v = vol.world_to_voxel(w)
            row = dict(seriesuid=s.seriesuid, flipped=bool(s.flip_x < 0 or s.flip_y < 0), diameter=a.diameter_mm,
                       aligned=sphere_mean(vol, w), in_lung=point_in_lung(p.lung, v, dilate_vox=int(np.ceil(5 / vol.spacing[0]))))
            if row["flipped"]:  # negative control: mirror x,y about the volume centre
                nx, ny, _ = vol.shape_xyz
                vm = np.array([nx - 1 - v[0], ny - 1 - v[1], v[2]])
                row["mirrored"] = sphere_mean(vol, vol.voxel_to_world(vm))
            # control: random lung voxel
            i = rng.integers(len(zl))
            row["random_lung"] = sphere_mean(vol, vol.voxel_to_world([xl[i], yl[i], zl[i]]))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("data/metadata/preprocessing_alignment.csv", index=False)
    res = {
        "scans": int(len(sample)), "annotations": int(len(df)), "flipped_scans": int(len(flipped)),
        "frac_aligned_dense(>-500HU)": round(float((df.aligned > THR).mean()), 3),
        "frac_random_lung_dense": round(float((df.random_lung > THR).mean()), 3),
        "frac_annotation_in_lung(5mm dilation)": round(float(df.in_lung.mean()), 3),
        "flipped_frac_aligned_dense": round(float((df[df.flipped].aligned > THR).mean()), 3),
        "flipped_frac_mirrored_dense": round(float((df[df.flipped].mirrored > THR).mean()), 3),
    }
    Path("data/metadata/preprocessing_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
