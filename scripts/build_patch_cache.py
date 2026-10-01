"""Phase 4: cache 2.5D candidate patches so training does not re-preprocess CT scans.

Output (data/processed/): patches_{train,val,test}.npy  uint8 (N, C, H, W)  + patches_{split}.csv metadata.
  * train: all positive candidates + a seeded random sample of negatives
  * val / test: ALL candidates (needed for the FROC protocol)
Patches are the normalized [0,1] CT quantized to uint8 (1/255 of the 1400 HU window = 5.5 HU steps).
Positive candidates carry the matched nodule box: offset (box_dx, box_dy, in patch pixels from the
patch centre) and diameter box_d (pixels; 1 px = 1 mm after resampling).
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.patches import extract_25d_patch
from src.preprocessing.pipeline import preprocess_scan
from src.utils.config import load_config

DATA, OUT = Path("data"), Path("data/processed")
CFG = None


def _init(cfg):
    global CFG
    CFG = cfg


def _work(job):
    split, subset, uid, rows, coords, ann = job   # coords (n,3) world; ann (m,4) x,y,z,d
    vol = preprocess_scan(DATA, subset, uid, CFG).volume
    pc, sc = CFG["patches"], CFG["slice_context"]
    patches = np.empty((len(rows), sc["input_slices"], pc["size"], pc["size"]), np.uint8)
    boxes = np.full((len(rows), 3), np.nan, np.float32)
    for i, w in enumerate(coords):
        p, c = extract_25d_patch(vol, w, pc["size"], sc["input_slices"], sc["boundary"], pc["pad_value"], sc["slice_stride"])
        patches[i] = np.rint(p * 255).astype(np.uint8)
        if len(ann):
            d = np.linalg.norm(ann[:, :3] - w, axis=1)
            j = int(d.argmin())
            if d[j] < max(ann[j, 3] / 2, 1.0) * 1.0 + 1e-6:   # candidate lies inside the nodule radius
                a = vol.world_to_voxel(ann[j, :3]) - np.round(c)
                boxes[i] = (a[0], a[1], ann[j, 3] / vol.spacing[0])
    return split, rows, patches, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-neg", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config("configs/preprocessing.yaml")
    idx = pd.read_csv(DATA / "candidates" / "candidate_index.csv")
    scans = pd.read_csv(DATA / "metadata" / "scans.csv").set_index("seriesuid")
    ann = pd.read_csv(DATA / "annotations.csv")
    rng = np.random.default_rng(args.seed)

    parts = {}
    tr = idx[idx.split == "train"]
    neg = tr[tr["class"] == 0]
    keep = np.sort(rng.choice(len(neg), size=min(args.n_train_neg, len(neg)), replace=False))
    parts["train"] = pd.concat([tr[tr["class"] == 1], neg.iloc[keep]])
    for s in ("val", "test"):
        parts[s] = idx[idx.split == s]
    OUT.mkdir(parents=True, exist_ok=True)
    mm, jobs = {}, []
    C, P = cfg["slice_context"]["input_slices"], cfg["patches"]["size"]
    for s, df in parts.items():
        df = df.sort_values(["seriesuid", "coordZ", "coordY", "coordX"]).reset_index(drop=True)
        df["box_dx"] = df["box_dy"] = df["box_d"] = np.nan
        parts[s] = df
        mm[s] = np.lib.format.open_memmap(OUT / f"patches_{s}.npy", mode="w+", dtype=np.uint8, shape=(len(df), C, P, P))
        for uid, g in df.groupby("seriesuid"):
            a = ann[ann.seriesuid == uid][["coordX", "coordY", "coordZ", "diameter_mm"]].to_numpy(float)
            jobs.append((s, int(scans.loc[uid, "subset"]), uid, g.index.to_numpy(), g[["coordX", "coordY", "coordZ"]].to_numpy(float), a))
    print({s: len(d) for s, d in parts.items()}, "scans:", len(jobs), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for n, (s, rows, patches, boxes) in enumerate(pool.imap_unordered(_work, jobs), 1):
            mm[s][rows] = patches
            parts[s].loc[rows, ["box_dx", "box_dy", "box_d"]] = boxes
            if n % 25 == 0:
                print(f"{n}/{len(jobs)} scans, {time.time() - t0:.0f}s", flush=True)
    for s, df in parts.items():
        mm[s].flush()
        df = df.rename(columns={"class": "label"})
        df[["seriesuid", "coordX", "coordY", "coordZ", "label", "in_lung", "box_dx", "box_dy", "box_d"]].to_csv(OUT / f"patches_{s}.csv", index=False)
    info = {"channels": C, "size": P, "dtype": "uint8 (x/255 -> [0,1])", "seed": args.seed, "train_negatives": int(args.n_train_neg),
            "counts": {s: {"n": len(d), "pos": int((d['class'] == 1).sum()),
                           "pos_with_box": int(d.loc[d['class'] == 1, 'box_d'].notna().sum())} for s, d in parts.items()}}
    (OUT / "patch_cache_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
