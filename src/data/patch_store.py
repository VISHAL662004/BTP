"""In-memory patch store + GPU-side dihedral augmentation with box-target adjustment."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class PatchStore:
    """Loads cached candidate patches. `channels`: list of channel indices to keep
    (e.g. [2] = centre slice of 5 for the 2D baseline) or None for all."""

    def __init__(self, root, split: str, channels=None):
        root = Path(root)
        arr = np.load(root / f"patches_{split}.npy", mmap_mode="r")
        self.meta = pd.read_csv(root / f"patches_{split}.csv")
        if len(arr) != len(self.meta):
            raise ValueError(f"{split}: patches ({len(arr)}) and metadata ({len(self.meta)}) differ")
        sel = slice(None) if channels is None else list(channels)
        self.x = torch.from_numpy(np.ascontiguousarray(arr[:, sel]))       # uint8 (N, C, H, W)
        self.y = torch.from_numpy(self.meta["label"].to_numpy(np.float32))
        self.box = torch.from_numpy(self.meta[["box_dx", "box_dy", "box_d"]].to_numpy(np.float32))

    def __len__(self):
        return len(self.y)


def dihedral(x: torch.Tensor, box: torch.Tensor, k: int, flip: bool):
    """Apply (flip along x if `flip`) then k * 90deg counter-clockwise rotation to images
    (N, C, H, W) and the matching offsets in box[:, :2] = (dx, dy). Diameter is unchanged.

    The patch centre is pixel index P/2 (even P), whereas flips/rotations pivot about
    (P-1)/2; a one-pixel roll after each op moves the centre pixel back onto itself so
    image and box targets stay aligned (one border row/column wraps around)."""
    box = box.clone()
    if flip:
        x = torch.roll(torch.flip(x, dims=[-1]), 1, dims=-1)
        box[:, 0] = -box[:, 0]
    for _ in range(k % 4):
        x = torch.roll(torch.rot90(x, 1, dims=[-2, -1]), 1, dims=-2)
        box[:, 0], box[:, 1] = box[:, 1].clone(), -box[:, 0].clone()   # (dx, dy) -> (dy, -dx)
    return x, box


def random_dihedral(x: torch.Tensor, box: torch.Tensor, gen: torch.Generator = None):
    """Per-sample random element of the dihedral group D4 (8 transforms)."""
    n = x.shape[0]
    code = torch.randint(0, 8, (n,), generator=gen).to(x.device)
    out_x, out_b = x.clone(), box.clone()
    for c in range(1, 8):
        m = code == c
        if m.any():
            xi, bi = dihedral(x[m], box[m], c % 4, c >= 4)
            out_x[m], out_b[m] = xi, bi
    return out_x, out_b
