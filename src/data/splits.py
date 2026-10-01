"""Scan-level train/val/test splits (no leakage between partitions).

LUNA16's 10 subsets are already disjoint at scan (patient) level, so the split
is defined on subsets: whole scans move together, no slice/patch-level splitting.
"""
import json
from pathlib import Path

import pandas as pd

SPLIT_SUBSETS = {"train": list(range(0, 8)), "val": [8], "test": [9]}
SPLIT_SEED = None  # deterministic by subset; no randomness involved


def make_splits(scan_df: pd.DataFrame) -> dict:
    splits = {}
    for name, subsets in SPLIT_SUBSETS.items():
        splits[name] = sorted(scan_df.loc[scan_df.subset.isin(subsets), "seriesuid"])
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits: dict) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise ValueError(f"Leakage: {len(overlap)} scans in both {a} and {b}")


def save_splits(splits: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"method": "subset-level (scan-level) split", "subsets": SPLIT_SUBSETS,
            "seed": SPLIT_SEED, "counts": {k: len(v) for k, v in splits.items()}}
    (out_dir / "split_info.json").write_text(json.dumps(meta, indent=2))
    for name, uids in splits.items():
        pd.Series(uids, name="seriesuid").to_csv(out_dir / f"{name}.csv", index=False)
