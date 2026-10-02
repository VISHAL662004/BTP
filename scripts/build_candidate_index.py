"""Phase 3: build data/candidates/candidate_index.csv (candidates_V2 + split + voxel coords + lung flag)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import load_candidates
from src.preprocessing.candidates import build_candidate_index

DATA = Path("data")


def main():
    scans = pd.read_csv(DATA / "metadata" / "scans.csv")
    split_of = {}
    for name in ("train", "val", "test"):
        for u in pd.read_csv(DATA / "splits" / f"{name}.csv").seriesuid:
            split_of[u] = name
    cands = load_candidates(DATA / "candidates" / "candidates_V2.csv")
    idx = build_candidate_index(DATA, cands, scans, split_of)
    idx.to_csv(DATA / "candidates" / "candidate_index.csv", index=False)
    summ = idx.groupby("split").agg(candidates=("class", "size"), positives=("class", "sum"), in_lung=("in_lung", "mean"))
    pos = idx[idx["class"] == 1]
    out = {
        "rows": len(idx), "per_split": summ.round(4).reset_index().to_dict("records"),
        "positives_in_lung_frac": round(float(pos.in_lung.mean()), 4),
        "negatives_in_lung_frac": round(float(idx[idx['class'] == 0].in_lung.mean()), 4),
        "positives_outside_lung": int((~pos.in_lung).sum()),
    }
    (DATA / "metadata" / "candidate_index_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
