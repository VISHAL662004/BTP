"""Phase 2: validate LUNA16 and write statistics to data/metadata and splits to data/splits.

Usage: python scripts/validate_dataset.py [--crc]
  --crc  additionally CRC-test every zip member (reads all ~66 GB; slow)
"""
import argparse
import sys
import zipfile
import zipfile as _zf
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import validation as V
from src.data.annotations import load_annotations, load_candidates
from src.data.luna16 import index_all
from src.data.splits import make_splits, save_splits

DATA = Path("data")
META = DATA / "metadata"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crc", action="store_true")
    args = ap.parse_args()
    META.mkdir(parents=True, exist_ok=True)
    issues = []

    scans = index_all(DATA)
    scan_df = V.scans_to_frame(scans)
    scan_df.to_csv(META / "scans.csv", index=False)
    issues += V.check_scans(scan_df)

    ann = load_annotations(DATA / "annotations.csv")
    issues += V.check_annotations(ann, scan_df)
    ann_vox = V.annotation_voxel_coords(ann, scan_df)
    ann_vox.drop(columns=["raw_name"]).to_csv(META / "annotations_voxel.csv", index=False)

    cand1 = load_candidates(DATA / "candidates.csv")
    issues += V.check_candidates(cand1, ann, scan_df, "candidates")
    with zipfile.ZipFile(DATA / "candidates_V2.zip") as z:
        out = DATA / "candidates"
        out.mkdir(exist_ok=True)
        z.extract("candidates_V2.csv", out)
    cand2 = load_candidates(DATA / "candidates" / "candidates_V2.csv")
    issues += V.check_candidates(cand2, ann, scan_df, "candidates_V2")

    # lung segmentation masks
    with zipfile.ZipFile(DATA / "seg-lungs-LUNA16.zip") as z:
        seg = {Path(n).stem for n in z.namelist() if n.endswith(".mhd")}
    missing_seg = sorted(set(scan_df.seriesuid) - seg)
    if missing_seg:
        issues.append(f"{len(missing_seg)} scans lack a lung segmentation mask")

    # annotation / candidate relationships
    nod_per_scan = ann.groupby("seriesuid").size().reindex(scan_df.seriesuid, fill_value=0)
    scan_df["n_nodules"] = nod_per_scan.values
    scan_df["n_cand_v1"] = scan_df.seriesuid.map(cand1.groupby("seriesuid").size()).fillna(0).astype(int)
    scan_df["n_cand_v2"] = scan_df.seriesuid.map(cand2.groupby("seriesuid").size()).fillna(0).astype(int)
    scan_df.to_csv(META / "scans.csv", index=False)

    if args.crc:
        for i in range(10):
            with _zf.ZipFile(DATA / f"subset{i}.zip") as z:
                bad = z.testzip()
            print(f"subset{i} CRC:", "OK" if bad is None else f"BAD member {bad}")
            if bad:
                issues.append(f"subset{i}.zip CRC failure at {bad}")

    stats = {
        "scans": len(scan_df),
        "scans_per_subset": scan_df.groupby("subset").size().to_dict(),
        "annotations": len(ann),
        "scans_with_nodules": int((nod_per_scan > 0).sum()),
        "scans_without_nodules": int((nod_per_scan == 0).sum()),
        "candidates_v1": len(cand1), "candidates_v1_pos": int(cand1["class"].sum()),
        "candidates_v2": len(cand2), "candidates_v2_pos": int(cand2["class"].sum()),
        "scans_with_flipped_xy": int(((scan_df.flip_x < 0) | (scan_df.flip_y < 0)).sum()),
        "slices": scan_df.n_slices.describe().round(2).to_dict(),
        "dim_xy_unique": sorted({(int(a), int(b)) for a, b in zip(scan_df.dim_x, scan_df.dim_y)}),
        "spacing_xy_mm": scan_df.sp_x.describe().round(4).to_dict(),
        "spacing_z_mm": scan_df.sp_z.describe().round(4).to_dict(),
        "nodule_diameter_mm": ann.diameter_mm.describe().round(3).to_dict(),
        "nodules_per_scan": nod_per_scan.describe().round(3).to_dict(),
    }
    pd.Series(stats).to_json(META / "dataset_stats.json", indent=2)

    splits = make_splits(scan_df)
    save_splits(splits, DATA / "splits")
    split_ann = {k: int(ann.seriesuid.isin(v).sum()) for k, v in splits.items()}

    print(pd.Series(stats).to_string())
    print("split scans:", {k: len(v) for k, v in splits.items()}, "annotations:", split_ann)
    (META / "validation_issues.txt").write_text("\n".join(issues) if issues else "No issues found.\n")
    print(f"\n{len(issues)} issue(s)")
    for i in issues[:40]:
        print(" -", i)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
