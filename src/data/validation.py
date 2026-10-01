"""Dataset integrity checks. Each check returns a list of human-readable issues."""
import numpy as np
import pandas as pd

from .luna16 import RAW_BYTES_PER_VOXEL

EXPECTED_SCANS = 888
EXPECTED_ANNOTATIONS = 1186


def scans_to_frame(scans) -> pd.DataFrame:
    rows = []
    for s in scans:
        rows.append(dict(
            seriesuid=s.seriesuid, subset=s.subset,
            dim_x=s.dim_size[0], dim_y=s.dim_size[1], n_slices=s.dim_size[2],
            sp_x=s.spacing[0], sp_y=s.spacing[1], sp_z=s.spacing[2],
            origin_x=s.origin[0], origin_y=s.origin[1], origin_z=s.origin[2],
            flip_x=float(s.transform[0]), flip_y=float(s.transform[4]), flip_z=float(s.transform[8]),
            axis_aligned=bool(np.allclose(np.abs(np.asarray(s.transform).reshape(3, 3)), np.eye(3))),
            element_type=s.element_type, raw_name=s.raw_name,
            raw_bytes=s.raw_bytes_in_zip, expected_raw_bytes=s.expected_raw_bytes,
        ))
    return pd.DataFrame(rows)


def check_scans(scan_df) -> list:
    issues = []
    if len(scan_df) != EXPECTED_SCANS:
        issues.append(f"Expected {EXPECTED_SCANS} scans, found {len(scan_df)}")
    dup = scan_df.seriesuid[scan_df.seriesuid.duplicated()].tolist()
    if dup:
        issues.append(f"Duplicate seriesuids: {dup}")
    bad = scan_df[~scan_df.element_type.isin(RAW_BYTES_PER_VOXEL)]
    for u in bad.seriesuid:
        issues.append(f"{u}: unsupported ElementType")
    for r in scan_df[scan_df.raw_bytes != scan_df.expected_raw_bytes].itertuples():
        issues.append(f"{r.seriesuid}: raw size {r.raw_bytes} != expected {r.expected_raw_bytes}")
    for r in scan_df[~scan_df.axis_aligned].itertuples():
        issues.append(f"{r.seriesuid}: orientation not axis-aligned (+/-1 diagonal expected)")
    for r in scan_df[(scan_df[["sp_x", "sp_y", "sp_z"]] <= 0).any(axis=1)].itertuples():
        issues.append(f"{r.seriesuid}: non-positive voxel spacing")
    for r in scan_df[scan_df.raw_name != scan_df.seriesuid + ".raw"].itertuples():
        issues.append(f"{r.seriesuid}: ElementDataFile {r.raw_name} does not match scan name")
    return issues


def check_table_vs_scans(df, scan_df, name) -> list:
    issues = []
    unknown = sorted(set(df.seriesuid) - set(scan_df.seriesuid))
    if unknown:
        issues.append(f"{name}: {len(unknown)} seriesuids have no scan, e.g. {unknown[:3]}")
    if df[["coordX", "coordY", "coordZ"]].isna().any().any():
        issues.append(f"{name}: NaN coordinates")
    return issues


def annotation_voxel_coords(ann, scan_df) -> pd.DataFrame:
    """Annotate each row with voxel coordinates and an in-bounds flag."""
    m = ann.merge(scan_df, on="seriesuid", how="inner")
    for ax, d in zip("xyz", ("dim_x", "dim_y", "n_slices")):
        # axis-aligned direction: diagonal entries are +/-1 (flip_*), so divide by them
        m[f"vox_{ax}"] = (m[f"coord{ax.upper()}"] - m[f"origin_{ax}"]) / (m[f"sp_{ax}"] * m[f"flip_{ax}"])
        m[f"in_{ax}"] = (m[f"vox_{ax}"] >= 0) & (m[f"vox_{ax}"] <= m[d] - 1)
    m["in_bounds"] = m.in_x & m.in_y & m.in_z
    return m


def check_annotations(ann, scan_df) -> list:
    issues = check_table_vs_scans(ann, scan_df, "annotations")
    if len(ann) != EXPECTED_ANNOTATIONS:
        issues.append(f"Expected {EXPECTED_ANNOTATIONS} annotations, found {len(ann)}")
    if (ann.diameter_mm <= 0).any():
        issues.append("annotations: non-positive diameter")
    m = annotation_voxel_coords(ann, scan_df)
    out = m[~m.in_bounds]
    if len(out):
        issues.append(f"annotations: {len(out)} outside volume bounds after world->voxel conversion")
    return issues


def check_candidates(cand, ann, scan_df, name) -> list:
    issues = check_table_vs_scans(cand, scan_df, name)
    if not set(cand["class"].unique()) <= {0, 1}:
        issues.append(f"{name}: class values outside {{0,1}}")
    return issues
