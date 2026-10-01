import numpy as np
import pandas as pd
import pytest

from src.data.luna16 import ScanInfo, parse_mhd
from src.data.splits import check_no_leakage, make_splits
from src.data import validation as V


def _scan(transform=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    return ScanInfo("uid", 0, (512, 512, 100), (0.7, 0.7, 2.5), (-100.0, -200.0, -300.0),
                    transform, "MET_SHORT", "uid.raw", 512 * 512 * 100 * 2, "subset0/uid.mhd")


def test_parse_mhd():
    h = parse_mhd("NDims = 3\nDimSize = 512 512 100\nElementType = MET_SHORT\n")
    assert h["DimSize"] == "512 512 100"


def test_expected_raw_size():
    assert _scan().expected_raw_bytes == 512 * 512 * 100 * 2


def test_world_to_voxel_identity():
    v = _scan().world_to_voxel([-100 + 0.7 * 10, -200 + 0.7 * 20, -300 + 2.5 * 5])
    assert np.allclose(v, [10, 20, 5])


def test_world_to_voxel_flipped_xy():
    s = _scan((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    v = s.world_to_voxel([-100 - 0.7 * 10, -200 - 0.7 * 20, -300 + 2.5 * 5])
    assert np.allclose(v, [10, 20, 5])


def test_vectorised_matches_scan_conversion():
    s = _scan((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    df = V.scans_to_frame([s])
    ann = pd.DataFrame(dict(seriesuid=["uid"], coordX=[-107.0], coordY=[-214.0], coordZ=[-287.5], diameter_mm=[5.0]))
    m = V.annotation_voxel_coords(ann, df)
    assert np.allclose(m[["vox_x", "vox_y", "vox_z"]].iloc[0], s.world_to_voxel([-107, -214, -287.5]))
    assert bool(m.in_bounds.iloc[0])


def test_no_leakage_detected():
    with pytest.raises(ValueError):
        check_no_leakage({"train": ["a", "b"], "val": ["b"]})


def test_make_splits_disjoint_and_complete():
    df = pd.DataFrame({"seriesuid": [f"s{i}" for i in range(10)], "subset": range(10)})
    sp = make_splits(df)
    assert sum(len(v) for v in sp.values()) == 10
    assert sp["val"] == ["s8"] and sp["test"] == ["s9"]
