import numpy as np
import pytest

from src.preprocessing.hu import to_hu
from src.preprocessing.io import Volume, _canonicalize
from src.preprocessing.lung_region import lung_bbox_voxels, point_in_lung
from src.preprocessing.normalization import window_normalize
from src.preprocessing.patches import extract_25d_patch
from src.preprocessing.resampling import resample
from src.preprocessing.slice_context import slice_indices, stack_slices


def test_hu_clip_removes_padding():
    hu = to_hu(np.array([-2000, -1024, 0, 4000], np.int16))
    assert hu.tolist() == [-1024.0, -1024.0, 0.0, 3071.0]


def test_window_normalize_range():
    out = window_normalize(np.array([-2000.0, -1000.0, -300.0, 400.0, 900.0], np.float32))
    assert np.allclose(out, [0, 0, 0.5, 1, 1])
    with pytest.raises(ValueError):
        window_normalize(out, 5, 5)


def test_canonicalize_flip_keeps_world_mapping():
    arr = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)          # z,y,x
    ox, oy, oz, sp = 10.0, 20.0, 30.0, (0.5, 0.5, 2.0)
    flipped_stored = arr[:, ::-1, ::-1]                                   # as stored in a flipped scan
    v = _canonicalize(flipped_stored, sp, (ox, oy, oz), (-1, 0, 0, 0, -1, 0, 0, 0, 1))
    # stored voxel (x=0,y=0,z=1) has world (ox, oy, oz+2); in the canonical grid same world -> same value
    world = np.array([ox, oy, oz + 2.0])
    xi, yi, zi = np.round(v.world_to_voxel(world)).astype(int)
    assert v.array[zi, yi, xi] == flipped_stored[1, 0, 0]
    assert v.origin[0] == ox - 3 * 0.5 and v.origin[1] == oy - 2 * 0.5


def test_resample_preserves_world_position():
    # intensity = world x coordinate -> resampled intensity must equal world x of that voxel
    sp = (2.0, 2.0, 2.0)
    arr = np.tile((np.arange(10) * 2.0 + 5.0).astype(np.float32), (6, 8, 1))   # z,y,x
    vol = Volume(arr, sp, (5.0, 0.0, 0.0))
    out = resample(vol, (1.0, 1.0, 1.0), order=1)
    assert out.array.shape == (12, 16, 20)
    xs = out.origin[0] + np.arange(out.array.shape[2]) * out.spacing[0]
    assert np.allclose(out.array[3, 3, :19], xs[:19], atol=1e-4)


def test_resample_nearest_keeps_labels():
    m = Volume(np.random.default_rng(0).integers(0, 6, (5, 6, 7)).astype(np.int16), (1.0, 1.0, 2.0), (0, 0, 0))
    out = resample(m, (1.0, 1.0, 1.0), order=0)
    assert set(np.unique(out.array)) <= set(np.unique(m.array))


def test_slice_indices_order_and_boundary():
    assert slice_indices(5, 3, 10) == [4, 5, 6]
    assert slice_indices(0, 3, 10) == [0, 0, 1]
    assert slice_indices(9, 3, 10) == [8, 9, 9]
    with pytest.raises(IndexError):
        slice_indices(0, 3, 10, boundary="error")
    with pytest.raises(ValueError):
        slice_indices(5, 4, 10)


def test_stack_slices_shape():
    assert stack_slices(np.zeros((10, 8, 8)), 4, 5).shape == (5, 8, 8)


def test_patch_centred_on_marker_and_padding():
    arr = np.zeros((20, 40, 40), np.float32)
    arr[10, 25, 12] = 1.0                                    # marker at x=12,y=25,z=10
    vol = Volume(arr, (1.0, 1.0, 1.0), (100.0, 200.0, 300.0))
    p, c = extract_25d_patch(vol, (112.0, 225.0, 310.0), size=16, n_slices=3)
    assert p.shape == (3, 16, 16) and np.allclose(c, [12, 25, 10])
    assert p[1, 8, 8] == 1.0                                 # centre channel, centre pixel
    p2, _ = extract_25d_patch(vol, (100.0, 200.0, 310.0), size=16, pad_value=-1.0)  # corner: padded
    assert p2[1, 0, 0] == -1.0 and p2[1, 8, 8] == arr[10, 0, 0]
    p3, _ = extract_25d_patch(vol, (100.0, 200.0, 999.0), size=16, pad_value=-1.0)  # outside z
    assert (p3 == -1.0).all()


def test_lung_helpers():
    b = np.zeros((10, 20, 20), bool)
    b[3:6, 5:15, 5:15] = True
    assert lung_bbox_voxels(b, (1, 1, 1), margin_mm=2) == (1, 8, 3, 17, 3, 17)
    assert point_in_lung(b, (10, 10, 4)) and not point_in_lung(b, (0, 0, 0))
    assert point_in_lung(b, (3, 10, 4), dilate_vox=2) and not point_in_lung(b, (99, 99, 99), dilate_vox=2)
    with pytest.raises(ValueError):
        lung_bbox_voxels(np.zeros((3, 3, 3), bool), (1, 1, 1))


def test_canonical_origin_matches_io():
    from types import SimpleNamespace as NS
    from src.preprocessing.candidates import canonical_origin
    row = NS(origin_x=10., origin_y=20., origin_z=30., flip_x=-1., flip_y=-1., flip_z=1.,
             dim_x=4, dim_y=3, n_slices=2, sp_x=.5, sp_y=.5, sp_z=2.)
    arr = np.zeros((2, 3, 4), np.int16)
    v = _canonicalize(arr, (.5, .5, 2.), (10., 20., 30.), (-1, 0, 0, 0, -1, 0, 0, 0, 1))
    assert canonical_origin(row) == v.origin
    row.flip_x = row.flip_y = 1.
    assert canonical_origin(row) == (10., 20., 30.)
