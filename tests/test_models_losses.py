import numpy as np
import torch

from src.data.patch_store import dihedral, random_dihedral
from src.losses.detection_loss import DetectionLoss, focal_loss, square_iou
from src.models.detector import build_detector
from src.models.heads.detection import decode_box
from src.utils.config import load_experiment_config


def test_model_forward_shapes_and_params():
    cfg = load_experiment_config("configs/experiments/baseline.yaml")
    m = build_detector(cfg)
    obj, box = m(torch.randn(4, 1, 64, 64))
    assert obj.shape == (4,) and box.shape == (4, 3)
    assert sum(p.numel() for p in m.parameters()) < 1_000_000


def test_fresh_head_predicts_prior():
    cfg = load_experiment_config("configs/experiments/baseline.yaml")
    m = build_detector(cfg).eval()
    obj, box = m(torch.randn(8, 1, 64, 64))
    assert torch.sigmoid(obj).max() < 0.05
    assert torch.allclose(decode_box(box)[:, 2], torch.full((8,), 8.0))


def test_square_iou_known_values():
    a = torch.tensor([[0., 0., 2.], [0., 0., 2.], [0., 0., 2.]])
    b = torch.tensor([[0., 0., 2.], [1., 0., 2.], [10., 0., 2.]])
    assert torch.allclose(square_iou(a, b), torch.tensor([1.0, 1 / 3, 0.0]), atol=1e-5)


def test_focal_downweights_easy_examples():
    easy = focal_loss(torch.tensor([-8.0]), torch.tensor([0.0]))
    hard = focal_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    assert easy < hard * 1e-3
    assert focal_loss(torch.tensor([0.0]), torch.tensor([1.0])) > 0


def test_detection_loss_total_and_nan_boxes_ignored():
    lossf = DetectionLoss(1.0, 2.0)
    obj = torch.tensor([0.5, -0.5, 0.2]); braw = torch.zeros(3, 3)
    y = torch.tensor([1., 0., 1.])
    box = torch.tensor([[1., 1., 8.], [float("nan")] * 3, [float("nan")] * 3])   # 3rd positive has no box
    total, parts = lossf(obj, braw, y, box)
    assert abs(total.item() - (parts["focal"] + 2.0 * parts["loc"])) < 1e-5
    assert parts["loc"] > 0


def test_loc_gradient_flows_to_box_only():
    lossf = DetectionLoss(0.0, 1.0)
    obj = torch.zeros(1, requires_grad=True); braw = torch.zeros(1, 3, requires_grad=True)
    total, _ = lossf(obj, braw, torch.ones(1), torch.tensor([[2., 0., 8.]]))
    total.backward()
    assert braw.grad.abs().sum() > 0 and obj.grad.abs().sum() == 0


def test_dihedral_moves_image_and_box_consistently():
    # a single bright pixel offset (dx, dy) from the centre must still sit at centre + new (dx, dy)
    P, c = 16, 8   # centre pixel index P/2
    rng = np.random.default_rng(0)
    for k in range(4):
        for flip in (False, True):
            dx, dy = int(rng.integers(-5, 6)), int(rng.integers(-5, 6))
            img = torch.zeros(1, 1, P, P); img[0, 0, c + dy, c + dx] = 1.0
            box = torch.tensor([[float(dx), float(dy), 6.0]])
            xi, bi = dihedral(img, box, k, flip)
            r, col = np.unravel_index(int(xi.flatten().argmax()), (P, P))
            assert (col - c, r - c) == (int(bi[0, 0]), int(bi[0, 1])), (k, flip)
            assert bi[0, 2] == 6.0


def test_random_dihedral_preserves_pixel_count():
    x = torch.rand(32, 1, 16, 16); b = torch.zeros(32, 3)
    xo, _ = random_dihedral(x, b, torch.Generator().manual_seed(0))
    assert torch.allclose(xo.sum((1, 2, 3)), x.sum((1, 2, 3)))


def test_centered_channels():
    import pytest
    from src.data.patch_store import centered_channels
    assert centered_channels(5, 1) == [2] and centered_channels(5, 3) == [1, 2, 3] and centered_channels(5, 5) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        centered_channels(5, 2)


def test_flops_scale_with_input_channels():
    from src.efficiency.flops import count_flops
    from src.utils.config import load_experiment_config
    cfg = load_experiment_config("configs/experiments/baseline.yaml")
    f1 = count_flops(build_detector(cfg), (1, 1, 64, 64))["flops_per_sample"]
    cfg["model"]["in_channels"] = 5
    f5 = count_flops(build_detector(cfg), (1, 5, 64, 64))["flops_per_sample"]
    assert 0 < f1 < f5 < 1.2 * f1       # only the first conv grows
