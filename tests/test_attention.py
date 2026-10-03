import pytest
import torch

from src.efficiency.memory import activation_memory
from src.efficiency.parameters import count_parameters
from src.models.attention.attention import CBAM, SqueezeExcitation, build_attention
from src.models.detector import build_detector
from src.utils.config import load_experiment_config


def _ref_cfg():
    cfg = load_experiment_config("configs/experiments/baseline.yaml")
    cfg["model"]["in_channels"] = 5
    return cfg


@pytest.mark.parametrize("mod", [SqueezeExcitation(32), CBAM(32)])
def test_attention_preserves_shape_and_gates_in_unit_interval(mod):
    mod.keep = True
    x = torch.randn(3, 32, 16, 16)
    y = mod(x)
    assert y.shape == x.shape
    assert 0 < mod.last_gate.min() and mod.last_gate.max() < 1
    assert mod.last_gate.shape == (3, 32)
    assert (y.abs() <= x.abs() + 1e-6).all()          # sigmoid gates can only attenuate


def test_cbam_spatial_map_shape_and_content_dependence():
    m = CBAM(16).eval(); m.keep = True
    x = torch.randn(2, 16, 10, 12)
    m(x)
    s1 = m.last_spatial.clone()
    assert s1.shape == (2, 1, 10, 12)
    x2 = x.clone(); x2[:, :, 0:3, 0:3] += 4 * torch.randn(2, 16, 3, 3)
    m(x2)
    assert not torch.allclose(s1, m.last_spatial)


def test_channel_gate_depends_on_content():
    m = SqueezeExcitation(8).eval(); m.keep = True
    m(torch.zeros(1, 8, 4, 4)); g0 = m.last_gate.clone()
    m(torch.randn(1, 8, 4, 4) * 5)
    assert not torch.allclose(g0, m.last_gate)


def test_build_attention_none_variants_and_errors():
    assert build_attention(None, 8) is None and build_attention(False, 8) is None and build_attention({"type": "none"}, 8) is None
    with pytest.raises(ValueError):
        build_attention({"type": "bogus"}, 8)
    with pytest.raises(ValueError):
        CBAM(8, spatial_kernel=4)


@pytest.mark.parametrize("name", ["attention_se", "attention_cbam"])
def test_detector_with_attention_runs_trains_and_budget(name):
    p_ref = count_parameters(build_detector(_ref_cfg()))["parameters"]
    m = build_detector(load_experiment_config(f"configs/experiments/{name}.yaml"))
    assert len(m.backbone.attention_modules()) == 4
    obj, box = m(torch.randn(4, 5, 64, 64))
    assert obj.shape == (4,) and box.shape == (4, 3)
    (obj.sum() + box.sum()).backward()
    for a in m.backbone.attention_modules():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.parameters())
    extra = count_parameters(m)["parameters"] - p_ref
    assert 0 < extra < 0.2 * p_ref                    # attention is lightweight (<20% extra params)


def test_stage_selection_and_old_configs_still_build():
    cfg = load_experiment_config("configs/experiments/attention_se.yaml")
    cfg["model"]["attention"]["stages"] = [3]
    assert len(build_detector(cfg).backbone.attention_modules()) == 1
    assert build_detector(_ref_cfg()).backbone.attention_modules() == []   # base.yaml stores `attention: false`


def test_activation_memory_grows_with_attention():
    a = activation_memory(build_detector(_ref_cfg()), (2, 5, 64, 64))
    b = activation_memory(build_detector(load_experiment_config("configs/experiments/attention_cbam.yaml")), (2, 5, 64, 64))
    assert b["activations_mb_per_sample"] > a["activations_mb_per_sample"] and b["parameters_mb"] > a["parameters_mb"]
