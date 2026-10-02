import pytest
import torch

from src.efficiency.flops import count_flops
from src.efficiency.parameters import count_parameters
from src.models.backbone.transformer import ConvContext, MultiHeadSelfAttention, TransformerContext
from src.models.detector import build_detector
from src.utils.config import load_experiment_config


def test_transformer_preserves_shape_and_uses_all_tokens():
    m = TransformerContext(128, (4, 4), depth=2, heads=4).eval()
    f = torch.randn(3, 128, 4, 4)
    out = m(f)
    assert out.shape == f.shape
    # self-attention mixes tokens: perturbing one token must change other positions' output
    f2 = f.clone(); f2[:, :, 0, 0] += torch.randn(3, 128) * 3   # (a constant shift would be removed by LayerNorm)
    d = (m(f2) - out).abs().amax(1)
    assert d[:, 3, 3].max() > 1e-4


def test_wrong_grid_rejected():
    with pytest.raises(ValueError):
        TransformerContext(128, (4, 4))(torch.randn(1, 128, 8, 8))


def test_attention_rows_sum_to_one_and_matches_torch_reference():
    torch.manual_seed(0)
    a = MultiHeadSelfAttention(32, 4).eval()
    x = torch.randn(2, 7, 32)
    ref = torch.nn.MultiheadAttention(32, 4, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(a.qkv.weight); ref.in_proj_bias.copy_(a.qkv.bias)
        ref.out_proj.weight.copy_(a.proj.weight); ref.out_proj.bias.copy_(a.proj.bias)
        assert torch.allclose(a(x), ref(x, x, x, need_weights=False)[0], atol=1e-5)


def test_gradients_reach_cnn_through_transformer():
    cfg = load_experiment_config("configs/experiments/cnn_transformer.yaml")
    m = build_detector(cfg)
    obj, box = m(torch.randn(4, 5, 64, 64))
    (obj.sum() + box.sum()).backward()
    assert m.backbone.blocks[0][0].weight.grad.abs().sum() > 0
    assert m.context.pos.grad.abs().sum() > 0


def test_configs_build_and_parameter_budget():
    ref = load_experiment_config("configs/experiments/baseline.yaml"); ref["model"]["in_channels"] = 5
    p_ref = count_parameters(build_detector(ref))["parameters"]
    p_tr = count_parameters(build_detector(load_experiment_config("configs/experiments/cnn_transformer.yaml")))["parameters"]
    p_cv = count_parameters(build_detector(load_experiment_config("configs/experiments/cnn_extra_conv.yaml")))["parameters"]
    assert p_tr > p_ref and p_cv > p_ref
    assert abs(p_tr - p_cv) / p_tr < 0.25     # control is capacity-matched within 25%


def test_transformer_flops_match_analytic():
    T, D, hid, layers = 16, 128, 256, 2
    per_layer = 2 * T * D * 3 * D + 2 * (2 * T * T * D) + 2 * T * D * D + 2 * (2 * T * D * hid)
    got = count_flops(TransformerContext(D, (4, 4), depth=layers, heads=4, mlp_ratio=2.0), (1, 128, 4, 4))["flops_per_sample"]
    assert abs(got - layers * per_layer) / (layers * per_layer) < 0.02


def test_conv_context_residual_identity_when_zeroed():
    c = ConvContext(8, 2).eval()
    for p in c.parameters():
        torch.nn.init.zeros_(p)
    f = torch.randn(1, 8, 4, 4)
    assert torch.equal(c(f), f)
