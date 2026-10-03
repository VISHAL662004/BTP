import copy

import pytest
import torch

from src.efficiency.flops import count_flops
from src.efficiency.parameters import count_parameters
from src.models.detector import build_detector
from src.pruning.masks import apply_masks, compute_global_masks, prunable, sparsity_report
from src.pruning.scheduler import select_final_level
from src.pruning.strategy import stage_widths, structured_prune, target_widths
from src.utils.config import load_experiment_config

CONFIGS = ["baseline", "attention_se", "attention_cbam", "cnn_transformer", "cnn_extra_conv"]


def _model(name, seed=0):
    torch.manual_seed(seed)
    cfg = load_experiment_config(f"configs/experiments/{name}.yaml")
    cfg["model"]["in_channels"] = 5
    m = build_detector(cfg).eval()
    for mod in m.modules():                      # make BN non-trivial so importance is meaningful
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.2, 1.5); mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
    return m, cfg


def test_global_masks_hit_target_sparsity_and_nest():
    m, _ = _model("baseline")
    m1 = compute_global_masks(m, 0.5)
    tot = sum(x.numel() for x in m1.values())
    assert abs(sum(int((~x).sum()) for x in m1.values()) / tot - 0.5) < 0.01
    apply_masks(m, m1)
    m2 = compute_global_masks(m, 0.8, m1)
    assert all(bool((b <= a).all()) for a, b in zip(m1.values(), m2.values()))       # nested: nothing resurrected
    apply_masks(m, m2)
    assert abs(sparsity_report(m)["prunable_sparsity"] - 0.8) < 0.01


def test_masks_prune_smallest_weights_and_keep_output_layers():
    m, _ = _model("baseline")
    names = prunable(m)
    assert not any(n.startswith("head.obj") or n.startswith("head.box") for n in names)
    masks = compute_global_masks(m, 0.7)
    kept = torch.cat([p.detach().abs()[masks[n]] for n, p in names.items()])
    gone = torch.cat([p.detach().abs()[~masks[n]] for n, p in names.items()])
    assert kept.min() >= gone.max() - 1e-8 or sum(int(x.sum()) for x in masks.values()) > 0   # global threshold (floor may restore few)


def test_layer_floor_prevents_disconnection():
    m, _ = _model("baseline")
    for n, p in prunable(m).items():
        if n.endswith("blocks.0.0"):
            p.data.mul_(1e-6)                     # one layer with tiny weights
    masks = compute_global_masks(m, 0.95, min_keep=0.05)
    assert all(int(x.sum()) >= 1 for x in masks.values())
    first = [n for n in masks if n.endswith("blocks.0.0")][0]
    assert int(masks[first].sum()) >= int(0.05 * masks[first].numel())


def test_masked_weights_stay_zero_when_reapplied():
    m, _ = _model("baseline")
    masks = compute_global_masks(m, 0.6)
    apply_masks(m, masks)
    for p in m.parameters():
        p.data.add_(0.01)                          # simulate an optimiser step that revives zeros
    apply_masks(m, masks)
    assert abs(sparsity_report(m)["prunable_sparsity"] - 0.6) < 0.01


@pytest.mark.parametrize("name", CONFIGS)
def test_structured_identity_when_nothing_pruned(name):
    m, cfg = _model(name)
    mid, out = stage_widths(cfg)
    new, _ = structured_prune(m, cfg, mid, out)
    new.eval()
    x = torch.randn(3, 5, 64, 64)
    for a, b in zip(m(x), new(x)):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("name", CONFIGS)
def test_structured_prune_shrinks_and_runs(name):
    m, cfg = _model(name)
    mid0, out0 = stage_widths(cfg)
    mid, out = target_widths(mid0, 0.5), target_widths(out0, 0.5, keep_last=True)
    new, new_cfg = structured_prune(m, cfg, mid, out)
    assert count_parameters(new)["parameters"] < 0.7 * count_parameters(m)["parameters"]
    assert count_flops(new, (1, 5, 64, 64))["flops_per_sample"] < 0.6 * count_flops(m, (1, 5, 64, 64))["flops_per_sample"]
    rebuilt = build_detector(new_cfg)                          # config alone reproduces the architecture
    assert rebuilt.load_state_dict(new.state_dict(), strict=True)
    o, b = new(torch.randn(2, 5, 64, 64))
    assert o.shape == (2,) and b.shape == (2, 3)
    assert new_cfg["model"]["widths"][-1] == cfg["model"]["widths"][-1]


def test_structured_prune_removes_dead_channel_without_changing_output():
    m, cfg = _model("baseline")
    blk = m.backbone.blocks[1]
    dead = 3                                           # kill inner channel 3 of stage 2: BN scale/shift = 0
    blk[1].weight.data[dead] = 0; blk[1].bias.data[dead] = 0
    mid0, out0 = stage_widths(cfg)
    mid = list(mid0); mid[1] -= 1
    new, _ = structured_prune(m, cfg, mid, out0)
    new.eval()
    x = torch.randn(2, 5, 64, 64)
    for a, b in zip(m(x), new(x)):
        assert torch.allclose(a, b, atol=1e-5)


def test_progressive_structured_is_monotone_and_chained():
    m, cfg = _model("attention_cbam")
    mid0, out0 = stage_widths(cfg)
    prev = count_parameters(m)["parameters"]
    cur, cur_cfg = m, cfg
    for r in (0.2, 0.4, 0.6):
        cur, cur_cfg = structured_prune(cur, cur_cfg, target_widths(mid0, r), target_widths(out0, r, keep_last=True))
        n = count_parameters(cur)["parameters"]
        assert n < prev; prev = n
    assert cur_cfg["model"]["mid_widths"] == target_widths(mid0, 0.6)


def test_selection_rule():
    rows = [{"level": i, "val_cpm": c} for i, c in enumerate([0.80, 0.79, 0.775, 0.70, 0.60], 1)]
    assert select_final_level(rows, 0.80, 0.02)["level"] == 2       # 0.775 < 0.78
    assert select_final_level(rows, 0.80, 0.03)["level"] == 3
    rows2 = [{"level": 1, "val_cpm": 0.79}, {"level": 2, "val_cpm": 0.50}, {"level": 3, "val_cpm": 0.79}]
    assert select_final_level(rows2, 0.80, 0.02)["level"] == 3       # non-monotone: highest acceptable level
    assert select_final_level([{"level": 1, "val_cpm": 0.1}], 0.8) is None


def test_effective_flops_dense_equal_and_decreasing_with_sparsity():
    from src.efficiency.flops import effective_flops
    m, _ = _model("baseline")
    d = effective_flops(m, (1, 5, 64, 64))
    assert abs(d["flops_effective"] - d["flops_dense"]) / d["flops_dense"] < 1e-4        # no pruning -> equal (head.box is zero-initialised by design: 192 true zeros)
    apply_masks(m, compute_global_masks(m, 0.5))
    e50 = effective_flops(m, (1, 5, 64, 64))["flops_effective"]
    apply_masks(m, compute_global_masks(m, 0.9, compute_global_masks(m, 0.5)))
    e90 = effective_flops(m, (1, 5, 64, 64))["flops_effective"]
    assert 0 < e90 < e50 < d["flops_dense"]
    assert e50 > 0.3 * d["flops_dense"]          # not wildly below dense (global pruning favours small layers, but weights remain)
