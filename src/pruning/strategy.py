"""Structured (channel) pruning: physically rebuild a smaller network.

Pruned channels (CNN backbone only):
  * every stage's inner channels (conv1 output / conv2 input);
  * stage output channels of stages 1..N-1 (they feed only the next stage's conv1 and, if present,
    the stage's own SE/CBAM channel MLP, which is sliced consistently).
NOT pruned: the last stage's output channels (interface to the context module / head), the input
channels, BatchNorm statistics of kept channels (copied), the Transformer/extra-conv context module and
the detection head. Importance = L1 norm of a filter times |BN scale| (a filter whose BN scale is ~0
contributes ~nothing). Kept channels keep their trained weights, so the pruned network starts close
to the original and is then fine-tuned (recovery).
"""
import copy

import torch
from torch import nn

from src.models.attention.attention import CBAM, SqueezeExcitation
from src.models.detector import build_detector


def _channel_importance(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> torch.Tensor:
    return conv.weight.detach().abs().sum((1, 2, 3)) * bn.weight.detach().abs()


def _keep_idx(importance: torch.Tensor, n_keep: int) -> torch.Tensor:
    return torch.sort(torch.topk(importance, n_keep).indices).values


def stage_widths(cfg: dict):
    """(mid_widths, out_widths) of a model config (mid defaults to out)."""
    out = list(cfg["model"]["widths"])
    return list(cfg["model"].get("mid_widths") or out), out


def target_widths(original, ratio: float, floor: int = 4, keep_last: bool = False) -> list:
    out = [max(int(round(w * (1 - ratio))), floor) for w in original]
    if keep_last:
        out[-1] = original[-1]
    return out


def _slice_conv_bn(conv_old, bn_old, conv_new, bn_new, keep_out, keep_in):
    conv_new.weight.data.copy_(conv_old.weight.data[keep_out][:, keep_in])
    for a in ("weight", "bias", "running_mean", "running_var"):
        getattr(bn_new, a).data.copy_(getattr(bn_old, a).data[keep_out])
    bn_new.num_batches_tracked.copy_(bn_old.num_batches_tracked)


def _slice_attention(old, new, keep_c):
    mlp_old = old.mlp if isinstance(old, SqueezeExcitation) else old.channel.mlp
    mlp_new = new.mlp if isinstance(new, SqueezeExcitation) else new.channel.mlp
    fc1, fc2 = mlp_old[0], mlp_old[2]
    h_new = mlp_new[0].out_features
    imp = fc1.weight.data[:, keep_c].abs().sum(1) + fc2.weight.data[keep_c].abs().sum(0)
    keep_h = _keep_idx(imp, h_new)
    mlp_new[0].weight.data.copy_(fc1.weight.data[keep_h][:, keep_c]); mlp_new[0].bias.data.copy_(fc1.bias.data[keep_h])
    mlp_new[2].weight.data.copy_(fc2.weight.data[keep_c][:, keep_h]); mlp_new[2].bias.data.copy_(fc2.bias.data[keep_c])
    if isinstance(old, CBAM):
        new.spatial.load_state_dict(old.spatial.state_dict())


@torch.no_grad()
def structured_prune(model: nn.Module, cfg: dict, mid_target, out_target):
    """Return (new_model, new_cfg) with inner widths `mid_target` and stage-output widths `out_target`
    (each a list per stage; last stage output must stay unchanged)."""
    old_bb = model.backbone
    n = len(old_bb.blocks)
    if out_target[-1] != cfg["model"]["widths"][-1]:
        raise ValueError("last stage output width must stay unchanged")
    new_cfg = copy.deepcopy(cfg)
    new_cfg["model"]["widths"] = [int(w) for w in out_target]
    new_cfg["model"]["mid_widths"] = [int(w) for w in mid_target]
    new = build_detector(new_cfg)
    keep_in = torch.arange(new_cfg["model"]["in_channels"])
    for s in range(n):
        ob, nb = old_bb.blocks[s], new.backbone.blocks[s]
        conv1, bn1, conv2, bn2 = ob[0], ob[1], ob[3], ob[4]
        cur_mid = conv1.out_channels
        keep_mid = _keep_idx(_channel_importance(conv1, bn1), min(mid_target[s], cur_mid))
        keep_out = _keep_idx(_channel_importance(conv2, bn2), min(out_target[s], conv2.out_channels))
        _slice_conv_bn(conv1, bn1, nb[0], nb[1], keep_mid, keep_in)
        _slice_conv_bn(conv2, bn2, nb[3], nb[4], keep_out, keep_mid)
        old_attn = next((m for m in ob if isinstance(m, (SqueezeExcitation, CBAM))), None)
        new_attn = next((m for m in nb if isinstance(m, (SqueezeExcitation, CBAM))), None)
        if old_attn is not None:
            _slice_attention(old_attn, new_attn, keep_out)
        keep_in = keep_out
    if model.context is not None:
        new.context.load_state_dict(model.context.state_dict())
    new.head.load_state_dict(model.head.state_dict())
    return new.to(next(model.parameters()).device), new_cfg
