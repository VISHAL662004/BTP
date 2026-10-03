"""Unstructured (weight-level) magnitude pruning with persistent masks.

Prunable tensors: weights of Conv2d / Linear layers, EXCEPT the final objectness/box output layers
(`head.obj`, `head.box`: tiny and the model's direct outputs). BatchNorm, biases, positional
embeddings and LayerNorm are never pruned. Masks only ever remove weights (progressive nesting).
"""
import torch
from torch import nn

EXCLUDE = ("head.obj", "head.box")


def prunable(model: nn.Module) -> dict:
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and not any(name == e or name.startswith(e + ".") for e in EXCLUDE):
            out[name] = m.weight
    return out


def compute_global_masks(model: nn.Module, sparsity: float, prev_masks: dict = None, min_keep: float = 0.02) -> dict:
    """Keep the largest-|w| fraction (1 - sparsity) of ALL prunable weights (global threshold),
    never resurrecting weights pruned earlier, and never leaving a layer below `min_keep` of its
    weights (avoids disconnecting a layer whose weights are globally small)."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    params = prunable(model)
    prev = prev_masks or {n: torch.ones_like(w, dtype=torch.bool) for n, w in params.items()}
    scores = {n: (w.detach().abs() * prev[n]).flatten() for n, w in params.items()}
    allv = torch.cat([s[prev[n].flatten()] for n, s in scores.items()])
    k = int(round(sparsity * sum(p.numel() for p in params.values())))
    already = sum(int((~m).sum()) for m in prev.values())
    n_remove = max(k - already, 0)
    masks = {n: m.clone() for n, m in prev.items()}
    if n_remove > 0:
        thr = torch.kthvalue(allv, min(n_remove, allv.numel())).values
        for n, s in scores.items():
            masks[n] = prev[n] & (s.view_as(prev[n]) > thr)
    for n, w in params.items():                     # per-layer safety floor
        floor = max(int(min_keep * w.numel()), 1)
        if int(masks[n].sum()) < floor:
            top = torch.topk((w.detach().abs() * prev[n]).flatten(), floor).indices
            m = torch.zeros(w.numel(), dtype=torch.bool); m[top] = True
            masks[n] = m.view_as(prev[n]) & prev[n]
    return masks


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict) -> None:
    params = prunable(model)
    for n, m in masks.items():
        params[n].mul_(m.to(params[n].device, params[n].dtype))


def sparsity_report(model: nn.Module) -> dict:
    """Sparsity over prunable tensors and over ALL parameters (measured from actual zeros)."""
    pr = prunable(model)
    pz = sum(int((w == 0).sum()) for w in pr.values()); pn = sum(w.numel() for w in pr.values())
    allz = sum(int((p == 0).sum()) for p in model.parameters()); alln = sum(p.numel() for p in model.parameters())
    return {"prunable_params": pn, "prunable_sparsity": pz / pn, "total_params": alln, "nonzero_params": alln - allz,
            "total_sparsity": allz / alln}
