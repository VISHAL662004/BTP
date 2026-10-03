"""Attention modules for the CNN backbone (Phase 8).

Interface (shared): nn.Module mapping a feature map (B, C, H, W) -> (B, C, H, W) of the same shape,
so attention can be switched on/off per stage for ablation.

* SqueezeExcitation : channel attention (Hu et al., SENet) -- global average pool -> bottleneck MLP -> sigmoid gate.
* CBAM              : channel attention (avg- and max-pooled descriptors through a shared MLP) followed by
                      spatial attention (7x7 conv over channel-wise avg/max maps), Woo et al. 2018.

These are existing techniques (see project literature: SE-ViT [4], DAFNet's channel+spatial attention [6]);
no novelty is claimed for them.
"""
import torch
from torch import nn


def _hidden(channels: int, reduction: int) -> int:
    return max(channels // reduction, 4)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        h = _hidden(channels, reduction)
        self.mlp = nn.Sequential(nn.Linear(channels, h), nn.ReLU(inplace=True), nn.Linear(h, channels))
        self.last_gate = None
        self.keep = False

    def forward(self, x):
        gate = torch.sigmoid(self.mlp(x.mean((2, 3))))              # (B, C)
        if self.keep:
            self.last_gate = gate.detach()
        return x * gate[:, :, None, None]


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        h = _hidden(channels, reduction)
        self.mlp = nn.Sequential(nn.Linear(channels, h), nn.ReLU(inplace=True), nn.Linear(h, channels))

    def forward(self, x):
        g = self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))
        return torch.sigmoid(g)[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("spatial kernel size must be odd")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        m = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return torch.sigmoid(self.conv(m))                           # (B, 1, H, W)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)
        self.last_gate = None
        self.last_spatial = None
        self.keep = False

    def forward(self, x):
        cg = self.channel(x)
        x = x * cg
        sg = self.spatial(x)
        if self.keep:
            self.last_gate, self.last_spatial = cg.flatten(1).detach(), sg.detach()
        return x * sg


def build_attention(cfg_attn, channels: int):
    """cfg_attn: None/False/{'type': 'none'} -> None; {'type': 'se'|'cbam', 'reduction': 4, 'spatial_kernel': 7}."""
    if not cfg_attn:
        return None
    kind = cfg_attn.get("type", "none")
    if kind == "none":
        return None
    r = cfg_attn.get("reduction", 4)
    if kind == "se":
        return SqueezeExcitation(channels, r)
    if kind == "cbam":
        return CBAM(channels, r, cfg_attn.get("spatial_kernel", 7))
    raise ValueError(f"Unknown attention type {kind!r}")
