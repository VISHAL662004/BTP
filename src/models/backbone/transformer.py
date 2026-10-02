"""Context modules inserted between the CNN backbone and the detection head.

Interface (shared): nn.Module mapping a feature map (B, C, H, W) -> (B, C, H, W), so that the
head and backbone are unchanged and the module can be switched on/off for ablation.

* TransformerContext : CNN -> tokens -> Transformer encoder -> feature map  (the Phase 7 component)
* ConvContext        : capacity-matched CNN-only control (extra 3x3 conv layers, no attention)

Token interface: every spatial position of the CNN map is one token (H*W tokens, dim C), i.e. 16
tokens x 128 dims for a 64x64 patch after four 2x poolings. Learned positional embeddings
restore spatial order. Attention is implemented with explicit matmuls (no fused kernels) so
FLOPs are countable.
"""
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.h, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) * self.dk ** -0.5   # (B, h, T, T)
        att = self.drop(att.softmax(-1))
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, D))


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (stable without warm-up)."""

    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TransformerContext(nn.Module):
    def __init__(self, channels: int, tokens_hw=(4, 4), depth: int = 2, heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.hw = tuple(tokens_hw)
        self.pos = nn.Parameter(torch.zeros(1, self.hw[0] * self.hw[1], channels))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([EncoderLayer(channels, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(channels)

    def forward(self, f):                                   # (B, C, H, W)
        B, C, H, W = f.shape
        if (H, W) != self.hw:
            raise ValueError(f"Feature map {(H, W)} != configured token grid {self.hw}")
        t = f.flatten(2).transpose(1, 2) + self.pos         # (B, H*W, C) tokens
        for layer in self.layers:
            t = layer(t)
        return self.norm(t).transpose(1, 2).reshape(B, C, H, W)


class ConvContext(nn.Module):
    """Capacity-matched control: `depth` residual 3x3 conv blocks at the same resolution."""

    def __init__(self, channels: int, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for _ in range(depth)])

    def forward(self, f):
        for b in self.blocks:
            f = f + b(f)
        return f


def build_context(cfg_model: dict, channels: int):
    kind = cfg_model.get("context", "none")
    if kind == "none":
        return None
    if kind == "transformer":
        t = cfg_model["transformer"]
        return TransformerContext(channels, tuple(t["tokens_hw"]), t["depth"], t["heads"], t["mlp_ratio"], t["dropout"])
    if kind == "conv_control":
        return ConvContext(channels, cfg_model["conv_control"]["depth"])
    raise ValueError(f"Unknown context {kind!r}")
