"""Simple configurable CNN backbone (baseline). Input (B, C, H, W) -> feature map.

Optional attention modules (src/models/attention) are inserted after the two convolutions of a
stage, before its max-pool; `attention_stages` selects which stages (default: all)."""
import torch
from torch import nn

from src.models.attention.attention import build_attention


def _block(cin, cout, attn=None):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if attn is not None:
        layers.append(attn)
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, widths=(16, 32, 64, 128), attention=None, attention_stages=None):
        super().__init__()
        stages = set(range(len(widths))) if attention_stages is None else set(attention_stages)
        chans = [in_channels, *widths]
        self.blocks = nn.Sequential(*[
            _block(a, b, build_attention(attention, b) if i in stages else None)
            for i, (a, b) in enumerate(zip(chans[:-1], chans[1:]))])
        self.out_channels = widths[-1]

    def attention_modules(self):
        return [m for m in self.modules() if hasattr(m, "keep")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
