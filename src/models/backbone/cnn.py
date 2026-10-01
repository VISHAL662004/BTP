"""Simple configurable CNN backbone (baseline). Input (B, C, H, W) -> feature map."""
import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, widths=(16, 32, 64, 128)):
        super().__init__()
        chans = [in_channels, *widths]
        self.blocks = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.out_channels = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
