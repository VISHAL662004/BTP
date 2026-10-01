"""Modular detector = backbone + (optional components added in later phases) + head."""
from torch import nn

from .backbone.cnn import SimpleCNN
from .heads.detection import DetectionHead


class Detector(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone(x))   # (obj_logit (B,), box_raw (B, 3))


def build_detector(cfg: dict) -> Detector:
    m = cfg["model"]
    if m["backbone"] != "simple_cnn":
        raise ValueError(f"Unknown backbone {m['backbone']!r}")
    bb = SimpleCNN(m["in_channels"], tuple(m["widths"]))
    return Detector(bb, DetectionHead(bb.out_channels, m.get("head_hidden", 64), m.get("dropout", 0.2)))
