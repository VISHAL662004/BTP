"""Modular detector = backbone [+ context module] [+ attention (later phases)] + head."""
from torch import nn

from .backbone.cnn import SimpleCNN
from .backbone.transformer import build_context
from .heads.detection import DetectionHead


class Detector(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, context: nn.Module = None):
        super().__init__()
        self.backbone, self.context, self.head = backbone, context, head

    def forward(self, x):
        f = self.backbone(x)
        if self.context is not None:
            f = self.context(f)
        return self.head(f)   # (obj_logit (B,), box_raw (B, 3))


def build_detector(cfg: dict) -> Detector:
    m = cfg["model"]
    if m["backbone"] != "simple_cnn":
        raise ValueError(f"Unknown backbone {m['backbone']!r}")
    attn = m.get("attention")            # False/None/{'type': ...}; older configs store `false`
    bb = SimpleCNN(m["in_channels"], tuple(m["widths"]), attn if isinstance(attn, dict) else None,
                   attn.get("stages") if isinstance(attn, dict) else None,
                   tuple(m["mid_widths"]) if m.get("mid_widths") else None)
    return Detector(bb, DetectionHead(bb.out_channels, m.get("head_hidden", 64), m.get("dropout", 0.2)),
                    build_context(m, bb.out_channels))
