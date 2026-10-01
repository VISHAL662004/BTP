"""Detection head: objectness logit + box (dx, dy, log-diameter) from pooled features.

Box decoding (patch pixels): centre offset = BOX_SCALE * (o0, o1) from the patch centre,
side length = BOX_SCALE * exp(o2). With the zero-initialised box layer a fresh model predicts
a BOX_SCALE-pixel box at the patch centre."""
import torch
from torch import nn

BOX_SCALE = 8.0


class DetectionHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(in_features, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.obj = nn.Linear(hidden, 1)
        self.box = nn.Linear(hidden, 3)
        nn.init.zeros_(self.box.weight)
        nn.init.zeros_(self.box.bias)
        # prior probability 0.01 for the positive class (standard focal-loss initialisation)
        nn.init.constant_(self.obj.bias, -4.595)

    def forward(self, feats: torch.Tensor):
        h = self.shared(self.pool(feats))
        return self.obj(h).squeeze(1), self.box(h)


def decode_box(raw: torch.Tensor) -> torch.Tensor:
    """raw (N, 3) -> (cx_offset, cy_offset, side) in patch pixels."""
    return torch.stack([raw[:, 0] * BOX_SCALE, raw[:, 1] * BOX_SCALE, torch.exp(raw[:, 2].clamp(-4, 4)) * BOX_SCALE], 1)
