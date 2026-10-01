"""L_det = lambda_cls * L_focal + lambda_loc * L_loc   (architecture.md section 10).

L_focal: binary focal loss on the objectness logit, summed over the batch and normalised by
the number of positives (>=1), as in RetinaNet.
L_loc  : 1 - IoU(B, B*) between predicted and ground-truth square boxes, averaged over positive
candidates that have a matched nodule box. Boxes are axis-aligned squares (cx, cy, side)."""
import torch
import torch.nn.functional as F

from src.models.heads.detection import decode_box


def focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).sum() / targets.sum().clamp(min=1.0)


def square_iou(a, b, eps: float = 1e-6):
    """a, b: (N, 3) = (cx, cy, side)."""
    def corners(t):
        h = t[:, 2] / 2
        return t[:, 0] - h, t[:, 1] - h, t[:, 0] + h, t[:, 1] + h
    ax0, ay0, ax1, ay1 = corners(a)
    bx0, by0, bx1, by1 = corners(b)
    iw = (torch.min(ax1, bx1) - torch.max(ax0, bx0)).clamp(min=0)
    ih = (torch.min(ay1, by1) - torch.max(ay0, by0)).clamp(min=0)
    inter = iw * ih
    union = a[:, 2] ** 2 + b[:, 2] ** 2 - inter
    return inter / (union + eps)


def iou_loss(box_raw, gt_box):
    """box_raw (N,3) raw head output; gt_box (N,3) = (dx, dy, d) in pixels. Empty -> 0."""
    if box_raw.shape[0] == 0:
        return box_raw.sum() * 0.0
    return (1.0 - square_iou(decode_box(box_raw), gt_box)).mean()


class DetectionLoss:
    def __init__(self, lambda_cls=1.0, lambda_loc=1.0, alpha=0.25, gamma=2.0):
        self.lc, self.ll, self.alpha, self.gamma = lambda_cls, lambda_loc, alpha, gamma

    def __call__(self, obj_logit, box_raw, target, gt_box):
        """target (B,) in {0,1}; gt_box (B,3) with NaN rows where no box is available."""
        l_cls = focal_loss(obj_logit, target, self.alpha, self.gamma)
        has = (target > 0.5) & ~torch.isnan(gt_box).any(1)
        l_loc = iou_loss(box_raw[has], gt_box[has])
        total = self.lc * l_cls + self.ll * l_loc
        return total, {"loss": total.item(), "focal": l_cls.item(), "loc": l_loc.item()}
