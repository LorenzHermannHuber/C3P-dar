"""The C3P-dar training loss: a marked Poisson point process likelihood on the BEV grid.

The model predicts (B,3,H,W) fields [log-intensity, RCS, Doppler]. Each detection of the
RadarGenBatch selects the grid cell it falls in (RadarGenBatch.raster_at); repeated cells
stay repeated (no deduplication).
"""

import math

import torch
from torch import nn


def _require_finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


class PPPLoss(nn.Module):
    """Per-frame Poisson Point Process negative log-likelihood plus L1 errors of the physical marks, averaged over frames.

    With cell masses m = exp(log-intensity) / (H*W) (what the sampler draws from), a frame's
    NLL is sum(m) - sum_i log m(cell_i). ``normalize_by_gt_count`` divides it by max(#detections, 1)
    the mark errors are always per-frame means. ``mark_link='cubic'`` predicts the cube root of the marks.
    """

    def __init__(self, normalize_by_gt_count=True, mark_link="cubic"):
        super().__init__()
        if mark_link not in ("cubic", "identity"):
            raise ValueError("Unknown mark link")
        self.normalize_by_gt_count = normalize_by_gt_count
        self.mark_link = mark_link

    def forward(self, fields, batch):
        if fields.ndim != 4 or fields.shape[1] != 3 or len(fields) != len(batch):
            raise ValueError("fields must be (B,3,H,W) for the B frames of the batch")
        b, _, h, w = fields.shape
        _require_finite(fields, "fields")
        _require_finite(batch.detections, "detections")
        with torch.autocast(device_type=fields.device.type, enabled=False):
            fields = fields.float()
            at_detections, inside = batch.raster_at(fields)
            detections, frames = batch.detections[inside].float(), batch.detection_frames()[inside]
            counts = torch.bincount(frames, minlength=b).float().clamp_min(1)

            def per_frame_sum(values):
                return values.new_zeros(b).scatter_add(0, frames, values)

            log_area = math.log(h * w)
            expected_count = (fields[:, 0].exp() / (h * w)).sum(dim=(1, 2))
            # \int_H\int_W intensity(x,y)/(H*W) dx dy - \sum_i log(intensity(x_i,y_i)/(H*W))
            nll = expected_count - per_frame_sum(at_detections[:, 0] - log_area)
            marks = at_detections[:, 1:].pow(3) if self.mark_link == "cubic" else at_detections[:, 1:]
            errors = (marks - detections[:, 2:]).abs()
            losses = dict(ppp=(nll / counts if self.normalize_by_gt_count else nll).mean(),
                          rcs=(per_frame_sum(errors[:, 0]) / counts).mean(),
                          doppler=(per_frame_sum(errors[:, 1]) / counts).mean())
        for name, value in losses.items():
            _require_finite(value, name)
        return losses


def weighted_loss(losses, weights):
    """Weighted total plus detached per-term logs."""
    total = sum(value * weights[name] for name, value in losses.items())
    logs = {"loss_total": total.detach(), **{f"loss_{name}": value.detach() for name, value in losses.items()}}
    return total, logs
