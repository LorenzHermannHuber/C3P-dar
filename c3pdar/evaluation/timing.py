"""Per-frame inference times at batch size 1 (GPU): every frame is predicted alone and draws one cloud.

Draw 0 uses draw_generator(seed, scene, frame, 0) as in generation.py; nothing is stored. predict_seconds
spans the model from the device inputs to its output on the CPU (the ``seconds`` of generation.py),
points_seconds the cloud from that output.
"""

import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from . import PROGRESS, radargen
from .generation import frame_info
from ..data import RadarGenBatch
from ..models.dit import NUM_STREAMS
from ..models.radargen import COORDINATE_RANGE, DOPPLER_RANGE, LATENT_CHANNELS, RCS_RANGE
from ..sampling import draw_generator, ppp_raster, sample_points


def _now(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _frames(dataset, indices, device, num_workers, desc):
    """(normalized device conditions, scene, frame, frame attrs) of the frames, one at a time, in order."""
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=num_workers,
                        collate_fn=RadarGenBatch.collate)
    for batch in tqdm(loader, desc=desc, unit="frame", **PROGRESS):
        yield batch.to(device, non_blocking=True).normalized_conditions(), *frame_info(dataset, batch.index.item())


def _row(scene, frame, attrs, start, predicted, end, points):
    return dict(scene_token=scene, frame_index=frame, sample_token=attrs["sample_token"],
                predict_seconds=predicted - start, points_seconds=end - predicted, seconds=end - start,
                points=len(points))


@torch.no_grad()
def time_ppp(model, dataset, indices, mark_link, seed, device, num_workers=4, desc="ppp"):
    """DataFrame with one row per frame in prediction order (bf16 autocast as in generation.generate_ppp)."""
    rows = []
    for maps, scene, frame, attrs in _frames(dataset, indices, device, num_workers, desc):
        start = _now(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            fields = model(*maps.unbind(1))
        fields = fields[0].float().cpu().numpy()
        predicted = _now(device)
        raster = ppp_raster(fields, dataset.point_limit, mark_link)
        points = sample_points(raster.masses, raster.rcs, raster.doppler, dataset.point_limit,
                               draw_generator(seed, scene, frame, 0))
        rows.append(_row(scene, frame, attrs, start, predicted, _now(device), points))
    return pd.DataFrame(rows)


@torch.no_grad()
def time_radargen(model, dataset, indices, seed, device, num_workers=4, desc="radargen"):
    """DataFrame with one row per frame in prediction order; points_seconds covers the sparse deconvolution."""
    if dataset.point_limit != COORDINATE_RANGE:
        raise ValueError(f"RadarGen covers +-{COORDINATE_RANGE} m, the dataset +-{dataset.point_limit} m")
    rows = []
    for maps, scene, frame, attrs in _frames(dataset, indices, device, num_workers, desc):
        start = _now(device)
        conditions = model.encode(maps)
        noise = torch.randn((NUM_STREAMS, LATENT_CHANNELS, *conditions.shape[-2:]),
                            generator=draw_generator(seed, scene, frame, 0))[None]
        images = model.decode(model.denoise(conditions, noise.to(device))).cpu().numpy()
        predicted = _now(device)
        points = radargen.recover_points(radargen.radar_maps(images[0], RCS_RANGE, DOPPLER_RANGE),
                                         COORDINATE_RANGE, device)
        rows.append(_row(scene, frame, attrs, start, predicted, _now(device), points))
    return pd.DataFrame(rows)
