"""CPU part of the evaluation: frame selection and scoring of the stored outputs (store.RawStore)."""

import json

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data import load_frame
from ..sampling import density_raster, ppp_raster
from . import radargen, PROGRESS
from .metrics import inside_raster, raster_metrics, sampled_metrics, summarize


def frame_ids(dataset):
    """(scene_token, frame_index) of every dataset frame."""
    return list(zip(dataset.frames.scene_token, map(int, dataset.frames.frame_index)))


def select_frames(dataset, count=None, seed=42):
    """Dataset indices of a scene-balanced frame selection; all frames if ``count`` is None.

    Same frames and order as RadarGen's select_keyframe_indices for a keyframes-only dataset:
    only scene tokens and frame indices enter, not the dataset order.
    """
    ids = frame_ids(dataset)
    if count is None or count >= len(ids):
        return list(range(len(ids)))
    scenes = {}
    for index, (scene, _) in enumerate(ids):
        scenes.setdefault(scene, []).append(index)
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(sorted(scenes[scene], key=lambda i: ids[i][1]))) for scene in sorted(scenes)]
    selected = []
    while len(selected) < count:
        for group in rng.permutation(len(groups)):
            if groups[group]:
                selected.append(int(groups[group].pop()))
                if len(selected) == count:
                    break
    return selected


def load_raster(store, scene, frame, draw, count, point_limit):
    """Raster of one draw (kind "ppp": the same for all draws) and its inference seconds."""
    if store.attrs["kind"] == "ppp":
        fields, attrs = store.read(scene, frame, "fields")
        return ppp_raster(fields, point_limit, store.attrs["mark_link"]), float(attrs["seconds"])
    maps, attrs = store.read(scene, frame, f"{draw}/maps")
    return density_raster(maps, count, point_limit), float(attrs["seconds"])


def evaluate(store, dataset, indices, draws, *, device, nll_floor=1e-3):
    """Frame table and box table (one row per draw) and summary of the selected frames' stored draws."""
    if not np.isclose(store.attrs["point_limit"], dataset.point_limit):
        raise ValueError(f"store point_limit {store.attrs['point_limit']} != dataset {dataset.point_limit}")
    ids = frame_ids(dataset)
    missing = [(*ids[i], d) for i in indices for d in range(draws) if not store.has(*ids[i], f"{d}/points")]
    if missing:
        raise ValueError(f"{len(missing)} draws not generated, e.g. {missing[:3]}; run scripts/generate.py")
    mmd = radargen.mmd_loss(device)
    with h5py.File(dataset.files[indices[0]], "r", locking=False) as f:
        normalization = radargen.normalization_config(json.loads(f["metadata/normalization_json"].asstr()[()]))
    frames, boxes, clouds = [], [], []
    for position, index in enumerate(tqdm(indices, desc=f"score {draws}x", unit="frame", **PROGRESS)):
        info = dataset.frames.iloc[index]
        scene, frame, key = info.scene_token, int(info.frame_index), bool(info.is_key_frame)
        sample = load_frame(dataset.files[index], index, load_boxes=key)
        # Non-keyframes carry their keyframe's boxes, which do not fit their detections.
        frame_boxes = sample.boxes(categories=None)[0] if key else []
        gt, outside = inside_raster(sample.detections.numpy(), dataset.point_limit)
        identity = dict(index=position, scene_token=scene, frame_index=frame, sample_token=info.sample_token,
                        is_key_frame=key)
        for draw in range(draws):
            points = store.read(scene, frame, f"{draw}/points")[0]
            if draw == 0 or store.attrs["kind"] != "ppp":
                raster, seconds = load_raster(store, scene, frame, draw, len(points), dataset.point_limit)
                raster_row, raster_boxes = raster_metrics(raster, gt, frame_boxes, nll_floor)
            row, box_rows, frame_clouds = sampled_metrics(points, gt, frame_boxes, normalization, mmd, device)
            ids_row = {**identity, "draw": draw}
            frames.append({**ids_row, "gt_count": len(gt), "gt_outside_raster": outside, **row, **raster_row,
                           "inference_seconds": seconds})
            boxes.extend({**ids_row, **box, **raster_box} for box, raster_box in zip(box_rows, raster_boxes))
            clouds.append(frame_clouds)
    frames, boxes = pd.DataFrame(frames), pd.DataFrame(boxes)
    print("summarizing (class MMD)", flush=True)
    return frames, boxes, summarize(frames, boxes, clouds, device)
