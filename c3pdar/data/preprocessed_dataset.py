"""TruckScenes frames preprocessed by RadarGen (create_hdf5_dataset_scene.py): condition maps, detections, boxes.

Planned rewrite of preprocessing and storage: one HDF5 per scene and content (~480 scenes) instead of one per frame.
- Every open costs ~45 ms on scratch (metadata latency, not locking); it dominates frame loading and the scan.
- <scene>/conditions.h5: (F,3,H,W,3) uint8 chunked per frame and condition, plus per-frame metadata arrays
  (frame_index, sample_token, split, is_key_frame, point_limit).
- <scene>/detections.h5: detections (G,4) with offsets (F+1,), mirroring RadarGenBatch; other RadarGen targets.
- <scene>/annotations.h5: box_geometry (K,15), box_labels (K,3) with offsets (F+1,) and the string table.
  Non-keyframes now copy their parent keyframe's boxes; store keyframe boxes once and reference them per frame.
- <scene>/depth.h5: the depth maps, rarely read.
- Workers keep their scene files open, so reading a frame is slicing straight into a RadarGenBatch.
- The preprocessor writes the frame summary itself, replacing the scan and the .data_cache CSV.
"""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

CONDITIONS = ("appearance", "semantics", "velocity")
FOREGROUND_CLASSES = ("car", "truck", "trailer")
# visibility.json tokens of levels 1 (0-40 %) and 2 (41-60 %); the evaluator drops them.
LOW_VISIBILITY = {"f95237cfe4e0480c899fab6d5e5a235c", "34fecb88f60d4842931ab1ce2187e6a0"}
CACHE_DIR = Path(__file__).resolve().parents[2] / ".data_cache"
SUMMARY_COLUMNS = dict(path=str, scene_token=str, frame_index="int64", sample_token=str, split=str,
                       is_key_frame=bool, point_limit="float64", num_detections="int64", num_boxes="int64")
X_OFFSETS = np.array([1, 1, 1, 1, -1, -1, -1, -1])
Y_OFFSETS = np.array([1, -1, -1, 1, 1, -1, -1, 1])
Z_OFFSETS = np.array([1, 1, -1, -1, 1, 1, -1, -1])


@dataclass
class Box:
    """Annotation box in the flat vehicle frame with the API used by RadarGen's evaluation code."""

    center: np.ndarray
    wlh: np.ndarray
    rotation_matrix: np.ndarray
    name: str
    token: str = ""

    def corners(self):
        w, l, h = self.wlh
        x = l / 2 * X_OFFSETS
        y = w / 2 * Y_OFFSETS
        z = h / 2 * Z_OFFSETS
        return self.rotation_matrix @ np.vstack((x, y, z)) + self.center[:, None]

    def bottom_corners(self):
        return self.corners()[:, [2, 3, 7, 6]]


def raster_cells(xy, shape, x_min, x_max, y_min, y_max):
    """Row, column and inside mask of the (H,W) raster cell containing each (N,2) [x,y].

    Columns span [x_min, x_max], rows [y_max, y_min] (row 0 is the largest y); points on any edge are inside.
    """
    h, w = shape
    x, y = xy.unbind(1)
    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    cols = ((x - x_min) / (x_max - x_min) * w).floor().long().clamp(0, w - 1)
    rows = h - 1 - ((y - y_min) / (y_max - y_min) * h).floor().long().clamp(0, h - 1)
    return rows, cols, inside


@dataclass
class RadarGenBatch:
    """B frames. Frame b owns detections[detection_offsets[b]:detection_offsets[b+1]], likewise for boxes.

    A DataLoader needs ``collate_fn=RadarGenBatch.collate``; torch's default_collate rejects this type.
    Without ``loaded_boxes`` the box fields are empty.
    """

    conditions: torch.Tensor  # (B,3,3,H,W) uint8: CONDITIONS x RGB
    index: torch.Tensor  # (B,) int64 dataset index; identity in dataset.frames
    detections: torch.Tensor  # (G,4) float32 [x, y, RCS dBsm, Doppler m/s], flat vehicle frame
    detection_offsets: torch.Tensor  # (B+1,) int64
    box_geometry: torch.Tensor  # (K,15) float64 [center xyz, wlh, row-major rotation], flat vehicle frame
    box_labels: torch.Tensor  # (K,3) int64 indices into strings: [category, visibility token, annotation token]
    box_offsets: torch.Tensor  # (B+1,) int64
    strings: tuple
    point_limit: float
    loaded_boxes: bool

    def __len__(self):
        return len(self.index)

    def normalized_conditions(self):
        """(B,3,3,H,W) float32 in [-1,1], bit-identical to RadarGen's default_train_from_np."""
        return (self.conditions.float() / 255 - 0.5) / 0.5

    def detection_frames(self):
        """(G,) batch-local frame of each detection, e.g. offsets [0,7,12,15] -> 7x 0, 5x 1, 3x 2."""
        frames = torch.arange(len(self), device=self.detection_offsets.device)
        return frames.repeat_interleave(self.detection_offsets.diff(), output_size=len(self.detections))

    def raster_at(self, raster, x_min=None, x_max=None, y_min=None, y_max=None):
        """Values of the (B,C,H,W) raster in the cell of each detection inside the extent, and the (G,) inside mask.

        Columns span [x_min, x_max] and rows [y_max, y_min] (see raster_cells), by default +-point_limit.
        Values (G_inside,C) pair with detections[inside], e.g. ``values - detections[inside, 2:]`` are mark errors.
        """
        limit = self.point_limit
        x_min, x_max = -limit if x_min is None else x_min, limit if x_max is None else x_max
        y_min, y_max = -limit if y_min is None else y_min, limit if y_max is None else y_max
        rows, cols, inside = raster_cells(self.detections[:, :2], raster.shape[-2:], x_min, x_max, y_min, y_max)
        return raster[self.detection_frames()[inside], :, rows[inside], cols[inside]], inside

    def boxes(self, categories=FOREGROUND_CLASSES):
        """Per frame the boxes with category in ``categories`` (all if None), visibility above 60 % and
        center within the point limit. RadarGen's evaluation scores all categories."""
        if not self.loaded_boxes:
            raise ValueError("batch was loaded without boxes")
        geometry = self.box_geometry.cpu().numpy()
        labels = [[self.strings[i] for i in row] for row in self.box_labels.tolist()]
        offsets = self.box_offsets.tolist()
        frames = []
        for start, stop in zip(offsets[:-1], offsets[1:]):
            frames.append([
                Box(box[:3], box[3:6], box[6:].reshape(3, 3), category, token)
                for box, (category, visibility, token) in zip(geometry[start:stop], labels[start:stop])
                if visibility and visibility not in LOW_VISIBILITY
                and (categories is None or category.split(".")[-1] in categories)
                and (np.abs(box[:2]) <= self.point_limit).all()])
        return frames

    def _map_tensors(self, function):
        return replace(self, **{f.name: function(getattr(self, f.name)) for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})

    def to(self, device, non_blocking=False):
        return self._map_tensors(lambda t: t.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        return self._map_tensors(torch.Tensor.pin_memory)

    @staticmethod
    def collate(batches: list["RadarGenBatch"]):
        first = batches[0]
        if any(b.point_limit != first.point_limit or b.loaded_boxes != first.loaded_boxes for b in batches):
            raise ValueError("all frames must share point_limit and loaded_boxes")
        strings = tuple(dict.fromkeys(s for b in batches for s in b.strings))
        lookup = {s: i for i, s in enumerate(strings)}
        remaps = [torch.tensor([lookup[s] for s in b.strings], dtype=torch.long) for b in batches]

        def cat(name):
            return torch.cat([getattr(b, name) for b in batches])

        def offsets(name):
            parts, total = [], 0
            for b in batches:
                parts.append(getattr(b, name)[:-1] + total)
                total += int(getattr(b, name)[-1])
            return torch.cat([*parts, torch.tensor([total])])

        def remapped(name):
            return torch.cat([remap[getattr(b, name)] for b, remap in zip(batches, remaps)])

        return RadarGenBatch(
            conditions=cat("conditions"), index=cat("index"), detections=cat("detections"),
            detection_offsets=offsets("detection_offsets"), box_geometry=cat("box_geometry"),
            box_labels=remapped("box_labels"),
            box_offsets=offsets("box_offsets"), strings=strings, point_limit=first.point_limit,
            loaded_boxes=first.loaded_boxes)


def quaternion_matrices(wxyz):
    """(K,4) [w,x,y,z] -> (K,3,3) rotation matrices."""
    q = np.asarray(wxyz, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)


def _read_boxes(f):
    """All stored boxes (a non-keyframe stores its parent keyframe's), mapped into the flat vehicle frame."""
    group = f["metadata/bounding_boxes"]
    transform = f["metadata/global_to_flat_up"][()]
    rotation, translation = transform[:3, :3], transform[:3, 3]
    text = [group[key].asstr()[()] for key in ("category_name", "visibility_token", "annotation_token")]
    strings = tuple(dict.fromkeys(s for values in text for s in values))
    lookup = {s: i for i, s in enumerate(strings)}
    center = group["center_global"][()].reshape(-1, 3) @ rotation.T + translation
    box_rotation = rotation @ quaternion_matrices(group["rotation_wxyz"][()])
    boxes = np.concatenate([center, group["size_wlh"][()].reshape(-1, 3), box_rotation.reshape(-1, 9)], axis=1)
    labels = torch.tensor([[lookup[s] for s in values] for values in text], dtype=torch.long).T
    return strings, torch.from_numpy(boxes.astype(np.float64)), labels.contiguous()


def load_frame(path, index=0, load_boxes=False):
    """One sample file as a RadarGenBatch of size 1."""
    with h5py.File(path, "r", locking=False) as f:
        conditions = np.stack([f[f"conditioning/{name}"][()] for name in CONDITIONS]).transpose(0, 3, 1, 2)
        detections = np.column_stack([f[f"ppp_targets/{key}"][()] for key in ("xy", "rcs", "doppler")])
        point_limit = float(f["metadata/point_limit"][()])
        if load_boxes:
            strings, boxes, labels = _read_boxes(f)
        else:
            strings, boxes, labels = (), torch.empty(0, 15, dtype=torch.float64), torch.empty(0, 3, dtype=torch.long)
    return RadarGenBatch(
        conditions=torch.from_numpy(np.ascontiguousarray(conditions))[None], index=torch.tensor([index]),
        detections=torch.from_numpy(detections), detection_offsets=torch.tensor([0, len(detections)]),
        box_geometry=boxes, box_labels=labels, box_offsets=torch.tensor([0, len(boxes)]), strings=strings,
        point_limit=point_limit, loaded_boxes=load_boxes)


def _scene_files(scene):
    # _SUCCESS.json: RadarGen's per-scene completion marker (file sizes, processing signature).
    return [scene / name for name in sorted(json.loads((scene / "_SUCCESS.json").read_text())["files"])]


def _summarize_file(path):
    with h5py.File(path, "r", locking=False) as f:
        meta = f["metadata"]
        return dict(path=f"{path.parent.name}/{path.name}", scene_token=meta["scene_token"].asstr()[()],
                    frame_index=int(meta["frame_index"][()]), sample_token=meta["sample_token"].asstr()[()],
                    split=meta["split"].asstr()[()], is_key_frame=bool(meta["reference_is_key_frame"][()]),
                    point_limit=float(meta["point_limit"][()]), num_detections=f["ppp_targets/rcs"].shape[0],
                    num_boxes=meta["bounding_boxes/category_name"].shape[0])


def _scan(root, scenes):
    rows = []
    if scenes:
        # Each HDF5 open costs ~45 ms on scratch and h5py serializes threads, so scan in processes.
        workers = min(32, len(os.sched_getaffinity(0)))
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            files = [path for paths in pool.map(_scene_files, [root / s for s in scenes]) for path in paths]
            rows = list(tqdm(pool.map(_summarize_file, files, chunksize=16), total=len(files),
                             desc=f"Summarizing {len(scenes)} scenes of {root}", unit="file"))
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)).astype(SUMMARY_COLUMNS)


def dataset_summary(root, rescan=False):
    """One row per sample file of every completed scene, cached in .data_cache/.

    Scenes added or removed since the cache was written are rescanned; ``rescan`` rebuilds it fully.
    """
    root = Path(root).resolve()
    path = CACHE_DIR / f"{str(root).strip('/').replace('/', '_')}.csv"
    path.parent.mkdir(exist_ok=True)
    summary = None if rescan or not path.exists() else pd.read_csv(path, dtype=SUMMARY_COLUMNS)
    known = set() if summary is None else set(summary.path.str.split("/").str[0])
    present = {entry.name for entry in os.scandir(root) if entry.is_dir()}
    new = sorted(s for s in present - known if (root / s / "_SUCCESS.json").is_file())
    if summary is not None and not new and known <= present:
        return summary
    kept = [] if summary is None else [summary[summary.path.str.split("/").str[0].isin(present)]]
    summary = pd.concat([*kept, _scan(root, new)]).sort_values("path").reset_index(drop=True)
    # Concurrent writers (e.g. DDP ranks) each publish a complete file.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    summary.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return summary


class PreprocessedRadarGenDataset(Dataset):
    """Frames of one split as RadarGenBatch items of size 1; collate with ``RadarGenBatch.collate``.

    ``frames`` holds the summary rows in dataset order. ``keyframes_only`` keeps the annotated keyframes.
    """

    def __init__(self, root, split, *, scenes=None, keyframes_only=False, load_boxes=False, rescan=False):
        self.root = Path(root).resolve()
        summary = dataset_summary(self.root, rescan)
        keep = summary.split == split
        if scenes is not None:
            keep &= summary.scene_token.isin(set(scenes))
        if keyframes_only:
            keep &= summary.is_key_frame
        self.frames = summary[keep].reset_index(drop=True)
        if self.frames.empty:
            raise ValueError(f"No completed {split} frames under {self.root}")
        limits = self.frames.point_limit.unique()
        if len(limits) != 1 or not np.isfinite(limits[0]):
            raise ValueError(f"frames must share one finite point_limit, found {limits}")
        self.point_limit = float(limits[0])
        self.files = [self.root / p for p in self.frames.path]
        self.load_boxes = load_boxes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return load_frame(self.files[index], index, self.load_boxes)
