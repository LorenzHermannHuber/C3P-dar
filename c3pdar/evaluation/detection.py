"""Downstream detection: a radar-only VoxelNeXt detector on the point clouds, scored with the TruckScenes devkit.

The detector (VOXELNEXT: checkpoint_epoch_400.pth, voxelnext.yaml, truckscenes_dataset.yaml,
evaluate_detection_config.json) was trained on measured single-sweep RADAR_LEFT_SIDE clouds
[x, y, z, RCS, |Doppler|] for car, truck and trailer. Needs OpenPCDet's ``pcdet`` (with its CUDA ops),
spconv 2, truckscenes-devkit, pyquaternion and easydict; all are imported lazily.
"""

import hashlib
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

CLASSES = ("car", "truck", "trailer")
ERRORS = ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")
SENSOR = "RADAR_LEFT_SIDE"
FLAT_UP = "RadarGen_vehicle_flat_up"


def _matrix(record):
    from pyquaternion import Quaternion

    out = np.eye(4)
    out[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    out[:3, 3] = record["translation"]
    return out


def sensor_from_global(trucksc, sample_data_token):
    data = trucksc.get("sample_data", sample_data_token)
    calibration = trucksc.get("calibrated_sensor", data["calibrated_sensor_token"])
    pose = trucksc.get("ego_pose", data["ego_pose_token"])
    return np.linalg.inv(_matrix(pose) @ _matrix(calibration))


def frame_inputs(path, trucksc):
    """Sample token, detector sample_data token, flat-up -> detector transform and measured cloud of a frame."""
    with h5py.File(path, "r", locking=False) as f:
        targets = f["ppp_targets"]
        if targets.attrs.get("xy_frame") != FLAT_UP:
            raise ValueError(f"{path}: ppp_targets are not in the {FLAT_UP} frame")
        sample_token = f["metadata/sample_token"].asstr()[()]
        flat_from_global = f["metadata/global_to_flat_up"][()]
        measured = np.column_stack([targets[name][()] for name in ("xy", "rcs", "doppler")])
    sensor_token = trucksc.get("sample", sample_token)["data"][SENSOR]
    detector_from_flat = sensor_from_global(trucksc, sensor_token) @ np.linalg.inv(flat_from_global)
    return sample_token, sensor_token, detector_from_flat, measured


def detector_points(points, detector_from_flat, height):
    """[x, y, RCS, Doppler] in the flat-up frame -> [x, y, z, RCS, |Doppler|] in the detector frame.

    The clouds have no height: z is the constant ``height`` in the flat-up frame.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("expected finite (N, 4) [x, y, RCS, Doppler] points")
    xyz = np.column_stack((points[:, :2], np.full(len(points), height)))
    xyz = xyz @ detector_from_flat[:3, :3].T + detector_from_flat[:3, 3]
    return np.column_stack((xyz, points[:, 2], np.abs(points[:, 3]))).astype(np.float32)


def point_seed(seed, sample_token):
    """Seed of the point shuffle before voxelization (only matters for voxels with too many points)."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{sample_token}".encode()).digest()[:4], "little")


def _load_checkpoint(path):
    """weights_only load; the checkpoint holds NumPy-2 scalars, which NumPy 1 knows under another module."""

    def scalar(dtype, data):
        return np.core.multiarray.scalar(dtype, data)

    scalar.__module__ = "numpy._core.multiarray"
    allowed = [scalar, np.dtype, *(type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


class Detector:
    def __init__(self, directory, device="cuda"):
        from easydict import EasyDict
        from pcdet.models import build_network
        from pcdet.utils.spconv_utils import find_all_spconv_keys
        from spconv.utils import Point2VoxelCPU3d

        cfg = yaml.safe_load((directory / "voxelnext.yaml").read_text())
        data = yaml.safe_load((directory / "truckscenes_dataset.yaml").read_text())
        data.update({k: v for k, v in cfg["DATA_CONFIG"].items() if k != "_BASE_CONFIG_"})
        cfg["DATA_CONFIG"] = data
        self.cfg = cfg = EasyDict(cfg)
        if tuple(cfg.CLASS_NAMES) != CLASSES:
            raise ValueError(f"detector classes {cfg.CLASS_NAMES} != {CLASSES}")
        voxel = next(p for p in cfg.DATA_CONFIG.DATA_PROCESSOR if p.NAME == "transform_points_to_voxels")
        self.bounds = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        dataset = SimpleNamespace(
            class_names=cfg.CLASS_NAMES, point_feature_encoder=SimpleNamespace(num_point_features=5),
            grid_size=np.round((self.bounds[3:] - self.bounds[:3]) / voxel.VOXEL_SIZE).astype(np.int64),
            point_cloud_range=self.bounds, voxel_size=np.asarray(voxel.VOXEL_SIZE), depth_downsample_factor=None)
        self.net = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
        checkpoint = directory / "checkpoint_epoch_400.pth"
        state, sparse = self.net.state_dict(), find_all_spconv_keys(self.net)
        weights = {}
        for name, value in _load_checkpoint(checkpoint)["model_state"].items():
            # spconv versions store sparse convolution kernels in different layouts.
            if name in sparse and name in state and value.shape != state[name].shape:
                layouts = [value.transpose(-1, -2)] + ([value.permute(4, 0, 1, 2, 3)] if value.ndim == 5 else [])
                value = next((v.contiguous() for v in layouts if v.shape == state[name].shape), value)
            weights[name] = value
        self.net.load_state_dict(weights, strict=True)
        self.device = torch.device(device)
        self.net.to(self.device).eval()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.voxelizer = Point2VoxelCPU3d(
            vsize_xyz=voxel.VOXEL_SIZE, coors_range_xyz=cfg.DATA_CONFIG.POINT_CLOUD_RANGE, num_point_features=5,
            max_num_points_per_voxel=voxel.MAX_POINTS_PER_VOXEL, max_num_voxels=voxel.MAX_NUMBER_OF_VOXELS["test"])

    @torch.inference_mode()
    def __call__(self, points, seed):
        """Boxes (K, 9) [x, y, z, dx, dy, dz, yaw, vx, vy] in the detector frame, scores (K,), labels (K,) 1..3."""
        import cumm.tensorview as tv

        inside = ((points[:, :3] >= self.bounds[:3]) & (points[:, :3] < self.bounds[3:])).all(axis=1)
        points = points[inside]
        points = points[np.random.default_rng(seed).permutation(len(points))]
        if not len(points):
            return np.empty((0, 9), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
        voxels, coords, counts = (x.numpy() for x in self.voxelizer.point_to_voxel(tv.from_numpy(points)))
        batch = dict(batch_size=1, voxels=torch.from_numpy(voxels).to(self.device),
                     voxel_coords=torch.from_numpy(np.pad(coords, ((0, 0), (1, 0)))).int().to(self.device),
                     voxel_num_points=torch.from_numpy(counts).to(self.device))
        prediction = self.net(batch)[0][0]
        return tuple(prediction[k].cpu().numpy() for k in ("pred_boxes", "pred_scores", "pred_labels"))


def global_boxes(trucksc, sample_token, sensor_token, boxes, scores, labels):
    """Detector-frame predictions -> devkit DetectionBoxes in the global frame (attribute vehicle.moving)."""
    from pyquaternion import Quaternion
    from truckscenes.eval.detection.data_classes import DetectionBox
    from truckscenes.utils.data_classes import Box

    to_global = np.linalg.inv(sensor_from_global(trucksc, sensor_token))
    rotation = Quaternion(matrix=to_global[:3, :3])
    result = []
    for b, score, label in zip(boxes, scores, labels):
        # pcdet sizes are [dx (length), dy (width), dz]; the devkit's are [width, length, height].
        box = Box(b[:3], b[[4, 3, 5]], Quaternion(axis=[0, 0, 1], radians=float(b[6])),
                  velocity=(float(b[7]), float(b[8]), 0.0))
        box.rotate(rotation)
        box.translate(to_global[:3, 3])
        result.append(DetectionBox(
            sample_token=sample_token, translation=box.center.tolist(), size=box.wlh.tolist(),
            rotation=box.orientation.elements.tolist(), velocity=box.velocity[:2].tolist(),
            detection_name=CLASSES[int(label) - 1], detection_score=float(score), attribute_name="vehicle.moving"))
    return result


def ground_truth(trucksc, sample_tokens):
    """Devkit GT boxes of the three classes; velocities do not use annotations of a scene's last sample."""
    from truckscenes.eval.common.data_classes import EvalBoxes
    from truckscenes.eval.detection.data_classes import DetectionBox
    from truckscenes.eval.detection.utils import category_to_detection_name

    last = {scene["last_sample_token"] for scene in trucksc.scene}
    result = EvalBoxes()
    for token in sample_tokens:
        boxes = []
        for annotation_token in trucksc.get("sample", token)["anns"]:
            a = trucksc.get("sample_annotation", annotation_token)
            name = category_to_detection_name(a["category_name"])
            if name not in CLASSES:
                continue
            if len(a["attribute_tokens"]) > 1:
                raise ValueError("multiple attributes on a GT box")
            attribute = trucksc.get("attribute", a["attribute_tokens"][0])["name"] if a["attribute_tokens"] else ""
            following = a["next"]
            if following and trucksc.get("sample_annotation", following)["sample_token"] in last:
                a["next"] = ""
            try:
                velocity = trucksc.box_velocity(annotation_token)[:2]
            finally:
                a["next"] = following
            boxes.append(DetectionBox(
                sample_token=token, translation=a["translation"], size=a["size"], rotation=a["rotation"],
                velocity=velocity, num_pts=a["num_lidar_pts"] + a.get("num_radar_pts", 0), detection_name=name,
                detection_score=-1.0, attribute_name=attribute))
        result.add_boxes(token, boxes)
    return result


def score(trucksc, predictions, config):
    """Per-class AP and TP errors and their three-class means mAP, mATE, ..., NDS.

    ``predictions``: {sample_token: [DetectionBox]}, ``config``: evaluate_detection_config.json as a dict.
    """
    from truckscenes.eval.common.data_classes import EvalBoxes
    from truckscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from truckscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from truckscenes.eval.detection.data_classes import DetectionConfig

    config = DetectionConfig.deserialize(config)
    tokens = sorted(predictions)
    pred = EvalBoxes()
    for token in tokens:
        pred.add_boxes(token, predictions[token])
    gt = ground_truth(trucksc, tokens)
    counts = dict(gt_before=len(gt.all), pred_before=len(pred.all))
    for boxes in (gt, pred):
        if boxes.all:  # the devkit filter cannot handle an empty box set
            add_center_dist(trucksc, boxes)
            filter_eval_boxes(trucksc, boxes, config.class_range)
    counts.update(gt_after=len(gt.all), pred_after=len(pred.all))
    classes = {}
    for name in CLASSES:
        matched = {d: accumulate(gt, pred, name, config.dist_fcn_callable, d) for d in config.dist_ths}
        aps = {str(d): float(calc_ap(m, config.min_recall, config.min_precision)) for d, m in matched.items()}
        errors = {k: float(calc_tp(matched[config.dist_th_tp], config.min_recall, k)) for k in ERRORS}
        classes[name] = dict(AP=float(np.mean(list(aps.values()))), AP_by_distance=aps, **errors,
                             gt_count=sum(b.detection_name == name for b in gt.all))
    summary = dict(mAP=float(np.mean([c["AP"] for c in classes.values()])))
    for label, key in zip(("mATE", "mASE", "mAOE", "mAVE", "mAAE"), ERRORS):
        summary[label] = float(np.mean([c[key] for c in classes.values()]))
    summary["NDS"] = (5 * summary["mAP"] + sum(1 - summary[k] for k in ("mATE", "mASE", "mAOE", "mAAE"))
                      + 1 - min(1.0, summary["mAVE"])) / 10
    return dict(summary=summary, classes=classes, box_counts=counts, sample_count=len(tokens))


def score_row(result):
    """A score() result as one flat row: summary, per-class scores (<class>_<score>) and box counts."""
    row = dict(result["summary"])
    for name, scores in result["classes"].items():
        for key, value in scores.items():
            if key == "AP_by_distance":
                row.update({f"{name}_AP_{distance}": ap for distance, ap in value.items()})
            else:
                row[f"{name}_{key}"] = value
    return {**row, **result["box_counts"], "sample_count": result["sample_count"]}
