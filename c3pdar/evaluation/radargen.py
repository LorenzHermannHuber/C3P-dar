"""RadarGen's own code (third_party/RadarGen): the reference for our metrics and the end of its sampling.

The only module that touches third_party: the scoring functions return RadarGen's own scores, which
metrics.py logs as radargen_<key> next to our reimplementation; radar_maps and recover_points turn the
images decoded by c3pdar.models.RadarGen into its maps and point cloud.

RadarGen's package inits pull in its dataset adapters and training dependencies, so its packages are
registered as empty shells and only the plain modules below are loaded, without touching sys.path.
"""

import inspect
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2] / "third_party" / "RadarGen"
for _name in ("evaluation", "radargen", "radargen.core", "radargen.inference", "radargen.radar_maps",
              "radargen.radar_maps.map_creation"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
        sys.modules[_name].__path__ = [str(ROOT.joinpath(*_name.split(".")))]

from evaluation.aggregation import PER_SAMPLE_METRICS, create_aggregator, finalize_global_metrics  # noqa: E402
from evaluation.metrics import MMDLoss, RBF, compute_box_metrics_and_stats, compute_per_sample_metrics  # noqa: E402
from radargen.core.normalization import NormalizationConfig  # noqa: E402
from radargen.inference.pcl_recovery import deconv_cell_recovery, recover_pcl_attributes, solve_irl1  # noqa: E402
from radargen.radar_maps.map_creation.doppler import decoded_to_doppler_map  # noqa: E402
from radargen.radar_maps.map_creation.point_density import decoded_to_pd_map  # noqa: E402
from radargen.radar_maps.map_creation.rcs import decoded_to_rcs_map  # noqa: E402

LOG = logging.getLogger(__name__)


def radar_maps(images, rcs_range, doppler_range):
    """(3,H,W) float32 [point density (sums to 1), RCS, Doppler] of the (3,3,H,W) decoded images in [0,255]."""
    pd, rcs, doppler = np.moveaxis(np.asarray(images, dtype=np.float32), 1, -1)
    return np.stack([decoded_to_pd_map(pd), decoded_to_rcs_map(rcs, *rcs_range),
                     decoded_to_doppler_map(doppler, *doppler_range)]).astype(np.float32)


def recover_points(maps, coordinate_range, device):
    """RadarGen's (N,4) [x, y, RCS, Doppler] cloud of the maps by sparse deconvolution (IRL1 + FISTA)."""
    locations, rcs, doppler = recover_pcl_attributes(maps[0], maps[1], maps[2], maps.shape[-1], coordinate_range,
                                                     device=device)
    # radar_maps_to_pcl returns float32.
    return np.column_stack([locations, rcs, doppler]).astype(np.float32).astype(np.float64)


DECONV = {k: p.default for k, p in inspect.signature(deconv_cell_recovery).parameters.items()
          if p.default is not inspect.Parameter.empty}


def recovered_map(pd_map, device):
    """(H,W) float32 deconvolved map of deconv_cell_recovery, whose cells above DECONV["threshold"] form the cloud."""
    if pd_map.max() <= 0:
        raise ValueError("Point density map is invalid")
    y = torch.from_numpy(pd_map / pd_map.max()).float().to(device)
    return solve_irl1(y, DECONV["sigma_pix"], DECONV["sigma_pix"], alpha=DECONV["lam"],
                      n_irl1_iter=DECONV["n_irl1_iter"], n_fista_iter=DECONV["n_fista_iter"]).cpu().numpy()


def cell_points(maps, rows, cols, coordinate_range):
    """(N,4) [x, y, RCS, Doppler] of the map cells (rows, cols), as recover_pcl_attributes converts them."""
    size = maps.shape[-1]
    locations = np.column_stack([cols, size - (np.asarray(rows) + 1)]).astype(np.float32) + 0.5
    locations = locations / size * (2 * coordinate_range) - coordinate_range
    rcs, doppler = np.rint(maps[1][rows, cols]), maps[2][rows, cols]
    return np.column_stack([locations, rcs, doppler]).astype(np.float32).astype(np.float64)


def _value(x):
    x = float(x)
    return x if np.isfinite(x) else np.nan


def normalization_config(values):
    """RadarGen's NormalizationConfig from a file's normalization_json dict."""
    return NormalizationConfig(**values)


def mmd_loss(device):
    return MMDLoss(kernel=RBF(device=device), device=device)


def cloud_scores(points, gt, normalization, mmd, device):
    """RadarGen's per-sample scores of the (N,4) generated vs the (M,4) GT cloud.

    NaN unless both clouds are nonempty: RadarGen's evaluator never scores an empty cloud.
    """
    if not (len(points) and len(gt)):
        return dict.fromkeys(PER_SAMPLE_METRICS, np.nan)
    return {k: _value(v) for k, v in compute_per_sample_metrics(points, gt, normalization, mmd, device).items()}


def box_scores(points, gt, box, normalization, device):
    """RadarGen's scores of one box and its box-local clouds ``{"gt": (M,4), "syn": (K,4)}``."""
    stats = compute_box_metrics_and_stats(points, gt, [box], device, normalization)
    local = {side: stats[f"per_box_invariant_pcl_{side}"][box.name][0] for side in ("gt", "syn")}
    hit = stats["n_hits"] > 0
    scores = dict(
        gt_count=len(local["gt"]), generated_count=len(local["syn"]),
        hit=int(hit), false_negative=int(stats["n_fn_boxes"]), false_positive=int(stats["n_fp_boxes"]),
        chamfer_distance_xy=_value(stats["per_box_chamfer_distance_xy"][0]) if hit else np.nan,
        chamfer_distance_all_norm=_value(stats["per_box_chamfer_distance_all_norm"][0]) if hit else np.nan,
        num_points_similarity=_value(stats["per_box_num_points_similarity"][0]))
    return scores, local


def class_mmd(frame_clouds, device, classes):
    """RadarGen's object-local class MMD (aggregate_metrics + finalize_global_metrics).

    ``frame_clouds``: per frame ``{side: [(category, (M,4) box-local cloud), ...]}``. Per frame, class and
    side the nonempty box clouds are pooled; finalize then uses every third pooled frame, at most 200.
    """
    aggregator = create_aggregator()
    for clouds in frame_clouds:
        for side, items in clouds.items():
            per_class = {}
            for category, cloud in items:
                category = category.split(".")[-1]
                if category in classes and len(cloud):
                    per_class.setdefault(category, []).append(np.asarray(cloud, dtype=np.float64))
            for category, pooled in per_class.items():
                aggregator["agg_points_per_class"][category][side].append(np.vstack(pooled))
    final = finalize_global_metrics(aggregator, device, list(classes), LOG)
    return {k: _value(v) for k, v in final.items() if k.startswith("invariant_mmd_")}
