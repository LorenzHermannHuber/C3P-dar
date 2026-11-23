"""Per-frame, per-box and summary metrics of a predicted raster (sampling.Raster) and the clouds drawn from it.

Raster scores (raster_metrics) need no sampling and have no RadarGen counterpart. Every point-cloud score is computed twice: by our implementation below (column ``key``) and by
RadarGen's original evaluation code in third_party (column ``radargen_key``, see .radargen). The
summary reports both and their mean absolute difference ``key_radargen_mae``. Our implementation
follows RadarGen's definitions so that the two stay comparable; each docstring names where ours deviates
and which problems the original has (some of them shared by ours on purpose). GT detections outside
the field raster are dropped before any scoring, for both implementations.

Every scored cloud is (N,4) [x, y, RCS dBsm, Doppler m/s]. A cloud score is NaN unless both clouds
are nonempty (RadarGen's evaluator never scores an empty cloud).
"""

import math
from itertools import chain

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from ..data.preprocessed_dataset import FOREGROUND_CLASSES
from . import radargen

REFERENCE = "radargen_"
# Printed by scripts/evaluate.py with RadarGen's value and the MAE; the tables keep every score.
# Frame and box scores are means over frames and boxes.
HEADLINE = (
    "gt_count", "generated_count", "count_abs_error",
    "chamfer_distance_xy", "chamfer_distance_xy_generated_to_gt", "chamfer_distance_xy_gt_to_generated",
    "chamfer_distance_all_norm",
    "iou_th1", "iou_th1_precision", "iou_th1_recall",
    "distance_attr_precision", "distance_attr_recall", "distance_attr_f1",
    "pcl_mmd_xy", "pcl_mmd_rcs", "pcl_mmd_doppler",
    "rcs_mae_dense", "doppler_mae_dense", "rcs_mae_nn", "doppler_mae_nn", "ppp_total_count_abs_error",
    "ppp_nll", "ppp_nll_spatial", "ppp_nll_count",
    "box_hit_rate", "false_positive_boxes", "box_chamfer_distance_xy", "box_num_points_similarity",
    "box_ppp_hit_probability",
    *(f"invariant_mmd_xy_{category}" for category in FOREGROUND_CLASSES),
)
MMD_COLUMNS = (("xy", slice(0, 2)), ("rcs", slice(2, 3)), ("doppler", slice(3, 4)))


def _reference(scores):
    return {REFERENCE + k: v for k, v in scores.items()}


def inside_raster(gt, point_limit):
    """The (N,4) GT detections inside the raster and the number dropped (``gt_outside_raster``).

    Applied before any scoring, for both implementations. RadarGen's evaluator filters its GT to
    +-coordinate_range itself, but here the GT comes from the preprocessed files. A detection outside
    the raster can never be generated, so it would be an unavoidable miss in every recall-like score.
    """
    inside = (np.abs(gt[:, :2]) <= point_limit).all(1)
    return gt[inside], int((~inside).sum())


def raster_metrics(raster, gt, boxes, nll_floor=1e-3):
    """Scores of the raster itself, free of sampling noise: the frame row and one row per box.

    ``ppp_expected_count`` is the expected count of the cells whose centers lie in the box and
    ``ppp_hit_probability`` = P(at least one point in the box) = 1 - exp(-expected count).
    """
    rows, cols = raster.cells(gt[:, :2])
    row = {**ppp_nll(raster, rows, cols, nll_floor), **field_metrics(raster, gt, rows, cols)}
    box_rows = []
    for box in boxes:
        expected, observed = expected_box_count(raster, box), int(box_local(gt, box)[1].sum())
        box_rows.append(dict(ppp_expected_count=expected, ppp_expected_count_abs_error=abs(expected - observed),
                             ppp_hit_probability=float(-np.expm1(-expected)) if observed else np.nan))
    return row, box_rows


def ppp_nll(raster, rows, cols, floor=1e-3):
    """Negative log-likelihood of the GT detections in the cells ``rows, cols`` under the raster's PPP.

    With expected count m_c in a cell of area a, total L = sum m_c, intensity l = m_c / a per m^2 and
    spatial density p = l / L, the NLL of n detections x_i splits into count and position:

        NLL = L - sum_i log l(x_i) = [L - n log L + log n!] + n [-1/n sum_i log p(x_i)] - log n!

    ``ppp_nll`` = NLL / n (nats per GT point), ``ppp_nll_count`` = the first bracket = -log Poisson(n; L),
    ``ppp_nll_spatial`` = the second (nats per point). Densities are per m^2, so rasters of different
    resolution and shift compare. Our training loss per point (PPPLoss) is ppp_nll - log a without floor.

    The floor mixes p with ``floor`` of the uniform density, so a cell without mass (RadarGen's truncated
    splats) costs a finite amount; ``ppp_zero_intensity_count`` counts GT detections in such cells.
    Changes the NLL by at most -log(1 - floor) per point where p > 0.
    """
    masses = raster.masses.numpy()
    total, n = float(masses.sum()), len(rows)
    at_gt = masses[rows, cols]
    density = ((1 - floor) * at_gt + floor * total / masses.size) / total / np.prod(raster.cell_size)
    count = total - (n * math.log(total) if n else 0.0) + math.lgamma(n + 1)
    spatial = float(-np.log(density).mean()) if n else np.nan
    return dict(ppp_nll=(count - math.lgamma(n + 1)) / n + spatial if n else np.nan, ppp_nll_count=count,
                ppp_nll_spatial=spatial, ppp_zero_intensity_count=int((at_gt == 0).sum()))


def sampled_metrics(points, gt, boxes, normalization, reference_mmd, device):
    """Scores of one drawn (N,4) cloud against the (M,4) GT detections inside the raster.

    Returns the frame row, one row per box and the box-local clouds of both implementations
    ``{"ours" | "radargen": {"gt" | "syn": [(category, (K,4) cloud), ...]}}`` for the class MMD.

    Empty clouds: RadarGen's evaluator has no policy. It raises on an empty GT cloud, and its per-sample
    code fails on an empty generated cloud (sklearn cannot fit NearestNeighbors on zero points, MMDLoss
    raises). The original PPP validation filled in 0 or 1 instead, which mixes an arbitrary policy into the means.
    Here both implementations give NaN. The summary means skip NaN and count it in ``<key>_nan_count``;
    read those together with ``count_abs_error`` and the summary's ``empty_generated_draws``.
    """
    row = dict(generated_count=len(points), count_abs_error=abs(len(points) - len(gt)))
    row.update(cloud_metrics(points, gt, normalization, device))
    row.update(_reference(radargen.cloud_scores(points, gt, normalization, reference_mmd, device)))
    return row, *box_metrics(points, gt, boxes, normalization, device)


def cloud_metrics(points, gt, normalization, device):
    """Our whole-cloud scores of the generated vs the GT cloud (details in the functions below).

    RadarGen's compute_per_sample_metrics reports only the symmetric chamfer distances and only the IoU
    of its threshold scores; here also their parts, which tell over- from under-generation apart.
    """
    return {**chamfer_distances(points, gt, normalization), **threshold_scores(points, gt),
            **matching_scores(points, gt), **cloud_mmd(points, gt, device), **nearest_mark_errors(points, gt)}


def nearest_mark_errors(points, gt):
    """Mean absolute RCS and Doppler error between each GT detection and its nearest generated point in xy.

    No RadarGen counterpart. The dense errors (field_metrics) read the marks in each detection's own
    cell, whether or not the sampled cloud has a point there. These errors score the marks as a consumer
    of the sampled cloud sees them. They therefore also include position errors: if the nearest
    generated point lies in another cell or far away, its marks are compared anyway (its distance is in
    ``chamfer_distance_xy_gt_to_generated``). Equidistant neighbours in different cells are broken
    arbitrarily; duplicates in one cell share their marks.
    """
    if not (len(points) and len(gt)):
        return dict.fromkeys(("rcs_mae_nn", "doppler_mae_nn"), np.nan)
    nearest = cKDTree(points[:, :2]).query(gt[:, :2])[1]
    rcs, doppler = np.abs(points[nearest, 2:] - gt[:, 2:]).mean(0)
    return dict(rcs_mae_nn=float(rcs), doppler_mae_nn=float(doppler))


def nearest_distances(source, target):
    """(N,) Euclidean distance from each source point to its nearest target point (float64 kd-tree).

    RadarGen's chamfer_distance uses sklearn NearestNeighbors(algorithm="kd_tree", leaf_size=1): the same
    distances, but leaf_size=1 makes building and querying the tree needlessly slow. Its docstring
    claims squared distances, while kneighbors returns plain Euclidean ones; pcl_f1_iou_np trusts the
    docstring and takes another square root (see threshold_scores).
    """
    return cKDTree(target).query(source)[0]


def normalized(cloud, normalization):
    """Cloud min-max scaled by the dataset ranges (x and y by [-coordinate_range, coordinate_range]).

    Same mapping as RadarGen's normalize_pcl_for_chamfer. RadarGen runs it in float32 torch, so its
    all_norm chamfer carries float32 rounding (~1e-9 here). The weighting of the columns is a problem
    of the definition, shared by ours: every column is scaled by its full dataset range. With the
    TruckScenes ranges, 1 m in x or y weighs as much as 0.86 dBsm RCS or 2.4 m/s Doppler. So the
    all_norm chamfer is almost blind to Doppler errors that matter physically, e.g. moving vs static.
    """
    n = normalization
    low = np.array([-n.coordinate_range, -n.coordinate_range, n.rcs_min, n.doppler_min])
    high = np.array([n.coordinate_range, n.coordinate_range, n.rcs_max, n.doppler_max])
    return (cloud - low) / (high - low)


def chamfer_distances(points, gt, normalization):
    """Chamfer distance on xy (meters) and on all four normalized columns, and its two halves.

    ``_generated_to_gt``: mean distance of a generated point to its nearest GT point (precision-like);
    ``_gt_to_generated``: the reverse (recall-like). The chamfer distance is their mean.

    Same definition as RadarGen's ``0.5 * chamfer_distance(syn, gt, direction="bi")``. RadarGen reports
    only the sum, which hides which side fails. A cloud that covers all GT but adds many spurious points
    (small gt_to_generated, large generated_to_gt) and a cloud concentrated on a few strong reflectors
    (the reverse) can get the same chamfer distance. For boxes, RadarGen casts the in-box points to
    float32 first; ours stays in float64.
    """
    scores = {}
    for name, a, b in (("xy", points[:, :2], gt[:, :2]),
                       ("all_norm", normalized(points, normalization), normalized(gt, normalization))):
        if len(a) and len(b):
            to_gt, to_generated = nearest_distances(a, b).mean(), nearest_distances(b, a).mean()
        else:
            to_gt = to_generated = np.nan
        scores[f"chamfer_distance_{name}"] = float((to_gt + to_generated) / 2)
        scores[f"chamfer_distance_{name}_generated_to_gt"] = float(to_gt)
        scores[f"chamfer_distance_{name}_gt_to_generated"] = float(to_generated)
    return scores


def threshold_scores(points, gt, threshold=1.0):
    """RadarGen's iou_th1 and its parts, on xy.

    precision: share of generated points within ``threshold`` m of a GT point; recall: share of GT points
    within ``threshold`` m of a generated point; F1; and IoU = PR / (P + R - PR).

    Problems of RadarGen's pcl_f1_iou_np:

    - Precision and recall are swapped: its P is the share of GT points near a generated point, i.e. the
      recall. F1 and IoU are symmetric, so iou_th1 is unaffected; ours reports the parts with correct names.
    - It takes the square root of the nearest-neighbour distances, believing chamfer_distance returns
      squared ones (see nearest_distances). It thus tests sqrt(d) < th, i.e. d < th^2. That is identical
      at th = 1 as used, but its default th = 0.5 would actually test d < 0.25 m. Ours thresholds d.
    - If P and R are both below 1e-3, it returns P as F1 and IoU instead of the formula. That is a small
      discontinuity; ours applies the formula and returns 0 if P * R = 0.
    - The "IoU" is no intersection over union of point sets. PR / (P + R - PR) = F1 / (2 - F1), a monotone
      transform of F1 (the Jaccard-Dice relation), so it adds no information beyond F1. RadarGen
      computes F1 and then discards it.

    Problem of the definition, shared by ours: there is no one-to-one constraint. A single generated
    point near a cluster of GT points "recalls" all of them, and generated points at the same cell
    center all count. A coarse cloud therefore scores well here; matching_scores is the one-to-one
    version.
    """
    if not (len(points) and len(gt)):
        return dict.fromkeys(("iou_th1", "iou_th1_precision", "iou_th1_recall", "iou_th1_f1"), np.nan)
    precision = float((nearest_distances(points[:, :2], gt[:, :2]) < threshold).mean())
    recall = float((nearest_distances(gt[:, :2], points[:, :2]) < threshold).mean())
    both = precision * recall
    return dict(iou_th1=both / (precision + recall - both) if both else 0.0, iou_th1_precision=precision,
                iou_th1_recall=recall, iou_th1_f1=2 * both / (precision + recall) if both else 0.0)


def matching_scores(points, gt, distance=1.0, rcs=8.0, doppler=2.5):
    """RadarGen's distance_attr scores: recall, precision and F1 of a maximum one-to-one matching of GT
    and generated points, where a pair may match if it is at most ``distance`` m apart and differs by at
    most ``rcs`` dBsm and ``doppler`` m/s.

    Problems of RadarGen's distance_attr_recall_precision_f1_hungarian:

    - It solves a Hungarian assignment (linear_sum_assignment) on dense M x N matrices: O(n^3) time and
      several dense M x N float32 tensors (distances, an (M,N,2) attribute difference, masks). This
      dominates the evaluation time at ~1k points and would be infeasible at ~10k.
    - The distance cost is irrelevant to the scores. Invalid pairs cost 1e5, so the assignment first
      maximizes the number of valid pairs; the distances then only choose among maximum matchings,
      which leaves TP unchanged. TP is therefore the maximum bipartite matching size of the valid-pair
      graph. Ours computes that with Hopcroft-Karp (O(E sqrt(V))) on the sparse graph, whose edges come
      from a kd-tree radius query. The 1e5 trick holds as long as the summed valid distances
      (<= min(M, N) m) stay below 1e5.
    - torch.cdist in float32 uses the matrix-product formula |a|^2 + |b|^2 - 2ab for inputs of more than
      25 points. Its cancellation at coordinates of ~50 m gives distance errors of ~1e-4 m, so pairs
      right at the 1 m threshold can flip. Ours computes exact float64 distances.
    - eps = 1e-10 in the denominators biases the scores by ~1e-8; this is the MAE between the two.
    - The empty-cloud branch returns e.g. recall 1 for an empty GT cloud. It is unreachable (see
      sampled_metrics), and the code's own comment calls these values undefined.

    Thresholds (both implementations, all inclusive): 1 m, 8 dBsm, 2.5 m/s.
    """
    if not (len(points) and len(gt)):
        return dict.fromkeys(("distance_attr_recall", "distance_attr_precision", "distance_attr_f1"), np.nan)
    neighbours = cKDTree(gt[:, :2]).query_ball_tree(cKDTree(points[:, :2]), distance)
    rows = np.repeat(np.arange(len(gt)), [len(n) for n in neighbours])
    cols = np.fromiter(chain.from_iterable(neighbours), dtype=np.int64, count=len(rows))
    valid = (np.abs(gt[rows, 2] - points[cols, 2]) <= rcs) & (np.abs(gt[rows, 3] - points[cols, 3]) <= doppler)
    graph = csr_array((np.ones(int(valid.sum())), (rows[valid], cols[valid])), shape=(len(gt), len(points)))
    matched = int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())
    recall, precision = matched / len(gt), matched / len(points)
    return dict(distance_attr_recall=recall, distance_attr_precision=precision,
                distance_attr_f1=2 * recall * precision / (recall + precision) if matched else 0.0)


def rbf_mmd(x, y, device):
    """Biased squared MMD between (N,D) and (M,D) samples with RadarGen's kernel: the sum of five
    Gaussian kernels whose bandwidths are 2^-2 ... 2^2 times the mean pairwise squared distance of the
    pooled samples (float64).

    Ours is the same estimator. Differences in implementation:

    - RadarGen's callers cast the clouds to float32 before MMDLoss casts them to float64, so the inputs
      are rounded. Ours uses the float64 clouds.
    - RadarGen's RBF stacks all five kernels: 5 (N+M)^2 float64, ~0.9 GB for a 4.7k-point pair. This
      is why its class MMD must subsample and silently gives NaN on OOM. Ours accumulates them into one
      matrix.
    - MMDLoss raises on a negative result. It has a ``negative_tolerance`` parameter but never uses it,
      so float rounding on near-identical inputs (~-1e-17) can abort an evaluation run. Ours returns
      the raw value.

    Problems of the definition, kept by ours for comparability:

    - The bandwidth is data-dependent. It comes from the pooled clouds, so it depends on the generated
      cloud itself: a generated cloud with a far outlier spread widens all kernels and lowers the MMD.
      MMDs of different frames or models are also computed with different kernels, so their mean mixes
      different scales.
    - The estimator is biased (a V-statistic). It includes the diagonal terms k(x, x) = 5, a positive
      bias of order 1/N + 1/M. Small clouds get a larger MMD for the same distribution, so frames with
      few points weigh more in the mean. The unbiased estimator would exclude the diagonals.
    """
    z = torch.as_tensor(np.concatenate([x, y]), dtype=torch.float64, device=device)
    squared = torch.cdist(z, z).square_()
    bandwidth = squared.sum() / (len(z) ** 2 - len(z)) + torch.finfo(torch.float64).eps
    kernel = torch.zeros_like(squared)
    for power in range(-2, 3):
        kernel += torch.exp(-squared / (bandwidth * 2.0 ** power))
    n = len(x)
    return float(kernel[:n, :n].mean() - 2 * kernel[:n, n:].mean() + kernel[n:, n:].mean())


def cloud_mmd(points, gt, device):
    """RadarGen's pcl_mmd scores: MMD of the xy, RCS and Doppler distributions of the two clouds."""
    nonempty = len(points) and len(gt)
    return {f"pcl_mmd_{name}": rbf_mmd(points[:, columns], gt[:, columns], device) if nonempty else np.nan
            for name, columns in MMD_COLUMNS}


def field_metrics(raster, gt, rows, cols):
    """Scores of the raster itself, without sampling: the physical marks in the cells ``rows, cols`` of
    the GT detections (several may share a cell) and the expected point count.

    No RadarGen counterpart. RadarGen can only score sampled clouds, so its scores carry Poisson
    sampling noise. These scores evaluate the predicted intensity and marks directly.
    ``rcs_mae_dense`` and ``doppler_mae_dense`` compare the marks of each detection's cell, whatever the
    intensity there (see nearest_mark_errors for the sampled version).
    ``target_collision_count`` counts GT detections that share a raster cell with another one; sampled
    points sit at cell centers, so this is the part of the GT the raster cannot resolve.
    """
    row = dict(target_collision_count=len(gt) - len(set(zip(rows.tolist(), cols.tolist()))))
    for column, name, mark in ((2, "rcs", raster.rcs), (3, "doppler", raster.doppler)):
        row[f"{name}_mae_dense"] = (float(np.abs(mark.numpy()[rows, cols] - gt[:, column]).mean())
                                   if len(gt) else np.nan)
    total = float(raster.masses.sum())
    return {**row, "ppp_expected_total_count": total, "ppp_total_count_abs_error": abs(total - len(gt))}


def box_local(cloud, box):
    """Points of the cloud in the box, in box coordinates [x along the length, y along the width,
    remaining columns] (RadarGen's: offset from the center times R[:2,:2]), and the inside mask.

    As in RadarGen, a point is in the box if it lies in the xy footprint of the bottom face.

    Problems of RadarGen's get_points_in_box and convert_pcl_to_box_coords:

    - matplotlib's Path.contains_points leaves the membership of points exactly on an edge unspecified
      (it depends on the polygon's orientation), and it pulls matplotlib into the evaluation. Ours
      solves for the point's coordinates in the footprint's half-edge basis and includes the edges.
    - The bottom face is not the box's vertical projection. TruckScenes boxes are tilted with road and
      ego pitch (3-5 degrees observed), so the bottom face is shifted against the box by
      h/2 sin(tilt), up to ~0.2 m for 4 m high trucks. Points within the projection but beyond the
      shifted face are missed. The tight footprint would be the convex hull of all eight projected
      corners; ours keeps RadarGen's for comparability. Before matching it, ours used the centered
      rectangle and disagreed on a few points per box.
    - The box-local coordinates use the 3D center as origin, not the bottom-face center used for
      membership. They also use R[:2,:2], which is no rotation for a tilted box: it shrinks by up to
      1 - cos(tilt) (~0.4 % at 5 degrees). The class-MMD clouds are therefore slightly offset and
      distorted. Ours keeps the same transform for comparability.
    """
    rotation, (width, length, height) = box.rotation_matrix, box.wlh
    half_edges = rotation[:2, :2] * [length / 2, width / 2]
    bottom = box.center[:2] - rotation[:2, 2] * height / 2
    inside = (np.abs(np.linalg.solve(half_edges, (cloud[:, :2] - bottom).T)) <= 1).all(0)
    local = (cloud[inside, :2] - box.center[:2]) @ rotation[:2, :2]
    return np.column_stack([local, cloud[inside, 2:]]), inside


def expected_box_count(raster, box):
    """Expected number of points in the cells whose centers lie in the box (box_local)."""
    rows, cols = raster.cells(box.bottom_corners()[:2].T)
    rows, cols = np.arange(rows.min(), rows.max() + 1), np.arange(cols.min(), cols.max() + 1)
    inside = box_local(raster.centers(rows, cols), box)[1]
    return float(raster.masses.numpy()[np.ix_(rows, cols)].ravel()[inside].sum())


def box_scores(points, gt, box, normalization):
    """Our scores of one box and its box-local clouds ``{"gt": (M,4), "syn": (K,4)}``.

    A box is a hit if both clouds have points in it, a false negative if only GT has, a false positive
    if only the generated cloud has. Chamfer distances need a hit.

    These are the same definitions as RadarGen's compute_box_metrics_and_stats. Differences and problems:

    - RadarGen casts the in-box points to float32 before the chamfer distances. Ours uses float64 world
      coordinates, not the box-local ones, so box_local's non-rigid transform cannot distort them.
    - Ours also reports both chamfer halves.
    - num_points_similarity is 1 for a box that is empty on both sides (RadarGen's n_points_similarity_np;
      kept for comparability). Correctly predicting nothing in a box without GT points thus scores
      perfectly, and such boxes pull the mean up. Filter the box table on gt_count > 0 for the
      similarity of actually observed objects.
    - RadarGen's hit rate only covers boxes with GT points, so a model that puts points into every box
      is not penalized there. Read it together with the false-positive boxes, which RadarGen counts
      but its evaluator does not print.
    """
    (gt_local, gt_inside), (local, inside) = box_local(gt, box), box_local(points, box)
    n_gt, n_generated = len(gt_local), len(local)
    scores = dict(gt_count=n_gt, generated_count=n_generated, hit=int(n_gt > 0 and n_generated > 0),
                  false_negative=int(n_gt > 0 and n_generated == 0), false_positive=int(n_gt == 0 and n_generated > 0),
                  **chamfer_distances(points[inside], gt[gt_inside], normalization),
                  num_points_similarity=min(n_gt, n_generated) / max(n_gt, n_generated) if n_gt or n_generated else 1.0)
    return scores, {"gt": gt_local, "syn": local}


def box_metrics(points, gt, boxes, normalization, device):
    """One row per box with our and RadarGen's box scores, and the box-local clouds of both implementations.

    The boxes are all categories, as RadarGen's adapter passes them. The original PPP validation scored only
    car, truck and trailer, which made its hit rate incomparable to RadarGen's.
    """
    rows, clouds = [], {name: {"gt": [], "syn": []} for name in ("ours", "radargen")}
    for box in boxes:
        ours, local = box_scores(points, gt, box, normalization)
        reference, reference_local = radargen.box_scores(points, gt, box, normalization, device)
        rows.append(dict(annotation_token=box.token, category=box.name, **ours, **_reference(reference)))
        for name, sides in (("ours", local), ("radargen", reference_local)):
            for side, cloud in sides.items():
                clouds[name][side].append((box.name, cloud))
    return rows, clouds


def class_mmd(frame_clouds, device, classes=FOREGROUND_CLASSES, max_points=4000, seed=0):
    """Per class, MMD of xy, RCS and Doppler between the box-local GT and generated points of all
    boxes of that class in all frames; each side is randomly subsampled to at most ``max_points``.

    ``frame_clouds``: per frame ``{side: [(category, cloud), ...]}``.

    Problems of RadarGen's version (aggregate_metrics + finalize_global_metrics, see .radargen.class_mmd):

    - The two sides are misaligned. Per frame and class, it pools the nonempty box clouds into a GT
      list and a generated list. A frame enters the GT list only if a box of the class has GT points,
      and the generated list only if one has generated points. Each list is then cut independently to
      ``[::3][:200]``. Once a frame contributes to only one side, the lists shift against each other,
      so the compared samples come from different frames, i.e. different objects, distances and
      viewing angles. The score then partly measures frame selection, not the model.
    - It uses only part of the data, depending on order. ``[::3][:200]`` reaches only the first ~600
      qualifying frames: in dataset order, the first scenes, out of 1906 val keyframes.
    - The subsample unit is a frame, so the point count and memory vary. It runs on the GPU with five
      stacked kernels and silently gives NaN on OOM.
    - It inherits rbf_mmd's problems (data-dependent bandwidth, biased estimator).

    Ours pools all boxes of the class in all frames for each side, so both sides cover the same
    frames. It then subsamples points uniformly at random (seeded) to at most ``max_points`` per side,
    which bounds the memory to a few (2 max_points)^2 float64 matrices.
    """
    rng = np.random.default_rng(seed)
    scores = {}
    for category in classes:
        sides = []
        for side in ("gt", "syn"):
            pooled = [cloud for clouds in frame_clouds for name, cloud in clouds[side]
                      if name.split(".")[-1] == category]
            cloud = np.vstack(pooled) if pooled else np.empty((0, 4))
            if len(cloud) > max_points:
                cloud = cloud[rng.choice(len(cloud), max_points, replace=False)]
            sides.append(cloud)
        gt, generated = sides
        for name, columns in MMD_COLUMNS:
            scores[f"invariant_mmd_{name}_{category}"] = (rbf_mmd(generated[:, columns], gt[:, columns], device)
                                                         if len(gt) and len(generated) else np.nan)
    return scores


def describe(table, prefix=""):
    """Mean, std and NaN count of every numeric column, and for each column with a radargen_ twin the
    mean absolute difference of the two over the rows where both are finite.

    A score is NaN where it is undefined (e.g. an empty cloud, the chamfer distance of a missed box). The
    mean and std skip those rows, so one NaN never turns a summary value into NaN; ``<key>_nan_count``
    says how many rows were skipped. The std is the population std, as in RadarGen's _safe_mean_std.
    RadarGen's finalize_global_metrics keeps only means and stds, so its output alone cannot show
    per-frame differences between implementations; hence the per-row comparison here.
    """
    summary = {}
    for column in table.select_dtypes("number"):
        reference = column.startswith(REFERENCE)
        name = REFERENCE * reference + prefix + column.removeprefix(REFERENCE)
        summary[f"{name}_mean"] = table[column].mean(skipna=True)
        summary[f"{name}_std"] = table[column].std(ddof=0, skipna=True)
        summary[f"{name}_nan_count"] = int(table[column].isna().sum())
    for column in table.select_dtypes("number"):
        if REFERENCE + column in table:
            summary[f"{prefix}{column}_radargen_mae"] = (table[column] - table[REFERENCE + column]).abs().mean()
    return summary


def summarize(frames, boxes, frame_clouds, device):
    """One summary row from the frame and box tables (pandas, one row per draw) and the box clouds.

    Means run over all draws of all frames. RadarGen's values are named radargen_<key>;
    ``<key>_radargen_mae`` is the mean absolute difference of the per-row values, or for summary-only
    values the absolute difference.
    """
    first = frames[frames.draw == 0]
    summary = dict(frames=len(first), draws=len(frames) // max(len(first), 1),
                   empty_gt_frames=int((first.gt_count == 0).sum()),
                   empty_generated_draws=int((frames.generated_count == 0).sum()))
    identity = ["index", "frame_index", "draw", "is_key_frame"]
    summary.update(describe(frames.drop(columns=identity, errors="ignore")))
    if "inference_seconds" in frames:
        summary["inference_seconds_median"] = frames["inference_seconds"].median()

    def total(key, rows=boxes):
        return int(rows[key].sum()) if key in rows else 0

    def hit_rate(name, rows=boxes):
        hits, misses = total(f"{name}hit", rows), total(f"{name}false_negative", rows)
        return hits / (hits + misses) if hits + misses else np.nan

    summary["evaluated_boxes"] = len(boxes)
    draws = [rows for _, rows in boxes.groupby("draw")] if len(boxes) else []
    for name in ("", REFERENCE):
        summary[f"{name}box_hit_rate"] = hit_rate(name)
        # sample std over draws, like voxelnext/<score>:std in make_table (a per-box std of a hit is binary)
        summary[f"{name}box_hit_rate_std"] = pd.Series([hit_rate(name, rows) for rows in draws], dtype=float).std()
        summary[f"{name}false_positive_boxes"] = total(f"{name}false_positive")
    summary.update(describe(boxes.drop(columns=identity, errors="ignore"), "box_"))
    summary.update(class_mmd([clouds["ours"] for clouds in frame_clouds], device))
    summary.update(_reference(radargen.class_mmd([clouds["radargen"] for clouds in frame_clouds], device,
                                                 FOREGROUND_CLASSES)))
    for key in [k for k in summary if REFERENCE + k in summary and not k.endswith(("_mean", "_std", "_nan_count"))]:
        summary[f"{key}_radargen_mae"] = abs(summary[key] - summary[REFERENCE + key])
    return summary


def headline(summary):
    """The HEADLINE scores (means for frame and box scores): ours, RadarGen's, their MAE and the number
    of frames (boxes) where ours is NaN."""
    rows = {}
    for key in HEADLINE:
        stat = "_mean" if f"{key}_mean" in summary else ""
        rows[key] = dict(ours=summary.get(key + stat, np.nan), radargen=summary.get(REFERENCE + key + stat, np.nan),
                         mae=summary.get(f"{key}_radargen_mae", np.nan), nan=summary.get(f"{key}_nan_count", 0))
    return pd.DataFrame.from_dict(rows, orient="index")
