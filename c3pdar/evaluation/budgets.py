"""Precision/recall at a fixed point budget per frame: how well do b points of a model recover the GT cloud?

Each draw yields an ordered candidate cloud and a budget b scores its first b points:

- "ppp": iid cells from the normalized intensity, so the first b points are the PPP conditioned on b points.
- "radargen": the cells of the deconvolved map (radargen.recovered_map) with a positive value, from highest
  to lowest, so the first b points are the cloud of the threshold between the b-th and (b+1)-th value.
  Budgets beyond the positive cells are infeasible (NaN), barely more than its native cloud.

Budget "native" scores the stored cloud, "exact" takes b = GT count. Frames without GT points score NaN.
Scores: one-to-one matching on position only (position_*) and jointly with RCS/Doppler (distance_attr_*),
RadarGen's coverage scores (iou_th1_*), the RCS MMD (pcl_mmd_rcs) and the box counts of all-category boxes
with GT points that receive a generated point (box_hits) or none (box_misses), and of boxes without GT
points that receive one (box_false_positives), as in metrics.box_scores. The summary pools the box counts
into box_hit_rate = hits / (hits + misses), like the evaluation's summary.
"""

import multiprocessing

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..data import load_frame
from ..models.radargen import COORDINATE_RANGE
from ..sampling import cell_centers, draw_generator, ppp_raster
from . import PROGRESS, radargen
from .metrics import box_local, inside_raster, matching_scores, rbf_mmd, threshold_scores
from .store import RawStore

SCORES = ("position_precision", "position_recall", "position_f1",
          "distance_attr_precision", "distance_attr_recall", "distance_attr_f1",
          "iou_th1_precision", "iou_th1_recall", "iou_th1_f1",
          "pcl_mmd_rcs", "box_hits", "box_misses", "box_false_positives")

_WORKER = {}


def ppp_candidates(raster, count, generator):
    """(count,4) iid points [x, y, RCS, Doppler] at cell centers, cells drawn with probability ~ mass."""
    cdf = raster.masses.flatten().cumsum(0)
    u = torch.rand(count, generator=generator, dtype=torch.float64) * cdf[-1]
    cells = torch.searchsorted(cdf, u, right=True).clamp_(max=cdf.numel() - 1)
    xy = cell_centers(raster.masses.shape, float(raster.point_limit))[cells]
    return torch.cat((xy, raster.rcs.flatten()[cells, None], raster.doppler.flatten()[cells, None]), 1).numpy()


def radargen_candidates(maps, device):
    """RadarGen's cells with a positive deconvolved value as (K,4) points, highest value first."""
    recovered = radargen.recovered_map(maps[0], device)
    rows, cols = np.nonzero(recovered > 0)
    order = np.argsort(-recovered[rows, cols], kind="stable")
    return radargen.cell_points(maps, rows[order], cols[order], COORDINATE_RANGE)


def scores(points, gt):
    position = matching_scores(points, gt, rcs=np.inf, doppler=np.inf)
    mmd = rbf_mmd(points[:, 2:3], gt[:, 2:3], "cpu") if len(points) and len(gt) else np.nan
    return {**{k.replace("distance_attr", "position"): v for k, v in position.items()},
            **matching_scores(points, gt), **threshold_scores(points, gt), "pcl_mmd_rcs": mmd}


def first_inside(cloud, boxes):
    """Per box, the index of the cloud's first point in it (len(cloud) if none): a prefix of b points hits
    the box iff the index is below b."""
    return np.array([np.argmax(inside) if (inside := box_local(cloud, box)[1]).any() else len(cloud)
                     for box in boxes], dtype=np.int64)


def box_counts(observed, generated):
    return dict(box_hits=int((observed & generated).sum()), box_misses=int((observed & ~generated).sum()),
                box_false_positives=int((~observed & generated).sum()))


def _init(raw, attrs, budgets):
    torch.set_num_threads(1)
    _WORKER.update(store=RawStore(raw) if attrs["kind"] == "ppp" else None, attrs=attrs, budgets=budgets)


def _score_frame(task):
    """Rows of one frame: every draw x budget. ``clouds``: per draw (candidates, native) for "radargen"."""
    identity, path, clouds = task
    store, attrs, budgets = _WORKER["store"], _WORKER["attrs"], _WORKER["budgets"]
    scene, frame = identity["scene_token"], identity["frame_index"]
    sample = load_frame(path, load_boxes=identity["is_key_frame"])
    # Non-keyframes carry their keyframe's boxes, which do not fit their detections.
    boxes = sample.boxes(categories=None)[0] if identity["is_key_frame"] else []
    gt = inside_raster(sample.detections.numpy(), attrs["point_limit"])[0]
    observed = first_inside(gt, boxes) < len(gt)
    targets = {"native": None, "exact": len(gt), **{str(b): b for b in budgets}}
    if clouds is None:
        raster = ppp_raster(store.read(scene, frame, "fields")[0], attrs["point_limit"], attrs["mark_link"])
        count = max(max(budgets), len(gt))
        clouds = [(ppp_candidates(raster, count, draw_generator(attrs["seed"], scene, frame, draw)),
                   store.read(scene, frame, f"{draw}/points")[0]) for draw in range(identity.pop("draws"))]
    rows = []
    for draw, (candidates, native) in enumerate(clouds):
        first = first_inside(candidates, boxes)
        for budget, target in targets.items():
            feasible = target is None or target <= len(candidates)
            points = native if target is None else candidates[:target]
            generated = first_inside(native, boxes) < len(native) if target is None else first < target
            values = ({**scores(points, gt), **box_counts(observed, generated)} if feasible
                      else dict.fromkeys(SCORES, np.nan))
            rows.append({**identity, "draw": draw, "budget": budget, "target": np.nan if target is None else target,
                         "available": len(candidates), "gt_count": len(gt), "count": len(points) if feasible else np.nan,
                         **values})
    return rows


def _tasks(store, dataset, indices, draws, device):
    for position, index in enumerate(indices):
        info = dataset.frames.iloc[index]
        identity = dict(index=position, scene_token=info.scene_token, frame_index=int(info.frame_index),
                        is_key_frame=bool(info.is_key_frame))
        clouds = None
        if store.attrs["kind"] == "radargen":
            clouds = []
            for draw in range(draws):
                maps = store.read(identity["scene_token"], identity["frame_index"], f"{draw}/maps")[0]
                clouds.append((radargen_candidates(maps, device),
                               store.read(identity["scene_token"], identity["frame_index"], f"{draw}/points")[0]))
        else:
            identity["draws"] = draws
        yield identity, dataset.files[index], clouds


def evaluate_budgets(raw, dataset, indices, draws, budgets, *, device, workers=16):
    """Frame table: one row per frame x draw x budget (``budgets`` plus "native" and "exact")."""
    with RawStore(raw) as store:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in store.attrs.items()}
        if not np.isclose(attrs["point_limit"], dataset.point_limit):
            raise ValueError(f"store point_limit {attrs['point_limit']} != dataset {dataset.point_limit}")
        context = multiprocessing.get_context("spawn")
        rows = []
        with context.Pool(workers, _init, (raw, attrs, list(budgets))) as pool:
            tasks = _tasks(store, dataset, indices, draws, device)
            for frame_rows in tqdm(pool.imap(_score_frame, tasks), total=len(indices), desc="budgets",
                                   unit="frame", **PROGRESS):
                rows.extend(frame_rows)
    return pd.DataFrame(rows)


def _clustered(numerator, denominator, scenes):
    """Ratio sum(numerator) / sum(denominator) over the non-NaN rows and its scene-clustered standard error
    (linearized; frames of a scene and draws of a frame are correlated), n and number of scenes."""
    valid = numerator.notna() & denominator.notna()
    numerator, denominator, scenes = numerator[valid], denominator[valid], scenes[valid]
    total = denominator.sum()
    ratio = float(numerator.sum() / total) if total else np.nan
    sums = (numerator - ratio * denominator).groupby(scenes).sum()
    clusters = len(sums)
    se = float(np.sqrt(clusters / (clusters - 1) * (sums ** 2).sum()) / total) if clusters > 1 and total else np.nan
    return ratio, se, int(valid.sum()), clusters


def summarize_budgets(frames):
    """Per budget and score: mean over the feasible frame draws (box_hit_rate: pooled over their boxes), its
    scene-clustered standard error, n of total."""
    rows = []
    for budget, table in frames.groupby("budget", sort=False):
        ratios = {score: (table[score].astype(float), pd.Series(1.0, index=table.index).where(table[score].notna()))
                  for score in ("count", "target", *SCORES) if score in table}
        if "box_hits" in table:
            ratios["box_hit_rate"] = (table.box_hits.astype(float), (table.box_hits + table.box_misses).astype(float))
        for score, (numerator, denominator) in ratios.items():
            mean, se, n, clusters = _clustered(numerator, denominator, table.scene_token)
            rows.append(dict(budget=budget, score=score, mean=mean, se=se, n=n, total=len(table), scenes=clusters))
    return pd.DataFrame(rows)
