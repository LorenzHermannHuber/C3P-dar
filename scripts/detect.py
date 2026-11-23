"""Downstream detection scores (mAP, NDS) of generated or measured radar point clouds.

python scripts/detect.py radargen                             # all draws stored in EVAL_RAW/radargen
python scripts/detect.py dit --frames 20 --draws 1            # quick check (dit holds one checkpoint)
python scripts/detect.py --real                               # the measured clouds (ppp_targets)

A radar-only VoxelNeXt detector, trained on measured RADAR_LEFT_SIDE clouds, runs on every draw of
the annotated val keyframes (c3pdar/evaluation/detection.py). The clouds have no height: points get
z = --height (default 1 m) in the flat-up vehicle frame and are moved into the RADAR_LEFT_SIDE frame.
Car, truck and trailer boxes within 50 m are matched by center distance (TruckScenes devkit); mAP
and NDS are means over these three classes, one score per draw.

Runs in a VoxelNeXt environment (OpenPCDet's pcdet with its CUDA ops, spconv 2, truckscenes-devkit,
pyquaternion, easydict, and torch, h5py, pandas, tqdm) with this repository on PYTHONPATH. Writes
voxelnext.csv (one row per draw: summary, per-class scores, box counts) and voxelnext_detections.h5
next to scripts/evaluate.py's tables in EVAL_METRICS/<name>/<selection>/; never overwrites them.
"""

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd
from tqdm.auto import tqdm

from c3pdar import DATA_ROOT, EVAL_METRICS, EVAL_RAW, TRUCKSCENES_ROOT, VOXELNEXT
from c3pdar.data import PreprocessedRadarGenDataset
from c3pdar.evaluation import PROGRESS
from c3pdar.evaluation.detection import (SENSOR, Detector, detector_points, frame_inputs, global_boxes, point_seed,
                                         score, score_row)
from c3pdar.evaluation.store import RawStore, resolve
from c3pdar.evaluation.validation import frame_ids, select_frames

COLUMNS = ("mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "NDS")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="directory below EVAL_RAW or a path (omit with --real)")
    parser.add_argument("--real", action="store_true", help="score the measured radar clouds")
    parser.add_argument("--frames", type=int, help="number of frames (default: all)")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--draws", type=int, help="point clouds per frame (default: all generated)")
    parser.add_argument("--height", type=float, default=1.0, help="constant point height in the flat-up frame (m)")
    parser.add_argument("--seed", type=int, default=42, help="point shuffle before voxelization")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--truckscenes-root", type=Path, default=TRUCKSCENES_ROOT)
    parser.add_argument("--version", default="v1.2-trainval")
    parser.add_argument("--detector", type=Path, default=VOXELNEXT)
    args = parser.parse_args()
    if args.real == (args.name is not None):
        parser.error("give either NAME or --real")
    if args.real and args.draws not in (None, 1):
        parser.error("the measured clouds have a single draw")

    dataset = PreprocessedRadarGenDataset(args.data_root, "val", keyframes_only=True)
    indices = select_frames(dataset, args.frames, args.selection_seed)
    ids = frame_ids(dataset)
    if args.real:
        raw, name, store, draws = None, Path("real_radar"), None, 1
    else:
        raw, name = resolve(args.name, EVAL_RAW)
        store = RawStore(raw)
        draws = store.draws([ids[i] for i in indices])
        if not draws or (args.draws or 0) > draws:
            raise SystemExit(f"{raw} has {draws} draws for all selected frames; run scripts/generate.py")
        draws = args.draws or draws
    selection = ("val_keyframes"
                 + ("" if args.frames is None else f"_n{args.frames}_sel{args.selection_seed}")
                 + f"_draws{draws}")
    output = EVAL_METRICS / name / selection
    stem = "voxelnext" + ("" if args.height == 1.0 else f"_z{args.height:g}")
    if (output / f"{stem}.csv").exists():
        raise SystemExit(f"{output / stem}.csv exists; evaluations are never overwritten")
    print(f"{selection}: {len(indices)} of {len(dataset)} frames from {raw or args.data_root} -> {output / stem}.csv",
          flush=True)

    from truckscenes import TruckScenes

    trucksc = TruckScenes(version=args.version, dataroot=str(args.truckscenes_root), verbose=False)
    detector = Detector(args.detector)
    print(f"VoxelNeXt checkpoint sha256 {detector.checkpoint_sha256}", flush=True)
    detections = [{} for _ in range(draws)]  # draw -> {sample_token: (sensor_token, boxes, scores, labels)}
    for index in tqdm(indices, desc=f"detect {draws}x", unit="frame", **PROGRESS):
        sample_token, sensor_token, detector_from_flat, measured = frame_inputs(dataset.files[index], trucksc)
        if sample_token != dataset.frames.sample_token.iloc[index]:
            raise ValueError(f"{dataset.files[index]}: sample token differs from the dataset index")
        for draw in range(draws):
            points = measured if store is None else store.read(*ids[index], f"{draw}/points")[0]
            points = detector_points(points, detector_from_flat, args.height)
            detections[draw][sample_token] = (sensor_token, *detector(points, point_seed(args.seed, sample_token)))
    if store is not None:
        store.close()

    results = []
    for draw, frames in enumerate(detections):
        print(f"scoring draw {draw}", flush=True)
        boxes = {token: global_boxes(trucksc, token, *frame) for token, frame in frames.items()}
        results.append(score(trucksc, boxes, json.loads((args.detector / "evaluate_detection_config.json").read_text())))
    table = pd.DataFrame([{"draw": d, **score_row(r)} for d, r in enumerate(results)])
    mean, std = table[list(COLUMNS)].mean(), table[list(COLUMNS)].std(ddof=0)

    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(output / f"{stem}_detections.h5", "x") as f:
        f.attrs.update(boxes_columns="x,y,z,dx,dy,dz,yaw,vx,vy", frame=SENSOR, labels="1=car,2=truck,3=trailer",
                       checkpoint_sha256=detector.checkpoint_sha256)
        for draw, frames in enumerate(detections):
            for token, (sensor_token, boxes, scores, labels) in frames.items():
                group = f.create_group(f"{draw}/{token}")
                group.attrs["sensor_token"] = sensor_token
                for key, value in dict(boxes=boxes, scores=scores, labels=labels).items():
                    group.create_dataset(key, data=value)
    table.to_csv(output / f"{stem}.csv", index=False)
    print(table[["draw", *COLUMNS]].to_string(index=False, float_format="{:.4f}".format))
    print("mean +- std over draws: " + ", ".join(f"{k} {mean[k]:.4f} +- {std[k]:.4f}" for k in COLUMNS))
    print(f"wrote {output / stem}.csv")


if __name__ == "__main__":
    main()
