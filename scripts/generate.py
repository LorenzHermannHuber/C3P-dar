"""Generate the raw evaluation outputs of a checkpoint (GPU): predicted fields and drawn point clouds.

python scripts/generate.py RUN/checkpoints/step_20000.ckpt                         # all val keyframes, 1 draw
python scripts/generate.py RUN/checkpoints/step_20000.ckpt --frames 50 --draws 3   # scene-balanced subset
python scripts/generate.py radargen --frames 50 --batch-size 8                     # the RadarGen baseline

The checkpoint is a path or relative to RUNS_ROOT; "radargen" is the original RadarGen at RADARGEN_CHECKPOINT,
where every draw is a diffusion sample. Outputs go to EVAL_RAW/<run>/<checkpoint>/<scene_token>.h5
(EVAL_RAW/radargen for RadarGen; c3pdar.evaluation.store). Existing frames and draws are kept, so the command
resumes after an interruption and extends a selection, the sweeps or the number of draws. Score them with
scripts/evaluate.py.
"""

import argparse
from pathlib import Path

import torch

from c3pdar import DATA_ROOT, EVAL_RAW, RADARGEN_CHECKPOINT, RUNS_ROOT
from c3pdar.data import PreprocessedRadarGenDataset
from c3pdar.evaluation.generation import generate_ppp, generate_radargen
from c3pdar.evaluation.store import RawStore
from c3pdar.evaluation.validation import select_frames
from c3pdar.models import RadarGen, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help='run checkpoint or "radargen"')
    parser.add_argument("--name", help="output directory below EVAL_RAW (default: <run>/<checkpoint> or radargen)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sweeps", action="store_true", help="all frames, not only the annotated keyframes")
    parser.add_argument("--frames", type=int, help="number of frames (default: all)")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=1, help="point clouds per frame")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (all cores stall on a busy node)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    dataset = PreprocessedRadarGenDataset(args.data_root, args.split, keyframes_only=not args.sweeps)
    if str(args.checkpoint) == "radargen":
        name = args.name or "radargen"
        model, generate = RadarGen.from_checkpoint(device=device), generate_radargen
        attrs = dict(kind="radargen", model=str(RADARGEN_CHECKPOINT), step=-1)
    else:
        checkpoint = (args.checkpoint if args.checkpoint.exists() else RUNS_ROOT / args.checkpoint).resolve()
        name = args.name or f"{checkpoint.parent.parent.name}/{checkpoint.stem}"
        model, config, step = load_model(checkpoint, device=device)
        generate = generate_ppp
        attrs = dict(kind="ppp", model=str(checkpoint), step=-1 if step is None else step,
                     mark_link=config["loss"]["mark_link"])
    indices = select_frames(dataset, args.frames, args.selection_seed)
    print(f"{args.split} {'sweeps' if args.sweeps else 'keyframes'}: {len(indices)} of {len(dataset)} frames, "
          f"{dataset.frames.scene_token.iloc[indices].nunique()} scenes, {args.draws} draws -> {EVAL_RAW / name}")
    with RawStore(EVAL_RAW / name, "a", **attrs, seed=args.seed, point_limit=dataset.point_limit) as store:
        predicted, drawn = generate(model, dataset, indices, store, args.draws, args.seed, device,
                                    args.batch_size, args.num_workers)
    print(f"{len(indices)} frames: predicted {predicted}, drew {drawn} clouds into {EVAL_RAW / name}")


if __name__ == "__main__":
    main()
