"""Per-frame inference times of models at batch size 1 (GPU), to choose the warm-up period afterwards.

python scripts/timing.py dino dit resnet50 radargen                 # the same 110 val keyframes for each
python scripts/timing.py RUN/checkpoints/step_20000.ckpt --frames 50

A model is a run with a single checkpoint, a checkpoint (a path or relative to RUNS_ROOT) or "radargen".
Models run one after another in one process; each predicts the selected frames one at a time, in the same
order, and draws one cloud (c3pdar.evaluation.timing). Nothing is discarded as warm-up: row ``position`` 0
is the first prediction of the freshly loaded model. Writes EVAL_METRICS/<name>/<selection>/timing.csv;
models that already have it are skipped unless --recompute is given.
"""

import argparse
import socket
from pathlib import Path

import torch

from c3pdar import DATA_ROOT, EVAL_METRICS, RUNS_ROOT
from c3pdar.data import PreprocessedRadarGenDataset
from c3pdar.evaluation.timing import time_ppp, time_radargen
from c3pdar.evaluation.validation import select_frames
from c3pdar.models import RadarGen, load_model


def locate(model):
    """Metric table name and checkpoint (None for RadarGen) of a model argument."""
    if model == "radargen":
        return "radargen", None
    path = Path(model) if Path(model).exists() else RUNS_ROOT / model
    if path.is_dir():
        checkpoints = sorted((path / "checkpoints").glob("*.ckpt"))
        if len(checkpoints) != 1:
            raise SystemExit(f"{path} holds {len(checkpoints)} checkpoints, pick one")
        path = checkpoints[0]
    if not path.is_file():
        raise SystemExit(f"no checkpoint {model}")
    path = path.resolve()
    return f"{path.parent.parent.name}/{path.stem}", path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help='runs, checkpoints or "radargen"')
    parser.add_argument("--split", default="val")
    parser.add_argument("--sweeps", action="store_true", help="all frames, not only the annotated keyframes")
    parser.add_argument("--frames", type=int, default=110, help="number of frames")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--warmup", type=int, default=10, help="frames left out of the printed mean only")
    parser.add_argument("--recompute", action="store_true", help="overwrite existing timing.csv files")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (all cores stall on a busy node)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    dataset = PreprocessedRadarGenDataset(args.data_root, args.split, keyframes_only=not args.sweeps)
    indices = select_frames(dataset, args.frames, args.selection_seed)
    selection = f"{args.split}_{'sweeps' if args.sweeps else 'keyframes'}_n{len(indices)}_sel{args.selection_seed}"
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"{selection}: {len(indices)} frames, batch size 1 on {hardware} ({socket.gethostname()})")
    for name, checkpoint in [locate(model) for model in args.models]:
        output = EVAL_METRICS / name / selection / "timing.csv"
        if output.exists() and not args.recompute:
            print(f"WARNING {name}: {output} exists, skipped; run with --recompute if the model changed")
            continue
        if checkpoint is None:
            network = RadarGen.from_checkpoint(device=device)
            times = time_radargen(network, dataset, indices, args.seed, device, args.num_workers, desc=name)
        else:
            network, config, _ = load_model(checkpoint, device=device)
            times = time_ppp(network, dataset, indices, config["loss"]["mark_link"], args.seed, device,
                             args.num_workers, desc=name)
        del network
        if device.type == "cuda":
            torch.cuda.empty_cache()
        times.insert(0, "position", range(len(times)))
        times.insert(0, "name", name)
        times["device"], times["host"] = hardware, socket.gethostname()
        output.parent.mkdir(parents=True, exist_ok=True)
        times.to_csv(output, index=False)
        head, rest = times.seconds[:args.warmup], times.seconds[args.warmup:]
        print(f"{name}: first {len(head)} frames {head.mean():.3f} s, then {rest.mean():.3f} +- {rest.std():.3f} s "
              f"(predict {times.predict_seconds[args.warmup:].mean():.3f} s) -> {output}")


if __name__ == "__main__":
    main()
