"""Score the raw outputs of a model against the ground truth (scripts/generate.py writes them).

python scripts/evaluate.py RUN/step_20000                         # all val keyframes, all draws
python scripts/evaluate.py RUN --frames 50 --draws 3              # RUN with a single evaluated checkpoint

NAME is a directory below EVAL_RAW or a path. The evaluation dispatches on the stored kind (our "ppp"
fields or RadarGen's "radargen" maps). Tables go to EVAL_METRICS/<name>/<selection>/{frames,boxes,summary}.csv
and are never overwritten.
"""

import argparse
from pathlib import Path

import torch

from c3pdar import DATA_ROOT, EVAL_METRICS, EVAL_RAW
from c3pdar.data import PreprocessedRadarGenDataset
from c3pdar.evaluation.metrics import headline
from c3pdar.evaluation.results import write_tables
from c3pdar.evaluation.store import RawStore, resolve
from c3pdar.evaluation.validation import evaluate, frame_ids, select_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sweeps", action="store_true", help="all frames, not only the annotated keyframes")
    parser.add_argument("--frames", type=int, help="number of frames (default: all)")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--draws", type=int, help="point clouds per frame (default: all generated)")
    parser.add_argument("--nll-floor", type=float, default=1e-3, help="uniform share of the density in ppp_nll")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (all cores stall on a busy node)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    raw, name = resolve(args.name, EVAL_RAW)
    dataset = PreprocessedRadarGenDataset(args.data_root, args.split, keyframes_only=not args.sweeps)
    indices = select_frames(dataset, args.frames, args.selection_seed)
    with RawStore(raw) as store:
        ids = frame_ids(dataset)
        draws = args.draws or store.draws([ids[i] for i in indices])
        if not draws:
            raise SystemExit(f"{raw} has no draws for all selected frames; run scripts/generate.py")
        selection = (f"{args.split}_{'sweeps' if args.sweeps else 'keyframes'}"
                     + ("" if args.frames is None else f"_n{args.frames}_sel{args.selection_seed}")
                     + f"_draws{draws}")
        output = EVAL_METRICS / name / selection
        if (output / "summary.csv").exists():
            raise SystemExit(f"{output} exists; evaluations are never overwritten")
        print(f"{selection}: {len(indices)} of {len(dataset)} frames from {raw} -> {output}")
        frames, boxes, summary = evaluate(store, dataset, indices, draws, device=torch.device(args.device),
                                          nll_floor=args.nll_floor)
        attrs = store.attrs
    identity = dict(name=str(name), kind=attrs["kind"], model=attrs["model"], step=int(attrs.get("step", -1)),
                    selection=selection, seed=int(attrs["seed"]), nll_floor=args.nll_floor)
    write_tables(output, identity, frames, boxes, summary)
    print(headline(summary).to_string(float_format="{:.4g}".format))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
