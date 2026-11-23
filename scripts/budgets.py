"""Precision/recall of each model at fixed point budgets per frame (c3pdar.evaluation.budgets).

python scripts/budgets.py dino                                     # all val keyframes, all draws
python scripts/budgets.py radargen --budgets 500 1000 --frames 20  # RadarGen needs a GPU (deconvolution)

Writes EVAL_METRICS/<name>/<selection>/budget_frames<_tag>.csv (frame x draw x budget) and budgets<_tag>.csv (per
budget and score: mean, scene-clustered standard error, n feasible of total); never overwritten.
"""

import argparse
import os
from pathlib import Path

# Workers are single-threaded; per-core BLAS pools in every worker exceed the login nodes' thread limit.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import torch  # noqa: E402

from c3pdar import DATA_ROOT, EVAL_METRICS, EVAL_RAW
from c3pdar.data import PreprocessedRadarGenDataset
from c3pdar.evaluation.budgets import evaluate_budgets, summarize_budgets
from c3pdar.evaluation.store import RawStore, resolve
from c3pdar.evaluation.validation import frame_ids, select_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(range(100, 2001, 100)))
    parser.add_argument("--split", default="val")
    parser.add_argument("--sweeps", action="store_true", help="all frames, not only the annotated keyframes")
    parser.add_argument("--frames", type=int, help="number of frames (default: all)")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--draws", type=int, help="point clouds per frame (default: all generated)")
    parser.add_argument("--workers", type=int, default=16, help="scoring processes (shared login node)")
    parser.add_argument("--tag", default="", help="suffix of the output tables, to keep earlier ones")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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
    suffix = f"_{args.tag}" if args.tag else ""
    frames_path, summary_path = output / f"budget_frames{suffix}.csv", output / f"budgets{suffix}.csv"
    if frames_path.exists() or summary_path.exists():
        raise SystemExit(f"{output} holds budget tables{suffix}; evaluations are never overwritten")
    print(f"{selection}: {len(indices)} frames, budgets {args.budgets} from {raw} -> {output}", flush=True)
    frames = evaluate_budgets(raw, dataset, indices, draws, args.budgets, device=torch.device(args.device),
                              workers=args.workers)
    summary = summarize_budgets(frames)
    output.mkdir(parents=True, exist_ok=True)
    frames.insert(0, "name", str(name))
    summary.insert(0, "name", str(name))
    frames.to_csv(frames_path, index=False)
    summary.to_csv(summary_path, index=False)
    table = summary[summary.score.isin(["count", "position_f1", "distance_attr_f1", "pcl_mmd_rcs", "box_hit_rate"])]
    print(table.pivot(index="budget", columns="score", values="mean").to_string(float_format="{:.4g}".format))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
