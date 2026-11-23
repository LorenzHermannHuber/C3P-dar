"""Evaluation tables as CSV: frames.csv, boxes.csv and summary.csv per model and frame selection.

Every row starts with the identity columns (name, kind, model, step, selection, seed, nll_floor), frames
and boxes add (index, scene_token, frame_index, sample_token, is_key_frame, draw) and boxes the
annotation_token, so the tables of different models concatenate and join directly.
"""

import argparse
from pathlib import Path

import pandas as pd

from .. import EVAL_METRICS

TABLES = ("frames", "boxes", "summary")


def write_tables(directory, identity, frames, boxes, summary):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / f"{name}.csv").exists() for name in TABLES):
        raise FileExistsError(f"{directory} already holds evaluation tables")
    for name, table in zip(TABLES, (frames, boxes, pd.DataFrame([summary]))):
        table = table.copy()
        for position, (key, value) in enumerate(identity.items()):
            table.insert(position, key, value)
        table.to_csv(directory / f"{name}.csv", index=False)


def collect(root=EVAL_METRICS, table="summary"):
    """All <table>.csv below ``root`` as one DataFrame."""
    paths = sorted(Path(root).rglob(f"{table}.csv"))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Concatenate the evaluation tables below a directory.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=EVAL_METRICS)
    parser.add_argument("--table", choices=TABLES, default="summary")
    args = parser.parse_args()
    collect(args.root, args.table).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
