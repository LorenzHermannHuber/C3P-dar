r"""LaTeX table of evaluation metrics, defined by a YAML config.

python visualisation/make_table.py visualisation/configs/table_core.yaml -o visualisation/plots/table_core.tex
python visualisation/make_table.py visualisation/configs/table_count.yaml -o visualisation/plots/table_count.tex


Config (see visualisation/configs/table_core.yaml):

    selection: val_keyframes_draws3          # default frame selection of the rows
    columns:                                 # column groups; a non-empty group gets a spanning header
      - group: Position
        columns:
          - {label: CD, unit: m, metric: summary/chamfer_distance_xy_mean, format: .3f, best: min}
    rows:                                    # row groups, separated by \midrule
      - group: ours                          # only a comment in the output
        rows:
          - {label: C3P-dar, run: dit/epoch_44_step_58057}
          - {label: Published, values: {summary/chamfer_distance_xy_mean: 1.68*}, rank: false}

A metric <table>/<column> is the mean of <column> over the rows of EVAL_METRICS/<run>/<selection>/<table>.csv
(summary.csv of scripts/evaluate.py has one row, voxelnext.csv of scripts/detect.py one row per draw);
<table>/<column>:std is the sample std over these rows instead. <table>/<column>?<key>=<value>&... only uses the
rows with these values, e.g. budgets/mean?budget=native&score=position_f1 of the long budgets.csv; <, <=, >, >=
compare numbers, e.g. timing/seconds?position>=10 drops the warm-up of timing.csv (scripts/timing.py).
A column's ``selection`` overrides the row's and the config's selection for that column.
``values`` sets cells directly: numbers are formatted and ranked, strings are printed as they are.
``best`` (min | max | closest) adds an arrow to the label and marks the best number of a column among the rows
with ``rank`` (default true); ``closest`` ranks by distance to the column's ``target``. Strings and unranked
rows never set the best or worst value. A column with ``rank: false`` keeps the arrow of its ``best`` but is
neither marked nor shaded.
A column's ``std`` metric is printed as value $\pm$ std for rows with a run, unless the config disables it.
A column's ``scale`` multiplies the metrics and stds of runs (e.g. 100 for percent), not ``values``:

    stds: true                           # false hides all stds
    units: displayed                     # none | displayed (label [unit]) | newrow (header row of units)
    style:
      best: underline                    # bold | underline | none (default bold)
      gradient: ['#808080', '#000000']   # optional: color worst -> best ranked number, linear in value;
                                         # other cells (also numbers in strings) are clamped to that range

The gradient needs \usepackage{xcolor}. Unavailable cells print as ``missing`` (default --).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from c3pdar import EVAL_METRICS


class Metrics:
    """Metric means of the evaluation tables; each table is read once."""

    def __init__(self, root=EVAL_METRICS):
        self.root, self.tables = Path(root), {}

    def __call__(self, run, selection, metric):
        table, column = metric.split("/", 1)
        column, _, reduction = column.partition(":")
        column, _, query = column.partition("?")
        path = self.root / run / selection / f"{table}.csv"
        if path not in self.tables:
            self.tables[path] = pd.read_csv(path) if path.exists() else None
            if self.tables[path] is None:
                print(f"missing {path}", file=sys.stderr)
        frame = self.tables[path]
        if frame is None or column not in frame:
            return None
        for condition in filter(None, query.split("&")):
            key, operator, value = CONDITION.fullmatch(condition).groups()
            if operator == "=":
                frame = frame[frame[key].astype(str) == value]
            else:
                frame = frame[COMPARE[operator](frame[key], float(value))]
        if frame.empty:
            return None
        return float(frame[column].std() if reduction == "std" else frame[column].mean())


NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)")
CONDITION = re.compile(r"(.+?)(<=|>=|<|>|=)(.*)")
COMPARE = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


def cells(config, metrics):
    """Row groups of (row config, [cell text or None], [number or None], [ranked]) per column."""
    columns = [column for group in config["columns"] for column in group["columns"]]
    groups = []
    for group in config["rows"]:
        rows = []
        for row in group["rows"]:
            texts, numbers, ranked = [], [], []
            for column in columns:
                selection = column.get("selection", row.get("selection", config.get("selection")))
                value = row.get("values", {}).get(column["metric"])
                if value is None and "run" in row:
                    value = metrics(row["run"], selection, column["metric"])
                    value = None if value is None else value * column.get("scale", 1)
                if isinstance(value, str):
                    match = NUMBER.search(value)
                    texts.append(value)
                    numbers.append(float(match.group()) if match else None)
                    ranked.append(False)
                elif value is None or not np.isfinite(value):
                    texts.append(None)
                    numbers.append(None)
                    ranked.append(False)
                else:
                    texts.append(format(value, column.get("format", ".3f")))
                    numbers.append(float(texts[-1]))  # rank what is shown, so displayed ties are all marked
                    ranked.append(row.get("rank", True))
                    std = None
                    if config.get("stds", True) and "std" in column and "run" in row:
                        std = metrics(row["run"], selection, column["std"])
                    if std is not None and np.isfinite(std):
                        texts[-1] += rf" $\pm$ {format(std * column.get('scale', 1), column.get('format', '.3f'))}"
            rows.append((row, texts, numbers, ranked))
        groups.append((group, rows))
    return columns, groups


EMPHASIS = {"bold": r"\textbf{{{}}}", "underline": r"\underline{{{}}}", "none": "{}"}


def _mix(worst, best, fraction):
    """Hex color (without #) at ``fraction`` of the way from worst to best."""
    worst, best = (np.array([int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]) for c in (worst, best))
    return "".join(f"{round(w + fraction * (b - w)):02X}" for w, b in zip(worst, best))


def highlight(columns, groups, style):
    emphasis = EMPHASIS[style.get("best", "bold")]
    gradient = style.get("gradient")
    for index, column in enumerate(columns):
        best = column.get("best")
        if best not in ("min", "max", "closest") or not column.get("rank", True):
            continue
        if best == "closest" and not np.isfinite(column.get("target", np.nan)):
            raise ValueError("best: closest requires a numeric target")
        rows = [r for _, rows in groups for r in rows if r[2][index] is not None]
        score = lambda value: abs(value - column["target"]) if best == "closest" else value
        values = [score(r[2][index]) for r in rows if r[3][index]]
        if not values:
            continue
        target, worst = (max(values), min(values)) if best == "max" else (min(values), max(values))
        for row in rows:
            value, text = row[2][index], row[1][index]
            value = score(value)
            if gradient:
                fraction = 1.0 if target == worst else min(max((value - worst) / (target - worst), 0.0), 1.0)
                text = rf"\textcolor[HTML]{{{_mix(*gradient, fraction)}}}{{{text}}}"
            row[1][index] = emphasis.format(text) if row[3][index] and value == target else text


def header(column, units):
    label = column["label"]
    if units == "displayed" and column.get("unit"):
        label += f" [{column['unit']}]"
    return label + {"min": r" $\downarrow$", "max": r" $\uparrow$"}.get(column.get("best"), "")


def latex(config, metrics):
    columns, groups = cells(config, metrics)
    missing = config.get("missing", "--")
    highlight(columns, groups, config.get("style", {}))

    environment = config.get("environment", "table*")
    lines = [rf"\begin{{{environment}}}[{config.get('placement', 't')}]", r"\centering"] if environment else []
    if environment and config.get("caption"):
        lines.append(rf"\caption{{{config['caption'].strip()}}}")
    if environment and config.get("label"):
        lines.append(rf"\label{{{config['label']}}}")
    lines += [r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{l{'c' * len(columns)}}}", r"\toprule"]
    spans, rules, start = [""], [], 2
    for group in config["columns"]:
        width = len(group["columns"])
        if group.get("group"):
            spans.append(rf"\multicolumn{{{width}}}{{c}}{{{group['group']}}}")
            rules.append(rf"\cmidrule(lr){{{start}-{start + width - 1}}}")
        else:
            spans.extend([""] * width)
        start += width
    lines += [" & ".join(spans) + r" \\", " ".join(rules)]
    units = config.get("units", "displayed")
    if units not in ("none", "displayed", "newrow"):
        raise ValueError(f"units must be none, displayed or newrow, not {units!r}")
    lines.append(" & ".join([config.get("corner", "Model"), *(header(c, units) for c in columns)]) + r" \\")
    if units == "newrow":
        # compensating for half the upwards or downwards pointing arrows
        lines.append(" & ".join(["", *("\\hspace{-5pt}"+f"[{c['unit']}]" if c.get("unit") else "" for c in columns)]) + r" \\")
    for group, rows in groups:
        lines.append(r"\midrule")
        if group.get("group"):
            lines.append(f"% {group['group']}")
        for row, texts, *_ in rows:
            lines.append(" & ".join([row["label"], *(missing if t is None else t for t in texts)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if not environment:  # caption and label belong to the surrounding float
        return "\n".join(lines) + "\n"
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="write the table here instead of printing it")
    parser.add_argument("--root", type=Path, default=EVAL_METRICS, help="evaluation outputs")
    args = parser.parse_args()
    table = latex(yaml.safe_load(args.config.read_text()), Metrics(args.root))
    print()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)
        print(f"wrote {args.output}")
    else:
        print("\n"+table, end="\n\n")


if __name__ == "__main__":
    main()
