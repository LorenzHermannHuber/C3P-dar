r"""Precision-recall trade-off over the point budget per frame (Pareto fronts), defined by a YAML config.

python visualisation/plot_tradeoff.py visualisation/configs/tradeoff.yaml

Reads EVAL_METRICS/<run>/<selection>/budgets.csv of scripts/budgets.py. Per run, the budgets form a curve, with
error bars of ``interval`` scene-clustered standard errors if set; budgets feasible for less than ``coverage``
of the frame draws (RadarGen cannot exceed its deconvolved cells) are drawn hollow and dashed, since their mean
covers a different, denser subset of frames. Every run shares the ``markers`` of its own cloud (native) and
of the budget b = GT count per frame (exact). Dashed gray lines are iso-F1 curves, labelled on the right axis.

Without ``grid``, the panels sit side by side and the figure height follows ``ratio``. With
``grid: {rows, cols, widths, heights}`` (relative widths/heights), each panel names its cell by ``row``, ``col``
and optionally ``rowspan``/``colspan``; the figure is ``width`` x ``height`` and panels with aspect equal widen
their free limits to their cell. ``iso_f1``, ``f1_axis``, ``labelled_budgets``, ``ticks``, ``xlim``, ``ylim`` and
``aspect`` can be set per panel; ``ticks`` takes {major, minor} for both axes or per axis {x: {...}, y: {...}}.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # noqa: E402

from c3pdar import EVAL_METRICS  # noqa: E402


def load(run, selection, table="budgets", root=EVAL_METRICS):
    """Wide table: index budget, columns (stat, score) with stat in mean, se, n, total; None if missing."""
    path = Path(root) / run / selection / f"{table}.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return None
    return pd.read_csv(path).pivot(index="budget", columns="score", values=["mean", "se", "n", "total"])


def draw_iso_f1(ax, axis, values):
    """Iso-F1 lines within the current limits, labelled on the secondary ``axis`` (if any); returns the lines."""
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    lines, ticks, labels = [], [], []
    for f1 in values:
        # P = f1 R / (2R - f1), starting where it enters through the top edge
        recall = np.linspace(max(left, f1 * top / (2 * top - f1)), right, 400)
        precision = f1 * recall / (2 * recall - f1)
        lines += ax.plot(recall, precision, color="0.6", lw=0.9, ls="--", zorder=0)
        if bottom <= precision[-1] <= top:
            ticks.append(precision[-1])
            labels.append(f"{f1:g}")
    if axis is not None:
        axis.set_yticks(ticks, labels=labels)
    return lines


MARKERS = {"native": dict(marker="o", size=9, label="own count"),
           "exact": dict(marker="*", size=13, label="GT count")}


def densify(xy, steps=20):
    """(N,2) polyline -> points along its segments, so labels also avoid the lines between the budgets."""
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 2:
        return xy
    t = np.linspace(0, 1, steps, endpoint=False)[:, None, None]
    return np.vstack([(xy[:-1] + t * (xy[1:] - xy[:-1])).reshape(-1, 2), xy[-1:]])


def budget_labels(ax, labels, obstacles, clearance):
    """Annotate (text, xy, color, run) only if its box, padded by ``clearance`` points, holds no placed label
    and no obstacle (run, (N,2) data points) of another run (run None: of every run); returns the annotations."""
    fig = ax.figure
    fig.canvas.draw()  # final layout, so the display coordinates hold
    renderer, pad = fig.canvas.get_renderer(), clearance * fig.dpi / 72
    obstacles = [(run, ax.transData.transform(xy)) for run, xy in obstacles if len(xy)]
    placed, annotations = [], []
    for text, xy, color, run in labels:
        label = ax.annotate(text, xy, textcoords="offset points", xytext=(-2, 5), fontsize=8, color=color)
        box = label.get_window_extent(renderer).padded(pad)
        blocked = any(box.overlaps(other) for other in placed) or any(
            ((points >= box.p0) & (points <= box.p1)).all(1).any() for owner, points in obstacles if owner != run)
        if blocked:
            label.remove()
        else:
            placed.append(box)
            annotations.append(label)
    return annotations


def widen(limits, bounds, span):
    """Grow (low, high) to ``span``, moving only the bounds that are null in the config ``bounds``."""
    (low, high), extra = limits, span - (limits[1] - limits[0])
    free = [bounds is None or bound is None for bound in bounds or (None, None)]
    if all(free):
        return low - extra / 2, high + extra / 2
    return (low, high + extra) if free[1] else (low - extra, high) if free[0] else (low, high)


def frame(ax, points, xlim, ylim, ratio, equal, margin=0.05):
    """Set the limits to the (N,2) ``points`` plus ``margin``, except the configured bounds; with ``equal``,
    widen the free limits to the panel ``ratio``. Returns the ratio the panel needs."""
    low, high = points.min(0), points.max(0)
    low, high = low - margin * (high - low), high + margin * (high - low)
    (left, right), (bottom, top) = ([d if b is None else b for b, d in zip(bounds or (None, None), (lo, hi))]
                                    for bounds, lo, hi in zip((xlim, ylim), low, high))
    if equal:
        if (top - bottom) < ratio * (right - left):
            bottom, top = widen((bottom, top), ylim, ratio * (right - left))
        else:
            left, right = widen((left, right), xlim, (top - bottom) / ratio)
        ratio = (top - bottom) / (right - left)  # differs from the config only if both bounds are fixed
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    return ratio


def styles(runs, grayscale):
    """(color, budget marker) per run: ``color`` or a gray level (``gray`` overrides), and distinct markers."""
    grays, cycle = np.linspace(0, 0.6, len(runs)), "osD^vP"
    return [(str(run.get("gray", round(gray, 2))) if grayscale else run["color"],
             run.get("marker", cycle[i % len(cycle)])) for i, (run, gray) in enumerate(zip(runs, grays))]


def panel(ax, config, tables, spec, styles):
    """Draw the curves of one panel; returns its state for the shared limits and labels, and whether any
    budget is infeasible."""
    x, y = spec["x"], spec["y"]
    coverage, interval = config.get("coverage", 0.95), config.get("interval")
    markers = {k: {**v, **config.get("markers", {}).get(k, {})} for k, v in MARKERS.items()}
    labelled = set(spec.get("labelled_budgets", config.get("labelled_budgets", [])))
    labels, obstacles, infeasible = [], [], False
    for index, ((color, marker), table) in enumerate(zip(styles, tables)):
        if table is None:
            continue
        budgets = sorted((b for b in table.index if b.isdigit()), key=int)
        mean, se = table["mean"].loc[budgets], table["se"].loc[budgets]
        covered = (table["n"][x] / table["total"][x]).loc[budgets].to_numpy() >= coverage
        if interval:
            ax.errorbar(mean[x], mean[y], xerr=interval * se[x], yerr=interval * se[y], fmt="none", ecolor=color,
                        alpha=0.4, lw=0.8)
        ax.plot(mean[x][covered], mean[y][covered], "-", marker=marker, ms=3, lw=1.2, color=color)
        if not covered.all():
            infeasible = True
            start = np.flatnonzero(covered).max() if covered.any() else 0
            ax.plot(mean[x].iloc[start:], mean[y].iloc[start:], "--", lw=1.0, color=color)
            ax.plot(mean[x][~covered], mean[y][~covered], marker, ms=3, mfc="white", color=color)
        obstacles.append((index, densify(np.column_stack([mean[x], mean[y]]))))
        labels += [(budget, (mean[x][budget], mean[y][budget]), color, index) for budget in budgets
                   if int(budget) in labelled and np.isfinite(mean[x][budget])]
        for budget, style in markers.items():
            if budget in table.index:
                obstacles.append((None, np.array([[table["mean"][x][budget], table["mean"][y][budget]]])))
                ax.scatter(table["mean"][x][budget], table["mean"][y][budget], color=color, edgecolor="white",
                           linewidth=1.2, zorder=5, marker=style["marker"], s=style["size"] ** 2)
    xlim, ylim = spec.get("xlim", config.get("xlim")), spec.get("ylim", config.get("ylim"))
    equal = spec.get("aspect", config.get("aspect", "auto")) == "equal"
    points = np.vstack([xy for _, xy in obstacles])
    points = points[np.isfinite(points).all(1)]
    ticks = {**(config.get("ticks") or {}), **(spec.get("ticks") or {})}
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        own = {**ticks, **(ticks.get(name) or {})}
        if own.get("major"):
            axis.set_major_locator(MultipleLocator(own["major"]))
        axis.set_minor_locator(MultipleLocator(own["minor"]) if own.get("minor") else AutoMinorLocator())
    f1_axis = None
    if spec.get("f1_axis", config.get("f1_axis", True)):
        f1_axis = ax.secondary_yaxis("right")
        f1_axis.tick_params(labelsize=8, labelcolor="0.5", color="0.6", length=2.5)
        f1_axis.set_ylabel(spec.get("f1_label", "F1 ↑"), color="0.5", fontsize=9)
    ax.set_title(spec.get("title", ""), fontsize=10)
    ax.set_xlabel(spec.get("xlabel", x))
    ax.set_ylabel(spec.get("ylabel", y))
    ax.grid(alpha=0.25, which="major")
    ax.grid(alpha=0.15, which="minor")
    labels.sort(key=lambda label: (int(label[0]), label[3]))
    iso_f1 = spec.get("iso_f1", config.get("iso_f1", np.arange(0.1, 1.0, 0.1).round(1)))
    state = dict(ax=ax, f1_axis=f1_axis, labels=labels, obstacles=obstacles, points=points, xlim=xlim, ylim=ylim,
                 equal=equal, iso_f1=iso_f1, lines=[], annotations=[])
    return state, infeasible


def legend(fig, clusters, rows, **kwargs):
    """Legend below the figure, its clusters side by side in ``rows`` rows, separated by an empty column."""
    def blanks(count):
        return [Line2D([], [], ls="none", label=" ") for _ in range(count)]
    handles = []
    for index, cluster in enumerate(clusters):
        handles += (blanks(rows) if index else []) + cluster + blanks(-len(cluster) % rows)
    # matplotlib fills the legend column by column
    fig.legend(handles=handles, ncol=len(handles) // rows, loc="outside lower center", frameon=False, **kwargs)


def axes(fig, config):
    """One axes per panel: side by side, or in the cells of ``grid``."""
    grid, specs = config.get("grid"), config["panels"]
    if grid is None:
        return fig.subplots(1, len(specs), squeeze=False)[0]
    cells = fig.add_gridspec(grid.get("rows", 1), grid.get("cols", 1), width_ratios=grid.get("widths"),
                             height_ratios=grid.get("heights"))
    return [fig.add_subplot(cells[spec["row"]:spec["row"] + spec.get("rowspan", 1),
                                  spec["col"]:spec["col"] + spec.get("colspan", 1)]) for spec in specs]


def box_ratio(ax):
    """Height / width of the axes box on the page, as of the last draw."""
    box = ax.get_position()
    return box.height * ax.figure.get_figheight() / (box.width * ax.figure.get_figwidth())


def figure(config, tables, grayscale=False):
    """All panels side by side or in a grid, with the shared title and legend."""
    specs, width, grid = config["panels"], config.get("width", 10), "grid" in config
    fig = plt.figure(figsize=(width, config.get("height", 0.75 * width) if grid else width / len(specs)),
                     layout="constrained")
    run_styles = styles(config["runs"], grayscale)
    states, infeasible = zip(*(panel(ax, config, tables, spec, run_styles)
                               for ax, spec in zip(axes(fig, config), specs)))
    if "title" in config:
        fig.suptitle(config["title"])
    markers = {k: {**v, **config.get("markers", {}).get(k, {})} for k, v in MARKERS.items()}
    coverage = config.get("coverage", 0.95)
    runs = [Line2D([], [], color=color, lw=1.5, marker=marker, ms=4, label=run["label"])
            for run, (color, marker), table in zip(config["runs"], run_styles, tables) if table is not None]
    keys = [Line2D([], [], color="0.4", marker="o", ms=3, lw=1.2,
                   label=config.get("budget_label", "budget b: points per frame"))]
    keys += [Line2D([], [], color="0.4", marker=m["marker"], ms=m["size"], mec="white", ls="none", label=m["label"])
             for m in markers.values()]
    if any(infeasible):
        keys.append(Line2D([], [], color="0.4", marker="o", ms=4, mfc="white", ls="--",
                           label=f"b feasible in < {coverage:.0%} of frames"))
    legend(fig, [runs, keys], config.get("legend_rows", 3), fontsize=9, columnspacing=1.0)
    height, base = map(float, str(config.get("ratio", "3:4")).split(":"))
    for _ in range(6):  # the limits must also hold the budget labels, whose data size depends on the limits
        if grid:
            fig.canvas.draw()
        for state in states:
            for artist in state["lines"] + state["annotations"]:
                artist.remove()
            state["ratio"] = frame(state["ax"], state["points"], state["xlim"], state["ylim"],
                                   box_ratio(state["ax"]) if grid else height / base, state["equal"])
            state["lines"] = draw_iso_f1(state["ax"], state["f1_axis"], state["iso_f1"])
        for _ in range(4 * (not grid)):  # fit the figure height to the panel ratio
            fig.canvas.draw()
            slots = [state["ax"].get_position() for state in states]
            needed = max(slot.width * width * state["ratio"] for slot, state in zip(slots, states))
            fig.set_figheight(fig.get_figheight() + needed - slots[0].height * fig.get_figheight())
        done = True
        for state in states:
            ax = state["ax"]
            state["annotations"] = budget_labels(ax, state["labels"], state["obstacles"],
                                                 config.get("label_clearance", 3))
            renderer, to_data = fig.canvas.get_renderer(), ax.transData.inverted()
            corners = np.vstack([to_data.transform(a.get_window_extent(renderer).get_points())
                                 for a in state["annotations"]] or [np.empty((0, 2))])
            (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
            if not ((corners >= (left, bottom)) & (corners <= (right, top))).all():
                state["points"], done = np.vstack([state["points"], corners]), False
        if grid:  # new tick labels can move the cells
            fig.canvas.draw()
            done &= all(abs(box_ratio(s["ax"]) / s["ratio"] - 1) < 0.01 for s in states if s["equal"])
        if done:
            break
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    tables = [load(run["run"], run.get("selection", config["selection"]), config.get("table", "budgets"))
              for run in config["runs"]]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.labelsize": 11, "axes.titlesize": 12})
    name = config.get("name", "tradeoff")
    for suffix, grayscale in [("", False)] + [("_bw", True)] * bool(config.get("grayscale", True)):
        fig = figure(config, tables, grayscale)
        for extension in config.get("formats", ["pdf", "svg", "jpg"]):
            fig.savefig(str(output) + f"/{name}{suffix}.{extension}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {output / name}{suffix}.*")


if __name__ == "__main__":
    main()
