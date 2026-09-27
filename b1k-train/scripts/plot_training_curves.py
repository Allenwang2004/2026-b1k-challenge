"""Plot a training run's curves from its log: python scripts/plot_training_curves.py <log> [-o out.png]

train.py writes one line per log_interval steps ("Step 400: action_loss=..., subtask_accuracy=..."),
which is the only record of a run once the progress bar has scrolled away. This turns that into the
six panels worth looking at.

Each measure gets its own axes -- never two y-scales on one plot, which is the standard way to make
two unrelated curves look related. subtask_loss and grad_norm span three orders of magnitude, so
they are drawn on a log scale, labelled as such.
"""

import argparse
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

LOSS = "#2a78d6"
ACC = "#1baf7a"
INK, INK_2, INK_3 = "#16171a", "#55565b", "#83848a"
SURFACE, GRID, LINE = "#fcfcfb", "#ececE7", "#e2e2dd"

# (key, title, colour, scale, formatter, y-from-zero)
# The system has no CJK font, so titles are English -- which is also what the metrics are called.
# Only a rate that genuinely starts at zero gets a 0-100% axis; pinning fast_accuracy (which never
# leaves 0.77-0.84) to the full range would flatten its curve into a straight line.
PANELS = [
    ("subtask_accuracy", "Stage prediction accuracy", ACC, "linear", "pct", True),
    ("action_loss", "Action loss (flow matching)", LOSS, "linear", "num", False),
    ("subtask_loss", "Stage loss (cross-entropy)", LOSS, "log", "num", False),
    ("fast_accuracy", "FAST token accuracy", ACC, "linear", "pct", False),
    ("total_loss", "Total loss", LOSS, "linear", "num", False),
    ("grad_norm", "Gradient norm", LOSS, "log", "num", False),
]


def parse(path: pathlib.Path) -> dict[str, list]:
    keys = [k for k, *_ in PANELS]
    out: dict[str, list] = {k: [] for k in keys}
    out["step"] = []
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"Step (\d+): (.*)", line)
        if not m:
            continue
        step, body = int(m.group(1)), m.group(2)
        found = {}
        for k in keys:
            # (?<![a-z_]) so action_loss does not also match action_loss_base_vel_x
            mm = re.search(rf"(?<![a-z_]){k}=(-?[\d.]+(?:e-?\d+)?)", body)
            if mm:
                found[k] = float(mm.group(1))
        if not found:
            continue
        out["step"].append(step)
        for k in keys:
            out[k].append(found.get(k))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("outputs/figures/training_curves.png"))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    d = parse(args.log)
    if not d["step"]:
        raise SystemExit(f"no 'Step N: ...' lines in {args.log}")
    steps = d["step"]

    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.edgecolor": LINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_3, "ytick.color": INK_3,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))
    title = args.title or args.log.stem
    fig.suptitle(title, x=0.008, ha="left", fontsize=13, fontweight="semibold", color=INK, y=0.985)
    fig.text(0.008, 0.945, f"{len(steps)} logged points  |  step {steps[0]}-{steps[-1]}", ha="left",
             fontsize=8.5, color=INK_3)

    for ax, (key, name, colour, scale, fmt, from_zero) in zip(axes.ravel(), PANELS):
        ys = d[key]
        pts = [(s, y) for s, y in zip(steps, ys) if y is not None]
        if not pts:
            ax.set_visible(False)
            continue
        xs, ys = zip(*pts)

        ax.plot(xs, ys, color=colour, linewidth=1.6, solid_capstyle="round", zorder=3)
        if scale == "linear":
            ax.fill_between(xs, ys, min(ys), color=colour, alpha=0.10, linewidth=0, zorder=2)
        ax.set_yscale(scale)

        # the endpoint is the number the reader came for: mark it and label it
        ax.plot([xs[-1]], [ys[-1]], "o", color=colour, markersize=5, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.4)
        label = f"{ys[-1]*100:.2f}%" if fmt == "pct" else f"{ys[-1]:.4g}"
        ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points", xytext=(-4, 9),
                    ha="right", fontsize=9, color=INK, fontweight="medium",
                    fontfamily=["DejaVu Sans Mono", "monospace"])

        suffix = "   log scale" if scale == "log" else ""
        ax.set_title(name + suffix, fontsize=9.5, fontweight="semibold", loc="left", pad=8)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(LINE)
        ax.spines["bottom"].set_color(LINE)
        ax.set_xlim(0, max(steps))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
        if fmt == "pct":
            if from_zero:
                ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.tick_params(labelsize=8, length=0)

    for ax in axes[1]:
        ax.set_xlabel("training step", fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out} ({args.out.stat().st_size/1000:.0f} kB), {len(steps)} points")


if __name__ == "__main__":
    main()
