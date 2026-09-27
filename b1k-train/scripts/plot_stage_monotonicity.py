"""Show why a BDDL stage label is not always monotone: plot one episode's stage curve above a
per-literal timeline, so a drop can be traced to the object that left its container.

    python scripts/plot_stage_monotonicity.py task-0020/bddl_00201050.hdf5 -o out.png

The stage is the count of satisfied goal literals in the best solution option, so on its own a dip
only says "one fewer". The timeline below names the literal: one row per literal of the option that
wins at the end, filled where it holds. A row with a gap is an object that was placed and then came
back out -- the dip's cause, visible without reading the HDF5 by hand.
"""

import argparse
import json
import pathlib
import re

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HELD = "#2a78d6"      # a literal that holds
BROKE = "#d03b3b"     # a literal that held, stopped holding, i.e. the cause of a dip
INK, INK_2, INK_3 = "#16171a", "#55565b", "#83848a"
SURFACE, GRID, LINE = "#fcfcfb", "#ecece7", "#e2e2dd"


def attr(group, key):
    value = group.attrs[key]
    return json.loads(value) if isinstance(value, (str, bytes)) else np.asarray(value)


def short(literal: str) -> str:
    """Compact a ground literal for a row label, keeping the predicate.

    inside(leek.n.02_1, mixing_bowl.n.01_1)  -> inside: leek_1 -> bowl_1
    on_fire(firewood.n.01_2)                 -> on_fire(firewood_2)
    not toggled_on(cigar_lighter.n.01_1)     -> NOT toggled_on(lighter_1)

    The predicate is kept because a task can mix several (ontop vs inside), and a leading "not" is
    shouted because those are the literals a demo must break on its way to the goal.
    """
    negated = literal.startswith("not ")
    body = literal[4:] if negated else literal

    def obj(name: str, idx: str) -> str:
        base = name.split(".")[0]
        base = {"mixing_bowl": "bowl", "wood_fireplace": "fireplace", "cigar_lighter": "lighter"}.get(base, base)
        return f"{base}_{idx}"

    m = re.match(r"([\w.]+)\(([\w.]+)_(\d+),\s*([\w.]+)_(\d+)\)$", body)
    if m:
        pred, a, ai, b, bi = m.groups()
        out = f"{pred}: {obj(a, ai)} -> {obj(b, bi)}"
    else:
        m = re.match(r"([\w.]+)\(([\w.]+)_(\d+)\)$", body)
        out = f"{m.group(1)}({obj(m.group(2), m.group(3))})" if m else body
    return ("NOT " + out) if negated else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecar", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("outputs/figures/stage_monotonicity.png"))
    args = ap.parse_args()

    with h5py.File(args.sidecar, "r") as f:
        d = f["data"]
        status = d["atom_status"][:]
        frame = d["frame"][:]
        option_of = np.asarray(attr(d, "atom_option")).reshape(-1)
        labels = list(attr(d, "atom_labels"))
        demo_id = int(d.attrs["demo_id"])
        task_name = str(d.attrs["task_name"])
        stride = int(d.attrs["stride"])

    options = sorted(set(option_of.tolist()))
    # int32: summing uint8 gives an UNSIGNED accumulator, and unsigned differences wrap, so a
    # decrease silently reads as a huge increase and every dip is missed.
    per_option = np.stack([status[:, option_of == o].sum(1) for o in options], 1).astype(np.int32)
    stage = per_option.max(1)
    winner = options[int(per_option[-1].argmax())]        # the option the demo actually satisfies
    cols = np.flatnonzero(option_of == winner)
    held = status[:, cols].astype(bool)                   # [N, L]

    # order rows by when each literal first holds, so the timeline reads as the demo's sequence
    first = [int(np.argmax(held[:, j])) if held[:, j].any() else len(frame) for j in range(held.shape[1])]
    order = np.argsort(first)
    held, names = held[:, order], [short(labels[cols[j]]) for j in order]
    broke = [bool(np.any(held[:-1, j] & ~held[1:, j])) for j in range(held.shape[1])]

    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"]})
    # constrained layout, not tight_layout: the row labels are long, and tight_layout cannot size a
    # margin for them here (it warns and clips them).
    fig, (ax_s, ax_t) = plt.subplots(
        2, 1, figsize=(12.5, 2.4 + 0.34 * len(names) + 2.2),
        height_ratios=[2.2, 0.34 * len(names) + 0.6],
        sharex=True, layout="constrained",
    )
    fig.get_layout_engine().set(hspace=0.04, h_pad=0.08)
    fig.patch.set_facecolor(SURFACE)

    dips = np.flatnonzero(np.diff(stage) < 0)
    fig.suptitle(
        f"{task_name}  |  demo {demo_id}  |  the stage label is not monotone\n"
        f"stage = satisfied literals of the winning option ({len(names)} of them), "
        f"sampled every {stride} steps  |  {len(dips)} drop(s)",
        x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK, linespacing=1.6)

    # --- stage over time -----------------------------------------------------------------------
    ax_s.step(frame, stage, where="post", color=HELD, linewidth=1.8, zorder=3)
    ax_s.fill_between(frame, stage, step="post", color=HELD, alpha=0.10, linewidth=0, zorder=2)
    # Drops cluster (an object bounces out and back within a few samples), so a fixed label offset
    # overlaps. Stack labels of a cluster vertically instead of letting them collide.
    span = (frame[-1] - frame[0]) or 1
    level, prev_x = 0, None
    for i in dips:
        ax_s.axvspan(frame[i], frame[i + 1], color=BROKE, alpha=0.16, zorder=1, linewidth=0)
        level = level + 1 if prev_x is not None and (frame[i] - prev_x) < 0.06 * span else 0
        prev_x = frame[i]
        ax_s.annotate(f"{stage[i]}→{stage[i+1]}", (frame[i], stage[i]),
                      textcoords="offset points", xytext=(6, 5 + 13 * level), fontsize=8.5,
                      color=BROKE, fontweight="bold")
    ax_s.set_ylabel("stage", fontsize=9, color=INK_2)
    ax_s.set_ylim(-0.4, len(names) + 0.6)
    ax_s.set_yticks(range(0, len(names) + 1, max(1, len(names) // 6)))

    # --- one row per literal -------------------------------------------------------------------
    for row, (name, bad) in enumerate(zip(names, broke)):
        on = held[:, row]
        spans, start = [], None
        for k, v in enumerate(on):
            if v and start is None:
                start = frame[k]
            elif not v and start is not None:
                spans.append((start, frame[k] - start)); start = None
        if start is not None:
            spans.append((start, frame[-1] - start))
        colour = BROKE if bad else HELD
        ax_t.broken_barh(spans, (row - 0.32, 0.64), facecolors=colour,
                         alpha=1.0 if bad else 0.75, zorder=3)
    ax_t.set_yticks(range(len(names)))
    ax_t.set_yticklabels(names, fontsize=8.5,
                         fontfamily=["DejaVu Sans Mono", "monospace"])
    for tick, bad in zip(ax_t.get_yticklabels(), broke):
        tick.set_color(BROKE if bad else INK_2)
        if bad:
            tick.set_fontweight("bold")
    ax_t.set_ylim(len(names) - 0.5, -0.5)
    ax_t.set_xlabel("simulator step", fontsize=9, color=INK_2)
    ax_t.set_xlim(0, frame[-1])

    n_broke = sum(broke)
    ax_t.set_title(f"每個 literal 何時成立 — {n_broke} row(s) break" if False else
                   f"when each literal holds  —  {n_broke} row(s) break and recover",
                   fontsize=9.5, fontweight="bold", loc="left", pad=8, color=INK)

    for ax in (ax_s, ax_t):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
        ax.tick_params(labelsize=8, length=0, colors=INK_3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out} | winning option {winner} | drops at frames "
          f"{[int(frame[i]) for i in dips]} | breaking rows "
          f"{[n for n, b in zip(names, broke) if b]}")


if __name__ == "__main__":
    main()
