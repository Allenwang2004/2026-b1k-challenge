"""The finetune's learning rate destroys the checkpoint it starts from, then spends the run repairing it.

Both BDDL runs start from the 2025 champion's checkpoint_2. action_loss bottoms out around step 200 --
while the LR is still ~2e-5 and param_norm has barely moved off the champion's weights -- and then
climbs by ~70% as warmup drives the LR to its 1e-4 peak. The 20000-step run needs ~15000 steps to get
back to the level it already had at step 200; the 2500-step run stops while still in the hole, which
is why it rolled out worse despite being the "less overfit" model.

    python scripts/plot_lr_damage.py -o outputs/figures/lr_damage.png
"""

import argparse
import math
import pathlib
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNS = [
    ("peak_lr 1e-4, run A", REPO / "outputs/logs/bddl_ckpt2_task1.20260923-113351.log", 20_000, 1e-4, 1000, "#2a78d6"),
    ("peak_lr 1e-4, run B", REPO / "outputs/logs/bddl_2500.20260924-160120.log", 2_500, 1e-4, 1000, "#7fb2e8"),
    ("peak_lr 2e-5 (diagnostic)", REPO / "outputs/logs/lr2e5.log", 20_000, 2e-5, 200, "#1b8a5a"),
]
FIT, INK, INK_2, INK_3 = "#d03b3b", "#16171a", "#55565b", "#83848a"
SURFACE, GRID = "#fcfcfb", "#ecece7"


def read(path):
    steps, loss = [], []
    for m in re.finditer(r"Step (\d+): .*?action_loss=([0-9.]+)", path.read_text()):
        steps.append(int(m.group(1)))
        loss.append(float(m.group(2)))
    return np.array(steps), np.array(loss)


def lr(step, decay, peak=1e-4, warmup=1000, end=1e-5):
    if step < warmup:
        return peak * step / warmup
    p = min((step - warmup) / (decay - warmup), 1.0)
    return end + (peak - end) * 0.5 * (1 + math.cos(math.pi * p))


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=7, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_2, length=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=REPO / "outputs/figures/lr_damage.png")
    args = ap.parse_args()

    data = [(name, *read(p), decay, peak, warm, c) for name, p, decay, peak, warm, c in RUNS]
    floor = min(l[s <= 400].min() for _, s, l, *_ in data)   # the best the starting weights reach

    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"]})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    for col, (xmax, label) in enumerate([(2000, "first 2000 steps"), (20000, "the whole run")]):
        ax_lr, ax_l = axes[0][col], axes[1][col]
        for name, s, l, decay, peak, warm, c in data:
            m = s <= xmax
            ax_lr.plot(s[m], [lr(int(x), decay, peak, warm, peak / 10) for x in s[m]],
                       color=c, linewidth=1.8, label=name)
            ax_l.plot(s[m], l[m], color=c, linewidth=1.6, alpha=0.9, label=name)
        ax_lr.axhline(2e-5, color=FIT, linewidth=1.1, linestyle=":")
        ax_lr.annotate("2e-5 — the LR at which action_loss is lowest", (xmax * 0.015, 2.3e-5),
                       fontsize=8, color=FIT)
        ax_lr.set_yscale("log")
        style(ax_lr, "", "learning rate (log)", f"learning rate  —  {label}")

        ax_l.axhline(floor, color=FIT, linewidth=1.2, linestyle="--")
        ax_l.annotate(f"{floor:.3f}  the best the starting weights reach (step ~200, LR still ~2e-5)",
                      (xmax * 0.015, floor + 0.004), fontsize=8, color=FIT, fontweight="bold")
        style(ax_l, "training step", "action_loss", f"action_loss  —  {label}")
        ax_l.legend(frameon=False, fontsize=8, labelcolor=INK_2,
                    loc="lower right" if col == 0 else "upper right")

    # mark the two moments that carry the argument, on the wide panels
    a = axes[1][0]
    s, l = data[0][1], data[0][2]
    a.annotate("1e-4: +69% / +78% off the minimum,\nand it stays there", (1200, l[s == 1200][0]),
               textcoords="offset points", xytext=(-8, 26), fontsize=8.5, color=FIT, fontweight="bold",
               ha="center", arrowprops=dict(arrowstyle="->", color=FIT, linewidth=1.1))
    s3, l3 = data[2][1], data[2][2]
    a.annotate("2e-5: rises only 9%, then comes back\n(the cost of adapting to the new token)",
               (450, l3[s3 == 450][0]), textcoords="offset points", xytext=(70, 44), fontsize=8.5,
               color="#1b8a5a", fontweight="bold",
               arrowprops=dict(arrowstyle="->", color="#1b8a5a", linewidth=1.1))
    b = axes[1][1]
    back = int(s[(s > 1000) & (l <= floor)][0]) if (l[s > 1000] <= floor).any() else None
    if back:
        b.annotate(f"back to the step-200 level\nonly at step {back}", (back, floor),
                   textcoords="offset points", xytext=(-40, -48), fontsize=8.5, color=FIT,
                   fontweight="bold", ha="right",
                   arrowprops=dict(arrowstyle="->", color=FIT, linewidth=1.1))

    fig.suptitle(
        "Most of the loss blow-up is the learning rate, not the model adapting to its new conditioning\n"
        "Three runs from behavior_checkpoints/ilia/checkpoint_2, identical data, labels and BDDL token; only peak_lr differs.\n"
        "All bottom out near 0.19 while the LR is ~2e-5. Held at 2e-5 the loss rises 9% and recovers; driven to 1e-4 it\n"
        "rises 69-78% and stays up for 13800 steps. Adaptation is real but small; the learning rate is the larger effect.",
        x=0.006, ha="left", fontsize=10.5, fontweight="bold", color=INK, linespacing=1.55)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}; floor={floor:.4f}, recovered at step {back}")


if __name__ == "__main__":
    main()
