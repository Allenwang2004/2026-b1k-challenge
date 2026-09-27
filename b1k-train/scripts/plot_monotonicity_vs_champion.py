"""Is BDDL-stage non-monotonicity a proxy for task difficulty?

Regresses the 2025 champion's per-task scores (Robot Learning Collective, standard track) against
how badly each task's symbolic stage regresses. The champion was conditioned on the TIME-SPLIT
stage, so its score cannot be hurt by BDDL non-monotonicity directly -- a correlation here would
mean the two happen to measure the same underlying difficulty, which would be a confound to control
for when comparing label schemes. No correlation means the axes are independent.

    python scripts/plot_monotonicity_vs_champion.py -o outputs/figures/monotonicity_vs_champion.png
"""

import argparse
import json
import pathlib

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
SUBMISSION = REPO.parent / "BEHAVIOR-1K/docs/challenge_submissions" / \
    "standard.{testset}.Robot_Learning_Collective.Independent.20251114.json"
MONO = REPO / "outputs/assets/bddl_stage_monotonicity_all50.json"

POINT = "#2a78d6"
FIT = "#d03b3b"
INK, INK_2, INK_3 = "#16171a", "#55565b", "#83848a"
SURFACE, GRID = "#fcfcfb", "#ecece7"


def panel(ax, x, y, names, xlabel, ylabel, title):
    ax.scatter(x, y, s=34, color=POINT, alpha=0.62, linewidth=0.8,
               edgecolor=SURFACE, zorder=3)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    xs = np.linspace(0, max(x) * 1.05, 50)
    ax.plot(xs, intercept + slope * xs, color=FIT, linewidth=1.8, zorder=4)

    # a bootstrap band, so the eye sees how little the slope is pinned down
    rng = np.random.default_rng(0)
    fits = []
    for _ in range(600):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0:
            continue
        s, b, *_ = stats.linregress(x[i], y[i])
        fits.append(b + s * xs)
    lo, hi = np.percentile(np.array(fits), [2.5, 97.5], axis=0)
    ax.fill_between(xs, lo, hi, color=FIT, alpha=0.11, linewidth=0, zorder=2)

    # name the points that drive the eye: the extremes on each axis. Nearby labels are nudged
    # apart -- the high-regression tasks cluster at the right edge and otherwise overprint.
    marked = list(dict.fromkeys(sorted(range(len(x)), key=lambda i: -x[i])[:3]
                                + sorted(range(len(x)), key=lambda i: -y[i])[:2]))
    placed = []
    sx, sy = (np.ptp(x) or 1), (np.ptp(y) or 1)
    for i in marked:
        dy = 4
        while any(abs(x[i] - px) < 0.13 * sx and abs(y[i] + dy * sy / 260 - py) < 0.05 * sy
                  for px, py in placed):
            dy += 11
        placed.append((x[i], y[i] + dy * sy / 260))
        ax.annotate(names[i], (x[i], y[i]), textcoords="offset points", xytext=(6, dy),
                    fontsize=7.5, color=INK_3)

    stars = "" if p >= 0.05 else "*"
    ax.text(0.975, 0.995, f"Pearson r = {r:+.3f} (p = {p:.3f}){stars}\nSpearman ρ = {rho:+.3f} (p = {p_rho:.3f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK_2,
            linespacing=1.55, family=["DejaVu Sans Mono", "monospace"])
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=7, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_2, length=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=pathlib.Path,
                    default=REPO / "outputs/figures/monotonicity_vs_champion.png")
    args = ap.parse_args()

    mono = {r["task_name"]: r for r in json.loads(MONO.read_text())}

    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"]})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    for row, testset in enumerate(("public", "hidden")):
        scores = json.loads(SUBMISSION.with_name(SUBMISSION.name.format(testset=testset)).read_text())
        pt = scores["per_task_scores"]
        names = [n for n in pt["q_score"] if n in mono]
        if len(names) != 50:
            raise SystemExit(f"{testset}: joined {len(names)} of 50 tasks; check the task names")
        x = np.array([mono[n]["regressed_frac"] for n in names]) * 100
        short = [n.replace("_", " ")[:26] for n in names]
        for col, (key, label) in enumerate((("q_score", "q_score (partial credit)"),
                                            ("task_sr", "success rate"))):
            y = np.array([pt[key][n] for n in names])
            panel(axes[row][col], x, y, short,
                  "time the stage sits below its own running max (% of episode)", label,
                  f"{testset} test set  —  {label}")

    fig.suptitle(
        "BDDL stage non-monotonicity vs the 2025 champion's per-task score\n"
        "50 tasks, Robot Learning Collective, standard track. The champion was conditioned on the\n"
        "time-split stage, so this asks only whether symbolic regression tracks task difficulty.\n"
        "Band = 95% bootstrap CI of the fit.",
        x=0.006, ha="left", fontsize=11, fontweight="bold", color=INK, linespacing=1.55)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
