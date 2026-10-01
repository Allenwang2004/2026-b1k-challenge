#!/usr/bin/env python
"""Score a rollout's stage prediction against exact ground truth, without running more rollouts.

Why this exists
---------------
End-of-episode rollout scores cannot guide development here. Running the *same* checkpoint twice on
the *same* instance produced q = 0.692 and q = 0.385 on one instance and 0.231 vs 0.000 on another,
while the model effects being chased are around 0.02. Resolving an effect that size through that much
noise needs roughly 1,200 instances -- about 49 days of single-machine rollout for one task.

But ``q_score`` is the fraction of BDDL goal literals satisfied, so

    q_score x (num_stages - 1)  ==  the episode's final stage, exactly, as an integer

(verified across 10 episodes of sorting_vegetables: every product landed on a whole number). Every
rollout already on disk is therefore labelled data, and each episode contributes ~893 stage decisions
instead of one noisy number.

What it reports
---------------
Per stage readout, against the exact final stage:

  * MAE and bias.
  * **The constant baseline**: the best score obtainable by ignoring the camera and always emitting
    one fixed number. On sorting_vegetables the existing pipeline loses to it (3.90 vs 2.00), which is
    the sharpest statement available that the predictor contributes nothing. Any new readout has to
    beat this line before it is worth a rollout.
  * Volatility: how often the answer changes versus how often the truth changes. The 15-way head
    measured 38x too fast (178 changes per episode against 4.6).
  * Whether confidence separates right from wrong answers. For the 15-way head it does not (median
    margin 4.50 on provably wrong votes vs 5.25 on the rest), so the output cannot be filtered.

Usage
-----
    score_stage_predictor.py <run_dir> [<run_dir> ...] [--task sorting_vegetables]

A run dir is an eval_runs entry holding ``<task>/json/*.json`` and ``stage_logs/*.jsonl``.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np


def load_truth(run_dir: str, task: str | None) -> dict[tuple[str, int], tuple[float, int]]:
    """``(task, instance) -> (q_score, final_stage)`` for every rollout in the run."""
    out = {}
    pattern = os.path.join(run_dir, task or "*", "json", "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            j = json.load(fh)
        out[(j["task"], int(j["instance_id"]))] = (float(j["q_score"]["final"]), -1)
    return out


def resolve_final_stages(truth: dict, num_stages: dict[str, int]) -> dict:
    """Turn each q_score into an integer stage, and say so when it does not land on one.

    A non-integer means the assumed stage count is wrong for that task, which would silently corrupt
    every number below -- so it is reported rather than rounded away.
    """
    resolved, suspect = {}, []
    for (task, inst), (q, _) in truth.items():
        n = num_stages.get(task)
        if n is None:
            continue
        product = q * (n - 1)
        stage = int(round(product))
        if abs(product - stage) > 1e-6:
            suspect.append((task, inst, q, product))
        resolved[(task, inst)] = (q, stage)
    if suspect:
        print(f"⚠  {len(suspect)} rollout(s) whose q_score x (num_stages-1) is not an integer -- the")
        print("   assumed stage count is probably wrong for that task:")
        for task, inst, q, product in suspect[:5]:
            print(f"     {task} inst {inst}: q={q:.6f} -> {product:.4f}")
    return resolved


def load_votes(run_dir: str) -> list[list[dict]]:
    """Episodes of vote records, split on reset / task_change."""
    episodes: list[list[dict]] = []
    for path in sorted(glob.glob(os.path.join(run_dir, "stage_logs", "*.jsonl"))):
        cur: list[dict] | None = None
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") in ("reset", "task_change"):
                    if cur and len(cur) > 50:
                        episodes.append(cur)
                    cur = []
                elif rec.get("event") == "vote" and cur is not None:
                    cur.append(rec)
        if cur and len(cur) > 50:
            episodes.append(cur)
    return episodes


def constant_baseline(truth: np.ndarray) -> tuple[int, float]:
    """Best single number to emit while ignoring the observation entirely, and its MAE."""
    candidates = range(int(truth.min()), int(truth.max()) + 1)
    best = min(candidates, key=lambda c: np.abs(truth - c).mean())
    return best, float(np.abs(truth - best).mean())


def score(name: str, predicted: np.ndarray, truth: np.ndarray) -> dict:
    return {
        "name": name,
        "mae": float(np.abs(predicted - truth).mean()),
        "bias": float((predicted - truth).mean()),
        "under": int((predicted < truth).sum()),
        "over": int((predicted > truth).sum()),
        "exact": int((predicted == truth).sum()),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--task", default=None, help="restrict to one task name")
    ap.add_argument("--stages", type=int, default=None,
                    help="stage count for the task, if it cannot be imported from the model config")
    args = ap.parse_args(argv)

    # The stage count per task. Given explicitly, or -- by default -- recovered from the q_scores
    # themselves below, which avoids having to map task names onto BDDL_TASK_NUM_STAGES indices and
    # fails loudly rather than quietly dividing by the wrong number.
    num_stages = {args.task: args.stages} if (args.stages is not None and args.task) else None

    for run_dir in args.run_dirs:
        print("=" * 78)
        print(run_dir)
        print("=" * 78)

        truth_raw = load_truth(run_dir, args.task)
        if not truth_raw:
            print("  no rollout json found\n")
            continue

        tasks = sorted({t for t, _ in truth_raw})
        if num_stages is None:
            # Recover the stage count from the q_scores themselves: the true count is the smallest n
            # for which every q x (n-1) is a whole number. Cheaper than mapping task names to indices,
            # and it fails loudly rather than silently using the wrong divisor.
            resolved_counts = {}
            for task in tasks:
                qs = [q for (t, _), (q, _) in truth_raw.items() if t == task and q > 0]
                found = None
                for n in range(2, 31):
                    if all(abs(q * (n - 1) - round(q * (n - 1))) < 1e-6 for q in qs):
                        found = n
                        break
                resolved_counts[task] = found
                print(f"  {task}: stage count inferred as {found} from {len(qs)} non-zero q_scores")
            counts = resolved_counts
        else:
            counts = num_stages

        truth = resolve_final_stages(truth_raw, {k: v for k, v in counts.items() if v})
        episodes = load_votes(run_dir)
        print(f"  rollouts: {len(truth)}   episodes in stage log: {len(episodes)}")

        if len(episodes) != len(truth):
            print("  ⚠  counts differ -- pairing the first min(n) of each in order; a crashed instance")
            print("     leaves a stage log episode with no json and would shift the pairing.")

        keys = sorted(truth)
        n = min(len(keys), len(episodes))
        if n == 0:
            print()
            continue
        keys, episodes = keys[:n], episodes[:n]
        y = np.array([truth[k][1] for k in keys])

        # --- readouts -----------------------------------------------------------------------------
        tracked = np.array([ep[-1].get("stage_after", 0) for ep in episodes])
        argmax_last = np.array([ep[-1].get("argmax", 0) for ep in episodes])
        argmax_max = np.array([max(r.get("argmax", 0) for r in ep) for ep in episodes])
        argmax_median = np.array([int(np.median([r.get("argmax", 0) for r in ep])) for ep in episodes])

        rows = [
            score("tracked stage (what fed the model)", tracked, y),
            score("15-way argmax, last vote", argmax_last, y),
            score("15-way argmax, episode max", argmax_max, y),
            score("15-way argmax, episode median", argmax_median, y),
        ]

        has_event = any("event_count" in r for ep in episodes for r in ep)
        if has_event:
            ev = np.array([
                max((r.get("event_count", 0) for r in ep), default=0) for ep in episodes
            ])
            rows.append(score("event counter", ev, y))

        best_c, best_c_mae = constant_baseline(y)

        print(f"\n  真實最終 stage: {y.tolist()}  平均 {y.mean():.2f}")
        print(f"\n  {'readout':38s}{'MAE':>7}{'偏誤':>9}{'精確':>6}{'低估':>6}{'高估':>6}")
        print("  " + "-" * 72)
        for r in sorted(rows, key=lambda r: r["mae"]):
            flag = "  ✅" if r["mae"] < best_c_mae else "  ❌ 輸給常數"
            print(f"  {r['name']:38s}{r['mae']:>7.2f}{r['bias']:>+9.2f}"
                  f"{r['exact']:>6}{r['under']:>6}{r['over']:>6}{flag}")
        print("  " + "-" * 72)
        print(f"  {'常數基準（不看畫面，永遠輸出 ' + str(best_c) + '）':38s}{best_c_mae:>7.2f}"
              f"{(best_c - y.mean()):>+9.2f}")
        print("\n  任何讀法必須贏過常數基準才值得花一輪 rollout 驗證。")

        # --- volatility ---------------------------------------------------------------------------
        print("\n  === 波動：答案改變的頻率 vs 真相改變的頻率 ===")
        flips = np.array([float((np.diff([r.get("argmax", 0) for r in ep]) != 0).mean())
                          for ep in episodes])
        calls = np.array([len(ep) for ep in episodes])
        true_changes = y.astype(float)           # monotone labels: final stage == number of changes
        model_changes = flips * (calls - 1)
        print(f"  真實 stage 一場改變      {true_changes.mean():8.1f} 次")
        print(f"  15-way argmax 一場改變   {model_changes.mean():8.1f} 次"
              f"   ({model_changes.mean() / max(true_changes.mean(), 1e-6):.0f} 倍)")
        if has_event:
            ev_fires = np.array([
                float(np.mean([1.0 if r.get("event_logit", -99) > 0 else 0.0 for r in ep]))
                for ep in episodes
            ])
            print(f"  event head 說「有」的比例 {ev_fires.mean():8.1%}"
                  f"   （訓練分布約 3%，差太多就是校準跑掉了）")
            print(f"  event 計數一場           {ev.mean():8.1f} 次")

        # --- can confidence filter the wrong answers? ---------------------------------------------
        print("\n  === 信心度能不能分辨對錯 ===")
        wrong_margins, other_margins = [], []
        for ep, final in zip(episodes, y):
            for rec in ep:
                logits = rec.get("logits")
                if not logits:
                    continue
                arr = np.array([v if v is not None else -1e9 for v in logits], dtype=float)
                srt = np.sort(arr)
                margin = float(srt[-1] - srt[-2])
                # A vote above the stage the episode ever reached is provably wrong: the labels are
                # monotone, so the truth never exceeded its final value.
                (wrong_margins if rec.get("argmax", 0) > final else other_margins).append(margin)
        if wrong_margins and other_margins:
            print(f"  確定錯誤的票 {len(wrong_margins):6d} 筆   margin 中位數 {np.median(wrong_margins):6.2f}")
            print(f"  其餘的票     {len(other_margins):6d} 筆   margin 中位數 {np.median(other_margins):6.2f}")
            ratio = np.median(wrong_margins) / max(np.median(other_margins), 1e-6)
            if ratio > 0.8:
                print("  -> 幾乎一樣：它是有自信地答錯，沒辦法用信心度過濾")
            else:
                print("  -> 錯的票信心明顯較低：可以考慮設門檻")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
