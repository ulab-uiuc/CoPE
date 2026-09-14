#!/usr/bin/env python
"""Plot tau2 solve rate per epoch across runs.

The dense reward and the binary reward live on different scales, so
`critic/score/mean` is not comparable between runs. The fraction of trajectories that
actually solved their task is, and it is recoverable from the rollout logs -- this is
the metric to judge whether GRPO learned anything.

    python scripts/tau2_plot_solverate.py \\
        binary=runlogs/tau2_retail_grpo_20260906_181701 \\
        dense=runlogs/tau2_retail_grpo_20260907_031758
"""

import collections
import glob
import json
import math
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz default categorical slots 1 and 2; validated for CVD on the light surface.
SERIES_COLORS = ["#2a78d6", "#eb6834"]
SURFACE, INK, MUTED = "#fcfcfb", "#1a1a19", "#6b6b68"


def collect(run_dir: str, steps_per_epoch: int = 9):
    per_epoch = collections.defaultdict(lambda: [0, 0])
    degen = tot = 0
    for f in glob.glob(f"{run_dir}/rollout_logs/step*/*.json"):
        step = int(re.search(r"step(\d+)", f).group(1))
        ep = (step - 1) // steps_per_epoch
        groups = collections.defaultdict(list)
        for t in json.load(open(f)):
            r = t.get("reward", 0.0)
            per_epoch[ep][0] += 1
            per_epoch[ep][1] += 1 if r >= 1.0 else 0
            groups[t["item_id"]].append(r)
        for rs in groups.values():
            tot += 1
            degen += 1 if len(set(rs)) == 1 else 0
    return per_epoch, (degen / tot if tot else float("nan"))


def wilson(w, n, z=1.0):
    """Wilson interval -- at ~50 successes in ~576 the normal approximation is skewed."""
    if not n:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre, half


def main() -> int:
    runs = []
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        per_epoch, degen = collect(path)
        if per_epoch:
            runs.append((label or path, per_epoch, degen))
    if not runs:
        print(__doc__)
        return 1

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (label, per_epoch, degen) in enumerate(runs):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        eps = sorted(per_epoch)
        xs = [e + 1 for e in eps]
        ys, los, his = [], [], []
        for e in eps:
            n, w = per_epoch[e]
            centre, half = wilson(w, n)
            ys.append(100 * w / n)
            los.append(100 * max(0.0, centre - half))
            his.append(100 * (centre + half))
        ax.fill_between(xs, los, his, color=c, alpha=0.13, linewidth=0)
        ax.plot(xs, ys, color=c, lw=2.0, marker="o", ms=6,
                label=f"{label}   (degenerate groups {degen:.0%})")
        for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=INK)
        print(f"{label}: " + "  ".join(
            f"e{e+1}={100*per_epoch[e][1]/per_epoch[e][0]:.1f}%" for e in eps
        ) + f"   degenerate={degen:.1%}")

    ax.set_xlabel("epoch", color=MUTED, fontsize=10)
    ax.set_ylabel("tasks solved (%)", color=MUTED, fontsize=10)
    ax.set_title(
        "tau2-bench retail · GRPO · Qwen2.5-7B-Instruct\n"
        "solve rate per epoch, band = ±1 Wilson SE",
        color=INK, fontsize=12, pad=12, loc="left")
    ax.grid(True, axis="y", color="#e5e5e2", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d8d4")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xticks(sorted({e + 1 for _, pe, _ in runs for e in pe}))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    fig.tight_layout()
    out = "runlogs/tau2_solverate.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
