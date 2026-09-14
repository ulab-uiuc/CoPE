#!/usr/bin/env python
"""Turn tau2 GRPO slurm logs into a CSV + reward curve, for when wandb is unreachable.

wandb runs here are WANDB_MODE=offline and api.wandb.ai is blocked by the egress
allowlist, so the training metrics have to be read locally. Every metric verl logs is
already in the slurm stdout as `step:N - key:val - key:val - ...`; this parses that.

    python scripts/tau2_plot_metrics.py slurm_logs/tau2_grpo_20107.out
    python scripts/tau2_plot_metrics.py slurm_logs/tau2_grpo_2008{8,}.out --out reward.png

Writes <out>.csv alongside the image.
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz default categorical slots 1 and 2; validated for CVD on the light surface.
SERIES_COLORS = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#6b6b68"

STEP_RE = re.compile(r"step:(\d+)\s")
KV_RE = re.compile(r"([a-zA-Z][\w/]*):(-?[\d.]+(?:e-?\d+)?)")


def parse_log(path: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace")
            m = STEP_RE.search(line)
            if not m or "critic/score/mean" not in line:
                continue
            step = int(m.group(1))
            rec = {}
            for k, v in KV_RE.findall(line):
                try:
                    rec[k] = float(v)
                except ValueError:
                    pass
            rec.pop("step", None)
            rows[step] = rec
    return rows


def epoch_means(steps, vals, per_epoch):
    """Mean per epoch, plotted at the epoch's last step so it lines up with the raw run."""
    xs, ys = [], []
    for start in range(0, len(steps), per_epoch):
        chunk = vals[start : start + per_epoch]
        if chunk:
            xs.append(steps[start : start + per_epoch][-1])
            ys.append(sum(chunk) / len(chunk))
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--metric", default="critic/score/mean")
    ap.add_argument("--steps-per-epoch", type=int, default=9)
    ap.add_argument("--out", default="tau2_reward.png")
    args = ap.parse_args()

    runs = []
    for path in args.logs:
        rows = parse_log(path)
        if not rows:
            print(f"[warn] no metric lines in {path}", file=sys.stderr)
            continue
        steps = sorted(rows)
        vals = [rows[s].get(args.metric, float("nan")) for s in steps]
        runs.append((os.path.basename(path).replace(".out", ""), steps, vals, rows))

    if not runs:
        print("no data", file=sys.stderr)
        return 1

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    keys = sorted({k for _, _, _, rows in runs for r in rows.values() for k in r})
    with open(csv_path, "w") as f:
        f.write("run,step," + ",".join(keys) + "\n")
        for name, steps, _, rows in runs:
            for s in steps:
                f.write(f"{name},{s}," + ",".join(str(rows[s].get(k, "")) for k in keys) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (name, steps, vals, _) in enumerate(runs):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        n = len([v for v in vals if v == v])
        # Raw per-step is noisy; keep it recessive and let the epoch mean carry the trend.
        ax.plot(steps, vals, color=c, lw=1.0, alpha=0.30, zorder=2)
        ex, ey = epoch_means(steps, vals, args.steps_per_epoch)
        ax.plot(ex, ey, color=c, lw=2.0, marker="o", ms=5,
                label=f"{name}  (n={n} steps)", zorder=3)

    ax.set_xlabel("optimizer step", color=MUTED, fontsize=10)
    ax.set_ylabel(args.metric, color=MUTED, fontsize=10)
    ax.set_title("tau2-bench retail · GRPO · faint = per step, bold = epoch mean",
                 color=INK, fontsize=12, pad=12, loc="left")
    ax.grid(True, axis="y", color="#e5e5e2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d8d4")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    for name, steps, vals, _ in runs:
        ok = [v for v in vals if v == v]
        print(f"  {name}: {len(ok)} steps, mean={sum(ok)/max(len(ok),1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
