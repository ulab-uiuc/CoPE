#!/usr/bin/env python
"""Score a `tau2 run` sweep the way InfoPO's Table 1 does.

tau2 writes one results.json per domain under data/simulations/<save-to>/. Each entry is
one trial of one task, carrying tau2's own reward under EvaluationType.ALL. The paper
reports Avg@4: the mean success rate over 4 independent trials per task, i.e. the plain
mean of per-trial success -- not pass@4, which would count a task solved if any trial
succeeded and reads far higher.

    python scripts/tau2_align_score.py --tag base --domains airline retail telecom
"""

import argparse
import json
import math
import os
import sys

PAPER = {  # Table 1, Qwen2.5-7B-Instruct prompting row
    "telecom": 0.144,
    "retail": 0.131,
    "airline": 0.075,
}
SOLVED = 0.99  # tau2 rewards are 0/1 in practice; guard against float noise


def load(tag: str, domain: str, data_dir: str):
    sims = os.path.join(data_dir, "simulations")
    if not os.path.isdir(sims):
        return None, f"no simulations dir at {sims}"
    # Save-path layout differs by tau2 version: v1.0.1 writes <save-to>/results.json,
    # the 2026-02 tree appends its own .json to the name it was given.
    stem = f"tau2_align_{tag}_{domain}.json"
    cands = [os.path.join(sims, stem, "results.json"),
             os.path.join(sims, stem + ".json", "results.json"),
             os.path.join(sims, stem + ".json"),
             os.path.join(sims, stem)]
    for c in cands:
        if os.path.isfile(c):
            with open(c) as f:
                return json.load(f), c
    return None, f"not found under {sims} for tag={tag} domain={domain}"


def score(blob):
    sims = blob.get("simulations", blob if isinstance(blob, list) else [])
    per_task = {}
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward", s.get("reward", 0.0)) or 0.0
        per_task.setdefault(s.get("task_id", s.get("id")), []).append(float(r))
    trials = [r for rs in per_task.values() for r in rs]
    if not trials:
        return None
    avg = sum(1 for r in trials if r >= SOLVED) / len(trials)
    # pass@4 is reported too, purely as a guard: if someone later compares the wrong
    # column, the gap between these two makes the mistake obvious rather than silent.
    p4 = sum(1 for rs in per_task.values() if any(r >= SOLVED for r in rs)) / len(per_task)
    return {"avg": avg, "pass": p4, "tasks": len(per_task), "trials": len(trials),
            "mean_reward": sum(trials) / len(trials)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--domains", nargs="+", default=["airline", "retail", "telecom"])
    ap.add_argument("--data-dir", default=os.environ.get(
        "TAU2_DATA_DIR", "${TAU2_BENCH_DIR}/data"))
    a = ap.parse_args()

    print(f"{'domain':10s} {'Avg@4':>8s} {'paper':>8s} {'delta':>8s} "
          f"{'pass@4':>8s} {'tasks':>6s} {'trials':>7s}")
    got = []
    for d in a.domains:
        blob, where = load(a.tag, d, a.data_dir)
        if blob is None:
            print(f"{d:10s}  -- {where}")
            continue
        m = score(blob)
        if not m:
            print(f"{d:10s}  -- empty results in {where}")
            continue
        # Warn rather than silently scoring a partial run: tau2 accumulates results in
        # task order, so an unfinished file is biased toward the short, easy tasks.
        # Guard on trials, not just tasks: a run can reach every task while still
        # missing most of their repeats, and the ones it has are the fast ones.
        expected = {"airline": 20, "retail": 40, "telecom": 40}.get(d)
        want_trials = expected * 4 if expected else None
        if expected and (m["tasks"] < expected or m["trials"] < want_trials):
            print(f"{d:10s}  -- PARTIAL: {m['tasks']}/{expected} tasks, "
                  f"{m['trials']}/{want_trials} trials; biased toward short episodes, "
                  f"do not compare")
            continue
        ref = PAPER.get(d)
        delta = f"{100*(m['avg']-ref):+.1f}" if ref is not None else "n/a"
        print(f"{d:10s} {100*m['avg']:7.1f}% {100*ref:7.1f}% {delta:>8s} "
              f"{100*m['pass']:7.1f}% {m['tasks']:6d} {m['trials']:7d}")
        got.append((d, m["avg"], ref))

    if got:
        mine = sum(x[1] for x in got) / len(got)
        ref = sum(x[2] for x in got) / len(got)
        print(f"{'mean':10s} {100*mine:7.1f}% {100*ref:7.1f}% {100*(mine-ref):+7.1f}")
        # A single domain within ~3 points is consistent with 4-trial sampling noise on
        # 20-40 tasks; a gap much larger than that is a protocol difference, not noise.
        se = math.sqrt(0.15 * 0.85 / max(1, sum(1 for _ in got) * 80))
        print(f"\n(1 se at these rates and counts is roughly {100*se:.1f} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
