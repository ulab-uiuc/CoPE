#!/usr/bin/env python3
"""Which τ² tasks a do-nothing agent already solves.

The training reward (EvaluationType.ALL) is deterministic: database state, env
assertions, expected actions and communicated info -- no LLM judge. So "would an empty
policy get full marks" needs no rollout: hand tau2's own evaluate_simulation an episode
with no agent turns that ended normally (USER_STOP) and read the reward.

A task that scores 1.0 this way is won by talking, by transferring, or by anything that
leaves the database untouched; its wins say nothing about the actions that solve it, so
the action forecast should not learn from them.

Item ids follow the env server's order: domains in the order given, tasks in the
loader's order for the split, item id = position (agentenv_tau2 `_task_id_for`).

Usage (tau2 env):
  envs/tau2/bin/python scripts/tau2_null_policy_tasks.py \\
      --domains retail,airline,telecom --split train --out data/tau2_null_policy_train.json
"""
import argparse
import json
import uuid

from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="retail,airline,telecom")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, item = [], 0
    for dom in a.domains.split(","):
        for task in registry.get_tasks_loader(dom)(a.split):
            sim = SimulationRun(id=str(uuid.uuid4()), task_id=task.id, start_time="", end_time="",
                                duration=0.0, termination_reason=TerminationReason.USER_STOP,
                                messages=[])
            info = evaluate_simulation(sim, task, EvaluationType.ALL, solo_mode=False, domain=dom)
            crit = task.evaluation_criteria
            rows.append(dict(
                item_id=item, domain=dom, task_id=task.id, null_reward=float(info.reward),
                reward_basis=[str(getattr(b, "value", b)) for b in (crit.reward_basis if crit else [])],
                n_actions=len(crit.actions or []) if crit else 0,
                n_communicate=len(crit.communicate_info or []) if crit else 0,
                breakdown={str(getattr(k, "value", k)): v
                           for k, v in (info.reward_breakdown or {}).items()}))
            item += 1

    null = [r for r in rows if r["null_reward"] >= 1.0]
    json.dump(dict(domains=a.domains, split=a.split, n_tasks=len(rows),
                   null_full_score_item_ids=[r["item_id"] for r in null], tasks=rows),
              open(a.out, "w"), indent=1)
    print(f"{len(null)}/{len(rows)} tasks: a do-nothing agent scores 1.0")
    for dom in a.domains.split(","):
        n = sum(r["domain"] == dom for r in rows)
        k = sum(r["domain"] == dom for r in null)
        print(f"  {dom:8s} {k:3d}/{n}")


if __name__ == "__main__":
    main()
