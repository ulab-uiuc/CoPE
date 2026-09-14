#!/usr/bin/env python
"""Smoke test for the agentenv-tau2 env server (Stage 1).

Drives the HTTP contract directly -- no model, no verl -- and replays each task's
ground-truth tool calls, which should score 1.0 under the default DB/env reward basis.
That exercises create/reset/step, tool-call vs message routing, delta observations and
the evaluator in one shot.

    # terminal 1
    python scripts/tau2_fake_usersim.py --port 38001
    # terminal 2
    TAU2_USER_API_BASE=http://127.0.0.1:38001/v1 tau2-env --host 127.0.0.1 --port 36201
    # terminal 3
    python scripts/smoke_tau2_env.py --addr http://127.0.0.1:36201 --n-tasks 3

Run it with the *tau2* interpreter (it reads ground-truth actions out of tau2), and
point --addr at a server started on the same TAU2_DOMAIN / TAU2_TASK_SPLIT.
"""

import argparse
import json
import sys

import requests

from tau2.registry import registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="http://127.0.0.1:36201")
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-tasks", type=int, default=3)
    args = parser.parse_args()

    s = requests.Session()
    s.trust_env = False  # ignore any http(s)_proxy in the environment

    prompt = s.get(f"{args.addr}/system_prompt", timeout=60).json()
    print(f"[smoke] system_prompt: {len(prompt)} chars")
    assert "# Policy" in prompt and "# Tools" in prompt, "system prompt looks wrong"

    tasks = registry.get_tasks_loader(args.domain)(args.split)
    failures = []

    for idx in range(min(args.n_tasks, len(tasks))):
        task = tasks[idx]
        gt_actions = task.evaluation_criteria.actions
        env_idx = s.post(f"{args.addr}/create", timeout=60).json()
        first_obs = s.post(
            f"{args.addr}/reset", json={"env_idx": env_idx, "session_id": idx}, timeout=600
        ).json()
        print(f"\n[smoke] task {task.id} (item {idx}) env_idx={env_idx}")
        print(f"        reset obs: {first_obs[:120]!r}")
        assert first_obs, "reset returned an empty observation"

        reward, done = 0.0, False
        for act in gt_actions:
            action = json.dumps({"name": act.name, "arguments": act.arguments})
            r = s.post(
                f"{args.addr}/step",
                json={"env_idx": env_idx, "action": action},
                timeout=600,
            ).json()
            reward, done = r["reward"], r["done"]
            assert len(r["state"]) < 20000, "observation looks like full history, not a delta"
            print(f"        {act.name:<32} done={done} reward={reward} obs={r['state'][:80]!r}")
            if done:
                break

        if not done:
            r = s.post(
                f"{args.addr}/step",
                json={"env_idx": env_idx, "action": "done()"},
                timeout=600,
            ).json()
            reward, done = r["reward"], r["done"]
            print(f"        {'done()':<32} done={done} reward={reward}")

        s.post(f"{args.addr}/close", json={"env_idx": env_idx}, timeout=60)
        status = "OK" if (done and reward >= 1.0) else "FAIL"
        print(f"[smoke] task {task.id}: done={done} reward={reward} -> {status}")
        if status == "FAIL":
            failures.append((task.id, done, reward))

    print()
    if failures:
        print(f"[smoke] {len(failures)} task(s) did not reach reward 1.0: {failures}")
        return 1
    print(f"[smoke] all {min(args.n_tasks, len(tasks))} ground-truth replays scored 1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
