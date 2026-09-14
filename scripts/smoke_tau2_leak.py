#!/usr/bin/env python
"""Stage 1b: check that abandoned episodes do not leak orchestrator threads.

verl closes every env client at the end of each rollout, including trajectories that
simply ran out of max_rounds while tau2's simulation was still going. Each live
AgentGymEnv owns a daemon orchestrator thread parked on a threading.Event, so if close
does not drive that simulation to termination the thread never exits -- one leaked
thread per abandoned episode, every training step.

This opens N episodes, steps each only partway, closes them, and asserts the server's
thread count comes back down.

    python scripts/smoke_tau2_leak.py --addr http://127.0.0.1:36204 --n 5
"""

import argparse
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="http://127.0.0.1:36204")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    s = requests.Session()
    s.trust_env = False

    before = s.get(f"{args.addr}/debug_threads", timeout=60).json()
    print(f"[leak] threads before: {before}")

    env_idxs = []
    for i in range(args.n):
        env_idx = s.post(f"{args.addr}/create", timeout=60).json()
        s.post(f"{args.addr}/reset", json={"env_idx": env_idx, "session_id": i}, timeout=600)
        # One tool call, then walk away mid-episode -- this is the abandoned case.
        s.post(
            f"{args.addr}/step",
            json={"env_idx": env_idx, "action": "list_all_product_types()"},
            timeout=600,
        )
        env_idxs.append(env_idx)

    during = s.get(f"{args.addr}/debug_threads", timeout=60).json()
    print(f"[leak] threads with {args.n} live episodes: {during}")

    for env_idx in env_idxs:
        s.post(f"{args.addr}/close", json={"env_idx": env_idx}, timeout=600)

    after = s.get(f"{args.addr}/debug_threads", timeout=60).json()
    print(f"[leak] threads after close: {after}")

    live = s.get(f"{args.addr}/list_envs", timeout=60).json()
    print(f"[leak] slots still registered: {live}")

    ok = True
    if after["orchestrator_threads"] > before["orchestrator_threads"]:
        print(
            f"[leak] FAIL: {after['orchestrator_threads'] - before['orchestrator_threads']}"
            " orchestrator thread(s) leaked"
        )
        ok = False
    if live:
        print(f"[leak] FAIL: env slots not released: {live}")
        ok = False

    print("[leak] ok" if ok else "[leak] FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
