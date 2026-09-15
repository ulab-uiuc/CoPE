#!/usr/bin/env python
"""Evaluate a policy on tau2-bench without training. Used to A/B prompt variants.

Drives the same ReAct loop the verl rollout uses (Tau2EnvClient + Thought/Action), but
against an OpenAI-compatible endpoint instead of an in-process vLLM engine, so a prompt
change can be scored in ~30 minutes rather than by launching a 2-hour GRPO run.

Reports overall solve rate, the per-task distribution (how many tasks are stuck at 0 --
that, not the mean, is what starves GRPO), and a failure-mode breakdown.

    python scripts/eval_tau2.py \\
        --policy-url http://127.0.0.1:38201/v1 --policy-model policy \\
        --env-addrs http://127.0.0.1:36401,http://127.0.0.1:36402 \\
        --n-tasks 74 --k 4 --out eval_strict.json

Run with the training interpreter and the repo's AgentGym on PYTHONPATH.
"""

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

from envs.tau2.tau2_client import Tau2EnvClient

WRITE_TOOLS = ("cancel_pending_order", "modify_pending_order", "modify_user_address",
               "return_delivered_order_items", "exchange_delivered_order_items")


def classify(convo_text: str, solved: bool) -> str:
    if solved:
        return "solved"
    if "transfer_to_human_agents" in convo_text:
        return "transferred (gave up)"
    if not any(t in convo_text for t in ("find_user_id_by_email", "find_user_id_by_name_zip")):
        return "never authenticated"
    if not any(t in convo_text for t in WRITE_TOOLS):
        return "read-only, never wrote"
    return "wrote but wrong"


def run_episode(args, env_addr, item_id):
    session = requests.Session()
    session.trust_env = False
    client = Tau2EnvClient(env_server_base=env_addr, data_len=1, timeout=args.timeout)
    try:
        client.reset(item_id)
        sys_prompt = client.conversation_start[0]["value"]
        msgs = [
            {"role": "user", "content": sys_prompt},
            {"role": "assistant", "content": client.conversation_start[1]["value"]},
            {"role": "user", "content": client.observe()},
        ]
        reward, done = 0.0, False
        for _ in range(args.max_rounds):
            r = session.post(
                f"{args.policy_url}/chat/completions",
                json={"model": args.policy_model, "messages": msgs,
                      "temperature": args.temperature, "max_tokens": args.max_tokens},
                timeout=args.timeout,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or ""
            msgs.append({"role": "assistant", "content": content})
            out = client.step(content)
            reward, done = out.reward, out.done
            msgs.append({"role": "user", "content": out.state})
            if done:
                break
        text = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
        return {"item_id": item_id, "reward": float(reward), "done": bool(done),
                "mode": classify(text, reward >= 1.0), "turns": sum(1 for m in msgs if m["role"] == "assistant")}
    except Exception as e:
        return {"item_id": item_id, "reward": 0.0, "done": False,
                "mode": f"error: {type(e).__name__}", "turns": 0}
    finally:
        try:
            client.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-url", required=True)
    ap.add_argument("--policy-model", default="policy")
    ap.add_argument("--env-addrs", required=True, help="comma-separated env server URLs")
    ap.add_argument("--n-tasks", type=int, default=74)
    ap.add_argument("--k", type=int, default=4, help="rollouts per task")
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default="eval_tau2.json")
    args = ap.parse_args()

    addrs = [a.strip() for a in args.env_addrs.split(",") if a.strip()]
    jobs = [(i, addrs[n % len(addrs)])
            for n, (i, _) in enumerate((i, k) for i in range(args.n_tasks) for k in range(args.k))]

    print(f"[eval] {args.n_tasks} tasks x {args.k} rollouts = {len(jobs)} episodes "
          f"over {len(addrs)} env servers, concurrency={args.concurrency}")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_episode, args, addr, item) for item, addr in jobs]
        for n, f in enumerate(futs, 1):
            results.append(f.result())
            if n % 25 == 0 or n == len(futs):
                solved = sum(1 for r in results if r["reward"] >= 1.0)
                print(f"  {n}/{len(futs)}  solved={solved} ({solved/n:.1%})", flush=True)

    by_task = collections.defaultdict(list)
    for r in results:
        by_task[r["item_id"]].append(r["reward"])

    zero = sum(1 for v in by_task.values() if all(x <= 0 for x in v))
    always = sum(1 for v in by_task.values() if all(x >= 1.0 for x in v))
    # Under GRPO the advantage is the within-group deviation from the group mean, so a
    # group only produces gradient if its rewards differ at all. With a binary reward
    # that means "some solved, some not"; with dense partial credit any spread counts.
    informative = sum(1 for v in by_task.values() if len(set(v)) > 1)
    solved = sum(1 for r in results if r["reward"] >= 1.0)
    mean_r = sum(r["reward"] for r in results) / max(len(results), 1)

    summary = {
        "episodes": len(results),
        "solved": solved,
        "solve_rate": solved / max(len(results), 1),
        "mean_reward": mean_r,
        "tasks": len(by_task),
        "tasks_never_solved": zero,
        "tasks_always_solved": always,
        "tasks_informative": informative,
        "informative_frac": informative / max(len(by_task), 1),
        "modes": dict(collections.Counter(r["mode"] for r in results).most_common()),
    }

    print("\n=== summary ===")
    print(f"solve rate      : {solved}/{len(results)} = {summary['solve_rate']:.1%}")
    print(f"mean reward     : {mean_r:.4f}")
    print(f"tasks all-zero  : {zero}/{len(by_task)}")
    print(f"tasks always    : {always}/{len(by_task)}")
    print(f"tasks INFORMATIVE: {informative}/{len(by_task)} = {summary['informative_frac']:.1%}   <-- GRPO gradient comes only from these")
    print("failure modes   :")
    for k, v in summary["modes"].items():
        print(f"    {k:26s} {v:5d}  {v/len(results):.1%}")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
