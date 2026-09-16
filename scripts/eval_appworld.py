#!/usr/bin/env python
"""AppWorld checkpoint evaluation.

Mirrors the training rollout (verl agent_vllm_rollout) so evaluation and training
measure the same thing:
  - imports agentenv's AppWorldEnvClient directly, so the system prompt, code
    extraction and env protocol are the ones training used, not a reimplementation
  - generates in synchronous rounds like training: each round feeds the prompts of
    every still-running trajectory to vLLM together
  - gives every trajectory its own env server process. AppWorld's supervisor
    "active task" is process-global; episodes sharing a process mark each other done.

Environment:
  APPWORLD_ROOT    AppWorld data directory (required to count tasks per split)
  APPWORLD_PYTHON  python of the agentenv-appworld env (same purpose)

Usage:
  python eval_appworld.py --model-path <hf_dir> --split test_normal \
      --env-addrs "http://127.0.0.1:36301,..." --output-dir runs/eval/step50_normal
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTENV_PKG = REPO_ROOT / "AgentGym" / "agentenv"
if AGENTENV_PKG.exists() and str(AGENTENV_PKG) not in sys.path:
    sys.path.insert(0, str(AGENTENV_PKG))

DEFAULT_MAX_ROUNDS = 30
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.4   # G2PO validation temperature
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_MODEL_LEN = 32768   # Qwen2.5-14B max_position_embeddings (training forces 34816; eval stays in range)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--split", required=True,
                   choices=["train", "dev", "test_normal", "test_challenge"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--env-addrs", required=True,
                   help="comma-separated env server URLs; must be >= concurrency")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N tasks; 0 = all")
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"{summary_path} exists; pass --overwrite to replace it")
        return 0

    from agentenv.envs.appworld import AppWorldEnvClient, _APPWORLD_SYSTEM
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    addrs = [a.strip() for a in args.env_addrs.split(",") if a.strip()]
    n_tasks = _split_size(args.split)
    if args.limit:
        n_tasks = min(n_tasks, args.limit)
    conc = min(args.concurrency, len(addrs), n_tasks)
    assert conc >= 1, "concurrency is 0"
    print(f"[eval] split={args.split} tasks={n_tasks} concurrency={conc} "
          f"({len(addrs)} env servers, one per trajectory)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              dtype="bfloat16", enforce_eager=True, trust_remote_code=True)
    sp = SamplingParams(temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens)

    results = []
    t0 = time.time()
    # batches of `conc` trajectories, generated in synchronous rounds like the training rollout
    for base in range(0, n_tasks, conc):
        ids = list(range(base, min(base + conc, n_tasks)))
        results += _run_batch(ids, addrs, llm, tok, sp, args,
                              AppWorldEnvClient, _APPWORLD_SYSTEM)
        done = len(results)
        succ = sum(1 for r in results if r["reward"] > 0)
        el = time.time() - t0
        print(f"[eval] {done}/{n_tasks}  success {succ} ({succ/done:.1%})  "
              f"elapsed {el/60:.1f} min  eta {(el/done)*(n_tasks-done)/60:.1f} min",
              flush=True)

    succ = sum(1 for r in results if r["reward"] > 0)
    summary = {
        "model_path": str(args.model_path),
        "split": args.split,
        "num_tasks": len(results),
        "num_success": succ,
        "success_rate": succ / len(results) if results else 0.0,
        "temperature": args.temp,
        "max_rounds": args.max_rounds,
        "max_tokens_per_turn": args.max_tokens,
        "mean_rounds": sum(r["rounds"] for r in results) / len(results) if results else 0,
        "hit_round_cap": sum(1 for r in results if r["rounds"] >= args.max_rounds),
        "env_errors": sum(1 for r in results if r.get("error")),
        "elapsed_min": (time.time() - t0) / 60,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.output_dir / "trajectories.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("[eval] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def _require_env(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(f"ERROR: set {name} (see the module docstring)")
    return val


def _split_size(split):
    """Task count for a split, read from the AppWorld dataset itself (not an eval json,
    so the denominator cannot drift from what AppWorld considers the split)."""
    import subprocess
    code = (
        "import os;"
        "from appworld import load_task_ids;print(len(load_task_ids('%s')))" % split
    )
    out = subprocess.run(
        [_require_env("APPWORLD_PYTHON"), "-c", code],
        capture_output=True, text=True, cwd=_require_env("APPWORLD_ROOT"))
    return int(out.stdout.strip().splitlines()[-1])


def _run_batch(ids, addrs, llm, tok, sp, args, ClientCls, system_prompt):
    """Advance one batch of trajectories in synchronous rounds, like the verl rollout loop."""
    n = len(ids)
    clients, convs, done, reward, err = [], [], [], [], []
    for k, tid in enumerate(ids):
        c = ClientCls(env_server_base=addrs[k % len(addrs)], data_len=1, timeout=600)
        clients.append(c)
        try:
            instr = c.reset(tid)
        except Exception as e:
            instr = ""
            err.append(str(e)[:200])
        convs.append([
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": instr},
        ])
        done.append(False)
        reward.append(0.0)
    err += [None] * (n - len(err))

    rounds_used = [0] * n
    for _ in range(args.max_rounds):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        prompts = [tok.apply_chat_template(convs[i], tokenize=False,
                                           add_generation_prompt=True) for i in active]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]

        def one(j):
            i = active[j]
            convs[i].append({"role": "assistant", "content": texts[j]})
            rounds_used[i] += 1
            try:
                so = clients[i].step(texts[j])
                convs[i].append({"role": "user", "content": so.state})
                reward[i] = so.reward
                return so.done
            except Exception as e:
                err[i] = str(e)[:200]
                return True

        with ThreadPoolExecutor(max_workers=len(active)) as ex:
            flags = list(ex.map(one, range(len(active))))
        for j, f in enumerate(flags):
            if f:
                done[active[j]] = True

    for c in clients:
        try:
            c.close()
        except Exception:
            pass

    return [{"task_index": ids[i], "reward": reward[i], "rounds": rounds_used[i],
             "error": err[i], "conversations": convs[i]} for i in range(n)]


if __name__ == "__main__":
    sys.exit(main())
