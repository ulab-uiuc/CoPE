#!/usr/bin/env python3
"""Prompting / ReAct baseline on WebShop, over an OpenAI-compatible endpoint.

The counterpart to ``scripts/eval_webshop.py``: that one loads a checkpoint
into vLLM in-process, this one drives any OpenAI-compatible endpoint -- a
hosted model, or a local base model served by
``scripts/run_vllm_openai_server.sh``. Nothing is trained; the number it
produces is what the environment's own ReAct prompt gets out of the model.

Settings:
  * max_round = 15, per-turn max_tokens = 256, temperature = 1, top_p = 1
  * env timeout = 2400s
  * the same ``WebshopEnvClient`` from ``agentenv.envs``, so the conversation
    (``conversation_start`` + observation/action loop) is byte-identical to the
    training rollout, including the ``search[...]``/``click[...]`` action
    grammar. The prompt lives in ``AgentGym/agentenv/agentenv/envs/webshop.py``
    under ``WebshopAdapter.conversation_start_dict[ActionFormat.REACT]``.

The report prints two scalars, matching ``Evaluator.eval`` in
``agentenv.controller.utils``:
    Score   = mean reward over completed items (WebShop's reward is the
              continuous 0-1 product-match score)
    Succ    = fraction of items with reward == 1.0 (perfect match)

Concurrency is via threads -- each worker holds its own ``WebshopEnvClient``
and shares one OpenAI client. Workers round-robin across the comma-separated
``--env-addrs`` (``run_webshop_env_service.sh`` starts at port 36101).

Resume: per-item JSONs are written to ``--output-dir``. Re-runs skip ids that
already have a JSON unless ``--overwrite`` is set.

Usage::

    # hosted model
    OPENAI_API_KEY=sk-... python scripts/eval_webshop_openai.py \\
        --model gpt-5-mini \\
        --env-addrs http://127.0.0.1:36101 \\
        --output-dir runs/webshop_gpt5mini --concurrency 16

    # local base model, served by scripts/run_vllm_openai_server.sh
    python scripts/eval_webshop_openai.py \\
        --model Qwen2.5-7B-Instruct --api-key EMPTY \\
        --base-url http://127.0.0.1:8100/v1 \\
        --env-addrs http://127.0.0.1:36101 \\
        --num-items 500 \\
        --output-dir runs/webshop_qwen7b_prompting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Setup path for internal packages
REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTENV_PKG = REPO_ROOT / "AgentGym" / "agentenv"
if AGENTENV_PKG.exists() and str(AGENTENV_PKG) not in sys.path:
    sys.path.insert(0, str(AGENTENV_PKG))

from agentenv.controller.types import APIConversationMessage  # noqa: E402
from agentenv.envs import WebshopEnvClient  # noqa: E402

try:
    from openai import BadRequestError, OpenAI, RateLimitError  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "openai package is required: pip install --upgrade openai"
    ) from exc


DEFAULT_TEST_FILE = REPO_ROOT / "data" / "test" / "webshop_test.json"
DEFAULT_ENV_ADDRS = "http://127.0.0.1:36101"
DEFAULT_MAX_ROUNDS = 15
DEFAULT_MAX_TOKENS_PER_TURN = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT = 2400


@dataclass
class GenerationConfig:
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    api_retry_max: int
    api_retry_backoff: float
    reasoning_max_tokens: int
    reasoning_effort: str

    def is_reasoning_model(self) -> bool:
        m = self.model.lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
            or m.startswith("gemini-")
        )


def chat_complete(
    client: OpenAI,
    cfg: GenerationConfig,
    messages: list[dict[str, str]],
) -> str:
    """One chat completion with bounded retry. Returns the assistant text."""

    base_kwargs: dict[str, Any] = {"model": cfg.model, "messages": messages}
    if cfg.is_reasoning_model():
        # max_completion_tokens counts hidden reasoning tokens too, so a budget
        # sized like --max-tokens returns an empty string that the env scores as
        # an invalid action -- a config error that reads as a bad baseline.
        base_kwargs["max_completion_tokens"] = cfg.reasoning_max_tokens
        if cfg.reasoning_effort:
            base_kwargs["reasoning_effort"] = cfg.reasoning_effort
    else:
        base_kwargs["max_tokens"] = cfg.max_tokens
        base_kwargs["temperature"] = cfg.temperature
        base_kwargs["top_p"] = cfg.top_p

    last_err: Exception | None = None
    for attempt in range(1, cfg.api_retry_max + 1):
        try:
            resp = client.chat.completions.create(**base_kwargs)
            content = resp.choices[0].message.content
            return content if content is not None else ""
        except BadRequestError as exc:
            # Deterministic for this prompt; retrying replays the same failure.
            print(f"[api 400 -- no retry] {type(exc).__name__}: {exc}", flush=True)
            raise
        except RateLimitError as exc:
            msg = str(exc)
            if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                # Not transient -- it fires until billing changes, so retrying it
                # costs the full backoff ladder on every remaining item.
                print(
                    f"[api quota -- no retry] {type(exc).__name__}: {msg[:200]}",
                    flush=True,
                )
                raise
            last_err = exc
            sleep_s = cfg.api_retry_backoff * (2 ** (attempt - 1))
            print(
                f"[api retry {attempt}/{cfg.api_retry_max}] {type(exc).__name__}: {msg[:200]}",
                flush=True,
            )
            time.sleep(min(sleep_s, 60.0))
            continue
        except Exception as exc:  # pragma: no cover - network-dependent
            last_err = exc
            sleep_s = cfg.api_retry_backoff * (2 ** (attempt - 1))
            print(
                f"[api retry {attempt}/{cfg.api_retry_max}] {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(min(sleep_s, 60.0))
    raise RuntimeError(
        f"OpenAI call failed after {cfg.api_retry_max} retries: {last_err}"
    )


def run_trajectory(
    env_client: WebshopEnvClient,
    openai_client: OpenAI,
    cfg: GenerationConfig,
    item_idx: int,
    max_rounds: int,
) -> dict[str, Any]:
    """Run one WebShop trajectory."""

    env_client.reset(item_idx)
    state = env_client.observe()
    convo_start = env_client.conversation_start
    conversation: list[APIConversationMessage] = [
        {"role": "user", "content": convo_start[0]["value"], "reasoning_content": None},
        {
            "role": "assistant",
            "content": convo_start[1]["value"],
            "reasoning_content": None,
        },
        {"role": "user", "content": state, "reasoning_content": None},
    ]

    reward = 0.0
    done = False
    rounds = 0
    terminated_by = "max_rounds"
    while not done:
        api_messages = [
            {"role": m["role"], "content": m["content"]} for m in conversation
        ]
        try:
            generated = chat_complete(openai_client, cfg, api_messages)
        except BadRequestError as exc:
            print(
                f"[item {item_idx}] BadRequestError, terminating trajectory "
                f"at round {rounds}: {exc}"
            )
            terminated_by = "bad_request"
            break
        except RateLimitError as exc:
            print(
                f"[item {item_idx}] insufficient_quota, terminating trajectory "
                f"at round {rounds}: {exc}"
            )
            terminated_by = "insufficient_quota"
            break
        except Exception as exc:
            print(
                f"[item {item_idx}] generate failed, terminating trajectory "
                f"at round {rounds}: {exc}"
            )
            terminated_by = "api_error"
            break
        conversation.append(
            {"role": "assistant", "content": generated, "reasoning_content": None}
        )
        step = env_client.step(generated)
        state, reward, done = step.state, step.reward, step.done
        conversation.append(
            {"role": "user", "content": state, "reasoning_content": None}
        )
        rounds += 1
        if done:
            terminated_by = "env_done"
        elif rounds >= max_rounds:
            terminated_by = "max_rounds"
            break

    return {
        "item_id": f"webshop_{item_idx}",
        "reward": float(reward),
        "success": 1 if float(reward) == 1.0 else 0,
        "rounds": rounds,
        "terminated_by": terminated_by,
        "conversations": [
            {"role": m["role"], "content": m["content"]} for m in conversation
        ],
    }


class ClientPool:
    """Thread-local WebshopEnvClients, round-robin over env servers."""

    def __init__(self, env_addrs: list[str], timeout: int, action_format: str):
        self._addrs = env_addrs
        self._timeout = timeout
        self._action_format = action_format
        self._local = threading.local()
        self._counter = 0
        self._lock = threading.Lock()

    def get(self) -> WebshopEnvClient:
        client = getattr(self._local, "client", None)
        if client is None:
            with self._lock:
                addr = self._addrs[self._counter % len(self._addrs)]
                self._counter += 1
            # data_len is only used for __len__; we drive item ids ourselves.
            client = WebshopEnvClient(
                env_server_base=addr,
                data_len=1,
                timeout=self._timeout,
                action_format=self._action_format,
            )
            self._local.client = client
        return client


def load_test_ids(test_file: Path, num_items: int) -> list[int]:
    """Item ids to evaluate, in order.

    ``webshop_i`` is a direct index into the env server's own goal list, so
    ``--num-items N`` (ids 0..N-1) is a usable split when the generated
    ``data/test/webshop_test.json`` is not present -- item-id files are built
    per checkout, see the README. One of the two must resolve: failing loudly
    beats reporting an empty run as a successful evaluation.
    """
    if num_items > 0:
        return list(range(num_items))
    if not test_file.exists():
        print(
            f"ERROR: test id file {test_file} not found. Pass --test-file, or "
            f"--num-items N to evaluate webshop_0..N-1 (item-id files are "
            f"generated per checkout, see README).",
            file=sys.stderr,
        )
        sys.exit(1)
    with test_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    ids = [int(r["item_id"].split("_")[-1]) for r in rows]
    if not ids:
        print(f"ERROR: {test_file} contains no item ids.", file=sys.stderr)
        sys.exit(1)
    return ids


def write_per_item(out_dir: Path, item_idx: int, payload: dict[str, Any]) -> None:
    path = out_dir / f"webshop_{item_idx}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def already_done(out_dir: Path, item_idx: int) -> dict[str, Any] | None:
    path = out_dir / f"webshop_{item_idx}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def aggregate(results: dict[int, dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"score": float("nan"), "succ": float("nan"), "count": 0.0}
    rewards = [float(r["reward"]) for r in results.values()]
    succs = [float(r["success"]) for r in results.values()]
    return {
        "score": sum(rewards) / len(rewards),
        "succ": sum(succs) / len(succs),
        "count": float(len(rewards)),
    }



def partition_cached(
    test_ids: list[int], out_dir: Path, overwrite: bool
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Split the split into what is already on disk and what still has to run."""
    results: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    for idx in test_ids:
        cached = None if overwrite else already_done(out_dir, idx)
        if cached is not None:
            results[idx] = cached
        else:
            pending.append(idx)
    return results, pending


def run_pending(
    pending: list[int],
    run_one: Callable[[int], dict[str, Any]],
    out_dir: Path,
    concurrency: int,
    results: dict[int, dict[str, Any]],
) -> None:
    """Run ``pending`` through a thread pool, writing each item as it lands.

    A failed item is reported and skipped rather than raising: the per-item JSON
    is the unit of work, so the rest of the split still completes and a re-run
    picks up only what is missing.
    """

    def worker(item_idx: int) -> tuple[int, dict[str, Any] | None, str | None]:
        try:
            payload = run_one(item_idx)
            write_per_item(out_dir, item_idx, payload)
            return item_idx, payload, None
        except Exception as exc:
            return (
                item_idx,
                None,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, idx) for idx in pending]
        done_count = 0
        for fut in as_completed(futures):
            idx, payload, err = fut.result()
            done_count += 1
            if err is not None:
                print(f"[{done_count}/{len(pending)}] item {idx} FAILED: {err}", flush=True)
                continue
            results[idx] = payload
            print(
                f"[{done_count}/{len(pending)}] item {idx} "
                f"reward={payload['reward']:.3f} success={payload['success']} "
                f"rounds={payload['rounds']}",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model name, e.g. gpt-5-mini")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (defaults to $OPENAI_API_KEY; use EMPTY for a local vLLM server)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--env-addrs",
        default=DEFAULT_ENV_ADDRS,
        help="Comma-separated WebShop env server URLs (run_webshop_env_service.sh)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help="JSON list with item_id strings (defines eval order)",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=0,
        help="Evaluate webshop_0..N-1 instead of --test-file (0 = use the file)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS_PER_TURN)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--env-timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--api-retry-max", type=int, default=6)
    parser.add_argument("--api-retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=4096,
        help="Total max_completion_tokens for reasoning models (gpt-5*/o-series).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="minimal",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", ""],
        help="reasoning_effort for reasoning models. Empty string omits the kwarg.",
    )
    parser.add_argument(
        "--action-format",
        default="react",
        choices=["react", "function_calling", "code_as_action"],
        help=(
            "Prompt/action protocol, selecting which conversation_start the env "
            "adapter serves and how the reply is parsed. react is the baseline "
            "and the format training uses; the other two are wired through but "
            "not part of any reported number."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run items that already have a JSON in --output-dir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of items (0 = all). Useful for smoke tests.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "ERROR: --api-key not provided and OPENAI_API_KEY is unset",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_addrs = [s.strip() for s in args.env_addrs.split(",") if s.strip()]
    if not env_addrs:
        print("ERROR: --env-addrs is empty", file=sys.stderr)
        return 2

    test_ids = load_test_ids(args.test_file, args.num_items)
    if args.limit > 0:
        test_ids = test_ids[: args.limit]

    cfg = GenerationConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        api_retry_max=args.api_retry_max,
        api_retry_backoff=args.api_retry_backoff,
        reasoning_max_tokens=args.reasoning_max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    if cfg.is_reasoning_model() and (
        args.temperature != DEFAULT_TEMPERATURE or args.top_p != DEFAULT_TOP_P
    ):
        print(
            f"[note] {args.model} is a reasoning model; ignoring custom temperature/top_p",
            flush=True,
        )

    openai_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    pool = ClientPool(
        env_addrs, timeout=args.env_timeout, action_format=args.action_format
    )

    results, pending = partition_cached(test_ids, args.output_dir, args.overwrite)

    print(
        f"Eval plan: model={args.model} concurrency={args.concurrency} "
        f"env_servers={len(env_addrs)} total={len(test_ids)} "
        f"cached={len(test_ids) - len(pending)} todo={len(pending)} "
        f"action_format={args.action_format}",
        flush=True,
    )

    start = time.time()
    run_pending(
        pending,
        lambda item_idx: run_trajectory(
            env_client=pool.get(),
            openai_client=openai_client,
            cfg=cfg,
            item_idx=item_idx,
            max_rounds=args.max_rounds,
        ),
        args.output_dir,
        args.concurrency,
        results,
    )
    elapsed = time.time() - start
    summary = aggregate(results)
    print(f"\n==== EVALUATION (model={args.model}) elapsed={elapsed:.1f}s ====")
    print(f"Score (mean reward):  {summary['score']:.4f}")
    print(f"Success rate:         {summary['succ']:.4f}")
    print(f"Items completed:      {int(summary['count'])} / {len(test_ids)}")
    print(
        "METRICS_JSON:",
        json.dumps(
            {
                "overall": {"score": summary["score"], "succ": summary["succ"]},
                "count": int(summary["count"]),
            }
        ),
    )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "elapsed_sec": elapsed,
                "total_items": len(test_ids),
                "completed_items": int(summary["count"]),
                "max_rounds": args.max_rounds,
                "max_tokens_per_turn": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "action_format": args.action_format,
                "env_addrs": env_addrs,
                "score": summary["score"],
                "succ": summary["succ"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
