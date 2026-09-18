#!/usr/bin/env python3
"""Prompting / ReAct baseline on ALFWorld, over an OpenAI-compatible endpoint.

This is the *untrained* baseline: no checkpoint, no weights, just the
environment's own ReAct prompt driven turn by turn. It covers both hosted
models (``--model gpt-5-mini``) and a local base model served by
``scripts/run_vllm_openai_server.sh``, so ``Qwen2.5-7B-Instruct`` prompted and
the same model after GRPO are measured through one code path.

Settings are aligned with ``scripts/batch_eval_alfworld.sh``:
  * test split from ``AgentGym/agentenv-alfworld/configs/mappings_test.json``
  * max_round = 30, per-turn max_tokens = 200, temperature = 1, top_p = 1
  * env timeout = 2400s
  * the same ``AlfWorldEnvClient`` from ``agentenv.envs``, so the conversation
    (``conversation_start`` + observation/action loop) is byte-identical to the
    training rollout in ``src/verl/workers/rollout/``. The prompt itself lives in
    ``AgentGym/agentenv/agentenv/envs/alfworld.py`` under
    ``AlfWorldAdapter.conversation_start_dict[ActionFormat.REACT]`` -- it is not
    restated here, so there is exactly one copy of it in the repo.

The report aggregates per-trajectory success into the six ALFWorld task
families plus the overall mean:
    Pick   = pick_and_place_simple
    Look   = look_at_obj_in_light
    Clean  = pick_clean_then_place_in_recep
    Heat   = pick_heat_then_place_in_recep
    Cool   = pick_cool_then_place_in_recep
    Pick2  = pick_two_obj_and_place

Concurrency is via threads -- each worker holds its own AlfWorldEnvClient and a
shared OpenAI client. Workers round-robin across the comma-separated
``--env-addrs`` (matches the multi-server cluster from
``scripts/run_alfworld_env_service.sh``, base port 36001).

Resume: per-item JSONs are written to ``--output-dir``. Re-runs skip ids that
already have a JSON unless ``--overwrite`` is set.

Usage::

    # hosted model
    OPENAI_API_KEY=sk-... python scripts/eval_alfworld_openai.py \\
        --model gpt-5-mini \\
        --env-addrs http://127.0.0.1:36001,http://127.0.0.1:36002 \\
        --output-dir runs/alfworld_gpt5mini --concurrency 16

    # local base model, served by scripts/run_vllm_openai_server.sh
    python scripts/eval_alfworld_openai.py \\
        --model Qwen2.5-7B-Instruct --api-key EMPTY \\
        --base-url http://127.0.0.1:8100/v1 \\
        --env-addrs http://127.0.0.1:36001 \\
        --output-dir runs/alfworld_qwen7b_prompting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
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
from agentenv.envs import AlfWorldEnvClient  # noqa: E402

try:
    from openai import BadRequestError, OpenAI, RateLimitError  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "openai package is required: pip install --upgrade openai"
    ) from exc


# ---------------------------------------------------------------------------
# Task-type -> reporting bucket. Mirrors the six ALFWorld task families.
TASK_TYPE_TO_BUCKET = {
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick2",
}
BUCKET_ORDER = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]

# Defaults aligned with scripts/batch_eval_alfworld.sh.
DEFAULT_MAPPING_FILE = (
    REPO_ROOT / "AgentGym" / "agentenv-alfworld" / "configs" / "mappings_test.json"
)
DEFAULT_TEST_FILE = REPO_ROOT / "data" / "test" / "alfworld_test.json"
DEFAULT_ENV_ADDRS = "http://127.0.0.1:36001"
DEFAULT_MAX_ROUNDS = 30
DEFAULT_MAX_TOKENS_PER_TURN = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT = 2400


# ---------------------------------------------------------------------------
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
        # Models that accept ``max_completion_tokens`` + ``reasoning_effort``
        # via the OpenAI-compatible API. GPT-5 / o-series additionally reject
        # non-default temperature / top_p; the Gemini 2.5 family is routed here
        # too so we exercise its thinking budget rather than falling back to
        # the no-reasoning code path.
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
        # max_completion_tokens is a HARD cap that counts hidden reasoning
        # tokens too. If too small the model burns the budget on reasoning and
        # returns an empty string -- which the env then scores as an invalid
        # action, so a mis-set budget reads as a bad baseline rather than as a
        # configuration error. Use a larger budget plus a low reasoning_effort
        # so the visible answer still fits in roughly ``max_tokens``.
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
            # 400s are deterministic for this prompt -- moderation, schema, a
            # context overflow. Retrying replays the same failure; let the
            # caller end the trajectory instead.
            print(f"[api 400 -- no retry] {type(exc).__name__}: {exc}", flush=True)
            raise
        except RateLimitError as exc:
            # 429 splits in two. insufficient_quota is NOT transient: it keeps
            # firing until billing changes, so retrying it costs the full
            # backoff ladder on every remaining item of the split.
            msg = str(exc)
            if "insufficient_quota" in msg or "exceeded your current quota" in msg:
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


# ---------------------------------------------------------------------------
def run_trajectory(
    env_client: AlfWorldEnvClient,
    openai_client: OpenAI,
    cfg: GenerationConfig,
    item_idx: int,
    max_rounds: int,
    reinject_actions: bool,
) -> dict[str, Any]:
    """Run one ALFWorld trajectory.

    ``reinject_actions`` decides what each turn's user message contains, and it
    is the one knob that changes what the number means:

    * True  -- every turn comes from ``env_client.observe()``, i.e. observation
      *plus* the contextual ``AVAILABLE ACTIONS`` list. A cold-start API model
      has no other way to learn this env's verb/object vocabulary, so this is
      the honest prompting baseline.
    * False -- only turn 0 carries AVAILABLE ACTIONS (from the initial
      ``observe()``); later turns use the bare ``step.state``. This is what the
      training rollout does, so it is the right setting when the endpoint is
      serving an SFT/RFT/RL'd checkpoint that must stay in-distribution.

    The two are not comparable to each other. ``summary.json`` records which
    was used.
    """

    env_client.reset(item_idx)
    state = env_client.observe()
    convo_start = env_client.conversation_start  # tuple of two ConversationMessage
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
                f"[item {item_idx}] BadRequestError (likely moderation or context "
                f"overflow), terminating trajectory at round {rounds}: {exc}"
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
        reward, done = step.reward, step.done
        next_state = env_client.observe() if reinject_actions else step.state
        conversation.append(
            {"role": "user", "content": next_state, "reasoning_content": None}
        )
        rounds += 1
        if done:
            terminated_by = "env_done"
        elif rounds >= max_rounds:
            terminated_by = "max_rounds"
            break

    return {
        "item_id": f"alfworld_{item_idx}",
        "reward": float(reward),
        "success": 1 if float(reward) == 1.0 else 0,
        "rounds": rounds,
        "terminated_by": terminated_by,
        "conversations": [
            {"role": m["role"], "content": m["content"]} for m in conversation
        ],
    }


# ---------------------------------------------------------------------------
class ClientPool:
    """Thread-local AlfWorldEnvClients, round-robin over env servers."""

    def __init__(self, env_addrs: list[str], timeout: int, action_format: str):
        self._addrs = env_addrs
        self._timeout = timeout
        self._action_format = action_format
        self._local = threading.local()
        self._counter = 0
        self._lock = threading.Lock()

    def get(self) -> AlfWorldEnvClient:
        client = getattr(self._local, "client", None)
        if client is None:
            with self._lock:
                addr = self._addrs[self._counter % len(self._addrs)]
                self._counter += 1
            # data_len is only used for __len__; we drive item ids ourselves.
            client = AlfWorldEnvClient(
                env_server_base=addr,
                data_len=1,
                timeout=self._timeout,
                action_format=self._action_format,
            )
            self._local.client = client
        return client


# ---------------------------------------------------------------------------
def load_mappings(mapping_file: Path) -> OrderedDict[int, dict]:
    if not mapping_file.exists():
        print(
            f"ERROR: ALFWorld mapping file {mapping_file} not found. It ships with "
            f"the AgentGym submodule -- run `git submodule update --init AgentGym`.",
            file=sys.stderr,
        )
        sys.exit(1)
    with mapping_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out: OrderedDict[int, dict] = OrderedDict()
    for r in rows:
        out[int(r["item_id"])] = r
    return out


def load_test_ids(test_file: Path, mappings: OrderedDict[int, dict]) -> list[int]:
    """Item ids to evaluate, in order.

    ``data/test/alfworld_test.json`` is generated per checkout (see README), so
    fall back to the mapping file's own ids -- the same derivation
    ``scripts/batch_eval_alfworld.sh`` performs when that file is absent. This
    is a fallback, not a silent empty run: an unusable split should fail loudly
    rather than report 0 items as a successful evaluation.
    """
    if not test_file.exists():
        print(
            f"[note] {test_file} not found; deriving the eval order from "
            f"the mapping file ({len(mappings)} items), as batch_eval_alfworld.sh does."
        )
        return list(mappings.keys())
    with test_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    ids = [int(r["item_id"].split("_")[-1]) for r in rows]
    if not ids:
        print(f"ERROR: {test_file} contains no item ids.", file=sys.stderr)
        sys.exit(1)
    return ids


def write_per_item(out_dir: Path, item_idx: int, payload: dict[str, Any]) -> None:
    path = out_dir / f"alfworld_{item_idx}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def already_done(out_dir: Path, item_idx: int) -> dict[str, Any] | None:
    path = out_dir / f"alfworld_{item_idx}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
def aggregate(
    results: dict[int, dict[str, Any]],
    mappings: OrderedDict[int, dict],
) -> dict[str, dict[str, float]]:
    bucket_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item_idx, payload in results.items():
        meta = mappings.get(item_idx)
        if meta is None:
            continue
        family = meta["task_type"].split("-", 1)[0]
        bucket = TASK_TYPE_TO_BUCKET.get(family)
        if bucket is None:
            continue
        bucket_records[bucket].append(
            {"reward": float(payload["reward"]), "success": float(payload["success"])}
        )

    summary: dict[str, dict[str, float]] = {}
    for bucket in BUCKET_ORDER:
        recs = bucket_records.get(bucket, [])
        if recs:
            succ = sum(r["success"] for r in recs) / len(recs)
            score = sum(r["reward"] for r in recs) / len(recs)
        else:
            succ = float("nan")
            score = float("nan")
        summary[bucket] = {"success": succ, "score": score, "count": float(len(recs))}

    all_recs = [r for recs in bucket_records.values() for r in recs]
    if all_recs:
        all_succ = sum(r["success"] for r in all_recs) / len(all_recs)
        all_score = sum(r["reward"] for r in all_recs) / len(all_recs)
    else:
        all_succ = float("nan")
        all_score = float("nan")
    summary["All"] = {
        "success": all_succ,
        "score": all_score,
        "count": float(len(all_recs)),
    }
    return summary


def format_report(summary: dict[str, dict[str, float]]) -> str:
    cols = BUCKET_ORDER + ["All"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    succ_row = (
        "| "
        + " | ".join(
            f"{summary[c]['success'] * 100:.2f}" if summary[c]["count"] else "-"
            for c in cols
        )
        + " |"
    )
    cnt_row = "| " + " | ".join(f"{int(summary[c]['count'])}" for c in cols) + " |"

    return (
        "Success rate (%) by task family:\n"
        f"{header}\n{sep}\n{succ_row}\n"
        "Counts:\n"
        f"{header}\n{sep}\n{cnt_row}\n"
    )


# ---------------------------------------------------------------------------

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
                f"reward={payload['reward']:.2f} success={payload['success']} "
                f"rounds={payload['rounds']}",
                flush=True,
            )


# ---------------------------------------------------------------------------
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
        help="Comma-separated ALFWorld env server URLs (run_alfworld_env_service.sh)",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DEFAULT_MAPPING_FILE,
        help="ALFWorld test split mapping (item_id, task_type, task_id)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help="JSON list with item_id strings; falls back to --mapping-file order",
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
        help=(
            "Total max_completion_tokens for reasoning models (gpt-5*/o-series). "
            "Counts hidden reasoning + visible output, so set it comfortably "
            "above --max-tokens to leave room for the answer."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default="minimal",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", ""],
        help=(
            "reasoning_effort for reasoning models. Pick the lowest your model "
            "supports. Empty string omits the kwarg entirely."
        ),
    )
    parser.add_argument(
        "--no-reinject-actions",
        dest="reinject_actions",
        action="store_false",
        default=True,
        help=(
            "Don't append AVAILABLE ACTIONS to every turn; only the initial reset "
            "observation carries them. Use this when the endpoint serves a trained "
            "checkpoint, to match the training rollout. See run_trajectory."
        ),
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

    mappings = load_mappings(args.mapping_file)
    test_ids = load_test_ids(args.test_file, mappings)
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
        f"action_format={args.action_format} reinject_actions={args.reinject_actions}",
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
            reinject_actions=args.reinject_actions,
        ),
        args.output_dir,
        args.concurrency,
        results,
    )
    elapsed = time.time() - start
    summary = aggregate(results, mappings)
    print(f"\n==== EVALUATION (model={args.model}) elapsed={elapsed:.1f}s ====")
    print(format_report(summary))

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "elapsed_sec": elapsed,
                "total_items": len(test_ids),
                "completed_items": len(results),
                "max_rounds": args.max_rounds,
                "max_tokens_per_turn": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "action_format": args.action_format,
                "reinject_actions": args.reinject_actions,
                "env_addrs": env_addrs,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
