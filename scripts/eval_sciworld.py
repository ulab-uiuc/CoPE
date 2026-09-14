#!/usr/bin/env python3
"""Evaluate a local vLLM model on SciWorld.

This script is dedicated to local testing using checkpoints. It bypasses all
OpenAI API logic and uses the vLLM library directly for inference.

Features:
  * Uses vLLM for local model inference.
  * Supports SciWorld environment parallelism via multiple env servers.
  * Supports resuming from partially completed runs.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Setup path for internal packages
REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTENV_PKG = REPO_ROOT / "AgentGym" / "agentenv"
if AGENTENV_PKG.exists() and str(AGENTENV_PKG) not in sys.path:
    sys.path.insert(0, str(AGENTENV_PKG))

from agentenv.envs import SciworldEnvClient  # noqa: E402

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vllm is not installed. Please install it with 'pip install vllm'")
    sys.exit(1)

# Defaults
DEFAULT_TEST_FILE = (
    REPO_ROOT / "AgentItemId" / "test" / "sciworld_test.json"
)
DEFAULT_MAX_ROUNDS = 30
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT = 2400

# ---------------------------------------------------------------------------
class LocalModel:
    """Wrapper for vLLM model to handle thread-safe inference."""
    def __init__(self, model_path: str, tp: int = 1, gpu_util: float = 0.9):
        self._ensure_weights(model_path)
        print(f"[vllm] Loading model from {model_path} (tp={tp}, util={gpu_util})...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._lock = threading.Lock()

    def _ensure_weights(self, model_path: str):
        """Check for weights and try to merge shards if missing."""
        path = Path(model_path)
        weight_files = ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"]
        if any((path / f).exists() for f in weight_files):
            return

        # Weights missing, check for shards in parent directory
        actor_dir = path.parent
        shard_files = list(actor_dir.glob("model_world_size_*_rank_0.pt"))
        if not shard_files:
            print(f"ERROR: No weights found in {model_path} and no shards found in {actor_dir}")
            return

        print(f"[merge] Weights missing in {model_path}, attempting to merge shards from {actor_dir}...")
        import subprocess
        # Correct path to model_merger.py based on repository structure
        merger_script = REPO_ROOT / "AgentGym-RL" / "scripts" / "model_merger.py"
        if not merger_script.exists():
            print(f"ERROR: Merger script not found at {merger_script}")
            return
            
        try:
            # Force the working directory to the training code dir where the script expects to run
            train_code_dir = REPO_ROOT / "AgentGym-RL"
            subprocess.check_call([
                sys.executable, str(merger_script),
                "--local_dir", str(actor_dir)
            ], cwd=str(train_code_dir))
            print("[merge] Successfully merged weights.")
        except Exception as e:
            print(f"ERROR: Failed to merge weights: {e}")
            return # Exit if merging failed to avoid vLLM error later

        # Re-verify weight existence after merge attempt
        if not any((path / f).exists() for f in weight_files):
            print(f"ERROR: Weights still missing in {model_path} after merge attempt.")
            return

    def generate(self, messages: list[dict[str, str]], sampling_params: SamplingParams) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with self._lock:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

# ---------------------------------------------------------------------------
def run_trajectory(
    env_client: SciworldEnvClient,
    model: LocalModel,
    sampling_params: SamplingParams,
    item_idx: int,
    max_rounds: int,
) -> dict[str, Any]:
    """Run one SciWorld trajectory."""
    reset_info = env_client.reset(item_idx)
    state = env_client.observe()
    
    # Initialize conversation from environment starting point
    convo_start = env_client.conversation_start 
    conversation = [
        {"role": "user", "content": convo_start[0]["value"]},
        {"role": "assistant", "content": convo_start[1]["value"]},
        {"role": "user", "content": state},
    ]

    reward = 0.0
    done = False
    rounds = 0
    terminated_by = "max_rounds"
    
    while not done and rounds < max_rounds:
        try:
            generated = model.generate(conversation, sampling_params)
            generated = generated.strip()
        except Exception as exc:
            print(f"[item {item_idx}] Inference failed: {exc}")
            terminated_by = "model_error"
            break

        conversation.append({"role": "assistant", "content": generated})
        
        step = env_client.step(generated)
        reward, done = step.reward, step.done
        
        # Get observation
        state_with_actions = env_client.observe()
        conversation.append({"role": "user", "content": state_with_actions})
        
        rounds += 1
        if done:
            terminated_by = "env_done"

    return {
        "item_id": f"sciworld_{item_idx}",
        "reward": float(reward),
        "success": 1 if float(reward) >= 1.0 else 0, # SciWorld score is 0-1.0 or 0-100? Assuming 0-1.0 from StepOutput
        "rounds": rounds,
        "terminated_by": terminated_by,
        "conversations": conversation,
        "task_description": reset_info.get("task_description", ""),
    }

# ---------------------------------------------------------------------------
class EnvPool:
    """Thread-local SciworldEnvClients round-robining across servers."""
    def __init__(self, env_addrs: list[str], timeout: int):
        self._addrs = env_addrs
        self._timeout = timeout
        self._local = threading.local()
        self._counter = 0
        self._lock = threading.Lock()

    def get(self) -> SciworldEnvClient:
        client = getattr(self._local, "client", None)
        if client is None:
            with self._lock:
                addr = self._addrs[self._counter % len(self._addrs)]
                self._counter += 1
            client = SciworldEnvClient(env_server_base=addr, data_len=1, timeout=self._timeout)
            self._local.client = client
        return client

# ---------------------------------------------------------------------------
def load_test_ids(test_file: Path) -> list[int]:
    if not test_file.exists():
        print(f"Warning: {test_file} not found.")
        return []
    with test_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    
    ids = []
    for r in rows:
        if "official_goal_idx" in r:
            ids.append(int(r["official_goal_idx"]))
        else:
            ids.append(int(r["item_id"].split("_")[-1]))
    return ids

def aggregate(results: dict[int, dict[str, Any]]) -> dict[str, dict[str, float]]:
    all_recs = list(results.values())
    if all_recs:
        all_succ = sum(r["success"] for r in all_recs) / len(all_recs)
        all_score = sum(r["reward"] for r in all_recs) / len(all_recs)
    else:
        all_succ = all_score = float("nan")
    
    summary = {
        "All": {"success": all_succ, "score": all_score, "count": len(all_recs)}
    }
    return summary

def format_report(summary: dict[str, dict[str, float]]) -> str:
    cols = ["All"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    succ_row = "| " + " | ".join(f"{summary[c]['success']*100:.2f}" if summary[c]['count'] else "-" for c in cols) + " |"
    score_row = "| " + " | ".join(f"{summary[c]['score']:.4f}" if summary[c]['count'] else "-" for c in cols) + " |"
    return f"SciWorld Evaluation Results:\n{header}\n{sep}\nSuccess Rate (%): {succ_row}\nAverage Score: {score_row}"

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SciWorld with local vLLM.")
    parser.add_argument("--model-path", required=True, help="Path to local checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env-addrs", default="http://127.0.0.1:36101", help="Comma-separated env URLs")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-util", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_addrs = [s.strip() for s in args.env_addrs.split(",") if s.strip()]
    
    test_ids = load_test_ids(DEFAULT_TEST_FILE)
    if args.limit > 0:
        test_ids = test_ids[:args.limit]

    # Resolve absolute path to model
    abs_model_path = str(Path(args.model_path).resolve())
    model = LocalModel(abs_model_path, tp=args.tp, gpu_util=args.gpu_util)
    sampling_params = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=args.max_tokens)
    pool = EnvPool(env_addrs, timeout=DEFAULT_TIMEOUT)

    results = {}
    pending = []
    for idx in test_ids:
        out_path = args.output_dir / f"sciworld_{idx}.json"
        if out_path.exists() and not args.overwrite:
            with out_path.open("r") as f:
                results[idx] = json.load(f)
        else:
            pending.append(idx)

    print(f"Total: {len(test_ids)}, Cached: {len(results)}, Todo: {len(pending)}")

    def worker(idx):
        try:
            env = pool.get()
            payload = run_trajectory(env, model, sampling_params, idx, args.max_rounds)
            out_path = args.output_dir / f"sciworld_{idx}.json"
            with out_path.open("w") as f:
                json.dump(payload, f, indent=2)
            return idx, payload, None
        except Exception as e:
            return idx, None, f"{e}\n{traceback.format_exc()}"

    start_time = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(worker, idx) for idx in pending]
            for i, fut in enumerate(as_completed(futures), 1):
                idx, payload, err = fut.result()
                if err:
                    print(f"[{i}/{len(pending)}] Item {idx} FAILED: {err}")
                else:
                    results[idx] = payload
                    print(f"[{i}/{len(pending)}] Item {idx} reward={payload['reward']} rounds={payload['rounds']}")

    elapsed = time.time() - start_time
    summary = aggregate(results)
    print(f"\n==== EVAL COMPLETE ({elapsed:.1f}s) ====")
    print(format_report(summary))
    
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump({"summary": summary, "elapsed": elapsed, "model": args.model_path}, f, indent=2)

if __name__ == "__main__":
    main()
