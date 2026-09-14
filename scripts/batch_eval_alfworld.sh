#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentgym-rl}"

ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:36001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT}/FinalResults_ALFWorld.jsonl}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-1}"
SKIP_DONE_MODELS="${SKIP_DONE_MODELS:-0}"

MAX_ROUND="${MAX_ROUND:-30}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-200}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-2400}"
N_SAMPLES="${N_SAMPLES:-1}"

# Preferred high-concurrency batch size. Per-model auto-fallback is applied if OOM.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

BASE_3B_MODEL_PATH="${BASE_3B_MODEL_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct}"
BASE_7B_MODEL_PATH="${BASE_7B_MODEL_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
CKPT_ROOT_3B="${CKPT_ROOT_3B:-${ROOT}/checkpoints/alfworld_grpo_qwen2.5_3b_wm_clip_20260420_025703}"
CKPT_ROOT_7B="${CKPT_ROOT_7B:-${ROOT}/checkpoints/alfworld_grpo_qwen2.5_3b_add_20260506_100312}"

EVAL_DATA_DIR="${EVAL_DATA_DIR:-${ROOT}/AgentItemId/test}"
EVAL_TEST_FILE="${EVAL_TEST_FILE:-${EVAL_DATA_DIR}/alfworld_test.json}"

# Keep runtime files out of /home and /tmp.
TMPDIR="${ROOT}/tmp_eval/te"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${ROOT}/tmp_eval/he"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/tmp_eval/xe}"
WANDB_DIR="${ROOT}/tmp_eval/we"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
# Use /tmp/ray or a shorter path to avoid "AF_UNIX path length cannot exceed 107 bytes" error
RAY_TMPDIR="/tmp/ray_$(whoami)"

mkdir -p \
  "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${RAY_TMPDIR}" \
  "$(dirname "${RESULTS_FILE}")" "${EVAL_DATA_DIR}"

if [[ "${OVERWRITE_RESULTS}" == "1" ]]; then
  : > "${RESULTS_FILE}"
fi

should_skip_model() {
  local model_label="$1"
  local results_file="$2"

  if [[ "${SKIP_DONE_MODELS}" != "1" ]]; then
    return 1
  fi
  if [[ ! -s "${results_file}" ]]; then
    return 1
  fi

  python3 - "${results_file}" "${model_label}" <<'PY'
import json
import sys

path, target = sys.argv[1], sys.argv[2]
for raw in open(path, "r", encoding="utf-8"):
    line = raw.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get("model_label") == target:
        print("1")
        break
else:
    print("0")
PY
}

# Build official ALFWorld test split file for verl.main_generation if missing.
if [[ ! -f "${EVAL_TEST_FILE}" ]]; then
  echo "Generating ${EVAL_TEST_FILE} from AgentGym configs..."
  python3 - "${EVAL_TEST_FILE}" "${ROOT}/AgentGym/agentenv-alfworld/configs/mappings_test.json" <<'PY'
import json
import sys
import os

out_path = sys.argv[1]
mapping_path = sys.argv[2]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(mapping_path, "r", encoding="utf-8") as f:
    mappings = json.load(f)
rows = [{"item_id": f"alfworld_{int(m['item_id'])}"} for m in mappings]
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=True)
print(f"Wrote {len(rows)} test ids to {out_path}")
PY
fi

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES is empty." >&2
  exit 1
fi

MODEL_LABELS=()
MODEL_PATHS=()

# MODEL_LABELS+=("base_3b")
# MODEL_PATHS+=("${BASE_3B_MODEL_PATH}")
# MODEL_LABELS+=("base_7b")
# MODEL_PATHS+=("${BASE_7B_MODEL_PATH}")

# mapfile -t CKPTS_3B < <(find "${CKPT_ROOT_3B}" -maxdepth 1 -type d -name 'global_step_*' | sort -V)
mapfile -t CKPTS_7B < <(find "${CKPT_ROOT_7B}" -maxdepth 1 -type d -name 'global_step_*' | sort -V)

# for ckpt_dir in "${CKPTS_3B[@]}"; do
#   step_name="$(basename "${ckpt_dir}")"
#   MODEL_LABELS+=("3b_${step_name}")
#   MODEL_PATHS+=("${ckpt_dir}/actor/huggingface")
# done

for ckpt_dir in "${CKPTS_7B[@]}"; do
  step_name="$(basename "${ckpt_dir}")"
  MODEL_LABELS+=("7b_${step_name}")
  MODEL_PATHS+=("${ckpt_dir}/actor/huggingface")
done

is_7b_model() {
  local label="$1"
  local path="$2"
  local low="${label,,} ${path,,}"
  if [[ "${low}" == *"7b"* ]]; then
    return 0
  fi
  return 1
}

is_oom_log() {
  local log_file="$1"
  grep -Eiq \
    "cuda out of memory|outofmemoryerror|no available memory for the cache blocks|oom|allocation failed|device out of memory" \
    "${log_file}"
}

has_hf_weights() {
  local model_path="$1"
  [[ -f "${model_path}/model.safetensors.index.json" || -f "${model_path}/model.safetensors" || -f "${model_path}/pytorch_model.bin" ]]
}

ensure_model_weights() {
  local model_label="$1"
  local model_path="$2"
  local merge_log="$3"

  if has_hf_weights "${model_path}"; then
    return 0
  fi

  # For checkpoint models, try auto-merge FSDP shards into actor/huggingface.
  local actor_dir
  actor_dir="$(dirname "${model_path}")"
  if ! compgen -G "${actor_dir}/model_world_size_*_rank_0.pt" > /dev/null; then
    echo "Missing model weights for ${model_label}: ${model_path} (and no shard file found under ${actor_dir})" | tee -a "${merge_log}"
    return 1
  fi

  echo "Model weights missing for ${model_label}, start auto-merge from ${actor_dir} ..." | tee -a "${merge_log}"
  if (
    cd "${CODE_DIR}"
    python scripts/model_merger.py --local_dir "${actor_dir}"
  ) >> "${merge_log}" 2>&1; then
    if has_hf_weights "${model_path}"; then
      echo "Auto-merge success for ${model_label}: ${model_path}" | tee -a "${merge_log}"
      return 0
    fi
    echo "Auto-merge finished but weights still missing for ${model_label}: ${model_path}" | tee -a "${merge_log}"
    return 1
  fi

  echo "Auto-merge failed for ${model_label} (see log: ${merge_log})" | tee -a "${merge_log}"
  return 1
}

run_one_model_with_fallback() {
  local model_label="$1"
  local model_path="$2"
  local log_path="$3"
  local run_dir="$4"

  local -a candidates=()
  if is_7b_model "${model_label}" "${model_path}"; then
    # 7B: start aggressive then fallback.
    candidates=(
      "64:0.86:256:32768"
      "48:0.84:192:24576"
      "32:0.82:160:20480"
      "24:0.80:128:16384"
    )
  else
    # 3B: higher concurrency.
    candidates=(
      "${EVAL_BATCH_SIZE}:0.90:384:49152"
      "80:0.88:320:40960"
      "64:0.86:256:32768"
      "48:0.84:192:24576"
      "32:0.82:160:20480"
    )
  fi

  # Remove duplicate candidate lines while preserving order.
  local dedup_file="${run_dir}/candidate_dedup.txt"
  : > "${dedup_file}"
  for c in "${candidates[@]}"; do
    if ! grep -Fxq "${c}" "${dedup_file}"; then
      echo "${c}" >> "${dedup_file}"
    fi
  done
  mapfile -t candidates < "${dedup_file}"

  local ok=0
  local attempt=0
  for c in "${candidates[@]}"; do
    attempt=$((attempt + 1))
    IFS=':' read -r batch_size gpu_util max_num_seqs max_num_batched_tokens <<< "${c}"
    local attempt_log="${run_dir}/eval_attempt_${attempt}.log"

    {
      echo "===== ATTEMPT ${attempt} model=${model_label} ====="
      echo "batch_size=${batch_size} gpu_util=${gpu_util} max_num_seqs=${max_num_seqs} max_num_batched_tokens=${max_num_batched_tokens}"
    } | tee -a "${log_path}"

    if (
      cd "${CODE_DIR}"
      exec env \
        -u http_proxy -u https_proxy -u all_proxy \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        NO_PROXY="${NO_PROXY}" \
        no_proxy="${no_proxy}" \
        TMPDIR="${TMPDIR}" \
        TMP="${TMP}" \
        TEMP="${TEMP}" \
        HF_HOME="${HF_HOME}" \
        TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
        XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        WANDB_DIR="${WANDB_DIR}" \
        WANDB_CACHE_DIR="${WANDB_CACHE_DIR}" \
        WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR}" \
        RAY_TMPDIR="${RAY_TMPDIR}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        VLLM_USE_MODELSCOPE=0 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        HYDRA_FULL_ERROR=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m verl.agent_trainer.main_generation \
          data.path="${EVAL_DATA_DIR}" \
          data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
          data.max_response_length="${MAX_RESPONSE_LENGTH}" \
          data.n_samples="${N_SAMPLES}" \
          data.batch_size="${batch_size}" \
          agentgym.task_name=alfworld \
          agentgym.env_addr="${ENV_ADDR}" \
          agentgym.max_rounds="${MAX_ROUND}" \
          agentgym.timeout="${AGENT_TIMEOUT}" \
          model.path="${model_path}" \
          rollout.gpu_memory_utilization="${gpu_util}" \
          rollout.temperature=1 \
          rollout.max_model_len="${MAX_MODEL_LEN}" \
          rollout.max_tokens="${MAX_TOKENS_PER_TURN}" \
          rollout.max_num_seqs="${max_num_seqs}" \
          rollout.max_num_batched_tokens="${max_num_batched_tokens}" \
          rollout.tensor_model_parallel_size=1 \
          rollout.rollout_log_dir="${run_dir}/executer_logs" \
          trainer.nnodes=1 \
          trainer.n_gpus_per_node="${NUM_GPUS}"
    ) 2>&1 | tee "${attempt_log}" | tee -a "${log_path}"; then
      ok=1
      break
    else
      if is_oom_log "${attempt_log}"; then
        echo "OOM detected on attempt ${attempt}, fallback to lower concurrency..." | tee -a "${log_path}"
        continue
      fi
      echo "Non-OOM failure on attempt ${attempt}, aborting model ${model_label}." | tee -a "${log_path}"
      return 1
    fi
  done

  if [[ "${ok}" != "1" ]]; then
    echo "All fallback attempts failed for ${model_label}." | tee -a "${log_path}"
    return 1
  fi

  return 0
}

for idx in "${!MODEL_LABELS[@]}"; do
  model_label="${MODEL_LABELS[$idx]}"
  model_path="${MODEL_PATHS[$idx]}"

  if [[ "$(should_skip_model "${model_label}" "${RESULTS_FILE}")" == "1" ]]; then
    echo "===== EVAL SKIP model=${model_label} ====="
    continue
  fi

  if [[ ! -d "${model_path}" ]]; then
    echo "Missing model directory for ${model_label}: ${model_path}" >&2
    continue
  fi
  if ! ensure_model_weights "${model_label}" "${model_path}" "${ROOT}/runlogs/alfworld_eval_automerge.log"; then
    echo "Skip model due to unresolved weights: ${model_label}" >&2
    continue
  fi

  run_name="eval_alfworld_${model_label}_$(date -u +%Y%m%d_%H%M%S)"
  run_dir="${ROOT}/runlogs/${run_name}"
  log_path="${run_dir}/eval.log"
  metrics_json_path="${run_dir}/metrics.json"
  mkdir -p "${run_dir}"

  echo "===== EVAL START model=${model_label} =====" | tee -a "${log_path}"
  run_one_model_with_fallback "${model_label}" "${model_path}" "${log_path}" "${run_dir}"

  python3 - "${log_path}" "${metrics_json_path}" <<'PY'
import json
import sys

log_path, out_path = sys.argv[1], sys.argv[2]
metrics_line = None
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "METRICS_JSON:" in line:
            metrics_line = line.split("METRICS_JSON:", 1)[1].strip()
if metrics_line is None:
    raise SystemExit(f"Failed to find METRICS_JSON in log: {log_path}")
metrics = json.loads(metrics_line)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=True, sort_keys=True)
PY

  read -r score succ <<<"$(python3 - "${metrics_json_path}" <<'PY'
import json
import sys
metrics = json.load(open(sys.argv[1], "r", encoding="utf-8"))
overall = metrics.get("overall", {})
print(f"{float(overall.get('score', 0.0))} {float(overall.get('succ', 0.0))}")
PY
)"
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"timestamp":"%s","task":"alfworld","split":"test","model_label":"%s","model_path":"%s","score":%s,"succ":%s,"log_path":"%s"}\n' \
    "${ts}" "${model_label}" "${model_path}" "${score}" "${succ}" "${log_path}" >> "${RESULTS_FILE}"

  echo "===== EVAL DONE model=${model_label} score=${score} succ=${succ} =====" | tee -a "${log_path}"
done

echo "All ALFWorld evaluations completed."
echo "Results file: ${RESULTS_FILE}"
python3 - "${RESULTS_FILE}" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], "r", encoding="utf-8") if line.strip()]
print("===== SUMMARY =====")
for row in rows:
    print(
        f"{row.get('model_label', 'unknown'):>20} | "
        f"Score={float(row.get('score', 0.0)):.4f} | "
        f"Succ={float(row.get('succ', 0.0)):.4f}"
    )
PY