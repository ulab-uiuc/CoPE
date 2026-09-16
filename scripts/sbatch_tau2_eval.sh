#!/usr/bin/env bash
#SBATCH --job-name=tau2_eval
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/tau2_eval_%j.out
#SBATCH --error=slurm_logs/tau2_eval_%j.err
#
# A/B tau2-bench reward shaping (or any env-server flag) WITHOUT training.
#
# One node: policy vLLM on GPUs 0-3 (tp=4), user simulator on GPU 4. For each variant a
# fresh env-server cluster is started with TAU2_REWARD_SHAPE set, then eval_tau2.py
# runs every train task k times.
#
#   sbatch scripts/sbatch_tau2_eval.sh
#   VARIANTS="dense" K=8 sbatch scripts/sbatch_tau2_eval.sh

set -euo pipefail

# sbatch copies this script to a spool dir, so BASH_SOURCE is useless here.
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the checkpoint to evaluate}"
# The user simulator must be held FIXED when comparing policies, otherwise a change in
# score cannot be attributed to the policy. Defaults to MODEL_PATH for the single-model
# case; set it explicitly to the base model when evaluating a trained checkpoint.
USERSIM_MODEL="${USERSIM_MODEL:-${MODEL_PATH}}"
VARIANTS="${VARIANTS:-binary dense}"   # values of TAU2_REWARD_SHAPE
TAU2_DOMAIN="${TAU2_DOMAIN:-retail}"
N_TASKS="${N_TASKS:-74}"
TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-train}"
K="${K:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-15}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
NUM_ENVS="${NUM_ENVS:-16}"
BASE_PORT="${BASE_PORT:-36401}"
POLICY_PORT="${POLICY_PORT:-38201}"
USERSIM_PORT="${USERSIM_PORT:-38202}"
CONCURRENCY="${CONCURRENCY:-32}"
# telecom's system prompt alone is 6212 tokens; 16384 leaves too little for a 15-round
# conversation on top of it.
POLICY_MAX_LEN="${POLICY_MAX_LEN:-16384}"
# local -> vLLM user simulator on this node. hosted -> gpt-4o-mini via litellm, which is
# what the published tau2 numbers use; a 7B customer is a different benchmark.
USERSIM_MODE="${USERSIM_MODE:-local}"
USERSIM_LLM="${USERSIM_LLM:-openai/gpt-4o-mini}"
USERSIM_API_KEY_FILE="${USERSIM_API_KEY_FILE:-${ROOT}/.secrets/openai_api_key}"

TAG="${TAG:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${ROOT}/runlogs/tau2_eval_${TAG}"
mkdir -p "${OUT_DIR}" "${ROOT}/slurm_logs"

export HF_HOME="${ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
CONDA_SH=/opt/conda/etc/profile.d/conda.sh

echo "=== tau2 reward-shape A/B (${TAG}) on $(hostname) ==="
echo "variants: ${VARIANTS}   tasks=${N_TASKS} k=${K}"

cleanup() {
  echo "=== cleanup ==="
  pkill -f "agentenv_tau2:app" 2>/dev/null || true
  [[ -n "${POLICY_PID:-}" ]] && kill "${POLICY_PID}" 2>/dev/null || true
  [[ -n "${USERSIM_PID:-}" ]] && kill "${USERSIM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

wait_http() {  # url, tries
  for _ in $(seq 1 "$2"); do curl --noproxy '*' -sf "$1" >/dev/null && return 0; sleep 5; done
  return 1
}

# ---- policy server (GPUs 0-3) -------------------------------------------------------
echo "--- starting policy server ---"
(
  source "${CONDA_SH}"; conda activate ${TRAIN_ENV}
  exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${POLICY_GPUS:-0,1}" \
    python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" --served-model-name policy \
      --host 127.0.0.1 --port "${POLICY_PORT}" \
      --tensor-parallel-size "${POLICY_TP:-2}" --gpu-memory-utilization 0.85 \
      --max-model-len "${POLICY_MAX_LEN}" --disable-log-requests
) > "${OUT_DIR}/policy.log" 2>&1 &
POLICY_PID=$!

# ---- user simulator (GPU 4) ---------------------------------------------------------
if [[ "${USERSIM_MODE}" != "hosted" ]]; then
  echo "--- starting user simulator ---"
  USERSIM_MODEL="${USERSIM_MODEL}" SERVED_NAME=user-sim PORT="${USERSIM_PORT}" \
    HOST=127.0.0.1 USERSIM_GPU="${USERSIM_EVAL_GPU:-2}" LOG_DIR="${OUT_DIR}" \
    bash "${ROOT}/scripts/run_tau2_usersim_server.sh" &
fi
USERSIM_PID=$!

wait_http "http://127.0.0.1:${POLICY_PORT}/v1/models" 150 || { echo FATAL policy; tail -30 "${OUT_DIR}/policy.log"; exit 1; }
if [[ "${USERSIM_MODE}" == "hosted" ]]; then
  [[ -r "${USERSIM_API_KEY_FILE}" ]] || { echo "FATAL: no API key at ${USERSIM_API_KEY_FILE}"; exit 1; }
  TAU2_USER_LLM_ARG="${USERSIM_LLM}"; TAU2_USER_API_BASE_ARG=""
  echo "policy healthy; usersim = hosted ${USERSIM_LLM}"
else
  wait_http "http://127.0.0.1:${USERSIM_PORT}/v1/models" 150 || { echo FATAL usersim; tail -30 "${OUT_DIR}/usersim.log"; exit 1; }
  TAU2_USER_LLM_ARG="openai/user-sim"; TAU2_USER_API_BASE_ARG="http://127.0.0.1:${USERSIM_PORT}/v1"
  echo "policy + usersim healthy"
fi

VIDX=0
for VARIANT in ${VARIANTS}; do
  echo ""
  echo "################ variant: ${VARIANT} ################"
  # Each variant gets its OWN port block. Reusing one block and pkill-ing between
  # variants silently produced a bogus A/B once: the previous variant's uvicorn workers
  # had not released the sockets yet, every new server died with EADDRINUSE, and the
  # health check happily passed against the *old* processes -- so the second variant was
  # measured with the first variant's config.
  VBASE=$((BASE_PORT + VIDX * 100))
  VIDX=$((VIDX + 1))

  ENV_ADDRS=""
  for i in $(seq 0 $((NUM_ENVS - 1))); do
    ENV_ADDRS="${ENV_ADDRS:+${ENV_ADDRS},}http://127.0.0.1:$((VBASE + i))"
  done

  setsid env NUM_ENVS="${NUM_ENVS}" BASE_PORT="${VBASE}" \
    TAU2_DOMAIN="${TAU2_DOMAIN}" TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT}" TAU2_REWARD_BASIS=env \
    TAU2_PROMPT_VARIANT="${TAU2_PROMPT_VARIANT:-strict}" TAU2_REWARD_SHAPE="${VARIANT}" \
    TAU2_FORCE_DONE_AFTER="${TAU2_FORCE_DONE_AFTER:-${MAX_ROUNDS}}" \
    TAU2_USER_LLM="${TAU2_USER_LLM_ARG}" \
    ${TAU2_USER_API_BASE_ARG:+TAU2_USER_API_BASE="${TAU2_USER_API_BASE_ARG}"} \
    TAU2_USER_API_KEY_FILE="${USERSIM_API_KEY_FILE}" \
    LOG_DIR="${OUT_DIR}/env_${VARIANT}" \
    bash "${ROOT}/scripts/run_tau2_env_service.sh" > "${OUT_DIR}/envsvc_${VARIANT}.log" 2>&1 &

  ok=1
  for i in $(seq 0 $((NUM_ENVS - 1))); do
    wait_http "http://127.0.0.1:$((VBASE + i))/" 40 || { ok=0; break; }
  done
  [[ $ok == 1 ]] || { echo "FATAL: env servers for ${VARIANT} did not start"; tail -30 "${OUT_DIR}"/env_"${VARIANT}"/*.log; exit 1; }

  # Assert every server really is running this variant, not a survivor from the last one.
  for i in $(seq 0 $((NUM_ENVS - 1))); do
    got=$(curl --noproxy '*' -sf "http://127.0.0.1:$((VBASE + i))/config" | tr -d ' ' | grep -o "\"reward_shape\":\"[a-z]*\"" | cut -d'"' -f4)
    if [[ "${got}" != "${VARIANT}" ]]; then
      echo "FATAL: port $((VBASE + i)) reports reward_shape=${got}, expected ${VARIANT}"
      exit 1
    fi
  done
  echo "env servers up for ${VARIANT} on ${VBASE}..$((VBASE + NUM_ENVS - 1)) (config verified)"

  (
    source "${CONDA_SH}"; conda activate ${TRAIN_ENV}
    exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      PYTHONNOUSERSITE=1 PYTHONPATH="${ROOT}/AgentGym/agentenv" \
      NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      python "${ROOT}/scripts/eval_tau2.py" \
        --policy-url "http://127.0.0.1:${POLICY_PORT}/v1" --policy-model policy \
        --env-addrs "${ENV_ADDRS}" \
        --n-tasks "${N_TASKS}" --k "${K}" --max-rounds "${MAX_ROUNDS}" \
        --max-tokens "${MAX_TOKENS}" --concurrency "${CONCURRENCY}" \
        --out "${OUT_DIR}/eval_${VARIANT}.json"
  ) 2>&1 | tee "${OUT_DIR}/eval_${VARIANT}.log"
done

echo ""
echo "=== A/B done, results in ${OUT_DIR} ==="
for VARIANT in ${VARIANTS}; do
  echo "--- ${VARIANT} ---"
  grep -E "solve rate|mean reward|tasks INFORMATIVE|tasks all-zero" "${OUT_DIR}/eval_${VARIANT}.log" || true
done
