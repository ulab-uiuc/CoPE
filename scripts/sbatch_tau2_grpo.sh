#!/usr/bin/env bash
#SBATCH --job-name=tau2_grpo
#SBATCH --partition=a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=${PROJECT_ROOT}/slurm_logs/tau2_grpo_%j.out
#SBATCH --error=${PROJECT_ROOT}/slurm_logs/tau2_grpo_%j.err
#
# End-to-end GRPO on tau2-bench.
#
# These are 40GB A100s, so the user simulator gets its own node and all 8 GPUs on the
# batch node go to training. With a 1-node allocation the script falls back to
# user-sim on GPU 0 + training on GPUs 1-7.
#
# --mem=0 (all node memory) is required, not cosmetic: with param/optimizer offload the
# training side sits around 540GB of host RAM, and saving a checkpoint makes every FSDP
# rank gather the full state dict on CPU on top of that. Under slurm's default 768G
# allocation that spike is what killed run 20088 mid-write at the final step.
#
#   sbatch scripts/sbatch_tau2_grpo.sh                      # full run, 2 nodes
#   sbatch --nodes=1 --export=ALL,TINY=1 scripts/sbatch_tau2_grpo.sh   # smoke run

set -euo pipefail

ROOT=${PROJECT_ROOT}
cd "${ROOT}"

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/Qwen2.5-7B-Instruct}"
USERSIM_MODEL="${USERSIM_MODEL:-${MODEL_PATH}}"
USERSIM_PORT="${USERSIM_PORT:-38101}"
# 40GB cards: a 14B is 28GB of bf16 weights, so both the rollout engine and the user
# simulator need to be sharded. Defaults below are sized for 7B; override for larger.
# USERSIM_GPUS is derived from USERSIM_TP rather than passed in -- a comma-separated
# value cannot survive `sbatch --export` without being mangled into a literal backslash.
# local  -> start a vLLM user simulator on the aux node (needs 2 nodes)
# hosted -> use a hosted model (gpt-4o-mini) via litellm; no server, 1 node is enough
USERSIM_MODE="${USERSIM_MODE:-local}"
USERSIM_LLM="${USERSIM_LLM:-openai/gpt-4o-mini}"
USERSIM_API_KEY_FILE="${USERSIM_API_KEY_FILE:-${ROOT}/.secrets/openai_api_key}"
USERSIM_TP="${USERSIM_TP:-1}"
USERSIM_GPUS="$(seq -s, 0 $((USERSIM_TP - 1)))"
USERSIM_GPU_MEM_UTIL="${USERSIM_GPU_MEM_UTIL:-0.45}"
SERVED_NAME=user-sim
BASE_PORT="${BASE_PORT:-36301}"
ENVS_PER_GPU="${ENVS_PER_GPU:-4}"

TAU2_DOMAIN="${TAU2_DOMAIN:-retail}"
TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-train}"
TAU2_REWARD_BASIS="${TAU2_REWARD_BASIS:-env}"
TAU2_REWARD_SHAPE="${TAU2_REWARD_SHAPE:-dense}"
TAU2_PROMPT_VARIANT="${TAU2_PROMPT_VARIANT:-strict}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NODE_TRAIN="${NODES[0]}"
NODE_AUX="${NODES[1]:-}"

if [[ "${USERSIM_MODE}" == "hosted" ]]; then
  # No local server to make room for.
  TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
  USERSIM_HOST=""
elif [[ -n "${NODE_AUX}" ]]; then
  # Dedicated node for the user simulator -> all 8 local GPUs train.
  TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
  USERSIM_HOST="${NODE_AUX}"
else
  # Single node: the user simulator takes GPUs 0..USERSIM_TP-1, training gets the rest.
  # NUM_TRAIN_GPUS caps how many of the rest to use, for when the QOS leaves fewer than
  # a whole node free. It is a scalar because a comma-separated TRAIN_GPUS cannot
  # survive `sbatch --export` -- the escaping backslash arrives with it, and a
  # semicolon workaround silently collapses CUDA_VISIBLE_DEVICES to a single device,
  # which shows up as an OOM on GPU 0 rather than as a parse error.
  _last=$(( USERSIM_TP + ${NUM_TRAIN_GPUS:-$((8 - USERSIM_TP))} - 1 ))
  TRAIN_GPUS="${TRAIN_GPUS:-$(seq -s, "${USERSIM_TP}" "${_last}")}"
  USERSIM_HOST=127.0.0.1
fi
USERSIM_URL="http://${USERSIM_HOST}:${USERSIM_PORT}"

if [[ "${TINY:-0}" == "1" ]]; then
  export TRAIN_BATCH_SIZE=4 ROLLOUT_N=4 PPO_MINI_BATCH_SIZE=2
  export MAX_ROUNDS=10 TOTAL_EPOCHS=1 SAVE_FREQ=1000
  export MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=4096 MAX_MODEL_LEN=10240
  export MAX_TOKENS_PER_TURN=512
  export ROLLOUT_GPU_MEMORY_UTILIZATION=0.30
  ENVS_PER_GPU=2
else
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" ROLLOUT_N="${ROLLOUT_N:-8}"
  export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
  export MAX_ROUNDS="${MAX_ROUNDS:-20}" TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
  export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
  export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-14336}"
  export MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-512}"
  export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
fi

IFS=',' read -r -a _G <<< "${TRAIN_GPUS}"
NUM_ENVS=$(( ${#_G[@]} * ENVS_PER_GPU ))

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-tau2_${TAU2_DOMAIN}_grpo_${RUN_TS}}"
RUN_DIR="${ROOT}/runlogs/${EXP_NAME}"
mkdir -p "${RUN_DIR}" "${ROOT}/slurm_logs"

export HF_HOME="${ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
# ~/.local/lib/python3.10 carries a torch that shadows the conda env's and breaks
# vLLM's compiled extensions ("_core_C.ScalarType ... does not exist").
export PYTHONNOUSERSITE=1
export NO_PROXY="127.0.0.1,localhost,${NODE_TRAIN},${NODE_AUX}"
export no_proxy="${NO_PROXY}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "=== tau2 GRPO ${EXP_NAME} ==="
echo "train node : ${NODE_TRAIN}  GPUs=${TRAIN_GPUS}"
if [[ "${USERSIM_MODE}" == "hosted" ]]; then
  echo "usersim    : ${USERSIM_LLM} (hosted, key from ${USERSIM_API_KEY_FILE})"
else
  echo "usersim    : ${USERSIM_URL} (node ${NODE_AUX:-local GPU 0})"
fi
echo "env servers: ${NUM_ENVS} from port ${BASE_PORT}"
echo "model      : ${MODEL_PATH}"
echo "batch=${TRAIN_BATCH_SIZE} rollout_n=${ROLLOUT_N} rounds=${MAX_ROUNDS} epochs=${TOTAL_EPOCHS}"

cleanup() {
  echo "=== cleanup ==="
  [[ -n "${USERSIM_PID:-}" ]] && kill "${USERSIM_PID}" 2>/dev/null || true
  [[ -n "${ENVSVC_PID:-}" ]] && kill -- -"${ENVSVC_PID}" 2>/dev/null || true
  pkill -f "agentenv_tau2:app" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 1. user simulator -------------------------------------------------------------
if [[ "${USERSIM_MODE}" == "hosted" ]]; then
  echo "--- user simulator: hosted ${USERSIM_LLM} (no local server) ---"
  [[ -r "${USERSIM_API_KEY_FILE}" ]] || { echo "FATAL: no API key at ${USERSIM_API_KEY_FILE}"; exit 1; }
  TAU2_USER_LLM_ARG="${USERSIM_LLM}"
  TAU2_USER_API_BASE_ARG=""
else
  echo "--- starting user simulator ---"
if [[ -n "${NODE_AUX}" ]]; then
    srun --nodes=1 --ntasks=1 --overlap -w "${NODE_AUX}" \
      env USERSIM_MODEL="${USERSIM_MODEL}" SERVED_NAME="${SERVED_NAME}" \
          PORT="${USERSIM_PORT}" HOST=0.0.0.0 USERSIM_GPU="${USERSIM_GPUS}" \
          USERSIM_TP="${USERSIM_TP}" USERSIM_GPU_MEM_UTIL="${USERSIM_GPU_MEM_UTIL}" \
          LOG_DIR="${RUN_DIR}" \
      bash "${ROOT}/scripts/run_tau2_usersim_server.sh" &
  else
    USERSIM_MODEL="${USERSIM_MODEL}" SERVED_NAME="${SERVED_NAME}" \
      PORT="${USERSIM_PORT}" HOST=127.0.0.1 USERSIM_GPU="${USERSIM_GPUS}" \
      USERSIM_TP="${USERSIM_TP}" USERSIM_GPU_MEM_UTIL="${USERSIM_GPU_MEM_UTIL}" \
      LOG_DIR="${RUN_DIR}" \
      bash "${ROOT}/scripts/run_tau2_usersim_server.sh" &
  fi
  USERSIM_PID=$!
  
  for _ in $(seq 1 150); do
    curl --noproxy '*' -sf "${USERSIM_URL}/v1/models" >/dev/null && break
    sleep 5
  done
  if ! curl --noproxy '*' -sf "${USERSIM_URL}/v1/models" >/dev/null; then
    echo "FATAL: user simulator did not come up"; tail -40 "${RUN_DIR}/usersim.log" || true; exit 1
  fi
  echo "user simulator healthy at ${USERSIM_URL}"
  TAU2_USER_LLM_ARG="openai/${SERVED_NAME}"
  TAU2_USER_API_BASE_ARG="${USERSIM_URL}/v1"
fi

# ---- 2. env cluster (on the training node, so the client hop is loopback) -----------
echo "--- starting ${NUM_ENVS} tau2 env servers ---"
setsid env \
  NUM_ENVS="${NUM_ENVS}" BASE_PORT="${BASE_PORT}" \
  TAU2_DOMAIN="${TAU2_DOMAIN}" TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT}" \
  TAU2_REWARD_BASIS="${TAU2_REWARD_BASIS}" \
  TAU2_REWARD_SHAPE="${TAU2_REWARD_SHAPE}" \
  TAU2_PROMPT_VARIANT="${TAU2_PROMPT_VARIANT}" \
  TAU2_FORCE_DONE_AFTER="${MAX_ROUNDS}" \
  TAU2_USER_LLM="${TAU2_USER_LLM_ARG}" \
  ${TAU2_USER_API_BASE_ARG:+TAU2_USER_API_BASE="${TAU2_USER_API_BASE_ARG}"} \
  TAU2_USER_API_KEY_FILE="${USERSIM_API_KEY_FILE}" \
  TAU2_ENV="${TAU2_ENV:-${TAU2_ENV_DEFAULT}}" \
  NO_PROXY="${NO_PROXY}" no_proxy="${no_proxy}" \
  LOG_DIR="${RUN_DIR}/env_cluster" \
  bash "${ROOT}/scripts/run_tau2_env_service.sh" &
ENVSVC_PID=$!

for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  for _ in $(seq 1 60); do
    curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/" >/dev/null && break
    sleep 2
  done
  if ! curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/" >/dev/null; then
    echo "FATAL: env server on ${PORT} did not come up"
    tail -40 "${RUN_DIR}"/env_cluster/env_"${PORT}".log || true
    exit 1
  fi
done
echo "all ${NUM_ENVS} env servers healthy"

# ---- 3. training -------------------------------------------------------------------
echo "--- starting GRPO training ---"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
ENVS_PER_GPU="${ENVS_PER_GPU}" \
BASE_PORT="${BASE_PORT}" \
MODEL_PATH="${MODEL_PATH}" \
EXP_NAME="${EXP_NAME}" \
RUN_DIR="${RUN_DIR}" \
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/tau2_${TAU2_DOMAIN}_${TAU2_TASK_SPLIT}.json}" \
  bash "${ROOT}/scripts/run_tau2_grpo_train.sh" 2>&1 | tee "${RUN_DIR}/train.log"

echo "=== done ==="
