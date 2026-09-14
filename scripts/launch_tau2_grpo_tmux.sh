#!/usr/bin/env bash
#
# One-shot launcher for tau2-bench GRPO: user-sim server -> env cluster -> training,
# each in its own tmux session, with health gates between them.
#
# Usage:
#   MODEL_PATH=/path/to/policy USERSIM_MODEL=/path/to/user-sim \
#     CUDA_VISIBLE_DEVICES=1,2,3 USERSIM_GPU=0 bash launch_tau2_grpo_tmux.sh
#
# Note USERSIM_GPU must not appear in CUDA_VISIBLE_DEVICES -- the user simulator needs
# its own card, the training GPUs are already at ROLLOUT_GPU_MEMORY_UTILIZATION.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PORT="${BASE_PORT:-36201}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the policy model}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-tau2}"

TAU2_DOMAIN="${TAU2_DOMAIN:-retail}"
TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-train}"
TAU2_REWARD_BASIS="${TAU2_REWARD_BASIS:-env}"

# User simulator. Leave USERSIM_MODEL empty to reuse an already-running server.
USERSIM_MODEL="${USERSIM_MODEL:-}"
USERSIM_GPU="${USERSIM_GPU:-0}"
USERSIM_PORT="${USERSIM_PORT:-38001}"
SERVED_NAME="${SERVED_NAME:-user-sim}"
TAU2_USER_API_BASE="${TAU2_USER_API_BASE:-http://127.0.0.1:${USERSIM_PORT}/v1}"
TAU2_USER_LLM="${TAU2_USER_LLM:-openai/${SERVED_NAME}}"

ENVS_PER_GPU="${ENVS_PER_GPU:-4}"
IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NUM_ENVS=$(( ${#_GPU_ARR[@]} * ENVS_PER_GPU ))

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-tau2_${TAU2_DOMAIN}_grpo_${RUN_TS}}"

USERSIM_SESSION="tau2_usersim_${USERSIM_PORT}"
ENV_SESSION="tau2_env_cluster_${BASE_PORT}"
TRAIN_SESSION="tau2_grpo_train"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}"

for s in "${ENV_SESSION}" "${TRAIN_SESSION}"; do
  if tmux has-session -t "${s}" 2>/dev/null; then tmux kill-session -t "${s}"; fi
done

# ---- 1. user simulator -------------------------------------------------------------
if [[ -n "${USERSIM_MODEL}" ]]; then
  if tmux has-session -t "${USERSIM_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${USERSIM_SESSION}"
  fi
  echo "Starting user-simulator server on port ${USERSIM_PORT} (GPU ${USERSIM_GPU})..."
  tmux new-session -d -s "${USERSIM_SESSION}" \
    "cd ${ROOT} && USERSIM_MODEL=${USERSIM_MODEL} SERVED_NAME=${SERVED_NAME} PORT=${USERSIM_PORT} USERSIM_GPU=${USERSIM_GPU} bash ${ROOT}/scripts/run_tau2_usersim_server.sh"

  echo "Waiting for the user simulator to load (this takes a few minutes)..."
  for _ in $(seq 1 180); do
    if curl --noproxy '*' -sf "http://127.0.0.1:${USERSIM_PORT}/v1/models" >/dev/null; then
      echo "User simulator is healthy."
      break
    fi
    sleep 5
  done
  if ! curl --noproxy '*' -sf "http://127.0.0.1:${USERSIM_PORT}/v1/models" >/dev/null; then
    echo "User-simulator server failed to start. See runlogs/tau2_usersim/usersim.log"
    exit 1
  fi
else
  echo "USERSIM_MODEL unset -- expecting an existing server at ${TAU2_USER_API_BASE}"
  if ! curl --noproxy '*' -sf "http://127.0.0.1:${USERSIM_PORT}/v1/models" >/dev/null; then
    echo "No user simulator reachable on port ${USERSIM_PORT}."
    exit 1
  fi
fi

# ---- 2. env cluster ----------------------------------------------------------------
echo "Starting ${NUM_ENVS} tau2 environment services from port ${BASE_PORT}..."
tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && NUM_ENVS=${NUM_ENVS} BASE_PORT=${BASE_PORT} TAU2_DOMAIN=${TAU2_DOMAIN} TAU2_TASK_SPLIT=${TAU2_TASK_SPLIT} TAU2_REWARD_BASIS=${TAU2_REWARD_BASIS} TAU2_USER_LLM=${TAU2_USER_LLM} TAU2_USER_API_BASE=${TAU2_USER_API_BASE} bash ${ROOT}/scripts/run_tau2_env_service.sh"

echo "Waiting for env services to become healthy..."
for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  ADDR="http://127.0.0.1:${PORT}"
  for _ in $(seq 1 60); do
    if curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then break; fi
    sleep 2
  done
  if ! curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then
    echo "tau2 service on port ${PORT} failed to start."
    exit 1
  fi
  echo "Port ${PORT} is healthy."
done

# ---- 3. training -------------------------------------------------------------------
echo "Starting GRPO training..."
FWD=""
for v in KL_COEF ENTROPY_COEF POLICY_LR ROLLOUT_N TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE \
         PPO_MICRO_BATCH_SIZE_PER_GPU PPO_EPOCHS TOTAL_EPOCHS MAX_ROUNDS \
         MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN MAX_TOKENS_PER_TURN \
         ROLLOUT_GPU_MEMORY_UTILIZATION TENSOR_MODEL_PARALLEL_SIZE SAVE_FREQ TRAIN_FILE; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
echo "Forwarding overrides:${FWD:-<none>}"

tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ENVS_PER_GPU=${ENVS_PER_GPU} BASE_PORT=${BASE_PORT} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG}${FWD} bash ${ROOT}/scripts/run_tau2_grpo_train.sh"

echo "--------------------------------------------------"
echo "tau2 Training Cluster Launched!"
echo "Domain / split:      ${TAU2_DOMAIN} / ${TAU2_TASK_SPLIT}"
echo "Reward basis:        ${TAU2_REWARD_BASIS}"
echo "Number of Envs:      ${NUM_ENVS}"
echo "Base Port:           ${BASE_PORT}"
echo "User sim:            ${TAU2_USER_LLM} @ ${TAU2_USER_API_BASE}"
echo "Env Session:         ${ENV_SESSION}"
echo "Training Session:    ${TRAIN_SESSION}"
echo "Training Log:        ${TRAIN_LOG}"
echo "--------------------------------------------------"
