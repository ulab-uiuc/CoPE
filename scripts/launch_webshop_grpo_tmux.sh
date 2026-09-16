#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PORT="${BASE_PORT:-36101}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-webshop}"

# Env-server processes per GPU. Each rank round-robins over its own shard of
# servers, so >1 spreads a GPU's concurrent env stepping across processes/cores
# instead of GIL-serialising it in one process. Must match the training script,
# which builds ENVS_PER_GPU*NUM_GPUS addresses -- so ENVS_PER_GPU is forwarded.
ENVS_PER_GPU="${ENVS_PER_GPU:-4}"
IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NUM_ENVS=$(( ${#_GPU_ARR[@]} * ENVS_PER_GPU ))

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-webshop_grpo_qwen2.5_3b_add_${RUN_TS}}"

ENV_SESSION="webshop_env_cluster_${BASE_PORT}"
TRAIN_SESSION="webshop_grpo_train"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${ENV_SESSION}"
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi

echo "Starting ${NUM_ENVS} WebShop Environment Services starting at port ${BASE_PORT}..."
tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && NUM_ENVS=${NUM_ENVS} BASE_PORT=${BASE_PORT} bash ${ROOT}/scripts/run_webshop_env_service.sh"

echo "Waiting for services to become healthy..."
for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  ADDR="http://127.0.0.1:${PORT}"
  echo "Checking ${ADDR}..."
  for _ in $(seq 1 60); do
    if curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then
      echo "Port ${PORT} is healthy."
      break
    fi
    sleep 2
  done
  if ! curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then
    echo "WebShop service on port ${PORT} failed to start."
    exit 1
  fi
done

echo "Starting GRPO Training..."
# Forward tuning env vars into the training tmux command, but only those that
# are actually set in this shell — unset ones fall through to the defaults in
# run_webshop_grpo_train.sh.
FWD=""
for v in ACTION_FORECAST_ENABLE ACTION_FORECAST_COEF ACTION_FORECAST_K ACTION_FORECAST_GATE \
         ACTION_FORECAST_SUCCESS_THRESHOLD ACTION_FORECAST_MAX_LENGTH \
         POLICY_LR ENTROPY_COEF \
         KL_COEF; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
echo "Forwarding overrides:${FWD:-<none>}"
tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_ENVS=${NUM_ENVS} ENVS_PER_GPU=${ENVS_PER_GPU} BASE_PORT=${BASE_PORT} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG}${FWD} bash ${ROOT}/scripts/run_webshop_grpo_train.sh"

echo "--------------------------------------------------"
echo "WebShop Training Cluster Launched!"
echo "Number of Envs:      ${NUM_ENVS}"
echo "Base Port:           ${BASE_PORT}"
echo "Environment Session: ${ENV_SESSION}"
echo "Training Session:    ${TRAIN_SESSION}"
echo "Training Log:        ${TRAIN_LOG}"
echo "--------------------------------------------------"
