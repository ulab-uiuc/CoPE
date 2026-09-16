#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root; this script lives in scripts/
BASE_PORT="${BASE_PORT:-36101}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-sciworld}"

# Calculate number of envs based on GPUs
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
# Env servers per GPU; forwarded to the trainer so both sides build the same addresses.
ENVS_PER_GPU="${ENVS_PER_GPU:-1}"
NUM_ENVS=$((NUM_GPUS * ENVS_PER_GPU))
# SciWorld boots a JVM per server, so startup is slow; wait up to HEALTH_RETRIES*2s.
HEALTH_RETRIES="${HEALTH_RETRIES:-300}"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-sciworld_grpo_qwen2.5_3b_add_${RUN_TS}}"

# RUN_TAG suffixes the tmux session names so several runs can coexist on one host;
# unset, the names are unchanged.
ENV_SESSION="sciworld_env_cluster_${BASE_PORT}${RUN_TAG:+_${RUN_TAG}}"
TRAIN_SESSION="sciworld_grpo_train${RUN_TAG:+_${RUN_TAG}}"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${ENV_SESSION}"
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TRAIN_SESSION}"
fi

echo "Starting ${NUM_ENVS} SciWorld Environment Services starting at port ${BASE_PORT}..."
# The env and train scripts require these to be set (conda env names, data root).
# Pass them explicitly: a tmux server that is already running does not inherit this
# shell's environment, so relying on inheritance fails exactly when tmux is in use.
ENV_FWD=""
for v in CONDA_SH SCIWORLD_ENV; do
  if [ -n "${!v:-}" ]; then ENV_FWD="${ENV_FWD} ${v}=${!v}"; fi
done
tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && NUM_ENVS=${NUM_ENVS} BASE_PORT=${BASE_PORT}${ENV_FWD} bash ${ROOT}/scripts/run_sciworld_env_service.sh"

echo "Waiting for services to become healthy..."
for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  ADDR="http://127.0.0.1:${PORT}"
  echo "Checking ${ADDR}..."
  for _ in $(seq 1 "${HEALTH_RETRIES}"); do
    if curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then
      echo "Port ${PORT} is healthy."
      break
    fi
    sleep 2
  done
  if ! curl --noproxy '*' -sf "${ADDR}/" >/dev/null; then
    echo "SciWorld service on port ${PORT} failed to start."
    exit 1
  fi
done

echo "Starting GRPO Training..."
# Forward tuning env vars into the training tmux command, but only those that
# are actually set in this shell — unset ones fall through to the defaults in
# run_sciworld_grpo_train.sh. (Explicit in the command string to avoid tmux
# server env-inheritance staleness.)
FWD=""
for v in ACTION_FORECAST_ENABLE ACTION_FORECAST_COEF ACTION_FORECAST_K ACTION_FORECAST_GATE \
         ACTION_FORECAST_SUCCESS_THRESHOLD ACTION_FORECAST_MAX_LENGTH \
         POLICY_LR ENTROPY_COEF \
         KL_COEF \
         ACTION_FORECAST_GROUP_NORM \
         ACTION_FORECAST_SKIP_INVALID \
         TE_ENABLE TE_LAMBDA TE_ETA TE_KL_TYPE TE_MIX TE_CENTER TE_WARMUP_STEPS \
         TE_TRAJ_SUBSAMPLE TE_MICRO_BATCH_SIZE_PER_GPU \
         VLLM_PORT RESUME_MODE SAVE_FREQ TOTAL_EPOCHS TOTAL_TRAINING_STEPS \
         TRAIN_BATCH_SIZE ROLLOUT_N ROLLOUT_GPU_MEMORY_UTILIZATION; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
for v in TRAIN_ENV CONDA_SH; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
echo "Forwarding overrides:${FWD:-<none>}"
tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} BASE_PORT=${BASE_PORT} ENVS_PER_GPU=${ENVS_PER_GPU} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG}${FWD} bash ${ROOT}/scripts/run_sciworld_grpo_train.sh"

echo "--------------------------------------------------"
echo "SciWorld Training Cluster Launched!"
echo "Number of Envs:      ${NUM_ENVS}"
echo "Base Port:           ${BASE_PORT}"
echo "Environment Session: ${ENV_SESSION}"
echo "Training Session:    ${TRAIN_SESSION}"
echo "Training Log:        ${TRAIN_LOG}"
echo "--------------------------------------------------"
