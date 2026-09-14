#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PORT="${BASE_PORT:-36101}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
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
EXP_NAME="${EXP_NAME:-webshop_grpo_reb_baseline_prolong_${RUN_TS}}"

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
# run_webshop_grpo_train_rebuttal.sh.
FWD=""
for v in ENABLE_ERC ERC_CLIPPING_METHOD ERC_CLIPPING_TYPE ERC_MOMENTUM UNCERTAINTY_SCALE_KAPPA UNCERTAINTY_SCALE_MIN UNCERTAINTY_SCALE_RENORMALIZE SAFE_COMMIT_MODE SAFE_COMMIT_OMEGA SAFE_COMMIT_RECENCY SAFE_COMMIT_RECENCY_GAMMA SAFE_COMMIT_SUCCESS_THRESHOLD SAFE_COMMIT_KAPPA SAFE_COMMIT_GMAX SAFE_COMMIT_GMIN SAFE_COMMIT_RENORMALIZE SAFE_COMMIT_TEXT_GATE SAFE_COMMIT_GATE_MODE SAFE_COMMIT_CLF_ENV SAFE_COMMIT_CLF_WINS_ONLY SAFE_COMMIT_CLF_MAX_NEW \
         WMLOSS_ADD_COEF WMLOSS_ADD_COEF_END WMLOSS_ADD_HORIZON \
         WMLOSS_ADD_USE_GAP WMLOSS_ADD_USE_ENTROPY WMLOSS_ADD_ONLY_FAILED WMLOSS_ADD_TO_REWARD REF_NLL_ADD REF_NLL_COEF \
         EPISTEMIC_BASE EPISTEMIC_BASE_NEG EPISTEMIC_S_MAX EPISTEMIC_SHAPE EPISTEMIC_USE_REF EPISTEMIC_PRE_ADD_COEF EPISTEMIC_INVERT_ON_NEG EPI_INTRINSIC_COEF EPI_INTRINSIC_CAP EPI_INTRINSIC_USE_REF \
         \
         USE_HINDSIGHT_HCA HCA_RATIO_CLIP_MIN HCA_RATIO_CLIP_MAX HCA_TEMP HCA_OMEGA HCA_GAMMA HCA_SMOOTH_ALPHA HCA_Z_THRESHOLD HCA_FINAL_STATE_MAX_TOKENS HCA_PERSTEP HCA_HISTORY_LEN PE_CREDIT_ENABLE PE_OMEGA_PROGRESS PE_OMEGA_EXPLORE PE_CREDIT_WINS_ONLY PE_CLF_ENV \
         PLAN_FORECAST_ENABLE PLAN_FORECAST_COEF PLAN_FORECAST_K PLAN_FORECAST_GATE PLAN_FORECAST_SUCCESS_THRESHOLD PLAN_FORECAST_MAX_LENGTH PLAN_FORECAST_TARGET PLAN_FORECAST_SEQ PLAN_FORECAST_COEF_ANNEAL PLAN_FORECAST_COEF_END PLAN_FORECAST_COEF_HORIZON PLAN_FORECAST_COEF_POWER PLAN_FORECAST_COEF_CUTOFF_STEP PLAN_INLINE_ENABLE PLAN_INLINE_K PLAN_INLINE_STYLE PLAN_INLINE_PER_TURN PLAN_INLINE_WARMUP_STEPS THINK_REMINDER_ENABLE PLAN_FORMAT_REWARD_ENABLE PLAN_FORMAT_REWARD_COEF PLAN_FORMAT_REWARD_BASELINE PLAN_FORMAT_REWARD_CLIP PLAN_FORMAT_REWARD_PENALTY_ONLY PLAN_FORMAT_REWARD_WARMUP_STEPS \
         WMC_COEFF POLICY_LR ENTROPY_COEF KL_COEF; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
echo "Forwarding overrides:${FWD:-<none>}"
tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_ENVS=${NUM_ENVS} ENVS_PER_GPU=${ENVS_PER_GPU} BASE_PORT=${BASE_PORT} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG}${FWD} bash ${ROOT}/scripts/run_webshop_grpo_train_rebuttal.sh"

echo "--------------------------------------------------"
echo "WebShop Training Cluster Launched!"
echo "Number of Envs:      ${NUM_ENVS}"
echo "Base Port:           ${BASE_PORT}"
echo "Environment Session: ${ENV_SESSION}"
echo "Training Session:    ${TRAIN_SESSION}"
echo "Training Log:        ${TRAIN_LOG}"
echo "--------------------------------------------------"
