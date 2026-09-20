#!/usr/bin/env bash
#
# τ²-bench GRPO in a detached tmux session -- the same pipeline as
# scripts/run_tau2_pipeline.sh, but it survives the terminal closing.
# scripts/launch_tau2_cope_tmux.sh is this script with the action-forecast switch on.
#
#   bash scripts/launch_tau2_grpo_tmux.sh                     # InfoPO preset, plain GRPO
#   PRESET=repo bash scripts/launch_tau2_grpo_tmux.sh         # this repo's retail setup
#   DRY_RUN=1 bash scripts/launch_tau2_grpo_tmux.sh           # session prints the config and exits
#   USERSIM_MODE=local USERSIM_GPU=0 CUDA_VISIBLE_DEVICES=1,2,3 \
#     bash scripts/launch_tau2_grpo_tmux.sh                   # free, local customer
#
# The whole pipeline (customer, env cluster, training) runs as ONE process tree inside
# the session, so `tmux kill-session -t <name>` takes everything down cleanly; the
# pipeline's own EXIT trap stops the env servers and the customer. Output goes to
# runlogs/<exp>/pipeline.log, training metrics to runlogs/<exp>/train.log.
#
# Every knob of scripts/run_tau2_pipeline.sh is forwarded: set it in this shell.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v tmux >/dev/null || { echo "FATAL: tmux not found" >&2; exit 1; }

PRESET="${PRESET:-infopo}"
VARIANT="${VARIANT:-grpo}"          # grpo | cope (set by launch_tau2_cope_tmux.sh)
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-tau2_${PRESET}_${VARIANT}_${RUN_TS}}"
SESSION="${SESSION:-tau2_${VARIANT}${RUN_TAG:+_${RUN_TAG}}}"
RUN_DIR="${ROOT}/runlogs/${EXP_NAME}"
mkdir -p "${RUN_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists; kill it first or set RUN_TAG/SESSION" >&2
  exit 1
fi

# Forward the caller's settings verbatim: everything the pipeline reads, in one place.
FWD=""
for v in PRESET TRAIN_ENV TAU2_ENV CONDA_SH TAU2_BENCH_DIR TAU2_DATA_DIR MODEL_PATH \
         CUDA_VISIBLE_DEVICES TRAIN_GPUS USERSIM_MODE USERSIM_GPU USERSIM_MODEL USERSIM_LLM \
         USERSIM_PORT USERSIM_GPU_MEM_UTIL USERSIM_MAX_MODEL_LEN KEY_FILE \
         BASE_PORT ENVS_PER_GPU CKPT_DIR PROJECT_NAME WANDB_MODE HF_HUB_OFFLINE HF_HOME \
         TAU2_DOMAIN TAU2_TASK_SPLIT TRAIN_FILE TAU2_REWARD_SHAPE TAU2_REWARD_BASIS \
         TAU2_PROMPT_VARIANT NATIVE_TOOLS TAU2_USER_TEMPERATURE TAU2_MAX_STEPS \
         TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU \
         POLICY_LR USE_KL_LOSS KL_COEF ENTROPY_COEF TOTAL_EPOCHS MAX_ROUNDS \
         MAX_TOKENS_PER_TURN MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN \
         ROLLOUT_GPU_MEMORY_UTILIZATION SAVE_FREQ RESUME_MODE \
         ACTION_FORECAST_ENABLE ACTION_FORECAST_COEF ACTION_FORECAST_K \
         ACTION_FORECAST_GATE ACTION_FORECAST_SKIP_INVALID ACTION_FORECAST_GROUP_NORM \
         ACTION_FORECAST_SUCCESS_THRESHOLD ACTION_FORECAST_MAX_LENGTH ACTION_FORECAST_SEQ SFT_MINI_BATCH_SIZE ACTION_FORECAST_LR_SCALE \
         GRPO_FILTER_DEGENERATE DRY_RUN; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=$(printf '%q' "${!v}")"; fi
done

tmux new-session -d -s "${SESSION}" \
  "cd ${ROOT} && EXP_NAME=${EXP_NAME} RUN_DIR=${RUN_DIR}${FWD} bash ${ROOT}/scripts/run_tau2_pipeline.sh 2>&1 | tee ${RUN_DIR}/pipeline.log"

echo "--------------------------------------------------"
echo "tau2 ${VARIANT} launched in tmux session: ${SESSION}"
echo "preset      : ${PRESET}"
echo "forecast    : ${ACTION_FORECAST_ENABLE:-False}${ACTION_FORECAST_ENABLE:+ (coef ${ACTION_FORECAST_COEF:-0})}"
echo "experiment  : ${EXP_NAME}"
echo "pipeline log: ${RUN_DIR}/pipeline.log"
echo "train log   : ${RUN_DIR}/train.log"
echo "attach      : tmux attach -t ${SESSION}"
echo "stop        : tmux kill-session -t ${SESSION}"
echo "--------------------------------------------------"
