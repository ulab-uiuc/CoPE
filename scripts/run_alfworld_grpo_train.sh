#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/src"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:?set TRAIN_ENV to the agentgym-rl conda env}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
# Qwen/Qwen2.5-7B-Instruct
# Qwen/Qwen2.5-3B-Instruct
TASK_NAME="alfworld"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

ENV_ADDR_HOST="${ENV_ADDR_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"

# Automatically construct comma-separated list of environment addresses
ENV_ADDR_LIST=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((BASE_PORT + i))
  ADDR="http://${ENV_ADDR_HOST}:${PORT}"
  if [[ -z "${ENV_ADDR_LIST}" ]]; then
    ENV_ADDR_LIST="${ADDR}"
  else
    ENV_ADDR_LIST="${ENV_ADDR_LIST},${ADDR}"
  fi
done
ENV_ADDR="${ENV_ADDR:-${ENV_ADDR_LIST}}"
echo "Using ENV_ADDR: ${ENV_ADDR}"

WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-alfworld}"

KL_COEF="${KL_COEF:-0.001}"
ENTROPY_COEF="${ENTROPY_COEF:-0.002}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
# Hard cap on optimizer steps (null = use len(dataloader)*total_epochs). Set a
# small value (e.g. 12) for short validation runs.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"
SAVE_FREQ="${SAVE_FREQ:-25}"


# Plan-forecast auxiliary SFT (DEFAULT OFF): each step predict the realized next-K
# action commands (current included). Separate forward, CE loss * coef, no PG.
# gate=wins -> only winning trajectories; gate=all -> every trajectory.
PLAN_FORECAST_ENABLE="${PLAN_FORECAST_ENABLE:-True}"
PLAN_FORECAST_COEF="${PLAN_FORECAST_COEF:-0.01}"
PLAN_FORECAST_K="${PLAN_FORECAST_K:-3}"
# skip_invalid: build the forecast target from only EFFECTIVE actions — drop actions
# whose env result was invalid / no-effect ("Nothing happens." / "Invalid Action." /
# "No known action..."; per-env, auto-selected by task_name). Default off.
PLAN_FORECAST_SKIP_INVALID="${PLAN_FORECAST_SKIP_INVALID:-True}"
# gate=wins: forecast-SFT only on winning trajectories (block2-success, proven).
PLAN_FORECAST_GATE="${PLAN_FORECAST_GATE:-wins}"
# Group-weight NORMALIZATION (stability): give every kept GRPO group the SAME total
# plan-CE weight = the single PLAN_FORECAST_COEF, split EVENLY among its distilled
# successful trajectories. As success-rate rises mid/late training, per-traj weight
# shrinks and each group's contribution stays constant -> no SFT blow-up from more
# successful samples.
PLAN_FORECAST_GROUP_NORM="${PLAN_FORECAST_GROUP_NORM:-True}"
PLAN_FORECAST_SUCCESS_THRESHOLD="${PLAN_FORECAST_SUCCESS_THRESHOLD:-0.5}"
PLAN_FORECAST_MAX_LENGTH="${PLAN_FORECAST_MAX_LENGTH:-4096}"
# SFT-ablation (RFT) control — MUTUALLY EXCLUSIVE with plan_forecast (asserted at
# init). When ON: one extra SFT round behavior-cloning THIS step's winning trajs'
# real assistant turns (vs plan_forecast's constructed forecast target). Same
# optimizer path + win-gate; toggle only these two for a clean A/B. DEFAULT OFF.
SFT_ABLATION_ENABLE="${SFT_ABLATION_ENABLE:-False}"
SFT_ABLATION_COEF="${SFT_ABLATION_COEF:-0.01}"      # match PLAN_FORECAST_COEF for a fair control
SFT_ABLATION_GATE="${SFT_ABLATION_GATE:-wins}"      # wins (success trajs only) | all
# traj_lm: full-sequence next-token CE over the WHOLE trajectory (env obs AND the
# agent's own tokens), coef>0 = on. Default 0 = off.
TRAJ_LM_COEF="${TRAJ_LM_COEF:-0}"
# traj_lm trajectory gate: all (clone every rollout, default = old behavior) |
# wins (only clone trajectories with GRPO advantage>0, i.e. better than group mean).
# 'all' BC's losing behavior too -> anchors policy to base & slows early learning;
# 'wins' is recommended (mirrors block2 forecast's gate=wins).
TRAJ_LM_GATE="${TRAJ_LM_GATE:-wins}"


EXP_NAME="${EXP_NAME:-alfworld_grpo_qwen2.5_7b_$(date -u +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${RUN_DIR}/rollout_logs}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/train/alfworld_train.json}"
LOG_PATH="${LOG_PATH:-}"

mkdir -p "${CKPT_DIR}" "${RUN_DIR}" "${ROLLOUT_LOG_DIR}"
if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${TRAIN_CODE_DIR}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  VLLM_USE_MODELSCOPE=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  python -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
    data.train_file="${TRAIN_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
    actor_rollout_ref.agentgym.env_addr="'${ENV_ADDR}'" \
    actor_rollout_ref.agentgym.timeout=2400 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEF} \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_tokens="${MAX_TOKENS_PER_TURN}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    +actor_rollout_ref.actor.plan_forecast_enable="${PLAN_FORECAST_ENABLE}" \
    +actor_rollout_ref.actor.plan_forecast_coef="${PLAN_FORECAST_COEF}" \
    +actor_rollout_ref.actor.plan_forecast_k="${PLAN_FORECAST_K}" \
    +actor_rollout_ref.actor.plan_forecast_skip_invalid="${PLAN_FORECAST_SKIP_INVALID}" \
    +actor_rollout_ref.actor.plan_forecast_gate="${PLAN_FORECAST_GATE}" \
    +actor_rollout_ref.actor.plan_forecast_group_norm="${PLAN_FORECAST_GROUP_NORM}" \
    +actor_rollout_ref.actor.plan_forecast_success_threshold="${PLAN_FORECAST_SUCCESS_THRESHOLD}" \
    +actor_rollout_ref.actor.plan_forecast_max_length="${PLAN_FORECAST_MAX_LENGTH}" \
    +actor_rollout_ref.actor.sft_ablation_enable="${SFT_ABLATION_ENABLE}" \
    +actor_rollout_ref.actor.sft_ablation_coef="${SFT_ABLATION_COEF}" \
    +actor_rollout_ref.actor.sft_ablation_gate="${SFT_ABLATION_GATE}" \
    +actor_rollout_ref.actor.traj_lm_coef="${TRAJ_LM_COEF}" \
    +actor_rollout_ref.actor.traj_lm_gate="${TRAJ_LM_GATE}"
