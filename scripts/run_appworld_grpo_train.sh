#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/src"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:?set TRAIN_ENV to the agentgym-rl conda env}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}"
TASK_NAME="appworld"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

ENV_ADDR_HOST="${ENV_ADDR_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36301}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
# Env servers per GPU. Must match the launcher's ENVS_PER_GPU.
# AppWorld needs one process per concurrent trajectory, not merely for speed: its
# supervisor's "active task" is process-global state. When one episode calls
# complete_task, every other episode in that process is immediately marked done and
# evaluated against the polluted state. Concurrent trajectories are
# TRAIN_BATCH_SIZE * ROLLOUT_N = 128, so on 8 GPUs that is 16 per GPU.
ENVS_PER_GPU="${ENVS_PER_GPU:-16}"
NUM_ENVS=$((NUM_GPUS * ENVS_PER_GPU))

# Automatically construct comma-separated list of environment addresses
ENV_ADDR_LIST=""
for i in $(seq 0 $((NUM_ENVS - 1))); do
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
PROJECT_NAME="${PROJECT_NAME:-agentgym-appworld}"

KL_COEF="${KL_COEF:-0.01}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
# 14B with 20k-token responses: sequence packing, optimizer offload and the dynamic
# micro-batching knobs are what make it fit.
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
ULYSSES_SP="${ULYSSES_SP:-1}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
# Note: TE_MIX=fullvocab does not support remove_padding yet and fails fast if both are on.
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
# Exact step budget; null trains for TOTAL_EPOCHS.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20480}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-1024}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
SAVE_FREQ="${SAVE_FREQ:-25}"


# Action-forecast auxiliary SFT (DEFAULT OFF): each step predict the realized next-K
# action commands (current included). Separate forward, CE loss * coef, no PG.
ACTION_FORECAST_ENABLE="${ACTION_FORECAST_ENABLE:-True}"
ACTION_FORECAST_COEF="${ACTION_FORECAST_COEF:-0.01}"
ACTION_FORECAST_K="${ACTION_FORECAST_K:-3}"
ACTION_FORECAST_GATE="${ACTION_FORECAST_GATE:-wins}"
# Group-weight NORMALIZATION (stability): give every kept GRPO group the SAME total
# forecast-CE weight = the single ACTION_FORECAST_COEF, split EVENLY among its distilled
# successful trajectories. As success-rate rises mid/late training, per-traj weight
# shrinks and each group's contribution stays constant -> no SFT blow-up from more
# successful samples.
ACTION_FORECAST_GROUP_NORM="${ACTION_FORECAST_GROUP_NORM:-True}"
# skip_invalid: build the forecast target from only EFFECTIVE actions — drop actions
# whose env result was invalid / no-effect ("Nothing happens." / "Invalid Action." /
# "No known action..."; per-env, auto-selected by task_name). Default off.
ACTION_FORECAST_SKIP_INVALID="${ACTION_FORECAST_SKIP_INVALID:-True}"
ACTION_FORECAST_SUCCESS_THRESHOLD="${ACTION_FORECAST_SUCCESS_THRESHOLD:-0.5}"
ACTION_FORECAST_MAX_LENGTH="${ACTION_FORECAST_MAX_LENGTH:-4096}"

# Temporal Ensembling (see verl/agent_trainer/ppo/temporal_ensemble.py). Off by default;
# with TE_ENABLE=False none of the TE code runs and the batch is unchanged.
TE_ENABLE="${TE_ENABLE:-False}"
TE_LAMBDA="${TE_LAMBDA:-0.0}"          # 0 = dry-run: diagnostics only, no gradient
TE_ETA="${TE_ETA:-0.5}"                # forecast share of the target mixture
TE_KL_TYPE="${TE_KL_TYPE:-low_var_kl}"
# fullvocab: full-vocabulary per-token KL (the working form). token / seq are kept only
# to reproduce runs where they degenerated (response-length blow-up; entropy blow-up).
TE_MIX="${TE_MIX:-fullvocab}"
TE_CENTER="${TE_CENTER:-True}"         # legacy token/seq forms only
TE_WARMUP_STEPS="${TE_WARMUP_STEPS:-50}"
TE_TRAJ_SUBSAMPLE="${TE_TRAJ_SUBSAMPLE:-1.0}"
TE_MICRO_BATCH_SIZE_PER_GPU="${TE_MICRO_BATCH_SIZE_PER_GPU:-1}"

# Pin vLLM's port search. Unset, vllm's get_open_port() probes with bind("",0) and the
# port can be taken before it is used (TOCTOU) -- occasional with one run, frequent
# with two on the same host. Set, it takes the retry-on-OSError path.
export VLLM_PORT="${VLLM_PORT:-29700}"


EXP_NAME="${EXP_NAME:-appworld_grpo_qwen2.5_14b_$(date -u +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
# Checkpoint resume. 'auto' (default): auto-resume from the latest global_step_* in
# CKPT_DIR if present, else train from scratch. To actually resume a prior run you
# MUST reuse its EXP_NAME (CKPT_DIR is derived from it -- the default timestamped
# EXP_NAME makes a fresh dir every launch and thus never resumes). Set to a specific
# 'global_step_N' folder (abs or relative to cwd) to resume that exact step, or
# 'disable' to force from-scratch.
RESUME_MODE="${RESUME_MODE:-auto}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${RUN_DIR}/rollout_logs}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/train/appworld_train.json}"
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
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
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
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    +actor_rollout_ref.actor.action_forecast_enable="${ACTION_FORECAST_ENABLE}" \
    +actor_rollout_ref.actor.action_forecast_coef="${ACTION_FORECAST_COEF}" \
    +actor_rollout_ref.actor.action_forecast_k="${ACTION_FORECAST_K}" \
    +actor_rollout_ref.actor.action_forecast_gate="${ACTION_FORECAST_GATE}" \
    +actor_rollout_ref.actor.action_forecast_group_norm="${ACTION_FORECAST_GROUP_NORM}" \
    +actor_rollout_ref.actor.action_forecast_skip_invalid="${ACTION_FORECAST_SKIP_INVALID}" \
    +actor_rollout_ref.actor.action_forecast_success_threshold="${ACTION_FORECAST_SUCCESS_THRESHOLD}" \
    +actor_rollout_ref.actor.action_forecast_max_length="${ACTION_FORECAST_MAX_LENGTH}" \
    +actor_rollout_ref.actor.te_enable="${TE_ENABLE}" \
    +actor_rollout_ref.actor.te_lambda="${TE_LAMBDA}" \
    +actor_rollout_ref.actor.te_eta="${TE_ETA}" \
    +actor_rollout_ref.actor.te_kl_type="${TE_KL_TYPE}" \
    +actor_rollout_ref.actor.te_mix="${TE_MIX}" \
    +actor_rollout_ref.actor.te_center="${TE_CENTER}" \
    +actor_rollout_ref.actor.te_warmup_steps="${TE_WARMUP_STEPS}" \
    +actor_rollout_ref.actor.te_traj_subsample="${TE_TRAJ_SUBSAMPLE}" \
    +actor_rollout_ref.actor.te_micro_batch_size_per_gpu="${TE_MICRO_BATCH_SIZE_PER_GPU}" \
    +actor_rollout_ref.rollout.te_enable="${TE_ENABLE}"
