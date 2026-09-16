#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/src"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:?set TRAIN_ENV to the agentgym-rl conda env}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TASK_NAME="sciworld"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

ENV_ADDR_HOST="${ENV_ADDR_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36101}"
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
PROJECT_NAME="${PROJECT_NAME:-agentgym-sciworld}"

KL_COEF="${KL_COEF:-0.001}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
SAVE_FREQ="${SAVE_FREQ:-50}"


# Plan-forecast auxiliary SFT (DEFAULT OFF): each step predict the realized next-K
# action commands (current included). Separate forward, CE loss * coef, no PG.
PLAN_FORECAST_ENABLE="${PLAN_FORECAST_ENABLE:-True}"
PLAN_FORECAST_COEF="${PLAN_FORECAST_COEF:-0.01}"
PLAN_FORECAST_K="${PLAN_FORECAST_K:-3}"
PLAN_FORECAST_GATE="${PLAN_FORECAST_GATE:-wins}"
# --- Two ORTHOGONAL group knobs (compose; both distill successes only) ---
# Group-success GATING (curriculum): filter WHICH groups' successes to distill by the
# group's success-rate. off (use PLAN_FORECAST_GATE) | low (rate<=LOW) | low_high
# (rate<=LOW OR >=HIGH, skip mid-rate groups where GRPO signal is strong). Default off.
PLAN_FORECAST_GROUP_GATE="${PLAN_FORECAST_GROUP_GATE:-off}"
PLAN_FORECAST_GROUP_LOW_THRESH="${PLAN_FORECAST_GROUP_LOW_THRESH:-0.5}"
PLAN_FORECAST_GROUP_HIGH_THRESH="${PLAN_FORECAST_GROUP_HIGH_THRESH:-1.0}"
# Group-weight NORMALIZATION (stability): give every kept GRPO group the SAME total
# plan-CE weight = the single PLAN_FORECAST_COEF, split EVENLY among its distilled
# successful trajectories. As success-rate rises mid/late training, per-traj weight
# shrinks and each group's contribution stays constant -> no SFT blow-up from more
# successful samples. Applies to whatever gating keeps.
PLAN_FORECAST_GROUP_NORM="${PLAN_FORECAST_GROUP_NORM:-True}"
# group_dedup (default True): when group_norm on, split each group's weight over its
# DISTINCT successful action-sequences instead of per-trajectory -> duplicate rollouts
# don't inflate weight (within-group action repetition is heavy mid/late training).
PLAN_FORECAST_GROUP_DEDUP="${PLAN_FORECAST_GROUP_DEDUP:-False}"
# Horizon-growth schedule (curriculum): grow the forecast target length over
# training. Format "startStep:kMin:kMax,..." (start-step semantics, last stage
# persists); each per-step sample draws k uniformly in the active [kMin,kMax] and
# the prompt/plan-block align to the realized length. Empty = OFF (use PLAN_FORECAST_K).
# First stage MUST start at 0. Example: "0:2:3,40:2:4,80:3:5".
PLAN_FORECAST_K_SCHEDULE="${PLAN_FORECAST_K_SCHEDULE:-}"
# skip_invalid: build the forecast target from only EFFECTIVE actions — drop actions
# whose env result was invalid / no-effect ("Nothing happens." / "Invalid Action." /
# "No known action..."; per-env, auto-selected by task_name). Default off.
PLAN_FORECAST_SKIP_INVALID="${PLAN_FORECAST_SKIP_INVALID:-True}"
PLAN_FORECAST_SUCCESS_THRESHOLD="${PLAN_FORECAST_SUCCESS_THRESHOLD:-0.5}"
PLAN_FORECAST_MAX_LENGTH="${PLAN_FORECAST_MAX_LENGTH:-4096}"
# plan_forecast target: action (predict realized next-K actions; block2-success,
# grounded) | subgoal (predict next-K hindsight-confirmed achieved sub-goals).
PLAN_FORECAST_TARGET="${PLAN_FORECAST_TARGET:-action}"
# plan_forecast seq: separate (block2 — synthetic prompt + bare list, standalone;
# use with inline OFF) | inline_consistent (SFT sample matches a real rollout turn
# obs->Plan+Action; use with inline ON). Auto-falls-back to separate if inline off.
PLAN_FORECAST_SEQ="${PLAN_FORECAST_SEQ:-separate}"
# plan_forecast_coef anneal: fixed | linear | power | cutoff (start = PLAN_FORECAST_COEF).
PLAN_FORECAST_COEF_ANNEAL="${PLAN_FORECAST_COEF_ANNEAL:-fixed}"
PLAN_FORECAST_COEF_END="${PLAN_FORECAST_COEF_END:-0.0}"
PLAN_FORECAST_COEF_HORIZON="${PLAN_FORECAST_COEF_HORIZON:-50}"
PLAN_FORECAST_COEF_POWER="${PLAN_FORECAST_COEF_POWER:-2.0}"
PLAN_FORECAST_COEF_CUTOFF_STEP="${PLAN_FORECAST_COEF_CUTOFF_STEP:-0}"
# Plan FORMAT reward (DEFAULT OFF): per-turn shaping bonus on advantage for a
# well-formed Thought->Plan->Action turn (counters inline-plan decay under RL).
# bonus_t = COEF*(score-BASELINE) on the turn's tokens; baseline 0.5 = symmetric.
PLAN_FORMAT_REWARD_ENABLE="${PLAN_FORMAT_REWARD_ENABLE:-False}"
PLAN_FORMAT_REWARD_COEF="${PLAN_FORMAT_REWARD_COEF:-0.05}"
PLAN_FORMAT_REWARD_BASELINE="${PLAN_FORMAT_REWARD_BASELINE:-0.5}"
PLAN_FORMAT_REWARD_CLIP="${PLAN_FORMAT_REWARD_CLIP:-0.0}"
# penalty_only: only penalize turns that DROP the plan (never reward keeping it).
PLAN_FORMAT_REWARD_PENALTY_ONLY="${PLAN_FORMAT_REWARD_PENALTY_ONLY:-True}"
# warmup: keep format reward OFF until global_step >= this (learn task first).
PLAN_FORMAT_REWARD_WARMUP_STEPS="${PLAN_FORMAT_REWARD_WARMUP_STEPS:-10}"
# Block 1 (inline plan, DEFAULT OFF): model writes its next-K-action plan inside
# the THOUGHT each turn (auto-eats PG, env still parses Action:). Pairs with block 2.
PLAN_INLINE_ENABLE="${PLAN_INLINE_ENABLE:-False}"
PLAN_INLINE_K="${PLAN_INLINE_K:-${PLAN_FORECAST_K}}"
# inline plan style: actions (next-K actions) | todo (checkable sub-goal TODO
# list with (done) marks; pair with PLAN_FORECAST_TARGET=subgoal).
PLAN_INLINE_STYLE="${PLAN_INLINE_STYLE:-actions}"
# per-turn reminder: re-state the Plan request after EVERY obs (one-time decays).
PLAN_INLINE_PER_TURN="${PLAN_INLINE_PER_TURN:-False}"
# inline warmup: use ORIGINAL prompt until global_step >= this, then introduce
# the plan prompt (cold-start: let task competence build before planning).
PLAN_INLINE_WARMUP_STEPS="${PLAN_INLINE_WARMUP_STEPS:-0}"
# think reminder (alternative to inline plan, mutually exclusive): per-turn nudge
# to reason in a Thought before the Action, WITHOUT forcing a Plan.
THINK_REMINDER_ENABLE="${THINK_REMINDER_ENABLE:-False}"


EXP_NAME="${EXP_NAME:-sciworld_grpo_qwen2.5_3b_$(date -u +%Y%m%d_%H%M%S)}"
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
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/train/sciworld_train.json}"
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
    +data.plan_inline_enable="${PLAN_INLINE_ENABLE}" \
    +data.plan_inline_k="${PLAN_INLINE_K}" \
    +data.plan_inline_style="${PLAN_INLINE_STYLE}" \
    +data.plan_inline_per_turn="${PLAN_INLINE_PER_TURN}" \
    +data.plan_inline_warmup_steps="${PLAN_INLINE_WARMUP_STEPS}" \
    +data.think_reminder_enable="${THINK_REMINDER_ENABLE}" \
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
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    +actor_rollout_ref.actor.plan_forecast_enable="${PLAN_FORECAST_ENABLE}" \
    +actor_rollout_ref.actor.plan_forecast_coef="${PLAN_FORECAST_COEF}" \
    +actor_rollout_ref.actor.plan_forecast_k="${PLAN_FORECAST_K}" \
    +actor_rollout_ref.actor.plan_forecast_gate="${PLAN_FORECAST_GATE}" \
    +actor_rollout_ref.actor.plan_forecast_group_gate="${PLAN_FORECAST_GROUP_GATE}" \
    +actor_rollout_ref.actor.plan_forecast_group_low_thresh="${PLAN_FORECAST_GROUP_LOW_THRESH}" \
    +actor_rollout_ref.actor.plan_forecast_group_high_thresh="${PLAN_FORECAST_GROUP_HIGH_THRESH}" \
    +actor_rollout_ref.actor.plan_forecast_group_norm="${PLAN_FORECAST_GROUP_NORM}" \
    +actor_rollout_ref.actor.plan_forecast_group_dedup="${PLAN_FORECAST_GROUP_DEDUP}" \
    +actor_rollout_ref.actor.plan_forecast_k_schedule="'${PLAN_FORECAST_K_SCHEDULE}'" \
    +actor_rollout_ref.actor.plan_forecast_skip_invalid="${PLAN_FORECAST_SKIP_INVALID}" \
    +actor_rollout_ref.actor.plan_forecast_success_threshold="${PLAN_FORECAST_SUCCESS_THRESHOLD}" \
    +actor_rollout_ref.actor.plan_forecast_max_length="${PLAN_FORECAST_MAX_LENGTH}" \
    +actor_rollout_ref.actor.plan_forecast_target="${PLAN_FORECAST_TARGET}" \
    +actor_rollout_ref.actor.plan_forecast_seq="${PLAN_FORECAST_SEQ}" \
    +actor_rollout_ref.actor.plan_forecast_coef_anneal="${PLAN_FORECAST_COEF_ANNEAL}" \
    +actor_rollout_ref.actor.plan_forecast_coef_end="${PLAN_FORECAST_COEF_END}" \
    +actor_rollout_ref.actor.plan_forecast_coef_horizon="${PLAN_FORECAST_COEF_HORIZON}" \
    +actor_rollout_ref.actor.plan_forecast_coef_power="${PLAN_FORECAST_COEF_POWER}" \
    +actor_rollout_ref.actor.plan_forecast_coef_cutoff_step="${PLAN_FORECAST_COEF_CUTOFF_STEP}" \
    +actor_rollout_ref.actor.plan_format_reward_enable="${PLAN_FORMAT_REWARD_ENABLE}" \
    +actor_rollout_ref.actor.plan_format_reward_coef="${PLAN_FORMAT_REWARD_COEF}" \
    +actor_rollout_ref.actor.plan_format_reward_baseline="${PLAN_FORMAT_REWARD_BASELINE}" \
    +actor_rollout_ref.actor.plan_format_reward_clip="${PLAN_FORMAT_REWARD_CLIP}" \
    +actor_rollout_ref.actor.plan_format_reward_penalty_only="${PLAN_FORMAT_REWARD_PENALTY_ONLY}" \
    +actor_rollout_ref.actor.plan_format_reward_warmup_steps="${PLAN_FORMAT_REWARD_WARMUP_STEPS}"
