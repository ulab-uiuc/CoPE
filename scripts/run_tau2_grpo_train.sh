#!/usr/bin/env bash
#
# GRPO training on tau2-bench.
#
# Deliberately minimal: only the keys GRPO on this env needs. Every experimental knob
# in the webshop/alfworld scripts (wmc_erc, safe-commit, plan-forecast, HCA, ...) is
# left at its ppo_trainer.yaml default, which is off. Add them here if/when you want to
# run those ablations on tau2.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/src"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:?set TRAIN_ENV to the training conda env}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the policy model}"
TASK_NAME="tau2"

# The `agentenv` editable install in TRAIN_ENV points at a *different* AgentGym
# checkout, so Tau2EnvClient would not be importable. PYTHONPATH wins over the
# editable install's meta-path finder (which setuptools appends, i.e. after the normal
# sys.path finder), so this pins agentenv to this repo's submodule without mutating the
# shared conda env. PYTHONNOUSERSITE keeps a broken ~/.local transformers out of the way.
AGENTENV_PATH="${AGENTENV_PATH:-${ROOT}/AgentGym/agentenv}"

ENV_ADDR_HOST="${ENV_ADDR_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36201}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"

# Env-server processes per GPU. Each rank round-robins over its own shard of servers
# (verl/utils/agentgym/client.py::_select_env_addr), so >1 spreads a GPU's concurrent
# env stepping across processes instead of GIL-serialising it in one. tau2 spends most
# of its step time blocked on the user-sim HTTP call, so this matters here.
ENVS_PER_GPU="${ENVS_PER_GPU:-4}"
NUM_ENV_SERVERS=$((NUM_GPUS * ENVS_PER_GPU))

# Order matters: rank r owns addrs[r*ENVS_PER_GPU : (r+1)*ENVS_PER_GPU].
ENV_ADDR_LIST=""
for i in $(seq 0 $((NUM_ENV_SERVERS - 1))); do
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
PROJECT_NAME="${PROJECT_NAME:-agentgym-tau2}"
EXP_NAME="${EXP_NAME:-tau2_grpo_$(date -u +%Y%m%d_%H%M%S)}"

KL_COEF="${KL_COEF:-0.001}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
# InfoPO trains tau2 with the KL penalty off entirely (use_kl_loss=False, coef 0).
USE_KL_LOSS="${USE_KL_LOSS:-True}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"

# tau2 needs a lot more context than webshop: the retail policy plus the rendered tool
# signatures alone are ~3k tokens of prompt, and conversations run longer.
MAX_ROUNDS="${MAX_ROUNDS:-30}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-1024}"
# Tuned for 40GB A100s: vLLM and the FSDP actor share each card, and tau2's long
# context makes the KV cache expensive. Raise toward 0.8 on 80GB cards.
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
# Sequences here are long and highly variable in length; without this the actor update
# pays for MAX_PROMPT+MAX_RESPONSE padding on every sample and OOMs on a 40GB card.
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
SAVE_FREQ="${SAVE_FREQ:-25}"

RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${RUN_DIR}/rollout_logs}"
# Must have been generated from the same domain/split the env servers were started with.
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/tau2_retail_train.json}"
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

# ~/.cache/huggingface is a symlink to a path that no longer exists on these hosts, so
# datasets.load_dataset() dies in RLHFDataset before training starts. Point the caches
# somewhere writable unless the caller already has.
export HF_HOME="${HF_HOME:-${ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"

cd "${TRAIN_CODE_DIR}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${AGENTENV_PATH}" \
  HF_HOME="${HF_HOME}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
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
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    data.train_file="${TRAIN_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
    actor_rollout_ref.agentgym.env_addr="'${ENV_ADDR}'" \
    actor_rollout_ref.agentgym.timeout=2400 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-True}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEF}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}"
