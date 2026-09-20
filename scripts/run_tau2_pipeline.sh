#!/usr/bin/env bash
#
# τ²-bench GRPO, end to end, in the foreground: customer -> env cluster -> training.
# One process tree, one log directory, no scheduler. This is what
# scripts/launch_tau2_grpo_tmux.sh runs, and what
# scripts/launch_tau2_grpo_tmux.sh runs inside a tmux session.
#
#   PRESET=infopo bash scripts/run_tau2_pipeline.sh     # InfoPO's protocol (default)
#   PRESET=repo   bash scripts/run_tau2_pipeline.sh     # this repo's retail/ReAct setup
#   DRY_RUN=1     bash scripts/run_tau2_pipeline.sh     # check prerequisites + print config
#
# Presets fill every knob; any of them can still be overridden from the environment
# (TRAIN_BATCH_SIZE=..., MAX_ROUNDS=..., TAU2_DOMAIN=..., ...). Everything that
# decided whether a run works at all is handled here so a first run does not have to
# rediscover it:
#
#   * The env servers run in their own Python 3.12 env (tau2 needs >=3.12, the
#     trainer is on 3.10), reached over HTTP. TAU2_ENV points at it.
#   * TAU2_FORCE_DONE_AFTER is set to MAX_ROUNDS on the env side. tau2 only scores an
#     episode once its orchestrator terminates; without this every episode that merely
#     runs out of turns comes back as an unevaluated zero.
#   * Every listening port sits below 32768. The kernel hands out ephemeral source
#     ports from 32768-60999, and a listener inside that range can lose its port to an
#     outbound connection (EADDRINUSE on startup, which is how one run died).
#   * Checkpoints of a 7B are ~86GB each; CKPT_DIR defaults under the repo but should
#     point at a large disk.
#   * The rollout's context budget is min(MAX_MODEL_LEN, MAX_PROMPT_LENGTH +
#     MAX_RESPONSE_LENGTH); conversations that outgrow it are ended, not crashed.
#
# Customer (user simulator):
#   USERSIM_MODE=hosted   gpt-4o-mini through litellm; needs KEY_FILE. InfoPO's protocol
#                         and the only one whose absolute numbers are comparable.
#   USERSIM_MODE=local    a vLLM server of USERSIM_MODEL on USERSIM_GPU (not a training
#                         GPU). Free; only base-vs-trained deltas are meaningful.
#   Default: hosted if KEY_FILE is readable, else local.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ---- environments ---------------------------------------------------------------------
# Picked with a loop, not `ls ... | head -1`: under set -e a command substitution whose
# pipeline fails (ls exits 2 when any listed file is missing) kills the script silently.
if [[ -z "${CONDA_SH:-}" ]]; then
  for c in /opt/conda/etc/profile.d/conda.sh /usr/local/anaconda3/etc/profile.d/conda.sh \
           "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    [[ -f "${c}" ]] && { CONDA_SH="${c}"; break; }
  done
fi
[[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]] || { echo "FATAL: set CONDA_SH to your conda.sh" >&2; exit 1; }
TRAIN_ENV="${TRAIN_ENV:-${CONDA_PREFIX:-}}"
[[ -n "${TRAIN_ENV}" ]] || { echo "FATAL: set TRAIN_ENV to the training conda env (python 3.10, torch, vllm)" >&2; exit 1; }
TAU2_ENV="${TAU2_ENV:-${ROOT}/envs/tau2}"
[[ -x "${TAU2_ENV}/bin/tau2-env" ]] || { echo "FATAL: ${TAU2_ENV}/bin/tau2-env missing -- see README 'Setup' for the tau2 env" >&2; exit 1; }
TAU2_BENCH_DIR="${TAU2_BENCH_DIR:-${ROOT}/tau2-bench}"
[[ -d "${TAU2_BENCH_DIR}/data" ]] || { echo "FATAL: tau2-bench not found at ${TAU2_BENCH_DIR} (README 'Setup')" >&2; exit 1; }
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-${TAU2_BENCH_DIR}/data}"
[[ -f "${ROOT}/AgentGym/agentenv/agentenv/envs/tau2.py" ]] || { echo "FATAL: AgentGym submodule not initialized (git submodule update --init AgentGym)" >&2; exit 1; }
bash "${ROOT}/scripts/patch_tau2_bench.sh" "${TAU2_BENCH_DIR}"

export CONDA_SH TRAIN_ENV TAU2_ENV

# ---- GPUs -------------------------------------------------------------------------------
TRAIN_GPUS="${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a _G <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS=${#_G[@]}

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
if [[ ! -d "${MODEL_PATH}" && "${HF_HUB_OFFLINE:-1}" == "1" ]]; then
  echo "FATAL: MODEL_PATH=${MODEL_PATH} is not a local directory and HF_HUB_OFFLINE=1;" \
       "pass a local snapshot path or HF_HUB_OFFLINE=0" >&2; exit 1
fi

# ---- customer --------------------------------------------------------------------------
KEY_FILE="${KEY_FILE:-${ROOT}/.secrets/openai_api_key}"
if [[ -z "${USERSIM_MODE:-}" ]]; then
  if [[ -r "${KEY_FILE}" ]]; then USERSIM_MODE=hosted; else USERSIM_MODE=local; fi
fi
USERSIM_LLM="${USERSIM_LLM:-openai/gpt-4o-mini}"
USERSIM_MODEL="${USERSIM_MODEL:-${MODEL_PATH}}"
USERSIM_GPU="${USERSIM_GPU:-}"
USERSIM_PORT="${USERSIM_PORT:-20301}"
USERSIM_GPU_MEM_UTIL="${USERSIM_GPU_MEM_UTIL:-0.45}"
USERSIM_MAX_MODEL_LEN="${USERSIM_MAX_MODEL_LEN:-32768}"
if [[ "${USERSIM_MODE}" == "hosted" ]]; then
  [[ -r "${KEY_FILE}" ]] || { echo "FATAL: USERSIM_MODE=hosted needs an API key at ${KEY_FILE}" >&2; exit 1; }
elif [[ "${USERSIM_MODE}" == "local" ]]; then
  [[ -n "${USERSIM_GPU}" ]] || { echo "FATAL: USERSIM_MODE=local needs USERSIM_GPU (a GPU not in TRAIN_GPUS=${TRAIN_GPUS})" >&2; exit 1; }
  case ",${TRAIN_GPUS}," in *",${USERSIM_GPU},"*) echo "FATAL: USERSIM_GPU ${USERSIM_GPU} is inside TRAIN_GPUS ${TRAIN_GPUS}" >&2; exit 1;; esac
else
  echo "FATAL: USERSIM_MODE must be hosted|local, got ${USERSIM_MODE}" >&2; exit 1
fi

# ---- presets ----------------------------------------------------------------------------
PRESET="${PRESET:-infopo}"
case "${PRESET}" in
  infopo)
    # InfoPO's tau2 training protocol (examples/tau2/train.sh + Table 4 in their repo,
    # verified against their released train.parquet): all three domains jointly,
    # tau2's native tool-calling interface, tau2's stock agent prompt as the system
    # message, EvaluationType.ALL reward, gpt-4o-mini customer at temperature 0.7.
    # Only the algorithm is this repo's plain GRPO rather than info_grpo.
    : "${TAU2_DOMAIN:=retail+airline+telecom}"
    : "${TRAIN_FILE:=${ROOT}/data/tau2_retail-airline-telecom_train.json}"
    : "${TAU2_REWARD_SHAPE:=binary}" "${TAU2_REWARD_BASIS:=all}"
    : "${TAU2_PROMPT_VARIANT:=native}" "${NATIVE_TOOLS:=True}"
    : "${TAU2_USER_TEMPERATURE:=0.7}"
    : "${TRAIN_BATCH_SIZE:=32}" "${ROLLOUT_N:=5}" "${PPO_MINI_BATCH_SIZE:=16}"
    # InfoPO runs micro-batch 2 with activation offload, which this verl lacks. Native
    # tool-calling turns are long (responses average ~2.9k tokens, prompt + response
    # reach the 24576 window), and two such sequences in one backward OOM'd a 95GB card
    # at step 3. Micro-batch 1 is the same gradient, computed in two halves.
    : "${PPO_MICRO_BATCH_SIZE_PER_GPU:=1}" "${POLICY_LR:=1e-6}"
    : "${USE_KL_LOSS:=False}" "${KL_COEF:=0}" "${ENTROPY_COEF:=0.001}"
    : "${TOTAL_EPOCHS:=10}" "${MAX_ROUNDS:=50}" "${MAX_TOKENS_PER_TURN:=1024}"
    : "${MAX_PROMPT_LENGTH:=8192}" "${MAX_RESPONSE_LENGTH:=16384}" "${MAX_MODEL_LEN:=24576}"
    : "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.50}" "${SAVE_FREQ:=15}" "${TAU2_MAX_STEPS:=200}"
    ;;
  repo)
    # This repo's original retail setup: ReAct text protocol with the `strict` prompt,
    # env-state reward with dense partial credit, deterministic customer.
    : "${TAU2_DOMAIN:=retail}"
    : "${TRAIN_FILE:=${ROOT}/data/tau2_retail_train.json}"
    : "${TAU2_REWARD_SHAPE:=dense}" "${TAU2_REWARD_BASIS:=env}"
    : "${TAU2_PROMPT_VARIANT:=strict}"
    : "${TAU2_USER_TEMPERATURE:=0.0}"
    : "${TRAIN_BATCH_SIZE:=8}" "${ROLLOUT_N:=8}" "${PPO_MINI_BATCH_SIZE:=4}"
    : "${PPO_MICRO_BATCH_SIZE_PER_GPU:=1}" "${POLICY_LR:=1e-6}"
    : "${USE_KL_LOSS:=True}" "${KL_COEF:=0.001}" "${ENTROPY_COEF:=0.001}"
    : "${TOTAL_EPOCHS:=5}" "${MAX_ROUNDS:=20}" "${MAX_TOKENS_PER_TURN:=512}"
    : "${MAX_PROMPT_LENGTH:=8192}" "${MAX_RESPONSE_LENGTH:=16384}" "${MAX_MODEL_LEN:=32768}"
    : "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.45}" "${SAVE_FREQ:=25}" "${TAU2_MAX_STEPS:=200}"
    ;;
  *) echo "FATAL: PRESET must be infopo|repo, got ${PRESET}" >&2; exit 1 ;;
esac
export TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU POLICY_LR \
       USE_KL_LOSS KL_COEF ENTROPY_COEF TOTAL_EPOCHS MAX_ROUNDS MAX_TOKENS_PER_TURN \
       MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN ROLLOUT_GPU_MEMORY_UTILIZATION SAVE_FREQ
[[ -n "${NATIVE_TOOLS:-}" ]] && export NATIVE_TOOLS
TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-train}"
TAU2_DOMAIN_SLUG="${TAU2_DOMAIN//+/-}"
[[ -f "${TRAIN_FILE}" ]] || { echo "FATAL: TRAIN_FILE ${TRAIN_FILE} not found" >&2; exit 1; }

if [[ "${TAU2_PROMPT_VARIANT}" == "native" ]] && ! grep -q "_native_system_prompt" "${ROOT}/AgentGym/agentenv-tau2/agentenv_tau2/environment.py"; then
  echo "FATAL: this AgentGym checkout has no 'native' prompt variant. Update the submodule," \
       "(git submodule update --init AgentGym, or check out the tau2-native-protocol branch of PolarisDane/Agentgym#3)" >&2
  exit 1
fi

# verl asserts on both at startup; failing here is cheaper than after the engines load.
if (( (TRAIN_BATCH_SIZE * ROLLOUT_N) % NUM_TRAIN_GPUS != 0 )); then
  echo "FATAL: TRAIN_BATCH_SIZE*ROLLOUT_N = $((TRAIN_BATCH_SIZE * ROLLOUT_N)) is not divisible by ${NUM_TRAIN_GPUS} training GPUs" \
       "(3 GPUs: try TRAIN_BATCH_SIZE=30 PPO_MINI_BATCH_SIZE=15)" >&2; exit 1
fi
if (( (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % NUM_TRAIN_GPUS != 0 )); then
  echo "FATAL: PPO_MINI_BATCH_SIZE*ROLLOUT_N = $((PPO_MINI_BATCH_SIZE * ROLLOUT_N)) is not divisible by ${NUM_TRAIN_GPUS} training GPUs" >&2; exit 1
fi

# ---- run layout ------------------------------------------------------------------------
BASE_PORT="${BASE_PORT:-20401}"
ENVS_PER_GPU="${ENVS_PER_GPU:-4}"
NUM_ENVS=$(( NUM_TRAIN_GPUS * ENVS_PER_GPU ))
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-tau2_${PRESET}_${TAU2_DOMAIN_SLUG}_grpo_${RUN_TS}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
export CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
mkdir -p "${RUN_DIR}" "${CKPT_DIR}"

export HF_HOME="${HF_HOME:-${ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONNOUSERSITE=1
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
mkdir -p "${HF_DATASETS_CACHE}"

echo "=== tau2 GRPO ${EXP_NAME} ==="
echo "preset     : ${PRESET}"
echo "policy     : ${MODEL_PATH}"
echo "train GPUs : ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} ranks)   env servers: ${NUM_ENVS} from port ${BASE_PORT}"
if [[ "${USERSIM_MODE}" == "hosted" ]]; then echo "customer   : ${USERSIM_LLM} (hosted, key ${KEY_FILE})"; else echo "customer   : ${USERSIM_MODEL} on GPU ${USERSIM_GPU} (local, port ${USERSIM_PORT})"; fi
echo "domains    : ${TAU2_DOMAIN}/${TAU2_TASK_SPLIT}   file: $(basename "${TRAIN_FILE}")"
echo "reward     : ${TAU2_REWARD_SHAPE}/${TAU2_REWARD_BASIS}   prompt=${TAU2_PROMPT_VARIANT}   native_tools=${NATIVE_TOOLS:-False}   user_temp=${TAU2_USER_TEMPERATURE}"
echo "batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} mini=${PPO_MINI_BATCH_SIZE} rounds=${MAX_ROUNDS} epochs=${TOTAL_EPOCHS} lr=${POLICY_LR} kl=${USE_KL_LOSS}"
if [[ "${ACTION_FORECAST_ENABLE:-False}" == "True" ]]; then
  echo "forecast   : ON  coef=${ACTION_FORECAST_COEF:-0} k=${ACTION_FORECAST_K:-3} gate=${ACTION_FORECAST_GATE:-wins} skip_invalid=${ACTION_FORECAST_SKIP_INVALID:-True} group_norm=${ACTION_FORECAST_GROUP_NORM:-True} max_len=${ACTION_FORECAST_MAX_LENGTH:-4096} sft_mini_batch=${SFT_MINI_BATCH_SIZE:-${PPO_MINI_BATCH_SIZE}} lr_scale=${ACTION_FORECAST_LR_SCALE:-1.0}"
else
  echo "forecast   : off (plain GRPO)"
fi
echo "ckpt dir   : ${CKPT_DIR}"
echo "run dir    : ${RUN_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: configuration validated, nothing launched."
  exit 0
fi

cleanup() {
  echo "=== cleanup ==="
  [[ -n "${ENVSVC_PID:-}" ]] && kill -- -"${ENVSVC_PID}" 2>/dev/null || true
  pkill -f "agentenv_tau2:app" 2>/dev/null || true
  [[ -n "${USERSIM_PID:-}" ]] && kill "${USERSIM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 1. customer -------------------------------------------------------------------------
if [[ "${USERSIM_MODE}" == "local" ]]; then
  echo "--- starting local customer vLLM ---"
  USERSIM_MODEL="${USERSIM_MODEL}" SERVED_NAME=user-sim PORT="${USERSIM_PORT}" HOST=127.0.0.1 \
    USERSIM_GPU="${USERSIM_GPU}" USERSIM_TP=1 USERSIM_GPU_MEM_UTIL="${USERSIM_GPU_MEM_UTIL}" \
    USERSIM_MAX_MODEL_LEN="${USERSIM_MAX_MODEL_LEN}" USERSIM_ENV="${TRAIN_ENV}" CONDA_SH="${CONDA_SH}" \
    LOG_DIR="${RUN_DIR}" bash "${ROOT}/scripts/run_tau2_usersim_server.sh" &
  USERSIM_PID=$!
  for _ in $(seq 1 150); do
    curl --noproxy '*' -sf "http://127.0.0.1:${USERSIM_PORT}/v1/models" >/dev/null && break; sleep 5
  done
  curl --noproxy '*' -sf "http://127.0.0.1:${USERSIM_PORT}/v1/models" >/dev/null \
    || { echo "FATAL: customer vLLM did not come up"; tail -40 "${RUN_DIR}/usersim.log" || true; exit 1; }
  echo "customer healthy on ${USERSIM_PORT}"
  ENV_USER_ARGS=(TAU2_USER_LLM="openai/user-sim" TAU2_USER_API_BASE="http://127.0.0.1:${USERSIM_PORT}/v1")
else
  echo "--- customer: hosted ${USERSIM_LLM}, no local server ---"
  # The key reaches the user simulator via the file, and any other tau2 LLM call
  # (none at tau2 c5b2d22; the NL judge is never invoked there) via the variable.
  ENV_USER_ARGS=(TAU2_USER_LLM="${USERSIM_LLM}" TAU2_USER_API_KEY_FILE="${KEY_FILE}"
                 OPENAI_API_KEY="$(tr -d '\r\n' < "${KEY_FILE}")")
fi

# ---- 2. env cluster ----------------------------------------------------------------------
echo "--- starting ${NUM_ENVS} tau2 env servers ---"
setsid env \
  NUM_ENVS="${NUM_ENVS}" BASE_PORT="${BASE_PORT}" \
  TAU2_DOMAIN="${TAU2_DOMAIN}" TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT}" \
  TAU2_REWARD_BASIS="${TAU2_REWARD_BASIS}" TAU2_REWARD_SHAPE="${TAU2_REWARD_SHAPE}" \
  TAU2_PROMPT_VARIANT="${TAU2_PROMPT_VARIANT}" TAU2_MAX_STEPS="${TAU2_MAX_STEPS}" \
  TAU2_FORCE_DONE_AFTER="${MAX_ROUNDS}" TAU2_USER_TEMPERATURE="${TAU2_USER_TEMPERATURE}" \
  "${ENV_USER_ARGS[@]}" \
  TAU2_ENV="${TAU2_ENV}" CONDA_SH="${CONDA_SH}" TAU2_DATA_DIR="${TAU2_DATA_DIR}" \
  NO_PROXY="${NO_PROXY}" no_proxy="${no_proxy}" \
  LOG_DIR="${RUN_DIR}/env_cluster" \
  bash "${ROOT}/scripts/run_tau2_env_service.sh" &
ENVSVC_PID=$!
for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  for _ in $(seq 1 90); do curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/" >/dev/null && break; sleep 2; done
  curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/" >/dev/null \
    || { echo "FATAL: env server on ${PORT} did not come up"; tail -60 "${RUN_DIR}/env_cluster/env_${PORT}.log" || true; exit 1; }
done
echo "all ${NUM_ENVS} env servers healthy: $(curl --noproxy '*' -s "http://127.0.0.1:${BASE_PORT}/config")"

# ---- 3. training ---------------------------------------------------------------------------
echo "--- starting GRPO training ---"
# The trainer's step metrics are printed by a Ray task (main_task) and reach train.log
# only through Ray's worker->driver log forwarding, which was observed to stop
# mid-run while training carried on. Ray always writes that task's stdout/stderr to
# its session directory, so link them into the run directory as soon as they exist:
# runlogs/<exp>/main_task.out is the authoritative metrics log.
# The task is identified by the pid Ray prints on its forwarded lines ("(main_task
# pid=N)"), never by session_latest: for the first seconds that link still names the
# previous run's session, whose main_task log would satisfy any content check.
(
  RAY_ROOT="${RAY_TMPDIR:-/tmp}/ray"
  for _ in $(seq 1 180); do
    pid=$(grep -o -m1 'main_task pid=[0-9]*' "${RUN_DIR}/train.log" 2>/dev/null | grep -o '[0-9]*$' || true)
    f=""
    [[ -n "${pid}" ]] && f=$(ls -t "${RAY_ROOT}"/session_*/logs/worker-*-"${pid}".out 2>/dev/null | head -1 || true)
    if [[ -n "${f}" ]]; then
      ln -sfn "${f}" "${RUN_DIR}/main_task.out"; ln -sfn "${f%.out}.err" "${RUN_DIR}/main_task.err"
      echo "main_task logs linked: ${RUN_DIR}/main_task.out -> ${f}"; exit 0
    fi
    sleep 10
  done
  echo "WARNING: main_task log not found under ${RAY_ROOT}; metrics are only in ${RUN_DIR}/train.log"
) &
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" ENVS_PER_GPU="${ENVS_PER_GPU}" BASE_PORT="${BASE_PORT}" \
MODEL_PATH="${MODEL_PATH}" EXP_NAME="${EXP_NAME}" RUN_DIR="${RUN_DIR}" CKPT_DIR="${CKPT_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" PROJECT_NAME="${PROJECT_NAME:-agentgym-tau2}" \
  bash "${ROOT}/scripts/run_tau2_grpo_train.sh" 2>&1 | tee "${RUN_DIR}/train.log"

echo "=== done: ${RUN_DIR} ==="
