#!/usr/bin/env bash
#
# User-simulator LLM server for tau2-bench.
#
# tau2 calls an LLM for the customer side on every non-tool turn, so at
# TRAIN_BATCH_SIZE x ROLLOUT_N concurrent trajectories a training step issues on the
# order of a thousand user-sim requests. This serves that traffic from a local
# OpenAI-compatible vLLM server instead of a hosted API: free, offline, and (at
# temperature 0) deterministic, which keeps the GRPO group baseline from picking up
# user-simulator noise.
#
# The env server reaches it via TAU2_USER_API_BASE=http://HOST:PORT/v1 with
# TAU2_USER_LLM=openai/<SERVED_NAME>.

set -euo pipefail

USERSIM_MODEL="${USERSIM_MODEL:?set USERSIM_MODEL to the user-simulator model path}"
SERVED_NAME="${SERVED_NAME:-user-sim}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-38001}"
# Dedicated GPU(s): keep them off the training GPUs so the simulator is not fighting the
# actor for memory when vLLM re-inits its cache engine each rollout. A 14B in bf16 is
# 28GB of weights and will not fit one 40GB card alongside a KV cache -- set USERSIM_TP=2
# (and USERSIM_GPU to a matching comma list) for anything above ~8B.
USERSIM_GPU="${USERSIM_GPU:-0}"
USERSIM_TP="${USERSIM_TP:-1}"
USERSIM_ENV="${USERSIM_ENV:-${TRAIN_ENV:?set TRAIN_ENV or USERSIM_ENV}}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
GPU_MEM_UTIL="${USERSIM_GPU_MEM_UTIL:-0.45}"
MAX_MODEL_LEN="${USERSIM_MAX_MODEL_LEN:-16384}"
# Domains where the customer has their own tools (telecom: toggle_roaming,
# toggle_data_saver_mode, ...) make tau2 call the user LLM with tool_choice="auto".
# Without these two flags vLLM rejects that request outright, the orchestrator raises,
# and every episode dies after two turns with reward 0 -- which reads as "the model
# cannot do telecom" rather than as a server misconfiguration. Harmless for retail and
# airline, whose user simulators have no tools.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
LOG_DIR="${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runlogs/tau2_usersim}"

mkdir -p "${LOG_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${USERSIM_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "Starting user-sim vLLM server: ${USERSIM_MODEL} as '${SERVED_NAME}' on ${HOST}:${PORT} (GPU ${USERSIM_GPU})"

exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" no_proxy="${no_proxy}" \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="${USERSIM_GPU}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${USERSIM_MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${USERSIM_TP}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --disable-log-requests \
    > "${LOG_DIR}/usersim.log" 2>&1
