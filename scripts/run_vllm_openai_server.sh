#!/usr/bin/env bash
#
# Serve a local model on an OpenAI-compatible endpoint, so that
# scripts/eval_alfworld_openai.py and scripts/eval_webshop_openai.py can drive a
# base model (the prompting / ReAct baseline) through the same code path they
# use for hosted models. Point --base-url at http://HOST:PORT/v1 with
# --api-key EMPTY.
#
# Required env:
#   MODEL_PATH                - HF model directory or hub id (e.g. Qwen/Qwen2.5-7B-Instruct)
#   SERVED_MODEL_NAME         - public name, i.e. what the eval passes as --model
#   CUDA_VISIBLE_DEVICES      - GPU ids
#   PORT                      - listen port
# Optional env:
#   HOST                      (default 127.0.0.1)
#   TENSOR_PARALLEL_SIZE      (default 1; must divide the visible GPU count)
#   DTYPE                     (default bfloat16)
#   MAX_MODEL_LEN             (default 16384; ALFWorld transcripts reach ~12k)
#   GPU_MEMORY_UTILIZATION    (default 0.85)
#   MAX_NUM_SEQS              (default 256)
#   VLLM_ENV                  conda/venv prefix holding vllm; defaults to
#                             whatever `python` is on PATH
#   VLLM_EXTRA_ARGS           extra CLI flags appended verbatim
#   LOG_PATH                  - if set, redirect stdout/stderr there
#
# Example:
#   MODEL_PATH=Qwen/Qwen2.5-7B-Instruct SERVED_MODEL_NAME=Qwen2.5-7B-Instruct \
#   CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 PORT=8100 \
#     bash scripts/run_vllm_openai_server.sh

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
: "${PORT:?PORT is required}"

HOST="${HOST:-127.0.0.1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
VLLM_ENV="${VLLM_ENV:-}"
LOG_PATH="${LOG_PATH:-}"

if [[ -n "${VLLM_ENV}" ]]; then
  PY="${VLLM_ENV}/bin/python"
  [[ -x "${PY}" ]] || { echo "no python at ${PY}" >&2; exit 1; }
else
  PY="$(command -v python)"
fi

if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

# The env servers and this server are all local; a proxy in the environment
# would send loopback traffic through it and the eval would hang on connect.
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

echo "serving ${MODEL_PATH} as '${SERVED_MODEL_NAME}' on ${HOST}:${PORT} (tp=${TENSOR_PARALLEL_SIZE})"

exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  VLLM_USE_MODELSCOPE=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --trust-remote-code \
    --disable-log-requests \
    ${VLLM_EXTRA_ARGS:-}
