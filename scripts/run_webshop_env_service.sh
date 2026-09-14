#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
WEBSHOP_ENV="${WEBSHOP_ENV:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentenv-webshop}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36101}"
NUM_ENVS="${NUM_ENVS:-1}"
LOG_DIR="${LOG_DIR:-${ROOT}/runlogs/env_cluster/webshop_$(date +%Y%m%d_%H%M%S)}"
PID_DIR="${LOG_DIR}/pids"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "${PID_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${WEBSHOP_ENV}"
set -u

export WEBSHOP_DATASET_SIZE="${WEBSHOP_DATASET_SIZE:-all}"
export WEBSHOP_GOAL_SOURCE="${WEBSHOP_GOAL_SOURCE:-human}"
export WEBSHOP_HUMAN_GOAL_MODE="${WEBSHOP_HUMAN_GOAL_MODE:-official}"
export WEBSHOP_GOAL_SPLIT="${WEBSHOP_GOAL_SPLIT:-train}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${ROOT}"

# Handle cleanup
cleanup() {
  echo "Stopping all WebShop environment services..."
  for pid_file in "${PID_DIR}"/*.pid; do
    if [[ -f "$pid_file" ]]; then
      pid=$(cat "$pid_file")
      kill "$pid" 2>/dev/null || true
      rm "$pid_file"
    fi
  done
  exit 0
}

trap cleanup SIGINT SIGTERM EXIT

for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  LOG_PATH="${LOG_DIR}/env_${PORT}.log"
  
  echo "Starting WebShop service on port ${PORT}, logging to ${LOG_PATH}..."
  
  env \
    -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    NO_PROXY="${NO_PROXY}" \
    no_proxy="${no_proxy}" \
    WEBSHOP_DATASET_SIZE="${WEBSHOP_DATASET_SIZE}" \
    WEBSHOP_GOAL_SOURCE="${WEBSHOP_GOAL_SOURCE}" \
    WEBSHOP_HUMAN_GOAL_MODE="${WEBSHOP_HUMAN_GOAL_MODE}" \
    WEBSHOP_GOAL_SPLIT="${WEBSHOP_GOAL_SPLIT}" \
    webshop --host "${HOST}" --port "${PORT}" > "${LOG_PATH}" 2>&1 &
  
  echo $! > "${PID_DIR}/env_${PORT}.pid"
done

echo "All ${NUM_ENVS} environment services started. Use Ctrl+C to stop them."
wait
