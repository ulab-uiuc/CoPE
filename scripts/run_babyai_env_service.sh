#!/usr/bin/env bash
#
# BabyAI environment services, one process per port.
#
# The launcher (scripts/launch_babyai_grpo_tmux.sh) starts this in its own tmux
# window with NUM_ENVS and BASE_PORT set; running it directly works too.
#
#   BABYAI_ENV=/path/to/conda/env NUM_ENVS=4 bash scripts/run_babyai_env_service.sh
#
# The `babyai` entry point comes from AgentGym's agentenv-babyai package, which
# needs its own conda environment -- its dependencies conflict with the trainer's.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
BABYAI_ENV="${BABYAI_ENV:?set BABYAI_ENV to the agentenv-babyai conda env}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36001}"
NUM_ENVS="${NUM_ENVS:-1}"
LOG_DIR="${LOG_DIR:-${ROOT}/runlogs/env_cluster/babyai_$(date +%Y%m%d_%H%M%S)}"
PID_DIR="${LOG_DIR}/pids"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "${PID_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${BABYAI_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${ROOT}"

cleanup() {
  echo "Stopping all BabyAI environment services..."
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

  echo "Starting BabyAI service on port ${PORT}, logging to ${LOG_PATH}..."

  env \
    -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    NO_PROXY="${NO_PROXY}" \
    no_proxy="${no_proxy}" \
    babyai --host "${HOST}" --port "${PORT}" > "${LOG_PATH}" 2>&1 &

  echo $! > "${PID_DIR}/env_${PORT}.pid"
done

echo "All ${NUM_ENVS} environment services started. Use Ctrl+C to stop them."
wait
