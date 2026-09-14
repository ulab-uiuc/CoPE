#!/usr/bin/env bash
#
# Launch a cluster of tau2-bench env servers, one process per port.
#
# Runs in its own conda env because tau2 requires Python >= 3.12 while the training
# env is on 3.10 -- the training side only ever talks HTTP to these processes.
#
# TAU2_TASK_SPLIT here MUST match the split the training TRAIN_FILE was generated from:
# verl passes an integer item id and the server resolves it as
# task_ids[item_id % len(task_ids)] over this split.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TAU2_ENV="${TAU2_ENV:-${TAU2_ENV_DEFAULT}}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36201}"
NUM_ENVS="${NUM_ENVS:-1}"
LOG_DIR="${LOG_DIR:-${ROOT}/runlogs/env_cluster/tau2_$(date +%Y%m%d_%H%M%S)}"
PID_DIR="${LOG_DIR}/pids"

# --- tau2 env-server configuration (see agentenv_tau2/environment.py) --------------
export TAU2_DOMAIN="${TAU2_DOMAIN:-retail}"
export TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-train}"
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-60}"
export TAU2_SOLO_MODE="${TAU2_SOLO_MODE:-0}"
export TAU2_USER_LLM="${TAU2_USER_LLM:-openai/user-sim}"
# api_base is only meaningful for a self-hosted user simulator. It must stay UNSET for
# a hosted model (TAU2_USER_LLM=openai/gpt-4o-mini), otherwise litellm is pointed at a
# local port that does not serve it. Defaulted only when the caller did not decide.
if [[ -z "${TAU2_USER_API_BASE+x}" && "${TAU2_USER_LLM}" == "openai/user-sim" ]]; then
  export TAU2_USER_API_BASE="http://127.0.0.1:38001/v1"
elif [[ -n "${TAU2_USER_API_BASE:-}" ]]; then
  export TAU2_USER_API_BASE
fi
# Pass the key by file, not by value: exported variables are readable through
# `scontrol show job` and land in any `env` dump in the logs.
[[ -n "${TAU2_USER_API_KEY_FILE:-}" ]] && export TAU2_USER_API_KEY_FILE
[[ -n "${TAU2_USER_API_KEY:-}" ]] && export TAU2_USER_API_KEY
export TAU2_USER_TEMPERATURE="${TAU2_USER_TEMPERATURE:-0.0}"
# `env` = DB/env-state only. The official `all` basis pulls in an NL-assertion judge
# LLM on 112 of retail's 114 tasks, i.e. one extra judge call per rollout.
export TAU2_REWARD_BASIS="${TAU2_REWARD_BASIS:-env}"
# Partial credit from per-action checks. Measured on retail train with Qwen2.5-7B:
# binary leaves 64/74 tasks at a uniform zero so only 4/74 GRPO groups have any
# advantage; dense drops that to 14/74 all-zero and 54/74 informative, at an identical
# 10.1% solve rate. Without it GRPO has almost nothing to learn from.
export TAU2_REWARD_SHAPE="${TAU2_REWARD_SHAPE:-dense}"
export TAU2_DENSE_WEIGHT="${TAU2_DENSE_WEIGHT:-0.5}"
# tau2 only scores a run once its orchestrator terminates, so episodes that merely
# exhaust the caller's turn budget come back as unevaluated zeros. Must match the
# caller's MAX_ROUNDS.
export TAU2_FORCE_DONE_AFTER="${TAU2_FORCE_DONE_AFTER:-15}"
export TAU2_PROMPT_VARIANT="${TAU2_PROMPT_VARIANT:-strict}"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

mkdir -p "${PID_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${TAU2_ENV}"
set -u

cd "${ROOT}"

cleanup() {
  echo "Stopping all tau2 environment services..."
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

# A hosted user simulator is the only thing here that talks to the public internet, and
# it must use THIS node's egress. Proxy variables inherited from whoever submitted the
# job can point at an address that is meaningless on a compute node, which surfaces as
# `litellm.InternalServerError: Connection error` on every turn -- i.e. as a broken user
# simulator rather than as a networking problem. Set TAU2_USER_HTTP_PROXY to opt back in
# to a proxy the cluster actually requires.
if [[ -z "${TAU2_USER_API_BASE:-}" ]]; then
  if [[ -n "${TAU2_USER_HTTP_PROXY:-}" ]]; then
    export http_proxy="${TAU2_USER_HTTP_PROXY}" https_proxy="${TAU2_USER_HTTP_PROXY}"
    export HTTP_PROXY="${TAU2_USER_HTTP_PROXY}" HTTPS_PROXY="${TAU2_USER_HTTP_PROXY}"
  else
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  fi
  echo "egress: ${TAU2_USER_HTTP_PROXY:-direct (proxy vars cleared)}"
fi

echo "domain=${TAU2_DOMAIN} split=${TAU2_TASK_SPLIT} reward_basis=${TAU2_REWARD_BASIS}"
echo "user_llm=${TAU2_USER_LLM} api_base=${TAU2_USER_API_BASE:-<hosted, none>}"

for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  LOG_PATH="${LOG_DIR}/env_${PORT}.log"

  echo "Starting tau2 service on port ${PORT}, logging to ${LOG_PATH}..."

  env \
    -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    NO_PROXY="${NO_PROXY}" no_proxy="${no_proxy}" \
    tau2-env --host "${HOST}" --port "${PORT}" > "${LOG_PATH}" 2>&1 &

  echo $! > "${PID_DIR}/env_${PORT}.pid"
done

echo "All ${NUM_ENVS} environment services started. Use Ctrl+C to stop them."
wait
