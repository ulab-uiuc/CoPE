#!/usr/bin/env bash
# Batch evaluation of AppWorld checkpoints over several test splits.
# The env server cluster is restarted per split (APPWORLD_SPLIT selects the tasks).
#
#   EXP=<experiment name> TRAIN_ENV=<agentgym-rl env> APPWORLD_ENV=<agentenv-appworld env> \
#   APPWORLD_ROOT=<appworld data> APPWORLD_PYTHON=<that env's python> \
#     bash scripts/run_appworld_ckpt_eval.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root; this script lives in scripts/
EXP="${EXP:?set EXP to the experiment whose checkpoints to evaluate}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:?set TRAIN_ENV to the agentgym-rl conda env}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP}}"
STEPS="${STEPS:-25 50 75}"
SPLITS="${SPLITS:-test_normal test_challenge}"
BASE_PORT="${BASE_PORT:-36301}"
NUM_ENVS="${NUM_ENVS:-128}"        # must be >= CONCURRENCY (one process per trajectory)
CONCURRENCY="${CONCURRENCY:-128}"
TP="${TP:-8}"
GPU_UTIL="${GPU_UTIL:-0.85}"
TEMP="${TEMP:-0.4}"                # G2PO validation temperature
MAX_ROUNDS="${MAX_ROUNDS:-30}"
LIMIT="${LIMIT:-0}"                # >0: first N tasks only, for smoke tests
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/appworld_eval}"
SUMMARY="${OUT_ROOT}/summary.tsv"

if [ "${NUM_ENVS}" -lt "${CONCURRENCY}" ]; then
  echo "ERROR: NUM_ENVS=${NUM_ENVS} < CONCURRENCY=${CONCURRENCY}" >&2
  echo "       AppWorld's supervisor active task is process-global: one trajectory per process." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
[ -f "${SUMMARY}" ] || printf 'step\tsplit\tn\tsuccess\trate\tmean_rounds\tenv_err\tmin\n' > "${SUMMARY}"

ADDRS=""
for i in $(seq 0 $((NUM_ENVS - 1))); do
  a="http://127.0.0.1:$((BASE_PORT + i))"
  ADDRS="${ADDRS:+${ADDRS},}${a}"
done

start_envs() {   # $1 = split
  echo "--- starting ${NUM_ENVS} env servers (split=$1) ---"
  tmux kill-session -t appworld_eval_env 2>/dev/null
  # The previous split's servers were just killed and their ports sit in TIME_WAIT;
  # starting immediately fails to bind and the whole split gets skipped (seen when
  # test_challenge was skipped because port 36301 never came up). Wait for the old
  # processes to exit and the ports to free up first.
  pkill -f "appworld-env --host" 2>/dev/null
  for _ in $(seq 1 60); do
    pgrep -f "appworld-env --host" >/dev/null || break
    sleep 2
  done
  pkill -9 -f "appworld-env --host" 2>/dev/null
  sleep 15
  # Pass the env selection explicitly: a running tmux server does not inherit this shell.
  tmux new-session -d -s appworld_eval_env \
    "cd ${ROOT} && NUM_ENVS=${NUM_ENVS} BASE_PORT=${BASE_PORT} APPWORLD_SPLIT=$1 \
     CONDA_SH=${CONDA_SH} APPWORLD_ENV=${APPWORLD_ENV:?set APPWORLD_ENV} \
     APPWORLD_ROOT=${APPWORLD_ROOT:?set APPWORLD_ROOT} \
     bash ${ROOT}/scripts/run_appworld_env_service.sh"
  for i in $(seq 0 $((NUM_ENVS - 1))); do
    p=$((BASE_PORT + i))
    for _ in $(seq 1 300); do
      curl --noproxy '*' -sf "http://127.0.0.1:${p}/" >/dev/null && break
      sleep 2
    done
    curl --noproxy '*' -sf "http://127.0.0.1:${p}/" >/dev/null || { echo "port ${p} not ready" >&2; return 1; }
  done
  echo "--- ${NUM_ENVS} env servers ready ---"
}

stop_envs() {
  tmux kill-session -t appworld_eval_env 2>/dev/null
  sleep 3
  pkill -f "appworld-env --host" 2>/dev/null
  sleep 3
}

for split in ${SPLITS}; do
  start_envs "${split}" || { echo "env startup failed, skipping ${split}" >&2; continue; }
  for step in ${STEPS}; do
    MP="${CKPT_DIR}/global_step_${step}/actor/huggingface"
    OD="${OUT_ROOT}/step${step}_${split}"
    echo ""
    echo "############ EVAL step_${step} / ${split}  $(date '+%F %T') ############"
    if [ ! -f "${MP}/config.json" ]; then echo "missing ${MP}" >&2; continue; fi
    ( source "${CONDA_SH}" && conda activate "${TRAIN_ENV}" && \
      cd "${ROOT}" && \
      python3 scripts/eval_appworld.py \
        --model-path "${MP}" --split "${split}" --output-dir "${OD}" \
        --env-addrs "${ADDRS}" --tp "${TP}" --gpu-util "${GPU_UTIL}" \
        --concurrency "${CONCURRENCY}" --max-rounds "${MAX_ROUNDS}" \
        --temp "${TEMP}" --limit "${LIMIT}" --overwrite )
    rc=$?
    python3 - "$OD" "$step" "$split" "$SUMMARY" <<'PY'
import json,sys,os
od,step,split,summ = sys.argv[1:5]
f=os.path.join(od,"summary.json")
if os.path.exists(f):
    d=json.load(open(f))
    row=f"{step}\t{split}\t{d['num_tasks']}\t{d['num_success']}\t{d['success_rate']:.4f}\t{d['mean_rounds']:.1f}\t{d['env_errors']}\t{d['elapsed_min']:.1f}\n"
else:
    row=f"{step}\t{split}\tNO_SUMMARY\n"
open(summ,"a").write(row); print("  => "+row.strip())
PY
    echo "############ DONE rc=${rc} ############"
  done
done
stop_envs
echo ""
echo "=================== all done ==================="
column -t "${SUMMARY}"
