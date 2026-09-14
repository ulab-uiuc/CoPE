#!/usr/bin/env bash
# Serial evaluation of webshop rebuttal-baseline checkpoints.
# Starts N_SERVERS webshop env servers, then evaluates each (run, step) with
# eval_webshop.py (tp=4 vLLM over the 4 GPUs, env work spread over the servers).
# Each result is appended to the summary file AS SOON AS it finishes, so progress
# can be reported incrementally.
set -uo pipefail

ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/AgentGym-RL"
TRAIN_ENV="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentgym-rl"
WS_ENV="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentenv-webshop"
BASE_PORT="${BASE_PORT:-36101}"
N_SERVERS="${N_SERVERS:-16}"
CONCURRENCY="${CONCURRENCY:-16}"
TP="${TP:-4}"
SUMMARY="${ROOT}/runlogs/webshop_ckpt_eval_summary_r0new.txt"

export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1
cd "$ROOT"
: > "$SUMMARY"

# ---- env servers ------------------------------------------------------------
# START_SERVERS=1 launches them here; default 0 reuses servers already running on
# BASE_PORT..BASE_PORT+N_SERVERS-1 (e.g. started by the training launcher).
if [ "${START_SERVERS:-0}" = "1" ]; then
  echo "starting ${N_SERVERS} webshop env servers from port ${BASE_PORT}..."
  ( export PATH="${WS_ENV}/bin:$PATH"; export JAVA_HOME="${WS_ENV}/lib/jvm"
    for i in $(seq 0 $((N_SERVERS - 1))); do
      webshop --host 127.0.0.1 --port $((BASE_PORT + i)) > /tmp/wseval_$((BASE_PORT+i)).log 2>&1 &
    done )
else
  echo "reusing existing env servers on ${BASE_PORT}..$((BASE_PORT+N_SERVERS-1))"
fi

ADDRS=""
for i in $(seq 0 $((N_SERVERS - 1))); do
  P=$((BASE_PORT + i))
  [ -z "$ADDRS" ] && ADDRS="http://127.0.0.1:${P}" || ADDRS="${ADDRS},http://127.0.0.1:${P}"
done

ready=0
for _ in $(seq 1 150); do
  ready=0
  for i in $(seq 0 $((N_SERVERS - 1))); do
    [ "$(curl -s --noproxy 127.0.0.1 -m 3 http://127.0.0.1:$((BASE_PORT+i))/ 2>/dev/null)" = '"ok"' ] && ready=$((ready+1))
  done
  [ "$ready" -eq "$N_SERVERS" ] && break
  sleep 3
done
echo "env servers ready: ${ready}/${N_SERVERS}" | tee -a "$SUMMARY"

# ---- evaluations ----
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$TRAIN_ENV" 2>/dev/null || export PATH="${TRAIN_ENV}/bin:$PATH"

RUNS=(
  "webshop_grpo_reb_r0_new_20260728_144251"
)
STEPS=(100 150 200 250 300)

i=0
TOTAL=$(( ${#RUNS[@]} * ${#STEPS[@]} ))
for RUN in "${RUNS[@]}"; do
  for STEP in "${STEPS[@]}"; do
    i=$((i+1))
    OD="${ROOT}/runs/${RUN}/${STEP}"
    mkdir -p "$OD"
    echo ""
    echo "########### [${i}/${TOTAL}] ${RUN} step_${STEP}  $(date -u +%H:%M:%S)Z ###########"
    python "${ROOT}/scripts/eval_webshop.py" \
      --model-path "${ROOT}/checkpoints/${RUN}/global_step_${STEP}/actor/huggingface" \
      --output-dir "${OD}" \
      --env-addrs "${ADDRS}" \
      --tp "${TP}" \
      --concurrency "${CONCURRENCY}" \
      --overwrite 2>&1 | tail -25
    rc=$?
    LINE=$(python3 - "$OD/summary.json" "$RUN" "$STEP" <<'PY'
import json, sys
p, run, step = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(p))
    s = d.get("summary", d)
    o = s.get("All", s.get("overall", s))
    print(f"{run}\tstep_{step}\tsucc={o.get('success','?')}\tscore={o.get('score','?')}\tn={o.get('count','?')}")
except Exception as e:
    print(f"{run}\tstep_{step}\tNO_SUMMARY ({e})")
PY
)
    echo "$LINE" | tee -a "$SUMMARY"
    echo "########### DONE [${i}/${TOTAL}] rc=${rc} ###########"
  done
done

echo "" | tee -a "$SUMMARY"
echo "===== ALL ${TOTAL} WEBSHOP EVALS DONE =====" | tee -a "$SUMMARY"
cat "$SUMMARY"
# NOTE: do not kill env servers -- they may be owned by the training launcher.
[ "${START_SERVERS:-0}" = "1" ] && pkill -9 -f "webshop --host" 2>/dev/null
