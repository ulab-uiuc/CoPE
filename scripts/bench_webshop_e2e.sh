#!/usr/bin/env bash
# End-to-end webshop training speed A/B: identical config, only ENVS_PER_GPU differs.
# For each arm: start ENVS_PER_GPU*NGPU env servers, run training until N_STEPS steps
# have been logged, kill everything, then parse timing_s/gen and timing_s/step.
set -uo pipefail

ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/AgentGym-RL"
W="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentenv-webshop"
NGPU=4
N_STEPS="${N_STEPS:-4}"          # collect this many training steps per arm
MAX_WAIT="${MAX_WAIT:-2400}"     # per-arm wall-clock cap (s)
OUT="${ROOT}/runlogs/e2e_result.txt"

export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
cd "$ROOT"
: > "$OUT"

run_arm() {
  local EPG=$1 PORT_BASE=$2 TAG=$3
  local NSERV=$((NGPU * EPG))
  echo "===== ARM ${TAG}: ENVS_PER_GPU=${EPG}, ${NSERV} env servers =====" | tee -a "$OUT"

  # --- env servers (conda env's bin on PATH so pyserini finds its bundled JDK) ---
  ( export PATH="$W/bin:$PATH"; export JAVA_HOME="$W/lib/jvm"
    for i in $(seq 0 $((NSERV - 1))); do
      webshop --host 127.0.0.1 --port $((PORT_BASE + i)) > /tmp/e2e_ws_$((PORT_BASE+i)).log 2>&1 &
    done )
  # wait for all servers
  local ready=0
  for _ in $(seq 1 120); do
    ready=0
    for i in $(seq 0 $((NSERV - 1))); do
      [ "$(curl -s --noproxy 127.0.0.1 -m 3 http://127.0.0.1:$((PORT_BASE+i))/ 2>/dev/null)" = '"ok"' ] && ready=$((ready+1))
    done
    [ "$ready" -eq "$NSERV" ] && break
    sleep 3
  done
  echo "  env servers ready: ${ready}/${NSERV}" | tee -a "$OUT"

  # --- training ---
  local LOG="${ROOT}/runlogs/e2e_${TAG}_train.log"
  : > "$LOG"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=0,1,2,3 ENVS_PER_GPU="${EPG}" BASE_PORT="${PORT_BASE}" \
      EXP_NAME="e2e_${TAG}" LOG_PATH="${LOG}" WANDB_MODE=offline \
      bash scripts/run_webshop_grpo_train.sh > /dev/null 2>&1 ) &
  local TRAIN_PID=$!

  # wait until N_STEPS steps logged (or timeout)
  local t0=$(date +%s) seen=0
  while :; do
    seen=$(grep -ac "timing_s/step" "$LOG" 2>/dev/null || echo 0)
    [ "$seen" -ge "$N_STEPS" ] && break
    [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT" ] && { echo "  TIMEOUT after ${MAX_WAIT}s (got ${seen} steps)" | tee -a "$OUT"; break; }
    kill -0 "$TRAIN_PID" 2>/dev/null || { echo "  training exited early" | tee -a "$OUT"; break; }
    sleep 10
  done
  echo "  steps collected: ${seen}" | tee -a "$OUT"

  # --- parse timings (skip step 1: includes vLLM warmup / cache build) ---
  python3 - "$LOG" "$TAG" >> "$OUT" 2>&1 <<'PY'
import re, sys
log, tag = sys.argv[1], sys.argv[2]
vals = {k: [] for k in ("gen", "step")}
for line in open(log, "rb"):
    for k in vals:
        m = re.search(rb"timing_s/%s['\"]?\s*[:=]\s*([0-9.]+)" % k.encode(), line)
        if m:
            vals[k].append(float(m.group(1)))
for k, v in vals.items():
    v = v[1:]  # drop first step (warmup)
    if v:
        print(f"  timing_s/{k:5s}: n={len(v)} mean={sum(v)/len(v):7.2f}s  vals={[round(x,1) for x in v]}")
    else:
        print(f"  timing_s/{k:5s}: no data")
PY

  # --- teardown ---
  kill "$TRAIN_PID" 2>/dev/null
  pkill -9 -f "main_ppo" 2>/dev/null
  pkill -9 -f "webshop --host" 2>/dev/null
  sleep 20
}

run_arm 1 36301 "EPG1"
run_arm 4 36401 "EPG4"
echo "===== DONE =====" | tee -a "$OUT"
