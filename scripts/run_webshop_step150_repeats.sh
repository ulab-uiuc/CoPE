#!/usr/bin/env bash
# Repeat the step-150 webshop eval twice more for three runs, then report mean/std
# over n=3 (the existing runs/<run>/150 result counts as repeat 0).
# Sampling is temperature=1.0 with no fixed seed, so repeats genuinely resample.
# Reuses env servers already listening on BASE_PORT..BASE_PORT+N_SERVERS-1.
set -uo pipefail

ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/AgentGym-RL"
TRAIN_ENV="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentgym-rl"
BASE_PORT="${BASE_PORT:-36101}"
N_SERVERS="${N_SERVERS:-16}"
CONCURRENCY="${CONCURRENCY:-16}"
TP="${TP:-4}"
STEP=150
SUMMARY="${ROOT}/runlogs/webshop_step150_repeats.txt"

export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1
cd "$ROOT"
: > "$SUMMARY"

ADDRS=""
for i in $(seq 0 $((N_SERVERS - 1))); do
  P=$((BASE_PORT + i))
  [ -z "$ADDRS" ] && ADDRS="http://127.0.0.1:${P}" || ADDRS="${ADDRS},http://127.0.0.1:${P}"
done
ready=0
for i in $(seq 0 $((N_SERVERS - 1))); do
  [ "$(curl -s --noproxy 127.0.0.1 -m 3 http://127.0.0.1:$((BASE_PORT+i))/ 2>/dev/null)" = '"ok"' ] && ready=$((ready+1))
done
echo "env servers reachable: ${ready}/${N_SERVERS}" | tee -a "$SUMMARY"

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$TRAIN_ENV" 2>/dev/null || export PATH="${TRAIN_ENV}/bin:$PATH"

RUNS=(
  "webshop_grpo_reb_baseline0_20260726_110653"
  "webshop_grpo_reb_baseline1_20260726_112327"
  "webshop_grpo_reb_r0_new_20260728_144251"
)

i=0
TOTAL=$(( ${#RUNS[@]} * 2 ))
for RUN in "${RUNS[@]}"; do
  for REP in 1 2; do
    i=$((i+1))
    OD="${ROOT}/runs/${RUN}/${STEP}_rep${REP}"
    mkdir -p "$OD"
    echo ""
    echo "########### [${i}/${TOTAL}] ${RUN} step_${STEP} rep${REP}  $(date -u +%H:%M:%S)Z ###########"
    python "${ROOT}/scripts/eval_webshop.py" \
      --model-path "${ROOT}/checkpoints/${RUN}/global_step_${STEP}/actor/huggingface" \
      --output-dir "${OD}" \
      --env-addrs "${ADDRS}" \
      --tp "${TP}" \
      --concurrency "${CONCURRENCY}" \
      --overwrite 2>&1 | tail -12
    LINE=$(python3 - "$OD/summary.json" "$RUN" "$REP" <<'PY'
import json, sys
p, run, rep = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(p)); s = d.get("summary", d); o = s.get("All", s.get("overall", s))
    print(f"{run}\tstep_150\trep{rep}\tsucc={o.get('success','?')}\tscore={o.get('score','?')}")
except Exception as e:
    print(f"{run}\tstep_150\trep{rep}\tNO_SUMMARY ({e})")
PY
)
    echo "$LINE" | tee -a "$SUMMARY"
  done
done

echo "" | tee -a "$SUMMARY"
echo "===== MEAN / STD over n=3 (rep0 = original run) =====" | tee -a "$SUMMARY"
python3 - "$ROOT" "${RUNS[@]}" >> "$SUMMARY" 2>&1 <<'PY'
import json, os, statistics as st, sys
root, runs = sys.argv[1], sys.argv[2:]
print(f"{'run':46s} {'n':>2s} {'succ mean':>10s} {'succ std':>9s} {'score mean':>11s} {'score std':>10s}   values(succ)")
for run in runs:
    succ, score = [], []
    for sub in ("150", "150_rep1", "150_rep2"):
        p = os.path.join(root, "runs", run, sub, "summary.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); s = d.get("summary", d); o = s.get("All", s.get("overall", s))
        if o.get("success") is not None:
            succ.append(float(o["success"])); score.append(float(o["score"]))
    if not succ:
        print(f"{run[:46]:46s}  no data"); continue
    ss = st.stdev(succ) if len(succ) > 1 else 0.0
    cs = st.stdev(score) if len(score) > 1 else 0.0
    print(f"{run[:46]:46s} {len(succ):>2d} {st.mean(succ):>10.4f} {ss:>9.4f} "
          f"{st.mean(score):>11.4f} {cs:>10.4f}   {[round(x,3) for x in succ]}")
PY
echo "===== ALL ${TOTAL} REPEATS DONE =====" | tee -a "$SUMMARY"
cat "$SUMMARY"
