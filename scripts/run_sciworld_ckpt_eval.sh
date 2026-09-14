#!/usr/bin/env bash
# Serial eval of selected sciworld checkpoints. Each iteration is EXACTLY the command
# run by hand (no extra env vars), looped over all steps (step_264 excluded), auto-
# advancing to the next when one finishes.
set -uo pipefail

ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/AgentGym-RL"
ENV_ADDRS="http://127.0.0.1:36101,http://127.0.0.1:36102,http://127.0.0.1:36103,http://127.0.0.1:36104"

source /opt/conda/etc/profile.d/conda.sh
conda activate /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentgym-rl

SUMMARY="${ROOT}/runs/sciworld_ckpt_eval_summary.txt"
mkdir -p "${ROOT}/runs"
: > "${SUMMARY}"

# ordered (run : space-separated steps). step_264 intentionally excluded.
JOBS=(
  "sciworld_grpo_qwen2.5_3b_add_20260709_055129:50 100 150"
)

TOTAL=0; for e in "${JOBS[@]}"; do for _ in ${e#*:}; do TOTAL=$((TOTAL+1)); done; done
i=0
for entry in "${JOBS[@]}"; do
  run="${entry%%:*}"; steps="${entry#*:}"
  for step in ${steps}; do
    i=$((i+1))
    echo ""
    echo "############### [${i}/${TOTAL}] EVAL ${run} step_${step}  $(date -u +%H:%M:%S)Z ###############"
    cd "${ROOT}"
    python scripts/eval_sciworld.py \
      --model-path "${ROOT}/checkpoints/${run}/global_step_${step}/actor/huggingface" \
      --output-dir "runs/${run}/${step}" \
      --env-addrs "${ENV_ADDRS}" \
      --tp 4 \
      --concurrency 8 \
      --overwrite
    rc=$?
    sc=$(python3 -c "import json;d=json.load(open('runs/${run}/${step}/summary.json'));print(json.dumps(d.get('summary',d).get('overall','?')))" 2>/dev/null || echo "NO_SUMMARY rc=${rc}")
    printf '%s\tstep_%s\t%s\n' "${run}" "${step}" "${sc}" | tee -a "${SUMMARY}"
    echo "############### DONE [${i}/${TOTAL}] rc=${rc} ###############"
  done
done

echo ""
echo "=================== ALL ${TOTAL} EVALS DONE ==================="
cat "${SUMMARY}"
