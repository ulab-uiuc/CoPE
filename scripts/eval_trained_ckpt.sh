#!/usr/bin/env bash
# Merge the newest tau2 checkpoint to HF format and evaluate it against the paper's
# table, using the same aligned protocol the baseline was measured with.
#
#   bash scripts/eval_trained_ckpt.sh                         # newest step of the run
#   STEP=25 bash scripts/eval_trained_ckpt.sh                 # a specific step
#   EXP=tau2_infopo_align TAG=grpo bash scripts/eval_trained_ckpt.sh
#
# Merging is needed because verl saves FSDP shards; the merger reads the architecture
# from <actor>/huggingface, which the truncated save does not always write, so the base
# model's config is staged there first. The weights come from the shards either way.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EXP="${EXP:-tau2_infopo_align}"
TAG="${TAG:-grpo}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to the base checkpoint}"

CKPT_ROOT="${ROOT}/checkpoints/${EXP}"
[[ -d "${CKPT_ROOT}" ]] || { echo "FATAL: no checkpoints at ${CKPT_ROOT}"; exit 1; }

if [[ -n "${STEP:-}" ]]; then
  ACTOR="${CKPT_ROOT}/global_step_${STEP}/actor"
else
  ACTOR=$(ls -d "${CKPT_ROOT}"/global_step_*/actor 2>/dev/null \
          | sed 's/.*global_step_\([0-9]*\).*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2)
fi
[[ -n "${ACTOR}" && -d "${ACTOR}" ]] || { echo "FATAL: no actor dir found under ${CKPT_ROOT}"; exit 1; }
echo "checkpoint: ${ACTOR}"

HF="${ACTOR}/huggingface"
if [[ ! -f "${HF}/model.safetensors.index.json" ]]; then
  echo "--- merging FSDP shards ---"
  mkdir -p "${HF}"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [[ -f "${BASE_MODEL}/${f}" ]] && cp -n "${BASE_MODEL}/${f}" "${HF}/" || true
  done
  ( cd "${ROOT}/src" && PYTHONNOUSERSITE=1 HF_HOME="${ROOT}/.hf_cache" \
    ${TRAIN_ENV}/bin/python scripts/model_merger.py --local_dir "${ACTOR}" )
else
  echo "--- already merged ---"
fi

PYTHONNOUSERSITE=1 ${TRAIN_ENV}/bin/python - "${HF}" <<'PY'
import json, os, sys
h = sys.argv[1]
idx = json.load(open(f"{h}/model.safetensors.index.json"))
files = set(idx["weight_map"].values())
missing = [f for f in files if not os.path.exists(f"{h}/{f}")]
assert not missing, f"incomplete merge, missing {missing}"
print(f"merged model OK: {len(files)} shards, {len(idx['weight_map'])} tensors")
PY

echo "--- submitting aligned evaluation ---"
sbatch --gres=gpu:a100:2 --time=12:00:00 \
  --export=ALL,TAU2_VERSION=infopo,TAG="${TAG}",MODEL_PATH="${HF}",\
DOMAINS="airline retail telecom",NUM_TRIALS=4,MAX_STEPS=200,MAX_CONCURRENCY=16,\
POLICY_PORT="${POLICY_PORT:-8660}" \
  scripts/sbatch_tau2_align.sh

cat <<EOF

Once it finishes:
  python scripts/tau2_align_score.py --tag ${TAG} \\
    --data-dir ${TAU2_BENCH_DIR}/data

Reference (Avg@4, test split, paper Table 1, Qwen2.5-7B):
  base (measured here)   telecom  8.1   retail  9.4   airline  8.8
  RAGEN                  telecom 17.5   retail 17.5   airline 15.0
  InfoPO                 telecom 18.1   retail 18.8   airline 16.3
EOF
