#!/usr/bin/env bash
#
# InfoPO-aligned GRPO on tau2-bench.
#
# Every hyperparameter below is taken from InfoPO's examples/tau2/train.sh and Table 4
# (arXiv 2603.00656), so a run of this script is comparable to their published table
# rather than merely similar to it. Do not "improve" these values -- a tuned run is a
# different experiment and belongs in a separate script.
#
#   bash scripts/launch_tau2_infopo_aligned.sh                 # paper config, sbatch
#   PLAN_FORECAST_ENABLE=True bash scripts/launch_tau2_infopo_aligned.sh
#   DRY_RUN=1 bash scripts/launch_tau2_infopo_aligned.sh       # print, don't submit
#
# Deliberate, documented deviations from the paper:
#
#   rollout engine   vLLM, not SGLang. This fork's agent rollout is vLLM-only.
#   GPUs             8 (7 train + 1 local customer) vs their 4. Per-rank batch is what
#                    matters for the gradient and is matched below.
#   customer model   USERSIM_MODE=local uses a local Qwen rather than gpt-4o-mini.
#                    Absolute scores are then NOT comparable to the paper -- only
#                    base-vs-trained deltas from a paired run are. Set
#                    USERSIM_MODE=hosted for paper-comparable numbers (costs API credit).
#
# Evaluate with scripts/sbatch_tau2_align.sh, which drives tau2's own CLI at the
# paper-era commit; see docs/TAU2_GRPO.md for why both of those matter.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- paper hyperparameters (InfoPO Table 4, tau2-bench column) -----------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-28}"   # paper 32; 28 keeps 140/7 = 20 per rank
ROLLOUT_N="${ROLLOUT_N:-5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-14}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
POLICY_LR="${POLICY_LR:-1e-6}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"          # paper disables the KL penalty entirely
KL_COEF="${KL_COEF:-0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"               # paper Tmax
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-1024}"

# Sequence budget. The paper's 16384-token response does not fit a 40GB card in this
# fork -- their verl has enable_activation_offload, which this version lacks, so the
# activations for a 16k backward have nowhere to go.
#
# Two things are separable here and it is easy to conflate them:
#
#   memory     is driven by this cap, not by what the model writes -- verl pads to
#              max_response_length, so lowering the cap frees activation memory even
#              though generations are far shorter.
#   truncation is driven by what the model actually writes. Measured on training
#              rollouts (not eval -- eval runs under a 200-step orchestrator cap and
#              produces much longer transcripts that do not describe training):
#
#                  cap 8192 (job 22765):  response mean 1406, max 6243
#                  cap 5120 (job 23087):  response mean 1122, max 4008
#
# So 8192 truncates essentially nothing; 5120 clips only the tail. Both samples are
# from the first few steps of jobs that then OOMed, so treat them as indicative.
#
#   FIT=paper  16384 -- the published value; expect OOM on 40GB, fine on 80GB
#   FIT=40gb    8192 -- validated to train on 40GB without plan-forecast
#   FIT=40gb-pf 5120 -- plan-forecast adds a second full backward pass; needs this
FIT="${FIT:-40gb}"
case "${FIT}" in
  paper)    _resp=16384; _util=0.50 ;;
  40gb)     _resp=8192;  _util=0.30 ;;
  40gb-pf)  _resp=5120;  _util=0.25 ;;
  *) echo "FATAL: FIT must be paper|40gb|40gb-pf, got ${FIT}" >&2; exit 1 ;;
esac
# Prompt has a hard floor: telecom's policy document alone is 6212 tokens, and dropping
# below it trips verl's length assertion before the first step.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${_resp}}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-${_util}}"
[ "${FIT}" = "paper" ] || echo "NOTE: FIT=${FIT} trims response ${MAX_RESPONSE_LENGTH} <- 16384 (paper); absolute scores stay comparable, truncated episodes do not"

# ---- benchmark side ------------------------------------------------------------------
# 178 tasks = the train splits of all three domains, jointly -- the paper trains one
# model over all of them and reports per-domain, it does not train three models.
TAU2_DOMAIN="${TAU2_DOMAIN:-retail+airline+telecom}"
TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-train}"
TAU2_REWARD_SHAPE="${TAU2_REWARD_SHAPE:-binary}"   # tau2's native 0/1, not our dense variant
TAU2_REWARD_BASIS="${TAU2_REWARD_BASIS:-all}"      # EvaluationType.ALL, incl. NL judge
TAU2_PROMPT_VARIANT="${TAU2_PROMPT_VARIANT:-base}" # tau2 default prompt, not our `strict`
# TAU2_MAX_STEPS counts orchestrator steps (agent + customer + tool messages), not agent
# turns, so it is a looser bound than MAX_ROUNDS and never binds during training. 200 is
# the value InfoPO uses at *evaluation* time (eval_vllm.sh:19); their training script sets
# no step cap at all, only max_turns=50. Kept here so training and eval stop the same way.
TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-200}"
TAU2_FORCE_DONE_AFTER="${TAU2_FORCE_DONE_AFTER:-${MAX_ROUNDS}}"
# The 2026-02 tau2 tree. v1.0.1 rewrote the user simulator and changed 3.5M lines of
# task data; baselines measured against it are not the paper's baselines.
TAU2_ENV="${TAU2_ENV:?set TAU2_ENV to the tau2 conda env}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/tau2_retail-airline-telecom_train.json}"

# ---- cluster -------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
USERSIM_MODE="${USERSIM_MODE:-local}"
USERSIM_TP="${USERSIM_TP:-1}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-7}"
GPUS="${GPUS:-8}"
ENVS_PER_GPU="${ENVS_PER_GPU:-4}"
SAVE_FREQ="${SAVE_FREQ:-15}"
EXP_NAME="${EXP_NAME:-tau2_infopo_aligned}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# batch x n must divide the number of training GPUs, or verl asserts at startup.
if (( (TRAIN_BATCH_SIZE * ROLLOUT_N) % NUM_TRAIN_GPUS != 0 )); then
  echo "FATAL: batch*n = $((TRAIN_BATCH_SIZE * ROLLOUT_N) ) is not divisible by" \
       "NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS}" >&2
  exit 1
fi
echo "per-rank batch: $(( TRAIN_BATCH_SIZE * ROLLOUT_N / NUM_TRAIN_GPUS ))" \
     "(20 is the value validated on 40GB cards)"

EXPORTS="ALL,TAU2_ENV=${TAU2_ENV},USERSIM_MODE=${USERSIM_MODE},USERSIM_TP=${USERSIM_TP}"
EXPORTS="${EXPORTS},NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS},TAU2_DOMAIN=${TAU2_DOMAIN}"
EXPORTS="${EXPORTS},TAU2_TASK_SPLIT=${TAU2_TASK_SPLIT},TAU2_REWARD_SHAPE=${TAU2_REWARD_SHAPE}"
EXPORTS="${EXPORTS},TAU2_REWARD_BASIS=${TAU2_REWARD_BASIS},TAU2_PROMPT_VARIANT=${TAU2_PROMPT_VARIANT}"
EXPORTS="${EXPORTS},TAU2_MAX_STEPS=${TAU2_MAX_STEPS},TAU2_FORCE_DONE_AFTER=${TAU2_FORCE_DONE_AFTER}"
EXPORTS="${EXPORTS},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},ROLLOUT_N=${ROLLOUT_N}"
EXPORTS="${EXPORTS},PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}"
EXPORTS="${EXPORTS},PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
EXPORTS="${EXPORTS},TOTAL_EPOCHS=${TOTAL_EPOCHS},MAX_ROUNDS=${MAX_ROUNDS}"
EXPORTS="${EXPORTS},MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH},MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
EXPORTS="${EXPORTS},MAX_MODEL_LEN=${MAX_MODEL_LEN},MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN}"
EXPORTS="${EXPORTS},POLICY_LR=${POLICY_LR},USE_KL_LOSS=${USE_KL_LOSS},KL_COEF=${KL_COEF}"
EXPORTS="${EXPORTS},ENTROPY_COEF=${ENTROPY_COEF}"
EXPORTS="${EXPORTS},ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
EXPORTS="${EXPORTS},ENVS_PER_GPU=${ENVS_PER_GPU},SAVE_FREQ=${SAVE_FREQ}"
EXPORTS="${EXPORTS},MODEL_PATH=${MODEL_PATH},EXP_NAME=${EXP_NAME},RESUME_MODE=disable"
EXPORTS="${EXPORTS},TRAIN_FILE=${TRAIN_FILE}"

# Anything else already set in this shell (plan-forecast, info-grpo, ...) rides along, so
# an ablation is one extra variable rather than a forked copy of this script.
for v in PLAN_FORECAST_ENABLE PLAN_FORECAST_COEF PLAN_FORECAST_K PLAN_FORECAST_TARGET \
         PLAN_FORECAST_GATE PLAN_FORECAST_MAX_LENGTH PLAN_FORECAST_SEQ PLAN_INLINE_ENABLE \
         PLAN_INLINE_K GRPO_FILTER_DEGENERATE INFO_INTRINSIC_WEIGHT INFO_GATE_TEMP INFO_KL_BATCH \
         INFO_MAX_TURNS; do
  if [ -n "${!v:-}" ]; then EXPORTS="${EXPORTS},${v}=${!v}"; fi
done

CMD=(sbatch --nodes=1 --gres="gpu:a100:${GPUS}" --time=48:00:00
     --export="${EXPORTS}" "${ROOT}/scripts/sbatch_tau2_grpo.sh")

if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
