#!/usr/bin/env bash
# Launch TWO identical-setting AlfWorld GRPO runs in parallel, each on 2 GPUs
# (2 x 2 = 4 GPUs total), for a fast 2-seed comparison in one run's wall-clock.
#
# The two runs share every tuning override (forwarded below) but diverge by
# nondeterministic rollout sampling — verl exposes no explicit seed knob, and
# identical-config pairs reliably diverge in practice. Each run gets its own
# env-server cluster (CPU-only AlfWorld servers on a distinct port range),
# its own EXP_NAME, and its own tmux sessions.
#
# Usage (set tuning vars then launch):
#   REF_NLL_ADD=True REF_NLL_COEF=0.1 bash launch_alfworld_2seed_tmux.sh
#
# Overridable: GPUS_A/GPUS_B, BASE_PORT_A/BASE_PORT_B, MODEL_PATH, TAG.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
# /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
# /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-alfworld}"

# Two GPU groups (2 GPUs each) and two disjoint env-server port ranges.
GPUS_A="${GPUS_A:-0,1}"
GPUS_B="${GPUS_B:-2,3}"
BASE_PORT_A="${BASE_PORT_A:-36001}"
BASE_PORT_B="${BASE_PORT_B:-36021}"

# vLLM gpu_memory_utilization override for the 2-GPU layout. On 2 GPUs the FSDP
# per-GPU footprint roughly doubles vs the 4-GPU run, so vLLM must reserve less
# to avoid the KV-cache init OOM. Lower than the inner script's 0.80 default.
# Overridden ONLY here (passed via env) — the inner run_*.sh is left untouched.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
TAG="${TAG:-2seed}"

# Forward only the tuning vars that are actually set in this shell; unset ones
# fall through to the defaults in run_alfworld_grpo_train.sh. (Kept in sync
# with launch_alfworld_grpo_tmux.sh.)
FWD=""
for v in ENABLE_ERC ERC_CLIPPING_METHOD ERC_CLIPPING_TYPE ERC_MOMENTUM UNCERTAINTY_SCALE_KAPPA UNCERTAINTY_SCALE_MIN UNCERTAINTY_SCALE_RENORMALIZE SAFE_COMMIT_MODE SAFE_COMMIT_OMEGA SAFE_COMMIT_RECENCY SAFE_COMMIT_RECENCY_GAMMA SAFE_COMMIT_SUCCESS_THRESHOLD SAFE_COMMIT_KAPPA SAFE_COMMIT_GMAX SAFE_COMMIT_GMIN SAFE_COMMIT_RENORMALIZE SAFE_COMMIT_TEXT_GATE SAFE_COMMIT_GATE_MODE SAFE_COMMIT_CLF_ENV SAFE_COMMIT_CLF_WINS_ONLY SAFE_COMMIT_CLF_MAX_NEW \
         WMLOSS_ADD_COEF WMLOSS_ADD_COEF_END WMLOSS_ADD_HORIZON \
         WMLOSS_ADD_USE_GAP WMLOSS_ADD_USE_ENTROPY WMLOSS_ADD_ONLY_FAILED WMLOSS_ADD_TO_REWARD REF_NLL_ADD REF_NLL_COEF \
         EPISTEMIC_BASE EPISTEMIC_BASE_NEG EPISTEMIC_S_MAX EPISTEMIC_SHAPE EPISTEMIC_USE_REF EPISTEMIC_PRE_ADD_COEF EPISTEMIC_INVERT_ON_NEG EPI_INTRINSIC_COEF EPI_INTRINSIC_CAP EPI_INTRINSIC_USE_REF \
         \
         USE_HINDSIGHT_HCA HCA_RATIO_CLIP_MIN HCA_RATIO_CLIP_MAX HCA_TEMP HCA_OMEGA HCA_GAMMA HCA_SMOOTH_ALPHA HCA_Z_THRESHOLD HCA_FINAL_STATE_MAX_TOKENS HCA_PERSTEP HCA_HISTORY_LEN PE_CREDIT_ENABLE PE_OMEGA_PROGRESS PE_OMEGA_EXPLORE PE_CREDIT_WINS_ONLY PE_CLF_ENV \
         WMC_COEFF POLICY_LR ENTROPY_COEF KL_COEF; do
  if [ -n "${!v:-}" ]; then FWD="${FWD} ${v}=${!v}"; fi
done
echo "Forwarding overrides to BOTH seeds:${FWD:-<none>}"

launch_one() {
  local gpus="$1" base_port="$2" label="$3"
  local n_gpus; n_gpus="$(awk -F, '{print NF}' <<< "${gpus}")"
  local exp="alfworld_grpo_qwen2.5_3b_add_${RUN_TS}_${TAG}_${label}"
  local env_sess="alfworld_env_${base_port}"
  local train_sess="alfworld_train_${TAG}_${label}"
  local train_log="${ROOT}/runlogs/${exp}/train.log"
  mkdir -p "${ROOT}/runlogs/${exp}"

  tmux has-session -t "${env_sess}" 2>/dev/null && tmux kill-session -t "${env_sess}" || true
  tmux has-session -t "${train_sess}" 2>/dev/null && tmux kill-session -t "${train_sess}" || true

  # Distinct LOG_DIR per cluster — the env service's cleanup() trap kills every
  # PID under its PID_DIR, so two clusters MUST NOT share one (else stopping one
  # cluster kills the other's servers).
  local env_log_dir="${ROOT}/runlogs/env_cluster/${RUN_TS}_${TAG}_${label}_p${base_port}"
  echo "[${label}] starting ${n_gpus} AlfWorld env services at port ${base_port} (GPUs ${gpus})..."
  tmux new-session -d -s "${env_sess}" \
    "cd ${ROOT} && NUM_ENVS=${n_gpus} BASE_PORT=${base_port} LOG_DIR=${env_log_dir} bash ${ROOT}/scripts/run_alfworld_env_service.sh"

  echo "[${label}] waiting for env services to become healthy..."
  for i in $(seq 0 $((n_gpus - 1))); do
    local port=$((base_port + i)) addr
    addr="http://127.0.0.1:$((base_port + i))"
    for _ in $(seq 1 60); do
      if curl --noproxy '*' -sf "${addr}/" >/dev/null; then break; fi
      sleep 2
    done
    if ! curl --noproxy '*' -sf "${addr}/" >/dev/null; then
      echo "[${label}] env service on port ${port} failed to start." >&2
      return 1
    fi
    echo "[${label}] port ${port} healthy."
  done

  echo "[${label}] starting GRPO training (exp=${exp})..."
  tmux new-session -d -s "${train_sess}" \
    "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${gpus} BASE_PORT=${base_port} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} PROJECT_NAME=${PROJECT_NAME} EXP_NAME=${exp} ROLLOUT_GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL} LOG_PATH=${train_log}${FWD} bash ${ROOT}/scripts/run_alfworld_grpo_train.sh"

  echo "[${label}] launched. env_session=${env_sess} train_session=${train_sess} log=${train_log}"
}

launch_one "${GPUS_A}" "${BASE_PORT_A}" "sA"
launch_one "${GPUS_B}" "${BASE_PORT_B}" "sB"

echo "--------------------------------------------------"
echo "Two-seed AlfWorld cluster launched (2 GPUs each)."
echo "  seed A: GPUs ${GPUS_A}  ports ${BASE_PORT_A}+  exp ..._${TAG}_sA"
echo "  seed B: GPUs ${GPUS_B}  ports ${BASE_PORT_B}+  exp ..._${TAG}_sB"
echo "--------------------------------------------------"
