#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/conda_envs/agentgym-rl}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
# /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
# /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct
TASK_NAME="alfworld"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

ENV_ADDR_HOST="${ENV_ADDR_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-36001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"

# Automatically construct comma-separated list of environment addresses
ENV_ADDR_LIST=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((BASE_PORT + i))
  ADDR="http://${ENV_ADDR_HOST}:${PORT}"
  if [[ -z "${ENV_ADDR_LIST}" ]]; then
    ENV_ADDR_LIST="${ADDR}"
  else
    ENV_ADDR_LIST="${ENV_ADDR_LIST},${ADDR}"
  fi
done
ENV_ADDR="${ENV_ADDR:-${ENV_ADDR_LIST}}"
echo "Using ENV_ADDR: ${ENV_ADDR}"

WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-alfworld}"

KL_COEF="${KL_COEF:-0.001}"
ENTROPY_COEF="${ENTROPY_COEF:-0.002}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
# Hard cap on optimizer steps (null = use len(dataloader)*total_epochs). Set a
# small value (e.g. 12) for short validation runs.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-512}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"
SAVE_FREQ="${SAVE_FREQ:-25}"

ENABLE_ERC="${ENABLE_ERC:-1}"
ERC_MU_BASE="${ERC_MU_BASE:-1.0}"
ERC_MU_EXP="${ERC_MU_EXP:-1.5}"
ERC_ETA_WM="${ERC_ETA_WM:-2.0}"
ERC_LAMBDA_WM="${ERC_LAMBDA_WM:-1.0}"
ERC_CLIPPING_TYPE="${ERC_CLIPPING_TYPE:-global}"
# Default: epi_intrinsic_add (Design A) — pure non-negative additive intrinsic
# bonus on advantage:  A_t += EPI_INTRINSIC_COEF * min(U_t, EPI_INTRINSIC_CAP),
# with U_t = max(0, NLL_t - H_t). Calibration-gap ICM variant (noisy-TV-robust)
# applied at the advantage layer, parallel to and reinforcing WM-SFT
# (WM-SFT is active whenever WMC_COEFF > 0; default 0.01).
# Other methods: 'add' (baseline-style additive entropy/NLL/gap with batch-
# mean recentering), 'epistemic_scale' (multiplicative per-turn redistribution
# with sign-asymmetric base / invert / pre_add knobs), or 'mask'/'sigmoid'/etc.
ERC_CLIPPING_METHOD="${ERC_CLIPPING_METHOD:-safe_commit}"
ERC_MOMENTUM="${ERC_MOMENTUM:-0.8}"
# Uncertainty-scaling (set ERC_CLIPPING_METHOD=uncertainty_scale to use):
# shrink advantage on turns with high next-obs env entropy (model unsure
# what its action causes) to prevent action-entropy collapse there.
UNCERTAINTY_SCALE_KAPPA="${UNCERTAINTY_SCALE_KAPPA:-1.0}"
UNCERTAINTY_SCALE_MIN="${UNCERTAINTY_SCALE_MIN:-0.7}"
UNCERTAINTY_SCALE_RENORMALIZE="${UNCERTAINTY_SCALE_RENORMALIZE:-True}"
# Safe-Commit Sharpener (set ERC_CLIPPING_METHOD=safe_commit): AMPLIFY advantage on
# safe-to-commit turns (low next-obs env entropy U) to accelerate commit. Opposite
# sign of uncertainty_scale; mean-preserving (renorm) => trajectory-unbiased.
# Additive-boost mode: SAFE_COMMIT_GMIN=1.0 + SAFE_COMMIT_RENORMALIZE=False.
SAFE_COMMIT_MODE="${SAFE_COMMIT_MODE:-add}"            # add (HCA-scale additive) | redistribute (mean-preserving)
SAFE_COMMIT_OMEGA="${SAFE_COMMIT_OMEGA:-1.0}"          # add-mode magnitude dial; omega=1.0 ~= HCA scale
SAFE_COMMIT_RECENCY="${SAFE_COMMIT_RECENCY:-1.0}"         # add: 0=pure content; >0 tilts boost to late (success end-game) turns
SAFE_COMMIT_RECENCY_GAMMA="${SAFE_COMMIT_RECENCY_GAMMA:-0.95}"
SAFE_COMMIT_SUCCESS_THRESHOLD="${SAFE_COMMIT_SUCCESS_THRESHOLD:-0.0}"  # add-mode: win = traj A_GRPO > thr
SAFE_COMMIT_KAPPA="${SAFE_COMMIT_KAPPA:-1.0}"
# Text gate (DEFAULT OFF): per-turn ask model if outcome predictable, AND with low
# action-entropy -> only commit-boost turns that are BOTH predictable AND low-entropy.
SAFE_COMMIT_TEXT_GATE="${SAFE_COMMIT_TEXT_GATE:-False}"
SAFE_COMMIT_GATE_MODE="${SAFE_COMMIT_GATE_MODE:-and}"          # and (strict) | soft
SAFE_COMMIT_CLF_ENV="${SAFE_COMMIT_CLF_ENV:-alfworld}"         # domain slot for classifier prompt
SAFE_COMMIT_CLF_WINS_ONLY="${SAFE_COMMIT_CLF_WINS_ONLY:-True}" # classify only winning trajectories
SAFE_COMMIT_CLF_MAX_NEW="${SAFE_COMMIT_CLF_MAX_NEW:-24}"
SAFE_COMMIT_GMAX="${SAFE_COMMIT_GMAX:-1.0}"
SAFE_COMMIT_GMIN="${SAFE_COMMIT_GMIN:-0.2}"
SAFE_COMMIT_RENORMALIZE="${SAFE_COMMIT_RENORMALIZE:-False}"
WMLOSS_ADD_COEF="${WMLOSS_ADD_COEF:-0.2}"
WMLOSS_ADD_COEF_END="${WMLOSS_ADD_COEF_END:-0}"
WMLOSS_ADD_HORIZON="${WMLOSS_ADD_HORIZON:-0}"
WMLOSS_ADD_USE_ENTROPY="${WMLOSS_ADD_USE_ENTROPY:-True}"
# use_gap: build the additive advantage bonus from the epistemic calibration
# gap max(0, NLL - H) instead of raw entropy/NLL. Same add mechanism as the
# baseline, better (noisy-TV-robust) signal. Requires ERC_CLIPPING_METHOD=add.
WMLOSS_ADD_USE_GAP="${WMLOSS_ADD_USE_GAP:-False}"
WMLOSS_ADD_USE_EMA="${WMLOSS_ADD_USE_EMA:-False}"
WMLOSS_ADD_USE_GROUPED="${WMLOSS_ADD_USE_GROUPED:-True}"
WMLOSS_ADD_USE_REF_BASELINE="${WMLOSS_ADD_USE_REF_BASELINE:-False}"
WMLOSS_ADD_ONLY_FAILED="${WMLOSS_ADD_ONLY_FAILED:-False}"
WMLOSS_ADD_TO_REWARD="${WMLOSS_ADD_TO_REWARD:-False}"
# ref_nll_add: OVERRIDES all other wmc_erc paths. Adds REF_NLL_COEF *
# (frozen initial ref model env-token NLL) directly to advantage per turn.
REF_NLL_ADD="${REF_NLL_ADD:-False}"
REF_NLL_COEF="${REF_NLL_COEF:-0.4}"
# Epistemic advantage scaling (set ERC_CLIPPING_METHOD=epistemic_scale to use).
# Sign-preserving, trajectory-normalized credit redistribution driven by the
# world-model calibration gap (NLL - entropy). EPISTEMIC_BASE is the
# aggressiveness floor in units of the EMA mean signal (larger = milder),
# EPISTEMIC_S_MAX caps per-turn normalized signal, EPISTEMIC_USE_REF switches
# to the cross-model (current-vs-ref) epistemic proxy when ref_entropy exists.
EPISTEMIC_USE_REF="${EPISTEMIC_USE_REF:-False}"
EPISTEMIC_BASE="${EPISTEMIC_BASE:-1.0}"
EPISTEMIC_S_MAX="${EPISTEMIC_S_MAX:-5.0}"
# Concave U->weight shaping: linear | sqrt | log (sqrt = sharp at low U,
# saturating at high U). EPISTEMIC_BASE_NEG >= EPISTEMIC_BASE makes losing
# trajectories redistribute more gently (protect exploration); default ==
# EPISTEMIC_BASE i.e. symmetric. Set larger (e.g. 2x) to enable the asymmetry.
EPISTEMIC_SHAPE="${EPISTEMIC_SHAPE:-linear}"
EPISTEMIC_BASE_NEG="${EPISTEMIC_BASE_NEG:-5.0}"
# Optional small additive raw-entropy bonus applied BEFORE the epi_scale
# multiplicative redistribution (mimics baseline's net exploration push,
# which pure redistribution cannot supply). 0 = off (default).
EPISTEMIC_PRE_ADD_COEF="${EPISTEMIC_PRE_ADD_COEF:-0.0}"
# Design A: pure non-negative additive intrinsic bonus on advantage.
#   A_t += EPI_INTRINSIC_COEF * min(U_t, EPI_INTRINSIC_CAP), U=max(0, NLL-H).
# Use with ERC_CLIPPING_METHOD=epi_intrinsic_add. No recentering, no per-turn
# redistribution -- just a calibration-gap intrinsic reward at the advantage
# layer (noisy-TV-robust ICM variant), parallel to WM-SFT.
EPI_INTRINSIC_COEF="${EPI_INTRINSIC_COEF:-0.3}"
EPI_INTRINSIC_CAP="${EPI_INTRINSIC_CAP:-0.5}"
EPI_INTRINSIC_USE_REF="${EPI_INTRINSIC_USE_REF:-False}"
# On FAILURE trajectories (A_grpo<0) only: flip s_t -> (s_max - s_t) so that
# routine/loop turns (low U) get heaviest punishment and exploratory turns
# (high U) get protected. Corrects the structural anti-exploration bias on
# failures that we diagnosed in 174247 (stuck-loop dup% 73% by turn 20).
# Off by default; recovers symmetric behavior.
EPISTEMIC_INVERT_ON_NEG="${EPISTEMIC_INVERT_ON_NEG:-True}"

ERC_ENABLE_VALUE="False"
if [[ "${ENABLE_ERC}" == "1" ]]; then
  ERC_ENABLE_VALUE="True"
fi


# Hindsight Credit Assignment (HCAPO, arXiv:2603.08754) — training-free
# generative verification: re-prompt the frozen policy with the realized
# outcome, no separate head / SFT. Independent of the value baseline
# (HARD MUTEX). OFF (default) => behaviour matches pure GRPO.
USE_HINDSIGHT_HCA="${USE_HINDSIGHT_HCA:-False}"
# Self-normalized hindsight ratio rho = pi_hind / mean(pi_hind), clipped.
HCA_RATIO_CLIP_MIN="${HCA_RATIO_CLIP_MIN:-0.8}"
HCA_RATIO_CLIP_MAX="${HCA_RATIO_CLIP_MAX:-1.2}"
# Sharpening temperature T_temp in pi_hind = exp(mean_log_p / T_temp).
HCA_TEMP="${HCA_TEMP:-2.0}"
# Weight of the micro (hindsight) advantage added to the GRPO macro advantage.
HCA_OMEGA="${HCA_OMEGA:-1.0}"
# Discount for the per-turn return G_t = gamma^{T-1-t} * R.
HCA_GAMMA="${HCA_GAMMA:-1.0}"
# Temporal smoothing Q_t = a*Q_t + (1-a)*Q_{t+1}; set 1.0 to disable.
HCA_SMOOTH_ALPHA="${HCA_SMOOTH_ALPHA:-0.5}"
# Reward threshold to label success (binary R in {0,1} => 0.5).
HCA_Z_THRESHOLD="${HCA_Z_THRESHOLD:-0.5}"
# Max tokens of the realized FINAL STATE injected as hindsight (env-agnostic).
HCA_FINAL_STATE_MAX_TOKENS="${HCA_FINAL_STATE_MAX_TOKENS:-96}"
# HCAPO-aligned per-step LOCAL injection (paper §4.2). OFF=old front-once (no signal);
# ON=reconstruct truncated per-step prompts (history_len) so pi_hind carries hindsight.
HCA_PERSTEP="${HCA_PERSTEP:-True}"
HCA_HISTORY_LEN="${HCA_HISTORY_LEN:-0}"   # 0 = FULL history (consistent with our impl); >0 = paper truncation
# Progress/Exploration credit (DEFAULT OFF): classify P/E steps per traj, add all-positive
# advantage (omega_p on progress > omega_e on exploration; P/E overlap -> P; wins only).
PE_CREDIT_ENABLE="${PE_CREDIT_ENABLE:-False}"
PE_OMEGA_PROGRESS="${PE_OMEGA_PROGRESS:-1.0}"
PE_OMEGA_EXPLORE="${PE_OMEGA_EXPLORE:-0.4}"
PE_CREDIT_WINS_ONLY="${PE_CREDIT_WINS_ONLY:-True}"
PE_CLF_ENV="${PE_CLF_ENV:-alfworld}"
# Plan-forecast auxiliary SFT (DEFAULT OFF): each step predict the realized next-K
# action commands (current included). Separate forward, CE loss * coef, no PG.
# gate=wins -> only winning trajectories; gate=all -> every trajectory.
PLAN_FORECAST_ENABLE="${PLAN_FORECAST_ENABLE:-False}"
PLAN_FORECAST_COEF="${PLAN_FORECAST_COEF:-0}"
PLAN_FORECAST_K="${PLAN_FORECAST_K:-3}"
# Horizon-growth schedule (curriculum): grow the forecast target length over
# training. Format "startStep:kMin:kMax,..." (start-step semantics, last stage
# persists); each per-step sample draws k uniformly in the active [kMin,kMax] and
# the prompt/plan-block align to the realized length. Empty = OFF (use PLAN_FORECAST_K).
# First stage MUST start at 0. Example: "0:2:3,40:2:4,80:3:5".
PLAN_FORECAST_K_SCHEDULE="${PLAN_FORECAST_K_SCHEDULE:-}"
# skip_invalid: build the forecast target from only EFFECTIVE actions — drop actions
# whose env result was invalid / no-effect ("Nothing happens." / "Invalid Action." /
# "No known action..."; per-env, auto-selected by task_name). Default off.
PLAN_FORECAST_SKIP_INVALID="${PLAN_FORECAST_SKIP_INVALID:-True}"
# gate=wins: forecast-SFT only on winning trajectories (block2-success, proven).
PLAN_FORECAST_GATE="${PLAN_FORECAST_GATE:-wins}"
# --- Two ORTHOGONAL group knobs (compose; both distill successes only) ---
# Group-success GATING (curriculum): filter WHICH groups' successes to distill by the
# group's success-rate. off (use PLAN_FORECAST_GATE) | low (rate<=LOW) | low_high
# (rate<=LOW OR >=HIGH, skip mid-rate groups where GRPO signal is strong). Default off.
PLAN_FORECAST_GROUP_GATE="${PLAN_FORECAST_GROUP_GATE:-off}"
PLAN_FORECAST_GROUP_LOW_THRESH="${PLAN_FORECAST_GROUP_LOW_THRESH:-0.5}"
PLAN_FORECAST_GROUP_HIGH_THRESH="${PLAN_FORECAST_GROUP_HIGH_THRESH:-1.0}"
# Group-weight NORMALIZATION (stability): give every kept GRPO group the SAME total
# plan-CE weight = the single PLAN_FORECAST_COEF, split EVENLY among its distilled
# successful trajectories. As success-rate rises mid/late training, per-traj weight
# shrinks and each group's contribution stays constant -> no SFT blow-up from more
# successful samples. Applies to whatever gating keeps.
PLAN_FORECAST_GROUP_NORM="${PLAN_FORECAST_GROUP_NORM:-True}"
# group_dedup (default True): when group_norm on, split each group's weight over its
# DISTINCT successful action-sequences instead of per-trajectory -> duplicate rollouts
# don't inflate weight (within-group action repetition is heavy mid/late training).
PLAN_FORECAST_GROUP_DEDUP="${PLAN_FORECAST_GROUP_DEDUP:-False}"
PLAN_FORECAST_SUCCESS_THRESHOLD="${PLAN_FORECAST_SUCCESS_THRESHOLD:-0.5}"
PLAN_FORECAST_MAX_LENGTH="${PLAN_FORECAST_MAX_LENGTH:-4096}"
# plan_forecast target: action (predict realized next-K actions; block2-success,
# grounded) | subgoal (predict next-K hindsight-confirmed achieved sub-goals).
PLAN_FORECAST_TARGET="${PLAN_FORECAST_TARGET:-action}"
# plan_forecast seq: separate (block2 — synthetic prompt + bare list, standalone;
# use with inline OFF) | inline_consistent (SFT sample matches a real rollout turn
# obs->Plan+Action; use with inline ON). inline_consistent auto-falls-back to
# separate when inline is off.
PLAN_FORECAST_SEQ="${PLAN_FORECAST_SEQ:-separate}"
# plan_forecast_coef anneal: fixed | linear | power | cutoff (start = PLAN_FORECAST_COEF).
PLAN_FORECAST_COEF_ANNEAL="${PLAN_FORECAST_COEF_ANNEAL:-fixed}"
PLAN_FORECAST_COEF_END="${PLAN_FORECAST_COEF_END:-0.0}"
PLAN_FORECAST_COEF_HORIZON="${PLAN_FORECAST_COEF_HORIZON:-40}"
PLAN_FORECAST_COEF_POWER="${PLAN_FORECAST_COEF_POWER:-2.0}"
PLAN_FORECAST_COEF_CUTOFF_STEP="${PLAN_FORECAST_COEF_CUTOFF_STEP:-0}"
# SFT-ablation (RFT) control — MUTUALLY EXCLUSIVE with plan_forecast (asserted at
# init). When ON: one extra SFT round behavior-cloning THIS step's winning trajs'
# real assistant turns (vs plan_forecast's constructed forecast target). Same
# optimizer path + win-gate; toggle only these two for a clean A/B. DEFAULT OFF.
SFT_ABLATION_ENABLE="${SFT_ABLATION_ENABLE:-False}"
SFT_ABLATION_COEF="${SFT_ABLATION_COEF:-0.01}"      # match PLAN_FORECAST_COEF for a fair control
SFT_ABLATION_GATE="${SFT_ABLATION_GATE:-wins}"      # wins (success trajs only) | all
# Plan FORMAT reward (DEFAULT OFF): per-turn shaping bonus on advantage for a
# well-formed Thought->Plan->Action turn (counters inline-plan decay under RL).
# bonus_t = COEF*(score-BASELINE) on the turn's tokens; baseline 0.5 = symmetric.
PLAN_FORMAT_REWARD_ENABLE="${PLAN_FORMAT_REWARD_ENABLE:-False}"
PLAN_FORMAT_REWARD_COEF="${PLAN_FORMAT_REWARD_COEF:-0.05}"
PLAN_FORMAT_REWARD_BASELINE="${PLAN_FORMAT_REWARD_BASELINE:-0.5}"
PLAN_FORMAT_REWARD_CLIP="${PLAN_FORMAT_REWARD_CLIP:-0.0}"
# penalty_only: only penalize turns that DROP the plan (never reward keeping it).
PLAN_FORMAT_REWARD_PENALTY_ONLY="${PLAN_FORMAT_REWARD_PENALTY_ONLY:-True}"
# warmup: keep format reward OFF until global_step >= this (learn task first).
PLAN_FORMAT_REWARD_WARMUP_STEPS="${PLAN_FORMAT_REWARD_WARMUP_STEPS:-10}"
# Block 1 (inline plan, DEFAULT OFF): append a standing instruction so the model
# writes its next-K-action plan inside the THOUGHT each turn (auto-eats PG, the
# env still parses Action:). Pairs with block 2 (plan_forecast) — same K/framing.
PLAN_INLINE_ENABLE="${PLAN_INLINE_ENABLE:-False}"
PLAN_INLINE_K="${PLAN_INLINE_K:-${PLAN_FORECAST_K}}"
# inline plan style: actions (next-K actions) | todo (checkable sub-goal TODO
# list with (done) marks; pair with PLAN_FORECAST_TARGET=subgoal).
PLAN_INLINE_STYLE="${PLAN_INLINE_STYLE:-actions}"
# per-turn reminder: re-state the Plan request after EVERY obs (one-time decays).
PLAN_INLINE_PER_TURN="${PLAN_INLINE_PER_TURN:-False}"   # ARCHIVED: per-turn reminder off; opening prompt only
# inline warmup: use ORIGINAL prompt until global_step >= this, then introduce
# the plan prompt (cold-start: let task competence build before planning).
PLAN_INLINE_WARMUP_STEPS="${PLAN_INLINE_WARMUP_STEPS:-0}"
# think reminder (alternative to inline plan, mutually exclusive): per-turn nudge
# to reason in a Thought before the Action, WITHOUT forcing a Plan.
THINK_REMINDER_ENABLE="${THINK_REMINDER_ENABLE:-False}"
# HCA action-only ρ scoring: score on the action tokens (after the
# delimiter) instead of the whole Thought+Action turn. Default OFF.

WMC_COEFF="${WMC_COEFF:-0.01}"
# traj_lm: full-sequence next-token CE over the WHOLE trajectory (env obs AND the
# agent's own tokens), coef>0 = on. MUTUALLY EXCLUSIVE with WM-SFT (asserted at init):
# don't set this together with WMC_COEFF>0 or WM_ENABLE=True. Default 0 = off.
TRAJ_LM_COEF="${TRAJ_LM_COEF:-0}"
# traj_lm trajectory gate: all (clone every rollout, default = old behavior) |
# wins (only clone trajectories with GRPO advantage>0, i.e. better than group mean).
# 'all' BC's losing behavior too -> anchors policy to base & slows early learning;
# 'wins' is recommended (mirrors block2 forecast's gate=wins).
TRAJ_LM_GATE="${TRAJ_LM_GATE:-wins}"
WMC_TYPE="${WMC_TYPE:-fixed}"
WMC_START_COEFF="${WMC_START_COEFF:-0.001}"
WMC_END_COEFF="${WMC_END_COEFF:-0.0}"
WMC_HORIZON="${WMC_HORIZON:-100}"
WMC_POWER="${WMC_POWER:-2}"
WMC_CUTOFF_STEP="${WMC_CUTOFF_STEP:-50}"

WM_ENABLE="${WM_ENABLE:-False}"
# C3 placebo (WM-value experiment): shuffle obs targets so WM-SFT gets same dense
# gradient with NO real dynamics signal. DEFAULT False = normal WM-SFT. Only meaningful
# when WM-SFT is active (WMC_COEFF>0 or WM_ENABLE=True).
WM_PLACEBO_SHUFFLE="${WM_PLACEBO_SHUFFLE:-False}"
# Separate WM-SFT pass strength, decoupled from the inline WMC_COEFF. Set >0 (and
# WM_ENABLE=True, WMC_COEFF=0) to run the separate WM-SFT alone (WM-value experiment).
WM_SFT_COEF="${WM_SFT_COEF:-0.0}"
WM_LOSS_PI_DEDUP="${WM_LOSS_PI_DEDUP:-True}"

WM_ENV_PREDICT_PROMPT="${WM_ENV_PREDICT_PROMPT:-null}"
WM_MAX_LENGTH="${WM_MAX_LENGTH:-4096}"
WM_MAX_SAMPLES_PER_TRAJECTORY="${WM_MAX_SAMPLES_PER_TRAJECTORY:-null}"
WM_MIN_ENV_TOKENS="${WM_MIN_ENV_TOKENS:-1}"

EXP_NAME="${EXP_NAME:-alfworld_grpo_qwen2.5_7b_$(date -u +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${RUN_DIR}/rollout_logs}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/AgentItemId/train/alfworld_train.json}"
LOG_PATH="${LOG_PATH:-}"

mkdir -p "${CKPT_DIR}" "${RUN_DIR}" "${ROLLOUT_LOG_DIR}"
if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${TRAIN_CODE_DIR}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  VLLM_USE_MODELSCOPE=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  python -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
    data.train_file="${TRAIN_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    +data.plan_inline_enable="${PLAN_INLINE_ENABLE}" \
    +data.plan_inline_k="${PLAN_INLINE_K}" \
    +data.plan_inline_style="${PLAN_INLINE_STYLE}" \
    +data.plan_inline_per_turn="${PLAN_INLINE_PER_TURN}" \
    +data.plan_inline_warmup_steps="${PLAN_INLINE_WARMUP_STEPS}" \
    +data.think_reminder_enable="${THINK_REMINDER_ENABLE}" \
    actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
    actor_rollout_ref.agentgym.env_addr="'${ENV_ADDR}'" \
    actor_rollout_ref.agentgym.timeout=2400 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEF} \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_tokens="${MAX_TOKENS_PER_TURN}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    wmc_erc.enable="${ERC_ENABLE_VALUE}" \
    wmc_erc.mu_base="${ERC_MU_BASE}" \
    wmc_erc.mu_exp="${ERC_MU_EXP}" \
    wmc_erc.eta_wm="${ERC_ETA_WM}" \
    wmc_erc.lambda_wm="${ERC_LAMBDA_WM}" \
    wmc_erc.clipping_type="${ERC_CLIPPING_TYPE}" \
    wmc_erc.clipping_method="${ERC_CLIPPING_METHOD}" \
    wmc_erc.momentum="${ERC_MOMENTUM}" \
    +wmc_erc.uncertainty_scale_kappa="${UNCERTAINTY_SCALE_KAPPA}" \
    +wmc_erc.uncertainty_scale_min="${UNCERTAINTY_SCALE_MIN}" \
    +wmc_erc.uncertainty_scale_renormalize="${UNCERTAINTY_SCALE_RENORMALIZE}" \
    +wmc_erc.safe_commit_mode="${SAFE_COMMIT_MODE}" \
    +wmc_erc.safe_commit_omega="${SAFE_COMMIT_OMEGA}" \
    +wmc_erc.safe_commit_recency="${SAFE_COMMIT_RECENCY}" \
    +wmc_erc.safe_commit_recency_gamma="${SAFE_COMMIT_RECENCY_GAMMA}" \
    +wmc_erc.safe_commit_success_threshold="${SAFE_COMMIT_SUCCESS_THRESHOLD}" \
    +wmc_erc.safe_commit_kappa="${SAFE_COMMIT_KAPPA}" \
    +wmc_erc.safe_commit_gmax="${SAFE_COMMIT_GMAX}" \
    +wmc_erc.safe_commit_gmin="${SAFE_COMMIT_GMIN}" \
    +wmc_erc.safe_commit_renormalize="${SAFE_COMMIT_RENORMALIZE}" \
    +wmc_erc.safe_commit_text_gate="${SAFE_COMMIT_TEXT_GATE}" \
    +wmc_erc.safe_commit_gate_mode="${SAFE_COMMIT_GATE_MODE}" \
    +wmc_erc.safe_commit_clf_env="${SAFE_COMMIT_CLF_ENV}" \
    +wmc_erc.safe_commit_clf_wins_only="${SAFE_COMMIT_CLF_WINS_ONLY}" \
    +wmc_erc.safe_commit_clf_max_new_tokens="${SAFE_COMMIT_CLF_MAX_NEW}" \
    +wmc_erc.wmloss_add_coef="${WMLOSS_ADD_COEF}" \
    +wmc_erc.wmloss_add_use_entropy="${WMLOSS_ADD_USE_ENTROPY}" \
    +wmc_erc.wmloss_add_use_gap="${WMLOSS_ADD_USE_GAP}" \
    +wmc_erc.wmloss_add_use_ema="${WMLOSS_ADD_USE_EMA}" \
    +wmc_erc.wmloss_add_use_grouped="${WMLOSS_ADD_USE_GROUPED}" \
    +wmc_erc.wmloss_add_use_ref_baseline="${WMLOSS_ADD_USE_REF_BASELINE}" \
    +wmc_erc.wmloss_add_only_failed="${WMLOSS_ADD_ONLY_FAILED}" \
    +wmc_erc.wmloss_add_coef_end="${WMLOSS_ADD_COEF_END}" \
    +wmc_erc.wmloss_add_horizon="${WMLOSS_ADD_HORIZON}" \
    +wmc_erc.wmloss_add_to_reward="${WMLOSS_ADD_TO_REWARD}" \
    +wmc_erc.ref_nll_add="${REF_NLL_ADD}" \
    +wmc_erc.ref_nll_coef="${REF_NLL_COEF}" \
    +wmc_erc.epistemic_use_ref="${EPISTEMIC_USE_REF}" \
    +wmc_erc.epistemic_base="${EPISTEMIC_BASE}" \
    +wmc_erc.epistemic_base_neg="${EPISTEMIC_BASE_NEG}" \
    +wmc_erc.epistemic_s_max="${EPISTEMIC_S_MAX}" \
    +wmc_erc.epistemic_shape="${EPISTEMIC_SHAPE}" \
    +wmc_erc.epistemic_pre_add_coef="${EPISTEMIC_PRE_ADD_COEF}" \
    +wmc_erc.epistemic_invert_on_neg="${EPISTEMIC_INVERT_ON_NEG}" \
    +wmc_erc.epi_intrinsic_coef="${EPI_INTRINSIC_COEF}" \
    +wmc_erc.epi_intrinsic_cap="${EPI_INTRINSIC_CAP}" \
    +wmc_erc.epi_intrinsic_use_ref="${EPI_INTRINSIC_USE_REF}" \
    +actor_rollout_ref.actor.use_hindsight_hca="${USE_HINDSIGHT_HCA}" \
    +actor_rollout_ref.actor.hca_ratio_clip_min="${HCA_RATIO_CLIP_MIN}" \
    +actor_rollout_ref.actor.hca_ratio_clip_max="${HCA_RATIO_CLIP_MAX}" \
    +actor_rollout_ref.actor.hca_temp="${HCA_TEMP}" \
    +actor_rollout_ref.actor.hca_omega="${HCA_OMEGA}" \
    +actor_rollout_ref.actor.hca_gamma="${HCA_GAMMA}" \
    +actor_rollout_ref.actor.hca_smooth_alpha="${HCA_SMOOTH_ALPHA}" \
    +actor_rollout_ref.actor.hca_z_threshold="${HCA_Z_THRESHOLD}" \
    +actor_rollout_ref.actor.hca_final_state_max_tokens="${HCA_FINAL_STATE_MAX_TOKENS}" \
    +actor_rollout_ref.actor.hca_perstep="${HCA_PERSTEP}" \
    +actor_rollout_ref.actor.hca_history_len="${HCA_HISTORY_LEN}" \
    +actor_rollout_ref.actor.pe_credit_enable="${PE_CREDIT_ENABLE}" \
    +actor_rollout_ref.actor.pe_omega_progress="${PE_OMEGA_PROGRESS}" \
    +actor_rollout_ref.actor.pe_omega_explore="${PE_OMEGA_EXPLORE}" \
    +actor_rollout_ref.actor.pe_credit_wins_only="${PE_CREDIT_WINS_ONLY}" \
    +actor_rollout_ref.actor.pe_clf_env="${PE_CLF_ENV}" \
    +actor_rollout_ref.actor.plan_forecast_enable="${PLAN_FORECAST_ENABLE}" \
    +actor_rollout_ref.actor.plan_forecast_coef="${PLAN_FORECAST_COEF}" \
    +actor_rollout_ref.actor.plan_forecast_k="${PLAN_FORECAST_K}" \
    +actor_rollout_ref.actor.plan_forecast_k_schedule="'${PLAN_FORECAST_K_SCHEDULE}'" \
    +actor_rollout_ref.actor.plan_forecast_skip_invalid="${PLAN_FORECAST_SKIP_INVALID}" \
    +actor_rollout_ref.actor.plan_forecast_gate="${PLAN_FORECAST_GATE}" \
    +actor_rollout_ref.actor.plan_forecast_group_gate="${PLAN_FORECAST_GROUP_GATE}" \
    +actor_rollout_ref.actor.plan_forecast_group_low_thresh="${PLAN_FORECAST_GROUP_LOW_THRESH}" \
    +actor_rollout_ref.actor.plan_forecast_group_high_thresh="${PLAN_FORECAST_GROUP_HIGH_THRESH}" \
    +actor_rollout_ref.actor.plan_forecast_group_norm="${PLAN_FORECAST_GROUP_NORM}" \
    +actor_rollout_ref.actor.plan_forecast_group_dedup="${PLAN_FORECAST_GROUP_DEDUP}" \
    +actor_rollout_ref.actor.plan_forecast_success_threshold="${PLAN_FORECAST_SUCCESS_THRESHOLD}" \
    +actor_rollout_ref.actor.plan_forecast_max_length="${PLAN_FORECAST_MAX_LENGTH}" \
    +actor_rollout_ref.actor.plan_forecast_target="${PLAN_FORECAST_TARGET}" \
    +actor_rollout_ref.actor.plan_forecast_seq="${PLAN_FORECAST_SEQ}" \
    +actor_rollout_ref.actor.plan_forecast_coef_anneal="${PLAN_FORECAST_COEF_ANNEAL}" \
    +actor_rollout_ref.actor.plan_forecast_coef_end="${PLAN_FORECAST_COEF_END}" \
    +actor_rollout_ref.actor.plan_forecast_coef_horizon="${PLAN_FORECAST_COEF_HORIZON}" \
    +actor_rollout_ref.actor.plan_forecast_coef_power="${PLAN_FORECAST_COEF_POWER}" \
    +actor_rollout_ref.actor.plan_forecast_coef_cutoff_step="${PLAN_FORECAST_COEF_CUTOFF_STEP}" \
    +actor_rollout_ref.actor.sft_ablation_enable="${SFT_ABLATION_ENABLE}" \
    +actor_rollout_ref.actor.sft_ablation_coef="${SFT_ABLATION_COEF}" \
    +actor_rollout_ref.actor.sft_ablation_gate="${SFT_ABLATION_GATE}" \
    +actor_rollout_ref.actor.plan_format_reward_enable="${PLAN_FORMAT_REWARD_ENABLE}" \
    +actor_rollout_ref.actor.plan_format_reward_coef="${PLAN_FORMAT_REWARD_COEF}" \
    +actor_rollout_ref.actor.plan_format_reward_baseline="${PLAN_FORMAT_REWARD_BASELINE}" \
    +actor_rollout_ref.actor.plan_format_reward_clip="${PLAN_FORMAT_REWARD_CLIP}" \
    +actor_rollout_ref.actor.plan_format_reward_penalty_only="${PLAN_FORMAT_REWARD_PENALTY_ONLY}" \
    +actor_rollout_ref.actor.plan_format_reward_warmup_steps="${PLAN_FORMAT_REWARD_WARMUP_STEPS}" \
    actor_rollout_ref.actor.world_model_coeff="${WMC_COEFF}" \
    +actor_rollout_ref.actor.traj_lm_coef="${TRAJ_LM_COEF}" \
    +actor_rollout_ref.actor.traj_lm_gate="${TRAJ_LM_GATE}" \
    actor_rollout_ref.actor.world_model.enable="${WM_ENABLE}" \
    +actor_rollout_ref.actor.world_model.placebo_shuffle="${WM_PLACEBO_SHUFFLE}" \
    +actor_rollout_ref.actor.world_model.sft_coef="${WM_SFT_COEF}" \
    actor_rollout_ref.actor.world_model.env_predict_prompt="${WM_ENV_PREDICT_PROMPT}" \
    actor_rollout_ref.actor.world_model.max_length="${WM_MAX_LENGTH}" \
    actor_rollout_ref.actor.world_model.max_samples_per_trajectory="${WM_MAX_SAMPLES_PER_TRAJECTORY}" \
    actor_rollout_ref.actor.world_model.min_env_tokens="${WM_MIN_ENV_TOKENS}" \
    +actor_rollout_ref.actor.wm_loss_pi_dedup="${WM_LOSS_PI_DEDUP}" \
    algorithm.world_model_coeff_ctrl.type="${WMC_TYPE}" \
    algorithm.world_model_coeff_ctrl.start_coeff="${WMC_START_COEFF}" \
    algorithm.world_model_coeff_ctrl.end_coeff="${WMC_END_COEFF}" \
    algorithm.world_model_coeff_ctrl.horizon="${WMC_HORIZON}" \
    algorithm.world_model_coeff_ctrl.power="${WMC_POWER}" \
    algorithm.world_model_coeff_ctrl.cutoff_step="${WMC_CUTOFF_STEP}"
