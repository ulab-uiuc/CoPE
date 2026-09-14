#!/usr/bin/env bash
#SBATCH --job-name=tau2_align
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=${PROJECT_ROOT}/slurm_logs/tau2_align_%j.out
#SBATCH --error=${PROJECT_ROOT}/slurm_logs/tau2_align_%j.err
#
# Reproduce InfoPO's tau2-bench protocol exactly, by driving tau2's own CLI rather than
# this repo's env-server + ReAct client.
#
# InfoPO evaluates with `tau2 run` and overrides nothing (eval/tau2bench/evaluator.py in
# kfq20/InfoPO), so the agent prompt, the user-simulator prompt and the reward all stay
# at tau2's defaults -- in particular the reward is EvaluationType.ALL, which runs the
# NL-assertion judge. Our own harness scores with env-assertions only and injects a
# `strict` ReAct prompt, so its numbers are not comparable to the paper's table: base
# Qwen2.5-7B measured 18.8% on airline under our protocol against the paper's 7.5%.
# Going through the CLI removes every one of those differences at once.
#
#   sbatch scripts/sbatch_tau2_align.sh                                   # base model
#   sbatch --export=ALL,MODEL_PATH=<ckpt>,TAG=trained scripts/sbatch_tau2_align.sh
#   sbatch --export=ALL,DOMAINS=airline scripts/sbatch_tau2_align.sh      # one domain
#
# Target (paper Table 1, Qwen2.5-7B-Instruct prompting row):
#   telecom 14.4   retail 13.1   airline 7.5

set -euo pipefail
ROOT=${PROJECT_ROOT}
cd "${ROOT}"

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/Qwen2.5-7B-Instruct}"
TAG="${TAG:-base}"
DOMAINS="${DOMAINS:-airline retail telecom}"

# Values below are InfoPO's eval/tau2bench/eval_vllm.sh verbatim. Note MAX_STEPS=200,
# not the "Tmax = 50 turns" quoted in the paper body -- max-steps counts orchestrator
# steps (agent AND user messages), and the script that produced the table used 200.
TASK_SPLIT="${TASK_SPLIT:-test}"
NUM_TRIALS="${NUM_TRIALS:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
AGENT_TEMP="${AGENT_TEMP:-0.0}"
USER_MODEL="${USER_MODEL:-gpt-4o-mini-2024-07-18}"
# hosted -> gpt-4o-mini, the protocol the paper uses and the only one whose numbers are
# comparable. local -> a second vLLM on this node, for mechanical questions (does an
# episode fit in the step budget?) that do not depend on which model plays the customer,
# and which would otherwise be un-runnable without API credit.
USERSIM_BACKEND="${USERSIM_BACKEND:-hosted}"
USERSIM_LOCAL_MODEL="${USERSIM_LOCAL_MODEL:-${MODEL_DIR}/Qwen2.5-7B-Instruct}"
# The customer sees the whole transcript plus telecom's 6212-token policy, so it needs a
# bigger window than the agent: at 32768 a long telecom episode dies with
# ContextWindowExceededError and takes the whole run with it.
USERSIM_MAX_LEN="${USERSIM_MAX_LEN:-65536}"
USERSIM_PORT="${USERSIM_PORT:-8510}"
USER_TEMP="${USER_TEMP:-0.0}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-32768}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"

POLICY_PORT="${POLICY_PORT:-8500}"
POLICY_TP="${POLICY_TP:-2}"
SERVED_NAME=policy
KEY_FILE="${KEY_FILE:-${ROOT}/.secrets/openai_api_key}"
# Which tau2-bench to evaluate against. InfoPO's paper (2026-02-28) and repo push
# (2026-03-16) both predate tau2 v1.0.0, and v1.0.1 -- the checkout in tau2-bench/ --
# rewrites the user simulator, adds LLM-judge reviewers to the evaluator and changes
# 3.5M lines of task data. Those are different benchmarks; reproducing the paper's
# table needs the code that existed when it was written.
#   infopo : worktree at c5b2d22 (2026-02-09), the last main commit before their push
#   local  : tau2-bench/ as checked out (v1.0.1)
TAU2_VERSION="${TAU2_VERSION:-infopo}"
if [[ "${TAU2_VERSION}" == "infopo" ]]; then
  TAU2_BIN=${TAU2_ENV_PAPER}/bin/tau2
  TAU2_ROOT=${TAU2_BENCH_DIR}
else
  TAU2_BIN=${TAU2_ENV_DEFAULT}/bin/tau2
  TAU2_ROOT="${ROOT}/tau2-bench"
fi

OUT_DIR="${ROOT}/runlogs/tau2_align_${TAG}"
mkdir -p "${OUT_DIR}" "${ROOT}/slurm_logs"

export PYTHONNOUSERSITE=1
export HF_HOME="${ROOT}/.hf_cache"
export HF_HUB_OFFLINE=1
export LITELLM_LOG=ERROR
export TAU2_DATA_DIR="${TAU2_ROOT}/data"
if [[ "${USERSIM_BACKEND}" == "hosted" ]]; then
  [[ -r "${KEY_FILE}" ]] || { echo "FATAL: no API key at ${KEY_FILE}"; exit 1; }
  export OPENAI_API_KEY="$(tr -d '\r\n' < "${KEY_FILE}")"
else
  export OPENAI_API_KEY=EMPTY
fi

# The user simulator is hosted, so it needs this node's egress; a proxy inherited from
# the submitting shell can be meaningless here and turns every user turn into a
# connection error. Keep localhost off the proxy either way for the policy server.
if [[ -n "${TAU2_USER_HTTP_PROXY:-}" ]]; then
  export http_proxy="${TAU2_USER_HTTP_PROXY}" https_proxy="${TAU2_USER_HTTP_PROXY}"
  export HTTP_PROXY="${TAU2_USER_HTTP_PROXY}" HTTPS_PROXY="${TAU2_USER_HTTP_PROXY}"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
fi
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"

echo "=== tau2 alignment eval [${TAG}] ==="
echo "model    : ${MODEL_PATH}"
echo "domains  : ${DOMAINS}   split=${TASK_SPLIT}  trials=${NUM_TRIALS}  max_steps=${MAX_STEPS}"
echo "user     : ${USER_MODEL} @ temp ${USER_TEMP}"
echo "egress   : ${TAU2_USER_HTTP_PROXY:-direct (proxy vars cleared)}"
echo "tau2     : ${TAU2_VERSION} at ${TAU2_ROOT}"

cleanup() {
  [[ -n "${POLICY_PID:-}" ]] && kill "${POLICY_PID}" 2>/dev/null || true
  [[ -n "${USERSIM_PID:-}" ]] && kill "${USERSIM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- policy server ------------------------------------------------------------------
# tau2's own agent drives the model through native tool calling, not through this repo's
# ReAct text protocol, so the policy server needs a tool-call parser too. Without it
# every agent turn returns BadRequest and each task ends as `infrastructure_error` with
# zero messages -- which scores as 0% and looks like a terrible policy rather than a
# server flag. This is itself a protocol difference from our harness worth remembering.
# Telecom transcripts can outgrow Qwen2.5's native 32768 on the agent side too, and the
# whole run dies with ContextWindowExceededError when they do. Paper comparability needs
# MAX_CONTEXT_TOKENS=32768, so only enable YaRN when a caller deliberately asks for more.
POLICY_ROPE=""
if [[ "${MAX_CONTEXT_TOKENS}" -gt 32768 ]]; then
  POLICY_ROPE="--rope-scaling {\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}"
fi
echo "--- starting policy vLLM (max_len=${MAX_CONTEXT_TOKENS}) ---"
CUDA_VISIBLE_DEVICES=0,1 ${TRAIN_ENV}/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 --port "${POLICY_PORT}" \
  --tensor-parallel-size "${POLICY_TP}" --gpu-memory-utilization 0.85 \
  --max-model-len "${MAX_CONTEXT_TOKENS}" ${POLICY_ROPE} --disable-log-requests \
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-hermes}" \
  > "${OUT_DIR}/policy.log" 2>&1 &
POLICY_PID=$!

for _ in $(seq 1 180); do
  curl --noproxy '*' -sf "http://127.0.0.1:${POLICY_PORT}/v1/models" >/dev/null && break
  sleep 5
done
curl --noproxy '*' -sf "http://127.0.0.1:${POLICY_PORT}/v1/models" >/dev/null \
  || { echo "FATAL: policy server did not come up"; tail -30 "${OUT_DIR}/policy.log"; exit 1; }
echo "policy healthy on ${POLICY_PORT}"

# ---- local user simulator (diagnostic mode only) -------------------------------------
if [[ "${USERSIM_BACKEND}" != "hosted" ]]; then
  # Qwen2.5's native window is 32768; anything beyond needs YaRN or vLLM refuses to
  # start. Commas are safe here because this string is built inside the script, not
  # passed through `sbatch --export`.
  USERSIM_ROPE=""
  if [[ "${USERSIM_MAX_LEN}" -gt 32768 ]]; then
    USERSIM_ROPE="--rope-scaling {\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}"
  fi
  echo "--- starting local user simulator (max_len=${USERSIM_MAX_LEN}) ---"
  # Needs a tool-call parser for the same reason the policy does: telecom's customer
  # holds diagnostic tools and tau2 calls it with tool_choice="auto".
  CUDA_VISIBLE_DEVICES=2,3 ${TRAIN_ENV}/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model "${USERSIM_LOCAL_MODEL}" --served-model-name user-sim \
    --host 127.0.0.1 --port "${USERSIM_PORT}" \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
    --max-model-len "${USERSIM_MAX_LEN}" ${USERSIM_ROPE} --disable-log-requests \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-hermes}" \
    > "${OUT_DIR}/usersim.log" 2>&1 &
  USERSIM_PID=$!
  for _ in $(seq 1 180); do
    curl --noproxy '*' -sf "http://127.0.0.1:${USERSIM_PORT}/v1/models" >/dev/null && break
    sleep 5
  done
  curl --noproxy '*' -sf "http://127.0.0.1:${USERSIM_PORT}/v1/models" >/dev/null \
    || { echo "FATAL: local user simulator did not come up"; tail -30 "${OUT_DIR}/usersim.log"; exit 1; }
  echo "local user simulator healthy on ${USERSIM_PORT}"
fi

# ---- one `tau2 run` per domain ------------------------------------------------------
# InfoPO also passes max_context_tokens here, but that key only exists in the tau2
# version they vendor; this one forwards it verbatim to litellm and vLLM rejects the
# request with "extra_forbidden: max_context_tokens", so every turn fails and the run
# reports zero simulations. It only bounds context truncation, which --max-model-len on
# the policy server already pins to the same value.
# Greedy decoding collapses on this model in long telecom conversations -- the policy
# starts emitting bare `<tool_call>` tags with no JSON body and loops until the step cap.
# FREQ_PENALTY is a scalar rather than a JSON fragment on purpose: a fragment has to
# carry a comma, and a comma inside `sbatch --export` arrives with its escaping
# backslash still attached, which makes the JSON unparseable. 0 keeps the paper's
# protocol; set it only to diagnose that collapse.
# REP_PENALTY is vLLM's multiplicative repetition_penalty (1.0 = off), which bites much
# harder on loops than the additive frequency_penalty -- that one measured useless here
# (69% truncation vs a 62% baseline over 80 episodes).
FREQ_PENALTY="${FREQ_PENALTY:-0}"
REP_PENALTY="${REP_PENALTY:-1.0}"
AGENT_ARGS=$(printf '{"temperature": %s, "frequency_penalty": %s, "repetition_penalty": %s, "base_url": "http://127.0.0.1:%s/v1", "api_key": "EMPTY"}' \
             "${AGENT_TEMP}" "${FREQ_PENALTY}" "${REP_PENALTY}" "${POLICY_PORT}")
if [[ "${USERSIM_BACKEND}" == "hosted" ]]; then
  USER_LLM_STR="openai/${USER_MODEL}"
  USER_ARGS=$(printf '{"temperature": %s}' "${USER_TEMP}")
else
  USER_LLM_STR="openai/user-sim"
  USER_ARGS=$(printf '{"temperature": %s, "base_url": "http://127.0.0.1:%s/v1", "api_key": "EMPTY"}' \
              "${USER_TEMP}" "${USERSIM_PORT}")
fi

# Record exactly what this run used. Two evaluations are only comparable if these files
# match, and the settings cannot be recovered afterwards from tau2's boxed console
# output -- the panel truncates the JSON mid-field.
cat > "${OUT_DIR}/eval_config.json" <<JSON
{
  "tau2_version": "${TAU2_VERSION}", "tau2_root": "${TAU2_ROOT}",
  "model": "${MODEL_PATH}", "tag": "${TAG}",
  "domains": "${DOMAINS}", "task_split": "${TASK_SPLIT}",
  "num_trials": ${NUM_TRIALS}, "max_steps": ${MAX_STEPS},
  "agent_llm_args": ${AGENT_ARGS},
  "user_llm": "${USER_LLM_STR}", "user_llm_args": ${USER_ARGS},
  "usersim_backend": "${USERSIM_BACKEND}",
  "max_context_tokens": ${MAX_CONTEXT_TOKENS}
}
JSON
echo "--- eval config ---"; cat "${OUT_DIR}/eval_config.json"

SIM_DIR="${TAU2_DATA_DIR}/simulations"
for DOMAIN in ${DOMAINS}; do
  echo ""
  echo "################ ${DOMAIN} ################"
  # tau2 resumes when the save path already exists, and it asks for confirmation on
  # stdin -- under sbatch that is an immediate EOFError, so a rerun of the same tag dies
  # before it starts. Clear the previous attempt unless RESUME=1 is set deliberately.
  # Clear every layout this can land in: v1.0.1 writes <save-to>/results.json, the
  # 2026-02 tree appends another .json to the name. Missing one leaves tau2 thinking it
  # can resume, and it asks on stdin -- an instant EOFError under sbatch.
  if [[ "${RESUME:-0}" != "1" ]]; then
    for PREV in "${SIM_DIR}/tau2_align_${TAG}_${DOMAIN}.json" \
                "${SIM_DIR}/tau2_align_${TAG}_${DOMAIN}.json.json"; do
      if [[ -e "${PREV}" ]]; then
        echo "clearing previous results at ${PREV}"
        rm -rf "${PREV}"
      fi
    done
  fi
  "${TAU2_BIN}" run \
    --domain "${DOMAIN}" \
    --agent-llm "openai/${SERVED_NAME}" \
    --user-llm "${USER_LLM_STR}" \
    --num-trials "${NUM_TRIALS}" \
    --max-steps "${MAX_STEPS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --task-split-name "${TASK_SPLIT}" \
    --agent-llm-args "${AGENT_ARGS}" \
    --user-llm-args "${USER_ARGS}" \
    --save-to "tau2_align_${TAG}_${DOMAIN}.json" \
    2>&1 | tee "${OUT_DIR}/${DOMAIN}.log"
done

echo ""
echo "=== done; scoring ==="
${TAU2_ENV_DEFAULT}/bin/python "${ROOT}/scripts/tau2_align_score.py" \
  --tag "${TAG}" --domains ${DOMAINS} | tee "${OUT_DIR}/summary.txt"
