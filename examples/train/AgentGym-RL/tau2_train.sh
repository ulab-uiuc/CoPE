#!/usr/bin/env bash
#
# τ²-bench GRPO in one command, the way sciworld_train.sh is for SciWorld.
#
#   bash examples/train/AgentGym-RL/tau2_train.sh
#
# Unlike SciWorld, τ²-bench is not a single env server you start beforehand: the
# environment runs as a cluster of Python 3.12 processes and the customer is an LLM.
# This script therefore brings all of it up and tears it down again:
# customer -> env cluster (health-gated) -> GRPO training, logging to runlogs/<exp>/.
#
# Defaults reproduce InfoPO's tau2 training protocol (all three domains, tau2's native
# tool-calling interface, gpt-4o-mini customer) with this repo's plain GRPO. Point the
# three environment variables below at your setup; everything else has a default, and
# any training knob can be overridden the same way (see scripts/run_tau2_pipeline.sh).
#
#   TRAIN_ENV   conda env with the trainer (python 3.10, torch, vllm >= 0.6.6 or 0.6.3)
#   MODEL_PATH  local snapshot of Qwen/Qwen2.5-7B-Instruct (HF_HUB_OFFLINE=0 to download)
#   CUDA_VISIBLE_DEVICES  training GPUs; the customer needs one more if USERSIM_MODE=local
#
# One-time setup (README 'Setup'): clone tau2-bench at c5b2d22 into ./tau2-bench and
# build ./envs/tau2 (python 3.12) with tau2-bench + AgentGym/agentenv-tau2 installed.
# A hosted customer needs an OpenAI key in .secrets/openai_api_key; without one the
# script falls back to a local Qwen customer and asks for USERSIM_GPU.

set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

export PRESET="${PRESET:-infopo}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAIN_GPUS="${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES}}"

exec bash "${ROOT}/scripts/run_tau2_pipeline.sh"
