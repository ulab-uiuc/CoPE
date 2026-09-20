#!/usr/bin/env bash
#
# τ²-bench, GRPO + the action-forecast auxiliary loss, in one command. Identical to
# tau2_grpo_train.sh (same preset, protocol, customer, hyperparameters) plus the
# forecast objective: after each policy update the actor is also trained, on the
# winning trajectories of the batch, to list the next K actions it will take
# (src/verl/agent_trainer/ppo/action_forecast.py). Under τ²'s native tool-calling
# protocol an action is a tool call (forecast as one JSON line) or a customer message
# (forecast as "say: ..."); failed tool calls are skipped from the target.
#
#   bash examples/train/AgentGym-RL/tau2_grpo_forecast_train.sh
#   ACTION_FORECAST_COEF=0.02 ACTION_FORECAST_K=5 bash examples/train/AgentGym-RL/tau2_grpo_forecast_train.sh
#
# The forecast pass is a second backward per step over samples of at most
# ACTION_FORECAST_MAX_LENGTH tokens; on 80-97GB cards it fits next to the InfoPO
# sequence budget with the preset's micro-batch of 1. Everything tau2_grpo_train.sh
# says about setup (TRAIN_ENV, MODEL_PATH, GPUs, the customer) applies unchanged.

set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

export ACTION_FORECAST_ENABLE=True
export ACTION_FORECAST_COEF="${ACTION_FORECAST_COEF:-0.01}"      # weight of the forecast CE next to the PG loss
export ACTION_FORECAST_K="${ACTION_FORECAST_K:-3}"               # horizon: forecast the next K actions
export ACTION_FORECAST_GATE="${ACTION_FORECAST_GATE:-wins}"      # learn only from winning trajectories
export ACTION_FORECAST_SKIP_INVALID="${ACTION_FORECAST_SKIP_INVALID:-True}"   # drop failed tool calls from the target
export ACTION_FORECAST_GROUP_NORM="${ACTION_FORECAST_GROUP_NORM:-True}"       # equal weight per task group
export ACTION_FORECAST_MAX_LENGTH="${ACTION_FORECAST_MAX_LENGTH:-4096}"
export EXP_NAME="${EXP_NAME:-tau2_grpo_forecast_$(date +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/examples/train/AgentGym-RL/tau2_grpo_train.sh"
