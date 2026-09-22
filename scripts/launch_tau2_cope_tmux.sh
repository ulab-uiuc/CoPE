#!/usr/bin/env bash
#
# τ²-bench CoPE = GRPO + the action-forecast auxiliary loss, in a detached tmux session.
# This is scripts/launch_tau2_grpo_tmux.sh with exactly one difference: the
# action-forecast switch is on, weight 0.1. Everything else (preset, protocol, customer,
# hyperparameters, every forwarded knob) is the same, so a grpo run and a cope run
# launched the same way differ only in the auxiliary objective.
#
#   bash scripts/launch_tau2_cope_tmux.sh
#   ACTION_FORECAST_COEF=0.05 bash scripts/launch_tau2_cope_tmux.sh
#   DRY_RUN=1 bash scripts/launch_tau2_cope_tmux.sh

set -euo pipefail
export VARIANT=cope
export ACTION_FORECAST_ENABLE=True
export ACTION_FORECAST_COEF="${ACTION_FORECAST_COEF:-0.1}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_tau2_grpo_tmux.sh" "$@"
