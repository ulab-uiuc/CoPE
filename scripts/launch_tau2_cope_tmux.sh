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
# 0.01, the value sciworld's runs use, measured to be the SAFE side of that comparison
# rather than assumed. Forecast CE gradient of a mini-batch of 16 at coef 1, each env with
# its own policy, data and settings: tau2 25.1, sciworld 279. At coef 0.01 that is 0.251
# here against 2.79 there, and each env's own policy gradient is 0.43 / 0.49 -- so the same
# coefficient makes the forecast 0.59x the policy gradient in tau2 and 5.7x (clipped by
# grad_clip=1.0 to 2.05x) in sciworld.
#
# tau2's much longer targets (274 trained tokens per sample against sciworld's 13.5) do NOT
# make its gradient bigger: the SFT loss is a mean over trained tokens, so length sits in
# the denominator. Measured per sample, target length correlates NEGATIVELY with gradient
# norm in both envs (tau2 r=-0.43 over 20..635 tokens, sciworld r=-0.51 over 3..30). What
# drives the difference is the CE level -- 0.67 here against 4.30 there -- because tau2's
# target is a verbatim copy of the turn the policy just wrote, while sciworld's is a bare
# action list in a format the policy never emits.
#
# The clip binds at coef >= 0.0398 here, so 0.1 (the old default), 0.05 and 0.04 were all
# the same run. Raise toward 0.035 to match sciworld's post-clip dose if the forecast turns
# out to be too weak to do anything.
export ACTION_FORECAST_COEF="${ACTION_FORECAST_COEF:-0.01}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_tau2_grpo_tmux.sh" "$@"
