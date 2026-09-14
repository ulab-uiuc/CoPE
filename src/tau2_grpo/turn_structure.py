"""Turn-level structure for InfoPO-style advantage estimation.

The rollout already emits `observation_mask` (1 on tokens that came from the
environment, 0 on tokens the policy generated). Everything InfoPO needs about turn
structure is recoverable from it, so nothing in the rollout loop has to change:

    obs_mask   0 0 0 1 1 0 0 0 1 1 1 0 0
                     ^obs   ^action ^obs  ^action
    boundaries 1 0 0 0 0 1 0 0 0 0 0 1 0

A turn starts at the first action token after a run of observation tokens (and at
position 0). That is the granularity InfoPO assigns its information-gain reward at:
"did seeing this observation change what the policy does next".
"""

from typing import List

import torch


def turn_boundaries_from_observation_mask(
    observation_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Mark the first action token of every turn.

    Args:
        observation_mask: (bs, seq) 1 where the token is environment output.
        response_mask: (bs, seq) 1 where the token is part of the response at all;
            padding is 0 and must not start a turn.

    Returns:
        (bs, seq) with 1 at each turn start, 0 elsewhere.
    """
    obs = (observation_mask > 0).to(torch.long)
    resp = (response_mask > 0).to(torch.long)

    action = (1 - obs) * resp                      # policy-generated, non-padding
    prev_action = torch.zeros_like(action)
    prev_action[:, 1:] = action[:, :-1]
    # A boundary is an action token whose predecessor was not an action token: either
    # the sequence just started, or an observation block just ended.
    return (action * (1 - prev_action)).to(observation_mask.dtype)


def turn_spans(
    observation_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> List[List[tuple]]:
    """Per-sample [(action_start, action_end, obs_start, obs_end), ...].

    `obs_*` is the observation block immediately preceding the action, which is the
    span InfoPO masks out to build its counterfactual. It is empty (obs_start ==
    obs_end) for the opening turn, which has no prior feedback and therefore earns no
    information-gain reward.
    """
    obs = (observation_mask > 0)
    resp = (response_mask > 0)
    act = (~obs) & resp

    out = []
    for b in range(act.shape[0]):
        a, o = act[b], obs[b]
        spans, i, n = [], 0, act.shape[1]
        while i < n:
            if not a[i]:
                i += 1
                continue
            start = i
            while i < n and a[i]:
                i += 1
            # Walk back over the observation block that fed this action.
            oe = start
            os_ = start
            while os_ > 0 and o[os_ - 1]:
                os_ -= 1
            spans.append((start, i, os_, oe))
        out.append(spans)
    return out
