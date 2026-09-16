# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Turn segmentation over a rollout's response mask.

The response mask is 1 on the agent's own tokens and 0 on the environment's, so a
contiguous run of 1s is exactly one agent action -- one turn. Callers that need to
attribute something per-turn (a shaping term, a hindsight prompt) use these to find
the spans.

This is keyed on the response mask alone. ``verl.trainer.ppo.turn_structure`` solves
the same problem from the *observation* mask and is a separate thing.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def turn_spans(mask_row: torch.Tensor) -> List[Tuple[int, int]]:
    """Return [start, end) token spans of contiguous action-token runs in a
    1-D response mask. Each run is one agent action (one turn)."""
    idx = mask_row.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return []
    breaks = (idx[1:] - idx[:-1] > 1).nonzero(as_tuple=True)[0]
    bounds = [0] + (breaks + 1).tolist() + [idx.numel()]
    return [(int(idx[a].item()), int(idx[b - 1].item()) + 1)
            for a, b in zip(bounds[:-1], bounds[1:])]


def compute_turn_boundaries(response_mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """``turn_spans`` over a batch: one list of action spans per trajectory."""
    return [turn_spans(row) for row in response_mask.bool()]
