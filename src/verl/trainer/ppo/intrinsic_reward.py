"""Hindsight policy shift: how much did seeing the observation change what the agent did.

For each turn whose action followed environment feedback, run the actor twice over the
same sequence -- once as-is, once with that feedback's tokens masked out of attention --
and measure the KL between the two next-action distributions. A turn where the feedback
changed nothing scores ~0; a turn where it redirected the agent scores high. This is
InfoPO's intrinsic reward, and unlike a reshaped task reward it leaves the definition of
success untouched.

Cost is the reason this is batched and capped: each counterfactual is an extra forward
pass, and a telecom episode has ~127 turns. `kl_batch_size` bounds how many turns are
evaluated together and `max_turns_per_sample` caps how many are scored at all (the paper
runs 2 on tau2).
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from verl.trainer.ppo.turn_structure import turn_spans


@torch.no_grad()
def compute_intrinsic_rewards(
    actor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    observation_mask: torch.Tensor,
    response_mask: torch.Tensor,
    kl_batch_size: int = 2,
    max_turns_per_sample: Optional[int] = None,
    probe_tokens: int = 16,
) -> torch.Tensor:
    """Per-token intrinsic reward, spread over each turn's action tokens.

    Args:
        actor: callable returning logits (bs, seq, vocab) for (input_ids, attention_mask,
            position_ids); the FSDP-wrapped actor module works directly.
        observation_mask: 1 on environment-produced tokens.
        response_mask: 1 on response tokens (0 on padding).
        kl_batch_size: counterfactual forwards to run at once.
        max_turns_per_sample: score at most this many turns per trajectory, latest first;
            None scores all of them.
        probe_tokens: how many of the turn's leading action tokens the KL is measured
            over. The shift shows up immediately after the feedback, and scoring the
            whole turn would dilute it with tokens the observation cannot explain.

    Returns:
        (bs, seq) intrinsic reward, constant across each scored turn's action tokens.
    """
    device = input_ids.device
    out = torch.zeros_like(input_ids, dtype=torch.float32)
    spans_per_sample: List[List[tuple]] = turn_spans(observation_mask, response_mask)

    jobs = []  # (sample_idx, action_start, action_end, obs_start, obs_end)
    for b, spans in enumerate(spans_per_sample):
        scored = [s for s in spans if s[3] > s[2]]  # needs a preceding observation
        if max_turns_per_sample is not None:
            scored = scored[-max_turns_per_sample:]
        jobs.extend((b, *s) for s in scored)
    if not jobs:
        return out

    for i in range(0, len(jobs), kl_batch_size):
        chunk = jobs[i:i + kl_batch_size]
        rows = torch.tensor([j[0] for j in chunk], device=device)

        ids = input_ids[rows]
        am_real = attention_mask[rows].clone()
        am_cf = am_real.clone()
        for k, (_, _, _, os_, oe) in enumerate(chunk):
            # Mask the observation out of attention rather than editing tokens: the
            # positions stay put, so position_ids and the KV layout are identical
            # between the two passes and the only difference is visibility.
            am_cf[k, os_:oe] = 0

        pos = position_ids[rows]
        logits_real = actor(input_ids=ids, attention_mask=am_real, position_ids=pos).logits
        logits_cf = actor(input_ids=ids, attention_mask=am_cf, position_ids=pos).logits

        for k, (b, a_start, a_end, _, _) in enumerate(chunk):
            end = min(a_end, a_start + probe_tokens)
            if end <= a_start:
                continue
            # logits at t predict token t+1, so the distribution over the first action
            # token lives at index a_start-1.
            lo, hi = max(a_start - 1, 0), max(end - 1, 1)
            p = F.log_softmax(logits_real[k, lo:hi].float(), dim=-1)
            q = F.log_softmax(logits_cf[k, lo:hi].float(), dim=-1)
            kl = (p.exp() * (p - q)).sum(dim=-1).mean()
            out[b, a_start:a_end] = kl

    return out
