# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared plumbing for the auxiliary SFT objectives that run alongside PG.

Two kinds of caller live here:

* the *separate-pass* objectives (action-forecast, the sft-ablation control), which
  re-assemble their own SFT samples with ``tokenizer.apply_chat_template``, pad them
  with ``collate_sft_samples`` and take CE with ``compute_sft_loss_from_logits``;
* the *same-forward* objectives (``traj_lm``), which reuse the PG forward's token
  log-probs and only need a mask over the response region.

Nothing here knows what is being predicted -- the target is the caller's business.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def compute_observation_mask(
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    observation_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve the observation-token mask for the response region.

    If an explicit ``observation_mask`` is already carried by the batch we use
    it directly; otherwise we fall back to ``attention_mask_response & ~response_mask``.
    """
    if observation_mask is not None:
        return observation_mask.float()

    response_length = response_mask.shape[1]
    attention_mask_response = attention_mask[:, -response_length:].float()
    return attention_mask_response * (1.0 - response_mask.float())


def collate_sft_samples(
    samples: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    max_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Right-pad samples into a fixed-size batch suitable for FSDP forward."""
    if not samples:
        return None

    actual_max = max(s['input_ids'].size(0) for s in samples)
    target_len = min(max_length, actual_max)

    bsz = len(samples)
    input_ids = torch.full((bsz, target_len), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, target_len), dtype=torch.long)
    loss_mask = torch.zeros((bsz, target_len), dtype=torch.long)
    for i, s in enumerate(samples):
        L = min(s['input_ids'].size(0), target_len)
        input_ids[i, :L] = s['input_ids'][:L].to(torch.long)
        attention_mask[i, :L] = s['attention_mask'][:L].to(torch.long)
        loss_mask[i, :L] = s['loss_mask'][:L].to(torch.long)

    position_ids = (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)

    out = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'loss_mask': loss_mask,
    }
    # optional per-sample loss weight (e.g. action-forecast group-weight normalization)
    if any('loss_weight' in s for s in samples):
        out['loss_weight'] = torch.tensor(
            [float(s.get('loss_weight', 1.0)) for s in samples], dtype=torch.float32)
    return out


def compute_sft_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weight: torch.Tensor = None,
) -> torch.Tensor:
    """Next-token CE loss on the positions marked by ``loss_mask``.

    ``logits``: (B, T, V); ``labels``: (B, T); ``loss_mask``: (B, T).

    ``sample_weight`` (optional, shape [B]) scales each sample's token losses. It
    multiplies the numerator only; the denominator stays the unweighted token count.
    So ``None`` (or all ones) reproduces the unweighted loss exactly, and with a single
    sample in the micro-batch the result is exactly ``w * L``.

    Weighting the denominator too (a weighted mean) looks equivalent but is not:
    with a single sample per micro-batch -- ppo_micro_batch_size_per_gpu=1, the
    setting every action-forecast run uses -- w appears in both numerator and
    denominator and cancels, so the weights have no effect at all (measured:
    gradient cosine vs. unweighted = 1.000000). Scaling an already-averaged loss by
    ``sample_weight.mean()`` has the opposite failure: correct at one sample per
    micro-batch, but it averages the weights away as soon as there are more.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().to(torch.bool)

    vocab = shift_logits.size(-1)
    # ignore_index on the non-supervised positions so CrossEntropyLoss does NOT
    # evaluate them at all. (The old "compute CE everywhere then multiply by mask"
    # form risked inf*0 = NaN if a padded/ignored position's logits were non-finite.)
    ignored_labels = shift_labels.masked_fill(~shift_mask, -100)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    tok_loss = loss_fn(shift_logits.view(-1, vocab), ignored_labels.view(-1))
    tok_loss = tok_loss.view(shift_labels.shape)   # ignored positions already 0
    denom = shift_mask.sum().clamp(min=1.0)
    if sample_weight is None:
        return tok_loss.sum() / denom
    w = sample_weight.to(tok_loss.dtype).view(-1, 1)   # [B, 1] broadcasts over T
    return (tok_loss * w).sum() / denom                # denominator deliberately unweighted


def compute_traj_lm_loss(log_prob, response_mask, obs_mask, row_mask=None):
    """Full-sequence next-token CE over the ENTIRE trajectory response region —
    BOTH the agent's own tokens (``response_mask``) AND the env observation tokens
    (``obs_mask``), no distinction. = plain LM SFT on the whole rollout. Uses the
    already-computed ``log_prob`` (response-region token log-probs), so it is a mask
    swap over the same forward as the PG term.

    ``row_mask`` (optional, shape [B]): per-trajectory 0/1 keep-mask for
    gate='wins' — zero out whole trajectories we don't want to clone (e.g. losing
    ones). None = keep all (gate='all'). Returns the scalar CE (0 if nothing kept)."""
    full = (response_mask.bool() | obs_mask.bool()).to(log_prob.dtype)
    if row_mask is not None:
        full = full * row_mask.to(full.dtype).unsqueeze(-1)
    denom = full.sum().clamp(min=1.0)
    return -(log_prob * full).sum() / denom
