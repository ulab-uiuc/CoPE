# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""World-model auxiliary loss utilities.

This module hosts the plumbing used to train the actor to predict the
environment's next observation (world-model SFT).  Unlike the previous
in-place variant that reused the rollout sequence and masked the env tokens
on it, here we *re-assemble* each env turn as a standalone SFT sample via
``tokenizer.apply_chat_template``:

    [system?, u_0 (task), a_1, u_1, ..., a_{i-1}, <env_predict_prompt>]
    + assistant: u_i                # <- loss only on these tokens

Training on the re-assembled sample explicitly cues the model that it is
predicting the environment's response (not continuing its own action), which
is the objective we actually want from a "world model" auxiliary loss.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F


DEFAULT_WORLD_MODEL_PROMPT = (
    "You are now acting as a world model. Based on the conversation above "
    "and the agent's last action, predict the environment's next observation."
)


# ---------------------------------------------------------------------------
# Observation-mask helper (kept for backward compatibility with any remaining
# callers that still need to compute an env-token mask from the rollout batch).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Chat-template assembly of world-model SFT samples.
# ---------------------------------------------------------------------------
def _to_chat_list(messages) -> List[Dict[str, str]]:
    """Normalize a ``RolloutHandler.messages``-like list into plain dicts."""
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            out.append({'role': m['role'], 'content': m['content']})
        elif hasattr(m, 'to_dict'):
            out.append(m.to_dict())
        else:  # pragma: no cover - defensive
            out.append({'role': getattr(m, 'role'), 'content': getattr(m, 'content')})
    return out


def build_world_model_sft_samples(
    messages,
    tokenizer: PreTrainedTokenizer,
    env_predict_prompt: str = DEFAULT_WORLD_MODEL_PROMPT,
    max_length: int = 4096,
    min_env_tokens: int = 1,
) -> List[Dict[str, torch.Tensor]]:
    """Build per-env-turn SFT samples from one multi-turn trajectory.

    For every env observation (user turn that is preceded by at least one
    assistant turn) we emit a dict with keys
    ``input_ids``/``attention_mask``/``loss_mask`` where the loss mask covers
    only the env-observation tokens.  The env observation is rendered as an
    ``assistant`` turn in the chat template so the tokens are in a place the
    model is trained to generate.
    """
    convo = _to_chat_list(messages)

    samples: List[Dict[str, torch.Tensor]] = []
    for idx, msg in enumerate(convo):
        if msg['role'] != 'user':
            continue
        if not any(m['role'] == 'assistant' for m in convo[:idx]):
            continue

        prefix = list(convo[:idx])
        prefix.append({'role': 'user', 'content': env_predict_prompt})
        target = [{'role': 'assistant', 'content': msg['content']}]

        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                prefix + target, tokenize=False, add_generation_prompt=False)
        except Exception:  # pragma: no cover - tokenizer template missing
            continue

        if not full_text.startswith(prefix_text):
            # Some chat templates don't guarantee monotonic prefix growth; fall
            # back to tokenizing both and comparing id sequences.
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                                   return_tensors='pt')['input_ids'][0]
            full_ids = tokenizer(full_text, add_special_tokens=False,
                                 return_tensors='pt')['input_ids'][0]
            # Find the longest id-prefix match.
            common = 0
            for i in range(min(len(prefix_ids), len(full_ids))):
                if prefix_ids[i].item() != full_ids[i].item():
                    break
                common = i + 1
            prefix_len = common
        else:
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                                   return_tensors='pt')['input_ids'][0]
            full_ids = tokenizer(full_text, add_special_tokens=False,
                                 return_tensors='pt')['input_ids'][0]
            prefix_len = prefix_ids.size(0)

        target_len = full_ids.size(0) - prefix_len
        if target_len < min_env_tokens:
            continue

        input_ids = full_ids
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[prefix_len:] = 1

        # Left-truncate (preserving env-observation target at the end) if too long.
        if input_ids.size(0) > max_length:
            drop = input_ids.size(0) - max_length
            input_ids = input_ids[drop:]
            attention_mask = attention_mask[drop:]
            loss_mask = loss_mask[drop:]
            if loss_mask.sum().item() < min_env_tokens:
                continue

        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
        })

    return samples


def collate_world_model_samples(
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
    # optional per-sample loss weight (e.g. plan-forecast group-weight normalization)
    if any('loss_weight' in s for s in samples):
        out['loss_weight'] = torch.tensor(
            [float(s.get('loss_weight', 1.0)) for s in samples], dtype=torch.float32)
    return out


def build_world_model_sft_batch(
    messages_list,
    tokenizer: PreTrainedTokenizer,
    env_predict_prompt: str = DEFAULT_WORLD_MODEL_PROMPT,
    max_length: int = 4096,
    max_samples_per_trajectory: Optional[int] = None,
    min_env_tokens: int = 1,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build a padded SFT batch from a list of rollout trajectories."""
    all_samples: List[Dict[str, torch.Tensor]] = []
    for messages in messages_list:
        if messages is None:
            continue
        traj_samples = build_world_model_sft_samples(
            messages=messages,
            tokenizer=tokenizer,
            env_predict_prompt=env_predict_prompt,
            max_length=max_length,
            min_env_tokens=min_env_tokens,
        )
        if max_samples_per_trajectory is not None and len(traj_samples) > max_samples_per_trajectory:
            # Deterministically keep the last env turns (closer to terminal reward).
            traj_samples = traj_samples[-max_samples_per_trajectory:]
        all_samples.extend(traj_samples)

    return collate_world_model_samples(
        samples=all_samples,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        max_length=max_length,
    )


def build_world_model_placebo_batch(
    messages_list,
    tokenizer: PreTrainedTokenizer,
    env_predict_prompt: str = DEFAULT_WORLD_MODEL_PROMPT,
    max_length: int = 4096,
    max_samples_per_trajectory: Optional[int] = None,
    min_env_tokens: int = 1,
    seed: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """C3 PLACEBO for the WM-SFT value experiment: identical transitions as
    ``build_world_model_sft_batch`` (same prefixes, same per-traj cap, same count),
    but each prefix's target observation is REPLACED by another transition's
    observation (a derangement). Same token budget / loss shape / gradient
    magnitude, but NO real env-dynamics signal — so C2 (real WM) minus C3 (this)
    isolates the world-model contribution from a generic 'dense obs-token gradient'
    confound. Deterministic given ``seed`` (pass global_step). Reuses the exact
    plan-forecast encoder so encoding matches the real path."""
    from verl.agent_trainer.ppo.plan_forecast import encode_sft_sample
    import random as _random

    # Collect (prefix_msgs, obs_content) transitions, mirroring the real builder's
    # per-trajectory selection (keep-last cap) so the sample COUNT matches C2.
    pairs = []
    for messages in messages_list:
        if messages is None:
            continue
        convo = _to_chat_list(messages)
        traj = []
        for idx, msg in enumerate(convo):
            if msg['role'] != 'user':
                continue
            if not any(m['role'] == 'assistant' for m in convo[:idx]):
                continue
            prefix = list(convo[:idx]) + [{'role': 'user', 'content': env_predict_prompt}]
            traj.append((prefix, msg['content']))
        if max_samples_per_trajectory is not None and len(traj) > max_samples_per_trajectory:
            traj = traj[-max_samples_per_trajectory:]
        pairs.extend(traj)

    if len(pairs) < 2:
        return None

    obs_list = [p[1] for p in pairs]
    perm = list(range(len(obs_list)))
    _random.Random(seed).shuffle(perm)
    for i in range(len(perm)):          # break any fixed points -> true wrong obs
        if perm[i] == i:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]

    samples = []
    for i, (prefix, _real) in enumerate(pairs):
        wrong_obs = obs_list[perm[i]]
        s = encode_sft_sample(tokenizer, prefix,
                              [{'role': 'assistant', 'content': wrong_obs}],
                              max_length=max_length, min_target_tokens=min_env_tokens)
        if s is not None:
            samples.append(s)

    return collate_world_model_samples(
        samples=samples,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        max_length=max_length,
    )


# ---------------------------------------------------------------------------
# Cross-entropy SFT loss on the assembled batch.
# ---------------------------------------------------------------------------
def compute_world_model_sft_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Next-token CE loss on the positions marked by ``loss_mask``.

    ``logits``: (B, T, V); ``labels``: (B, T); ``loss_mask``: (B, T).
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
    return tok_loss.sum() / denom


def compute_traj_lm_loss(log_prob, response_mask, obs_mask, row_mask=None):
    """Full-sequence next-token CE over the ENTIRE trajectory response region —
    BOTH the agent's own tokens (``response_mask``) AND the env observation tokens
    (``obs_mask``), no distinction. = plain LM SFT on the whole rollout. Uses the
    already-computed ``log_prob`` (response-region token log-probs), so it is a mask
    swap over the same forward as the PG/WM terms. MUTUALLY EXCLUSIVE with WM-SFT
    (which is obs-only) — the trainer asserts this.

    ``row_mask`` (optional, shape [B]): per-trajectory 0/1 keep-mask for
    gate='wins' — zero out whole trajectories we don't want to clone (e.g. losing
    ones). None = keep all (gate='all'). Returns the scalar CE (0 if nothing kept)."""
    full = (response_mask.bool() | obs_mask.bool()).to(log_prob.dtype)
    if row_mask is not None:
        full = full * row_mask.to(full.dtype).unsqueeze(-1)
    denom = full.sum().clamp(min=1.0)
    return -(log_prob * full).sum() / denom


# Backward-compat alias used by the earlier inline implementation: compute
# the world-model NLL on an already-known token log-probability tensor and
# observation mask (useful when the world-model loss still needs to share the
# same forward pass as the PPO update).
def compute_world_model_loss(
    log_prob: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    observation_mask: Optional[torch.Tensor] = None,
):
    resolved = compute_observation_mask(attention_mask=attention_mask,
                                        response_mask=response_mask,
                                        observation_mask=observation_mask)
    if not resolved.any().item():
        return None, resolved
    wm_loss = -verl_F.masked_mean(log_prob, resolved)
    return wm_loss, resolved
