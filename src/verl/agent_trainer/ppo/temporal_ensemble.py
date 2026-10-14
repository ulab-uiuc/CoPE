"""Temporal Ensembling (TE): aggregate action_forecast's across-time predictions into a
reference distribution q_t for the policy.

    q_t^F(a)   = (1/K_t) Sum_{k=1..K_t} pi^k(a | h_{t-k})          temporal mixture
    q_t(a)     = (1-eta) pi^0_thetabar(a | h_t) + eta q_t^F(a)     frozen reference
    L_policy   = L_GRPO + lambda_TE * E_{h_t~d_thetabar} KL(pi_theta^0(.|h_t) || q_t)

Relation to the action_forecast SFT pass: SFT trains the members (the af-loss), TE
distils the members back into the policy. The two **must share the same forecast block
construction**, so pi^k is the block's chained conditional p(a_t | h_s, a_s..a_{t-1}),
not the marginal p(a_t | h_s). This matches regrouping the af-loss by target timestep —
the chained factors are exactly the terms the af-loss already contains.

Members are selected by **real turn number**, never by slot index. With
skip_invalid=True, if the emitting turn s has an invalid action of its own it is dropped
by its own filter, so action_turns[0] > s and slot 1 is already a legitimate k>=1 member.
The rule is: members of target t = {(s,j) : action_turns[j] == t and s < t}.
See tests/temporal_ensemble/test_te_action_turns.py.

Nothing here is called unless te_enable=True; with it off, ray_trainer does not even
import this module.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch

from verl.agent_trainer.ppo.action_forecast import (
    build_action_targets, _to_chat_list, DEFAULT_ACTION_FORECAST_PROMPT,
)


# --------------------------------------------------------------------------
# 1) Build the scoring samples
# --------------------------------------------------------------------------
def build_te_scoring_samples(messages, tokenizer, k: int = 3,
                             skip_invalid: bool = False, env: str = "sciworld",
                             max_length: int = 4096) -> List[Dict]:
    """One trajectory -> one scoring sample per action turn, carrying slot->turn and
    slot->token-span maps.

    The encoding path is **identical, verbatim**, to action_forecast's
    build_action_forecast_samples:
        prefix = convo[:obs_s+1] + {'role':'user', 'content': PROMPT.format(k=realized)}
        target = {'role':'assistant', 'content': "\\n".join(items)}
    Note the target is a bare newline join. Here we use
    tokenizer(full_text, return_offsets_mapping=True), which yields the same token ids as
    encode_sft_sample's tokenizer(full_text) plus the character offsets needed to locate
    each slot.

    Each returned sample:
        input_ids / attention_mask / loss_mask : torch.LongTensor, same shape as SFT
        slot_turns : List[int]        the real action turn of each slot
        slot_spans : List[(s,e)]      each slot's action body as [s,e) into input_ids
        src_turn   : int              the turn s this forecast was emitted from
    """
    convo = _to_chat_list(messages)
    out: List[Dict] = []
    for tgt in build_action_targets(messages, k=k, skip_invalid=skip_invalid, env=env):
        items = tgt.get('actions') or []
        turns = tgt.get('action_turns') or []
        src = tgt.get('src_turn')
        if not items or src is None or len(turns) != len(items):
            continue
        # Need at least one k>=1 member -- judged by turn number, not slot index.
        if not any(t > src for t in turns):
            continue

        prefix = list(convo[:tgt['prefix_end'] + 1])
        prefix.append({'role': 'user',
                       'content': DEFAULT_ACTION_FORECAST_PROMPT.format(k=len(items))})
        content = "\n".join(items)
        target_msgs = [{'role': 'assistant', 'content': content}]
        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                prefix + target_msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue

        enc = tokenizer(full_text, add_special_tokens=False,
                        return_offsets_mapping=True)
        ids = enc['input_ids']
        offs = enc['offset_mapping']
        if not ids:
            continue

        # Where content sits in full_text (take the last occurrence: it is near the end)
        try:
            cpos = full_text.rindex(content)
        except ValueError:
            continue
        # Character range of each item within content (items are joined by "\n")
        spans_tok: List[Tuple[int, int]] = []
        off_in_content = 0
        ok = True
        for it in items:
            cs, ce = cpos + off_in_content, cpos + off_in_content + len(it)
            off_in_content += len(it) + 1        # +1 for the separating "\n"
            idx = [j for j, (a, b) in enumerate(offs) if a < ce and b > cs]
            if not idx:
                ok = False
                break
            spans_tok.append((min(idx), max(idx) + 1))
        if not ok:
            continue

        # prefix_len: exactly the same determination encode_sft_sample makes
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)['input_ids']
        if full_text.startswith(prefix_text):
            prefix_len = len(prefix_ids)
        else:
            common = 0
            for i in range(min(len(prefix_ids), len(ids))):
                if prefix_ids[i] != ids[i]:
                    break
                common = i + 1
            prefix_len = common

        input_ids = torch.tensor(ids, dtype=torch.long)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[prefix_len:] = 1

        # Left truncation (same side encode_sft_sample drops from); shift spans with it
        if input_ids.size(0) > max_length:
            drop = input_ids.size(0) - max_length
            input_ids = input_ids[drop:]
            loss_mask = loss_mask[drop:]
            spans_tok = [(s - drop, e - drop) for (s, e) in spans_tok]
            if any(s < 0 for (s, _) in spans_tok):
                continue                        # a slot was cut off; drop the sample

        out.append(dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            loss_mask=loss_mask,
            slot_turns=list(turns),
            slot_spans=spans_tok,
            src_turn=int(src),
        ))
    return out


# --------------------------------------------------------------------------
# 2) Per-slot sequence-level log-probs from logits
# --------------------------------------------------------------------------
@torch.no_grad()
def slot_logprobs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                              slot_spans: List[List[Tuple[int, int]]],
                              ) -> List[List[float]]:
    """Per sample, per slot: the **sum** of the log-probs of that slot's action tokens.

    Shift convention: the logits at position i predict token i+1, so the log-prob of the
    token at position p is gathered[p-1] (gathered is already the shifted [B, T-1]).
    """
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tok = input_ids[:, 1:]
    gathered = torch.gather(logp, 2, tok.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
    res: List[List[float]] = []
    for b, spans in enumerate(slot_spans):
        row = []
        for (s, e) in spans:
            lo, hi = max(s - 1, 0), max(e - 1, 0)
            row.append(float(gathered[b, lo:hi].sum()) if hi > lo else float('nan'))
        res.append(row)
    return res


def slot_token_logprobs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                                    slot_spans: List[List[Tuple[int, int]]],
                                    ) -> List[List[List[float]]]:
    """As slot_logprobs_from_logits, but returning **per-token** log-probs, unsummed.

    Per-token mixing needs these: members and the rollout score the same action token
    string (measured: 98.0% token-for-token identical, 0% differing in length; the
    remaining 2.0% differ only in a trailing newline token and are all degenerate samples
    where extract_action found nothing), so positions align 1:1.

    Returns res[b][slot] = [log-prob of each token in that slot]; empty span gives [].
    """
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tok = input_ids[:, 1:]
    gathered = torch.gather(logp, 2, tok.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
    res: List[List[List[float]]] = []
    for b, spans in enumerate(slot_spans):
        row = []
        for (s, e) in spans:
            lo, hi = max(s - 1, 0), max(e - 1, 0)
            row.append([float(x) for x in gathered[b, lo:hi]] if hi > lo else [])
        res.append(row)
    return res


# --------------------------------------------------------------------------
# 3) Assembling q_t
# --------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Top-M extraction and mixing for the full-vocabulary per-token KL (te_mix='fullvocab')
#
# Why it is needed -- the shared root cause of two failed runs:
#   Computing log q - log pi only on the tokens that actually occurred gives an
#   **unnormalised** objective:
#     - Raw KL form: it has a fixed point, but it only covers action tokens, so the
#       model can dodge it by moving probability mass onto non-action text
#       -> response_length 931 -> 3264, success rate to zero.
#     - After subtracting the common mode: the escape route is closed, but the loss
#       degenerates into the linear term w*log pi (w constant). For w>0 the minimum is
#       at log pi -> -inf, for w<0 at pi -> 1; **both directions are unbounded with no
#       fixed point** -> entropy 0.37 -> 6.72, the policy emits word salad.
#   A true KL has to sum over the whole vocabulary: it is 0 with zero gradient at
#   pi_theta=q (a fixed point), and moving mass to any other token increases it (no
#   escape route). That fixes both problems at once.
#
# Implementation:
#   q(v) = [(1-eta)*p0(v) + eta*qF(v)] / Z, where
#     p0 = the frozen policy's next-token distribution -- **not transmitted**; taken in
#          dp_actor from the detached current logits (exactly equal on the first inner
#          epoch, where theta = thetabar);
#     qF = the mean of the member predictions, shipped only as a sparse top-M;
#     Z  = (1-eta) + eta*(qF's top-M mass), correcting the sub-normalisation that the
#          top-M truncation introduces.
#   Then q(v) >= (1-eta)p0(v)/Z holds everywhere, the log ratio is bounded
#   (<= -log(1-eta)), no smoothing is needed, and nothing divides by zero.
# ---------------------------------------------------------------------------

TE_TOPM = 64          # vocabulary top-M kept at each action token position
TE_TOPM_MAXTOK = 16   # max tokens kept per action (measured median 2; 16 covers 98.51%)


def slot_topk_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                          slot_spans: List[List[Tuple[int, int]]],
                          topm: int = TE_TOPM, maxtok: int = TE_TOPM_MAXTOK):
    """Per sample, per slot: the top-M of the next-token distribution at that slot's
    action token positions.

    Same shift convention as slot_logprobs_from_logits: the distribution for the token at
    position p comes from logits[p-1]. Returns (ids, probs), both shaped
    [B, n_slot_max, maxtok, topm], ids padded with -1 and probs with 0 (padding
    positions contribute nothing to the mixture).
    """
    B = logits.shape[0]
    n_slot = max((len(sp) for sp in slot_spans), default=0)
    n_slot = max(n_slot, 1)
    ids = torch.full((B, n_slot, maxtok, topm), -1, dtype=torch.int32)
    prs = torch.zeros((B, n_slot, maxtok, topm), dtype=torch.float32)
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)      # [B, T-1, V]
    for b, spans in enumerate(slot_spans):
        for j, (st, en) in enumerate(spans[:n_slot]):
            lo, hi = max(st - 1, 0), max(en - 1, 0)
            hi = min(hi, lo + maxtok)
            if hi <= lo:
                continue
            v, i = torch.topk(lp[b, lo:hi], k=min(topm, lp.shape[-1]), dim=-1)
            n = hi - lo
            m = v.shape[-1]
            prs[b, j, :n, :m] = v.exp().cpu()
            ids[b, j, :n, :m] = i.to(torch.int32).cpu()
    return ids, prs


def assemble_q(member_lp: Dict[int, List[float]], pi0_lp: Dict[int, float],
               eta: float) -> Dict[int, Dict[str, float]]:
    """Combine member log-probs into log q_t.

    member_lp[t] : [log pi^k(a_t|h_{t-k}) for k>=1] (already filtered to t > s)
    pi0_lp[t]    : log pi^0_thetabar(a_t|h_t), from old_log_probs summed by turn_ids
    Returns {t: {'log_q','log_qF','K_t'}}. With no members (including t=0) log_q = pi0,
    where the TE-KL degenerates to the trust-region term KL(pi_theta||pi_thetabar),
    consistent with q_0 = pi^0_thetabar.
    """
    out: Dict[int, Dict[str, float]] = {}
    for t, p0 in pi0_lp.items():
        ms = [x for x in member_lp.get(t, []) if x == x]          # drop NaN
        if not ms or eta <= 0.0:
            out[t] = dict(log_q=float(p0), log_qF=float('nan'), K_t=0)
            continue
        mt = torch.tensor(ms, dtype=torch.float64)
        log_qF = float(torch.logsumexp(mt, 0) - math.log(len(ms)))
        a = torch.tensor(math.log1p(-eta) + float(p0), dtype=torch.float64)
        b = torch.tensor(math.log(eta) + log_qF, dtype=torch.float64)
        out[t] = dict(log_q=float(torch.logaddexp(a, b)), log_qF=log_qF, K_t=len(ms))
    return out


def segment_sum_by_turn(values: torch.Tensor, turn_ids: torch.Tensor,
                        n_turn: int) -> torch.Tensor:
    """Segment-sum per-token values by turn -> [B, n_turn]. -1 in turn_ids is ignored.

    Uses one_hot+einsum rather than a for loop: faster, and it **keeps the gradient**,
    which the policy-side log pi_theta needs.
    """
    valid = (turn_ids >= 0)
    oh = torch.nn.functional.one_hot(turn_ids.clamp(min=0), n_turn).to(values.dtype)
    oh = oh * valid.unsqueeze(-1).to(oh.dtype)
    return torch.einsum('bt,btn->bn', values, oh)


# --------------------------------------------------------------------------
# 4) Batch assembly (called from the trainer)
# --------------------------------------------------------------------------
def build_te_batch(messages_list, tokenizer, k: int, skip_invalid: bool, env: str,
                   max_length: int = 4096, pad_token_id: int = 0,
                   traj_subsample: float = 1.0, rng=None):
    """Pad every trajectory's scoring samples into one DataProto and return the index.

    index[i] = (traj_idx, slot_turns, src_turn), used to attribute sample i's slot
    log-probs to (traj_idx, target_turn).

    With traj_subsample < 1, score a random subset of trajectories. The KL is an
    expectation, so subsampling is unbiased and only raises variance -- the preferred way
    to cut cost.
    """
    import random as _random
    from verl import DataProto
    from tensordict import TensorDict

    if rng is None:
        rng = _random
    samples, index = [], []
    n_traj_scored = 0
    for ti, msgs in enumerate(messages_list):
        if traj_subsample < 1.0 and rng.random() > traj_subsample:
            continue
        got = build_te_scoring_samples(msgs, tokenizer, k=k, skip_invalid=skip_invalid,
                                       env=env, max_length=max_length)
        if got:
            n_traj_scored += 1
        for smp in got:
            samples.append(smp)
            index.append((ti, smp['slot_turns'], smp['src_turn']))
    if not samples:
        return None, [], {'te/n_samples': 0.0, 'te/n_traj_scored': 0.0}

    L = max(int(s['input_ids'].size(0)) for s in samples)
    B = len(samples)
    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    shifted_spans = []
    for b, s in enumerate(samples):
        n = int(s['input_ids'].size(0))
        off = L - n                                    # left padding; shift spans to match
        input_ids[b, off:] = s['input_ids']
        attn[b, off:] = 1
        pos[b, off:] = torch.arange(n, dtype=torch.long)
        shifted_spans.append([(a + off, e + off) for (a, e) in s['slot_spans']])

    import numpy as np
    batch = DataProto(
        batch=TensorDict({'input_ids': input_ids, 'attention_mask': attn,
                          'position_ids': pos}, batch_size=B),
        non_tensor_batch={'slot_spans': np.array(shifted_spans, dtype=object)},
    )
    meta = {'te/n_samples': float(B), 'te/n_traj_scored': float(n_traj_scored),
            'te/max_len': float(L)}
    return batch, index, meta


TE_COV_MAX = 128      # max covered action tokens kept per trajectory


def pad_slot_dim(x: torch.Tensor, k: int) -> torch.Tensor:
    """Pad the slot dimension of [B, n_slot, ...] out to k (-1 for int32, 0 for float).

    Micro-batches can have different slot counts, and the width has to be uniform before
    concatenating across workers. This lives in the TE module rather than dp_actor,
    whose function inventory is under a lazy-audit constraint.
    """
    if x.shape[1] >= k:
        return x
    sh = list(x.shape)
    sh[1] = k - x.shape[1]
    fill = -1 if x.dtype == torch.int32 else 0
    return torch.cat([x, torch.full(sh, fill, dtype=x.dtype)], 1)


def assemble_te_topm(index, topm_ids, topm_prs, turn_ids: torch.Tensor,
                     n_members_needed=None, cov_max: int = TE_COV_MAX):
    """Merge the members' top-M distributions into compact tensors aligned to the rollout.

    Output (compact, to avoid a [bsz, resp_len, M] monster of 128x4096x64):
        te_cov_pos [bsz, C]      position of each covered token in the response, -1 = pad
        te_cov_ids [bsz, C, M]   qF's top-M token ids at that position, -1 = pad
        te_cov_prs [bsz, C, M]   their probabilities (averaged over members), 0 = pad
    qF is the **arithmetic mean** of the member distributions. Members' top-M ids overlap,
    so they have to be merged by id and summed, not simply concatenated -- hence the
    chunked dense scatter_add (64 positions per chunk, 64x152k float is about 39MB)
    followed by a top-M. Correctness over cleverness.
    """
    bsz = turn_ids.shape[0]
    M = topm_prs.shape[-1] if topm_prs is not None and topm_prs.numel() else TE_TOPM
    # (traj, turn) -> [(sample_idx, slot_idx), ...]
    members = {}
    for si, (ti, slot_turns, src) in enumerate(index):
        for j, t in enumerate(slot_turns):
            if t <= src:                      # k=0 is excluded by construction
                continue
            members.setdefault((int(ti), int(t)), []).append((si, j))

    rows_traj, rows_pos, rows_src = [], [], []
    for (ti, t), ms in sorted(members.items()):
        if ti >= bsz:
            continue
        pos = (turn_ids[ti] == t).nonzero(as_tuple=True)[0]
        n = int(pos.numel())
        if n == 0:
            continue
        # A member's token count must match the rollout's, else drop it -- never truncate
        # into a misalignment (same rule as the per-token path).
        good = []
        for (si, j) in ms:
            valid_tok = int((topm_ids[si, j, :, 0] >= 0).sum())
            if valid_tok == n:
                good.append((si, j))
        if not good or n > topm_ids.shape[2]:
            continue
        for u in range(min(n, cov_max)):
            rows_traj.append(ti)
            rows_pos.append(int(pos[u]))
            rows_src.append([(si, j, u) for (si, j) in good])

    P = len(rows_traj)
    te_cov_pos = torch.full((bsz, cov_max), -1, dtype=torch.int32)
    te_cov_ids = torch.full((bsz, cov_max, M), -1, dtype=torch.int32)
    te_cov_prs = torch.zeros((bsz, cov_max, M), dtype=torch.float32)
    meta = {'te/fv_positions': float(P)}
    if P == 0:
        return te_cov_pos, te_cov_ids, te_cov_prs, meta

    V = int(topm_ids.max().item()) + 1 if topm_ids.numel() else 1
    V = max(V, 1)
    slot_in_traj = {}
    CH = 64
    for c0 in range(0, P, CH):
        c1 = min(c0 + CH, P)
        dense = torch.zeros((c1 - c0, V), dtype=torch.float32)
        for r in range(c0, c1):
            srcs = rows_src[r]
            for (si, j, u) in srcs:
                ii = topm_ids[si, j, u].long()
                pp = topm_prs[si, j, u].float()
                keep = ii >= 0
                dense[r - c0].scatter_add_(0, ii[keep].clamp(max=V - 1),
                                           pp[keep] / len(srcs))
        v, i = torch.topk(dense, k=min(M, V), dim=-1)
        for r in range(c0, c1):
            ti = rows_traj[r]
            k = slot_in_traj.get(ti, 0)
            if k >= cov_max:
                continue
            slot_in_traj[ti] = k + 1
            te_cov_pos[ti, k] = rows_pos[r]
            m = v.shape[-1]
            te_cov_ids[ti, k, :m] = i[r - c0].to(torch.int32)
            te_cov_prs[ti, k, :m] = v[r - c0]
    meta['te/fv_mass_mean'] = float(te_cov_prs.sum(-1)[te_cov_pos >= 0].mean()) \
        if (te_cov_pos >= 0).any() else 0.0
    return te_cov_pos, te_cov_ids, te_cov_prs, meta


def fullvocab_te_kl(logits: torch.Tensor, cov_pos: torch.Tensor,
                    cov_ids: torch.Tensor, cov_prs: torch.Tensor,
                    eta: float):
    """Full-vocabulary per-token KL(pi_theta || q), q = [(1-eta)p0 + eta qF] / Z.

    logits  : [B, T, V] (the response segment, same coordinates as cov_pos), with grad
    cov_pos : [B, C]    position of each covered token in the response, -1 = padding
    cov_ids : [B, C, M] qF's top-M token ids, -1 = padding
    cov_prs : [B, C, M] their probabilities

    p0 is softmax(logits).detach() -- exactly the frozen policy on the first inner epoch,
    where theta = thetabar. That makes q(v) >= (1-eta)p0(v)/Z hold everywhere and the log
    ratio bounded (<= -log(1-eta)), so no smoothing is required.

    Identity:
        KL(pi||q) = KL(pi||p0) - log((1-eta)/Z)
                    - Sum_{v in topM} pi(v)*log1p( eta*qF(v) / ((1-eta)*p0(v)) )
    KL(pi||p0) has both value and gradient 0 at p0=pi.detach(); it is kept for the
    formula's completeness (and so tests can verify the decomposition with a
    non-self-referential p0).

    Gradient = -Sum_topM grad pi(v)*log1p(...), automatically centred because
    Sum_v grad pi(v) = 0: mass only moves between tokens, the total is conserved --
    exactly the normalisation constraint the common-mode-subtracted version lacked.
    The fixed point is pi=qF, where the bracket is constant and the gradient vanishes.

    Returns (kl_mean, n_pos). With n_pos=0, kl_mean is a zero tensor (lazily).
    """
    B, T, V = logits.shape
    valid = cov_pos >= 0
    n_pos = int(valid.sum())
    if n_pos == 0 or eta <= 0.0:
        return logits.sum() * 0.0, 0
    bi, ci = valid.nonzero(as_tuple=True)
    pos = cov_pos[bi, ci].long().clamp(0, T - 1)
    return fullvocab_te_kl_rows(logits[bi, pos], cov_ids[bi, ci], cov_prs[bi, ci], eta)


def fullvocab_te_kl_rows(z: torch.Tensor, ids: torch.Tensor, prs: torch.Tensor,
                         eta: float):
    """The core of fullvocab_te_kl, taking already-gathered logit rows.

    z   : [P, V] logits at the covered positions (with grad)
    ids : [P, M] qF's top-M token ids, -1 = padding
    prs : [P, M] their probabilities
    dp_actor calls this entry point so only those few dozen [P,V] rows leave the forward
    pass, instead of the whole [bsz, resp_len, V] block (gigabytes at resp_len=4096,
    V=152k).
    """
    if z.numel() == 0 or eta <= 0.0:
        return z.sum() * 0.0, 0
    n_pos = z.shape[0]
    logp = torch.log_softmax(z.float(), dim=-1)
    p = logp.exp()
    logp0 = logp.detach()
    ids = ids.long()                                     # [P, M]
    prs = prs.float()                                    # [P, M]
    keep = ids >= 0
    ids_c = ids.clamp(min=0)
    qf = prs * keep                                      # zero out padding
    mass = qf.sum(-1)                                    # mass covered by the top-M
    Z = (1.0 - eta) + eta * mass                         # [P] sub-normalisation fix
    p0_sel = logp0.gather(1, ids_c).exp()                # [P, M]
    ratio = (eta * qf) / ((1.0 - eta) * p0_sel.clamp_min(1e-20))
    corr = torch.log1p(ratio) * keep                     # [P, M]
    p_sel = p.gather(1, ids_c)
    kl_p_p0 = (p * (logp - logp0)).sum(-1)               # value and grad both 0 (self-ref)
    kl = kl_p_p0 - torch.log((1.0 - eta) / Z) - (p_sel * corr).sum(-1)
    return kl.mean(), n_pos


def assemble_te_tensors_token(index, slot_tok_lp, old_log_probs: torch.Tensor,
                              turn_ids: torch.Tensor, eta: float,
                              traj_return=None):
    """**Per-token** variant: returns te_log_q shaped [bsz, T], like old_log_probs.

    Why per-token is required (measured in a dry run):
      Under sequence-level mixing, gain = log qF - log pi^0 is about -7.6 (-2.3 per token
      times a 3.3-token action), so qF's weight in q=(1-eta)pi^0+eta*qF is only 3e-4 and
      q is essentially (1-eta)pi^0. Then log q - log pi^0 == log(1-eta), the same constant
      for every target, and the k3 gradient 1 - e^{kl} is constant too -- it degenerates
      into "push every action down equally", carrying no directional information.
      Per-token, the gain per token is about -0.6, qF's weight at eta=0.5 is 0.33-0.41,
      and win/fail trajectories genuinely separate (delta about 0.14 nats/token).

    The alignment premise was verified offline: of 1677 pairs of (member slot span token
    string, rollout action token string marked by turn_ids), 98.0% are token-for-token
    identical and 0% differ in length; the remaining 2.0% have equal length and differ
    only in a trailing newline token. Members whose length differs are dropped outright
    here (counted in te/token_len_mismatch) and never truncated into alignment --
    misaligned mixing is worse than no mixing.
    """
    bsz, T = old_log_probs.shape
    te_log_q = old_log_probs.float().clone()          # default = pi^0 -> KL identically 0
    te_valid = torch.zeros_like(te_log_q, dtype=torch.bool)

    members = {}          # (traj, t) -> [[tok logp...], ...]
    n_k0 = 0
    for (ti, slot_turns, src), row in zip(index, slot_tok_lp):
        for j, t in enumerate(slot_turns):
            if j >= len(row):
                continue
            toks = row[j]
            if not toks or any(v != v for v in toks):     # empty or containing NaN
                continue
            if t <= src:                                  # k=0, excluded by construction
                n_k0 += 1
                continue
            members.setdefault((ti, int(t)), []).append(list(toks))

    Ks, gains, n_mismatch = [], [], 0
    log1m_eta = math.log1p(-eta) if eta < 1.0 else float('-inf')
    log_eta = math.log(eta) if eta > 0.0 else float('-inf')
    for (ti, t), ms in members.items():
        if ti >= bsz:
            continue
        pos = (turn_ids[ti] == t).nonzero(as_tuple=True)[0]
        n = int(pos.numel())
        if n == 0:
            continue
        good = [m for m in ms if len(m) == n]
        n_mismatch += len(ms) - len(good)
        if not good or eta <= 0.0:
            continue
        p0_vec = old_log_probs[ti, pos].float()
        mt = torch.tensor(good, dtype=torch.float64)                  # [K, n]
        log_qF = torch.logsumexp(mt, 0) - math.log(mt.shape[0])       # [n]
        a = log1m_eta + p0_vec.to(torch.float64)
        b = log_eta + log_qF
        te_log_q[ti, pos] = torch.logaddexp(a, b).to(te_log_q.dtype)
        te_valid[ti, pos] = True
        Ks.append(mt.shape[0])
        gains.append((ti, float((log_qF - p0_vec.to(torch.float64)).mean())))

    # Common-mode baseline g_bar: the k3 estimator's gradient w.r.t. log pi is
    # g = 1 - exp(log q - log pi^0). Since the gain is always negative and q < pi^0, g is
    # always positive -- i.e. "push every action token down". Measured: the common mode is
    # about 0.20/token while the win/fail difference is only 0.13, so the common mode is
    # larger than the signal. And because TE covers only bare action tokens, the model can
    # dodge it by moving mass onto non-action text: response_length went 931 -> 3264,
    # success rate to zero, and te/kl self-servingly fell to 0.
    # Subtracting g_bar makes the term zero-mean: only tokens that are *harder than
    # average* to predict get pushed down, and padding with filler stops paying.
    # It must be computed over the **whole batch**. dp_actor runs
    # micro_batch_size_per_gpu=1, so centring within a micro-batch is centring per
    # trajectory, which erases the win/fail difference entirely.
    _g_bar = float('nan')
    if int(te_valid.sum()) > 0:
        with torch.no_grad():
            _kl_raw = (te_log_q - old_log_probs.float())[te_valid]
            _g_bar = float((1.0 - torch.exp(_kl_raw)).mean())
    meta = {
        'te/members_mean': float(sum(Ks) / len(Ks)) if Ks else 0.0,
        'te/targets_with_members': float(len(Ks)),
        'te/k0_slots_excluded': float(n_k0),
        'te/token_len_mismatch': float(n_mismatch),
        'te/tokens_covered': float(int(te_valid.sum())),
        'te/g_bar': _g_bar,
        'te/gain_tok_all': float(sum(g for _, g in gains) / len(gains)) if gains else float('nan'),
    }
    if traj_return is not None and gains:
        gw = [g for ti, g in gains if float(traj_return[ti]) > 0.5]
        gf = [g for ti, g in gains if float(traj_return[ti]) <= 0.5]
        meta['te/gain_tok_win'] = float(sum(gw) / len(gw)) if gw else float('nan')
        meta['te/gain_tok_fail'] = float(sum(gf) / len(gf)) if gf else float('nan')
        meta['te/n_win_targets'] = float(len(gw))
        meta['te/n_fail_targets'] = float(len(gf))
    return te_log_q, te_valid, meta


def assemble_te_tensors(index, slot_lp, old_log_probs: torch.Tensor,
                        turn_ids: torch.Tensor, eta: float,
                        traj_return=None):
    """Merge slot log-probs into te_log_q / te_valid shaped [bsz, n_turn].

    index[i]   = (traj_idx, slot_turns, src_turn)
    slot_lp[i] = [sequence-level log-prob of each slot in that sample]
    old_log_probs / turn_ids : [bsz, T], the rollout's frozen log-probs and turn numbers

    Member selection keeps only slot_turns[j] > src_turn (k>=1). It must not filter on the
    slot index j>=1: with skip_invalid=True, slot 1 may already be a k>=1 member.
    """
    bsz = old_log_probs.shape[0]
    n_turn = int(turn_ids.max().item()) + 1 if (turn_ids >= 0).any() else 1
    n_turn = max(n_turn, 1)
    # The pi^0 component: segment-sum old_log_probs by turn (no extra forward pass)
    pi0 = segment_sum_by_turn(old_log_probs.float(), turn_ids, n_turn)   # [bsz,n_turn]
    has_turn = segment_sum_by_turn(torch.ones_like(old_log_probs.float()),
                                   turn_ids, n_turn) > 0                 # [bsz,n_turn]

    members = {}          # (traj, t) -> [logp,...]
    n_k0 = 0
    for (ti, slot_turns, src), row in zip(index, slot_lp):
        for j, t in enumerate(slot_turns):
            if j >= len(row):
                continue
            v = row[j]
            if v != v:                       # NaN
                continue
            if t <= src:                     # k=0, excluded (the rollout actor covers it)
                n_k0 += 1
                continue
            members.setdefault((ti, int(t)), []).append(float(v))

    te_log_q = pi0.clone()                   # default = pi^0 (no members, or t=0)
    te_valid = has_turn.clone()
    # gains holds (traj_idx, gain, n_tok). The traj_idx must be stored, not recovered by
    # zipping against members.items(): gains is appended to only when an entry passes the
    # filters and log_qF is non-NaN, while members.items() is every entry, so a positional
    # zip attributes gains to the wrong trajectory and gain_win/gain_fail stop meaning
    # anything.
    # n_tok is the number of bare action tokens in that turn, used for a **per-token**
    # gain. A sequence-level gain scales linearly with action length (-1.6 nats/tok x 5
    # tok = -8), so on its own it cannot say how far off the forecast is; the per-token
    # figure is the comparable one, and it directly sets qF's weight in the mixture
    # q=(1-eta)pi^0+eta*qF.
    Ks, gains = [], []
    for (ti, t), ms in members.items():
        if t >= n_turn or ti >= bsz or not has_turn[ti, t]:
            continue
        r = assemble_q({t: ms}, {t: float(pi0[ti, t])}, eta)[t]
        te_log_q[ti, t] = r['log_q']
        Ks.append(r['K_t'])
        if r['log_qF'] == r['log_qF']:
            n_tok = int((turn_ids[ti] == t).sum())
            gains.append((ti, r['log_qF'] - float(pi0[ti, t]), max(n_tok, 1)))

    meta = {
        'te/members_mean': float(sum(Ks) / len(Ks)) if Ks else 0.0,
        'te/targets_with_members': float(len(Ks)),
        'te/targets_total': float(has_turn.sum()),
        'te/k0_slots_excluded': float(n_k0),
        'te/gain_all': float(sum(g for _, g, _ in gains) / len(gains)) if gains else float('nan'),
        'te/span_len_mean': float(sum(n for _, _, n in gains) / len(gains)) if gains else 0.0,
        'te/gain_tok_all': float(sum(g / n for _, g, n in gains) / len(gains)) if gains else float('nan'),
    }
    # Gain split by success/failure -- the key indicator of whether the method holds up
    if traj_return is not None and gains:
        gw = [(g, n) for ti, g, n in gains if float(traj_return[ti]) > 0.5]
        gf = [(g, n) for ti, g, n in gains if float(traj_return[ti]) <= 0.5]
        meta['te/gain_win'] = float(sum(g for g, _ in gw) / len(gw)) if gw else float('nan')
        meta['te/gain_fail'] = float(sum(g for g, _ in gf) / len(gf)) if gf else float('nan')
        meta['te/gain_tok_win'] = float(sum(g / n for g, n in gw) / len(gw)) if gw else float('nan')
        meta['te/gain_tok_fail'] = float(sum(g / n for g, n in gf) / len(gf)) if gf else float('nan')
        meta['te/n_win_targets'] = float(len(gw))    # sample size; too small and gain_win is noise
        meta['te/n_fail_targets'] = float(len(gf))
    return te_log_q, te_valid, meta
