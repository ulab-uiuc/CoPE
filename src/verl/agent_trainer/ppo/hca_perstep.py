"""HCAPO-aligned PER-STEP hindsight scoring (paper arXiv:2603.08754, §4.2).

The paper builds a SHORT per-step prompt (truncated history, history_length=2 for
ALFWorld/WebShop) and injects the realized final state s_final into THAT prompt,
local to each scored action. Our previous implementation injected s_final ONCE at
the front of the whole concatenated trajectory, which the self-normalisation ÷π̄
cancels (verified). This module reconstructs the per-step local context so π_hind
actually carries the hindsight signal.

PURE logic only (no torch/model). For each scored assistant action it yields the
reconstructed per-step context messages (truncated history + current obs + s_final
hint) and the action text to score. The worker tokenizes these and reads the mean
action-token log-prob (teacher forcing); that scalar is then broadcast back into
the (B,T) h_log_probs layout so apply_hca_advantage is unchanged.
"""
import re
from typing import List, Dict, Tuple

HINT_PRE = ("\n[Hindsight] Knowing in advance that this episode ultimately ends in the "
            "following final state:\n")
HINT_SUF = "\n"


def _norm(messages: List[dict]) -> List[dict]:
    return [c for c in messages if c.get("role") in ("system", "user", "assistant")]


def final_state_text(messages: List[dict], max_chars: int = 600) -> str:
    """s_final = the trajectory's most recent environment observation (last user
    message). Env-agnostic: works whether the trajectory ends on an action or obs."""
    conv = _norm(messages)
    for c in reversed(conv):
        if c.get("role") == "user":
            return (c.get("content", "") or "")[:max_chars].strip()
    return ""


def iter_step_inputs(messages: List[dict], history_len: int = 0,
                     inject_final_state: bool = True, max_chars_obs: int = 600
                     ) -> List[Tuple[List[dict], str]]:
    """For each scored assistant ACTION turn (chronological order, matching
    _turn_spans over response_mask), return (context_messages, action_text):

      context_messages = [instruction/system, ack?] + the trajectory prefix
        (obs,action) pairs before this turn + CURRENT obs (with s_final hint
        appended). The s_final hint is injected LOCAL to each action (this is the
        bit that makes ρ carry signal — the front-once injection is cancelled by
        ÷π̄). history_len controls how many (obs,action) pairs to keep:
          history_len <= 0  -> FULL history (consistent with our own trajectory
                               representation; NOT the paper's truncation).
          history_len > 0   -> last `history_len` pairs (paper-style truncation).
      action_text = the assistant turn content to score (teacher-forced).

    Order matches _turn_spans(response_mask): the k-th yielded item == k-th action
    span, so the worker can scatter the score back into h_log_probs[i, s:e].
    """
    conv = _norm(messages)
    # layout: [instr(user), ack(assistant), obs(user), action(assistant), ...].
    sf = final_state_text(messages, max_chars_obs) if inject_final_state else ""
    hint = (HINT_PRE + sf + HINT_SUF) if (inject_final_state and sf) else ""
    seed = conv[:2]                                  # instruction (+ ack)
    out = []
    for ai in range(3, len(conv), 2):                # action turns at conv idx 3,5,7,...
        if conv[ai].get("role") != "assistant":
            continue
        cur_obs_msg = dict(conv[ai - 1])             # the obs the action responded to
        if hint:
            cur_obs_msg["content"] = (cur_obs_msg.get("content", "") or "") + hint
        # FULL history (history_len<=0) or last `history_len` (obs,action) pairs
        start = 2 if history_len <= 0 else max(2, (ai - 1) - 2 * history_len)
        hist = []
        for j in range(start, ai - 1, 2):
            if j + 1 < ai and conv[j].get("role") == "user":
                hist.append(dict(conv[j]))           # obs
                hist.append(dict(conv[j + 1]))       # action taken then
        ctx = list(seed) + hist + [cur_obs_msg]
        out.append((ctx, conv[ai].get("content", "") or ""))
    return out
