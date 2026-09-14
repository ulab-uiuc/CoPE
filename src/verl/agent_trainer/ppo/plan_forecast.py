"""Plan-forecast auxiliary loss: at every step t the model predicts the NEXT K
action commands it will take (the "plan"), supervised by the REALIZED future —
i.e. the actual actions a_t, a_{t+1}, ..., a_{t+K-1} taken in the rollout (the
current action is INCLUDED, so plan[0] == the action committed this turn).

This is the post-hoc, teacher-forced half of the design. It is the sibling of the
world-model SFT loss (world_model_loss.py): same chat-template re-assembly, same
collate + CE-from-logits, but the target is the agent's own future ACTION string
instead of the environment's next observation. It is a SEPARATE forward pass
(``update_plan_forecast``) and does NOT touch PG. The OTHER half — the inline
``<plan>`` that conditions the action and eats PG — lives in the rollout / env
adapter; the two share the backbone but operate on different token spans.

Leakage is intentionally ignored (per design): the realized future IS the target.

Gating: ``gate='wins'`` (default) keeps only trajectories with reward above
``success_threshold`` so we never teach the model to foresee a flailing future;
``gate='all'`` uses every trajectory (more data, but pulls plans toward bad
futures on losing rollouts — kept for ablation).

PURE logic for sample assembly (stdlib + tokenizer only, CPU-testable). The CE
loss + collate are reused from world_model_loss to avoid divergence.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

DEFAULT_PLAN_PROMPT = (
    "Plan ahead: list the next {k} actions you will take to make progress on the "
    "task, starting with the action you take right now, one action per line."
)

# Subgoal target prompt (plan_forecast_target='subgoal'): predict the next K
# sub-goals you will actually COMPLETE (hindsight-confirmed milestones).
DEFAULT_SUBGOAL_PROMPT = (
    "List the next {k} sub-goals you will actually accomplish from here — the "
    "milestones you will complete — one per line."
)

# ALFWORLD action grammar + object grounding, appended to the inline instructions.
# GPU env-rollout probe (base 7B, 8 episodes): adding this brought plan-mode's
# 'Nothing happens' rate down to ~52% == the no-plan floor (52%), with plan 100%
# — i.e. the cold-start tax of invalid actions disappears, so NO warmup is needed.
# NOTE: alfworld-specific verbs; for other envs this block should be swapped.
_ALFWORLD_ACTION_GRAMMAR = (
    "\n\nValid actions are EXACTLY of these forms (use these verbs verbatim, fill the "
    "slots with objects/receptacles you actually see; do NOT invent verbs like 'look "
    "at', 'place', or 'pick up'):\n"
    "go to <receptacle>\ntake <object> from <receptacle>\nput <object> in/on <receptacle>\n"
    "open <receptacle>\nclose <receptacle>\nuse <object>\nheat <object> with <receptacle>\n"
    "cool <object> with <receptacle>\nclean <object> with <receptacle>\n"
    "examine <object-or-receptacle>\nlook\ninventory\n"
    "Only interact with objects and receptacles that ACTUALLY appear in the observations; "
    "never invent objects, receptacles, or numbers. The full list of available actions is "
    "given in the FIRST observation. If unsure what is available, use 'look' or 'inventory'."
)

# Block-1 standing instruction: appended to the task instruction so the model,
# EACH turn, first writes a K-action Plan (which precedes the Action and is part of
# the generated turn, so it eats PG), then Thought, then Action. No new parsing —
# the env still reads the line after "Action:".
#
# Format chosen by GPU probe (probe_inline_plan.py, Qwen2.5-3B alfworld step75):
# a tagged "Plan:" section with a bare numbered skeleton "1. ...\n2. ..." hits
# 100% plan-compliance + 100% action-extractable, and is env-agnostic. The old
# soft "in your THOUGHT, lay out..." phrasing got 0% (model ignored it); verbose
# "<placeholder>" examples also got 0%; concrete few-shot examples are env-specific.
def inline_plan_reminder(k: int = 3) -> str:
    """Short per-turn reminder appended to EVERY observation when block-1 inline
    plan is on (a one-time standing instruction decays over turns). Kept terse to
    limit per-turn token bloat; the full format lives in the standing instruction.
    Order: Thought -> Plan -> Action."""
    return (f"\n\n[Reminder] Respond with 'Thought:' then 'Plan:' (a numbered list of "
            f"your next {k} actions, step 1 = the action you take now), then 'Action:' "
            f"(= step 1 of the Plan).")


def think_reminder() -> str:
    """Per-turn THINK reminder (a lighter alternative to the inline-plan reminder,
    mutually exclusive with it): only nudges the model to reason in a 'Thought:'
    line before the 'Action:', without forcing a forward Plan. Keeps the model
    reflective/reactive (preserves recovery actions like 'help' and obs-dependent
    choices) without the plan's forward-commitment downsides."""
    return ("\n\n[Reminder] Think before you act: first write a brief 'Thought:' "
            "reasoning about the current observation and what to do next, then give "
            "your 'Action:'.")


def inline_plan_instruction(k: int = 3) -> str:
    """Block-1 standing instruction (actions style), used as a FULL REPLACEMENT of
    the env's instruction (not appended). GPU multi-turn probe: appending decays to
    ~15% after turn 1 (the env's THOUGHT/ACTION framing wins); REPLACING with this
    reference-style instruction sustains ~95% plan-compliance across turns. Re-think
    a fresh full plan each turn; Plan -> Action (no Thought)."""
    skeleton = "\n".join(f"{i}. ..." for i in range(1, max(1, k) + 1))
    return (
        "Interact with a household to solve a task. You are an intelligent agent in a "
        "household environment; act to complete the goal. At the start you are given the "
        "environment description, your goal, and the AVAILABLE ACTIONS; each turn the "
        "environment gives feedback.\n\n"
        "On EVERY turn, output in EXACTLY this format:\n"
        f"Plan:\n{skeleton}\nAction:\nyour next action\n\n"
        f"(1) Plan: re-think from scratch a short plan of your next {k} actions from your "
        "CURRENT situation to the goal (independent each turn).\n"
        "(2) Action: your next action.\n"
        "Reminder:\n"
        "1. The Action MUST be chosen from the given AVAILABLE ACTIONS, written EXACTLY as "
        "listed. Any action other than the provided available actions is ILLEGAL and does "
        "nothing.\n"
        "2. If the environment says 'Nothing happens', the previous action was invalid — "
        "revise your plan and try a DIFFERENT available action.\n"
        "3. Output the Plan and the Action every single turn; never skip the Plan."
        + _ALFWORLD_ACTION_GRAMMAR
    )


def todo_plan_instruction(k: int = 3) -> str:
    """Block-1 standing instruction (TODO style, pairs with plan_forecast_target=
    'subgoal'), used as a FULL REPLACEMENT of the env's instruction. The Plan is a
    running TODO list; completed sub-goals are marked '(done)', which we
    hindsight-relabel as the achieved-subgoal targets. REPLACE-mode sustains 100%
    plan-compliance across turns in the probe (append-mode collapses)."""
    return (
        "Interact with a household to solve a task. You are an intelligent agent in a "
        "household environment; act to complete the goal. At the start you are given the "
        "environment description, your goal, and the AVAILABLE ACTIONS; each turn the "
        "environment gives feedback.\n\n"
        "On EVERY turn, output in EXACTLY this format:\n"
        "Plan:\n1. <sub-goal> (done)\n2. <sub-goal>\n3. <sub-goal>\nAction:\nyour next action\n\n"
        "(1) Plan: a TODO list of the sub-goals needed to reach the goal; append ' (done)' "
        "to completed sub-goals and keep/revise the rest, carrying the same sub-goals "
        "across turns.\n"
        "(2) Action: your next action.\n"
        "Reminder:\n"
        "1. The Action MUST be chosen from the given AVAILABLE ACTIONS, written EXACTLY as "
        "listed. Any action other than the provided available actions is ILLEGAL and does "
        "nothing.\n"
        "2. If the environment says 'Nothing happens', the previous action was invalid — "
        "revise your plan and try a DIFFERENT available action.\n"
        "3. Output the Plan and the Action every single turn; never skip the Plan."
        + _ALFWORLD_ACTION_GRAMMAR
    )


def todo_plan_reminder(k: int = 3) -> str:
    """Short per-turn reminder for the TODO-list plan."""
    return ("\n\n[Reminder] Maintain your Plan as a TODO list of sub-goals: append "
            "' (done)' to completed ones and keep/revise the rest. Format: 'Thought:', "
            "then 'Plan:' (numbered sub-goals, '(done)' on finished ones), then 'Action:'.")


def _to_chat_list(messages) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            out.append({'role': m['role'], 'content': m['content']})
        elif hasattr(m, 'to_dict'):
            out.append(m.to_dict())
        else:  # pragma: no cover - defensive
            out.append({'role': getattr(m, 'role'), 'content': getattr(m, 'content')})
    return out


def extract_action(assistant_text: str) -> str:
    """The bare action command from an assistant turn (drops the Thought).

    Mirrors progress_credit_probe.parse_action: take the first non-empty line
    after ``Action:``; fall back to the last non-empty line for bare-action envs.
    Returns '' for empty/degenerate turns (e.g. the trailing terminal turn).
    """
    m = re.search(r"Action:\s*(.+)", assistant_text or "", re.S)
    if not m:
        lines = [l.strip() for l in (assistant_text or "").splitlines() if l.strip()]
        return lines[-1] if lines else ""
    for ln in m.group(1).strip().splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _action_turn_indices(convo: List[Dict[str, str]]) -> List[int]:
    """Indices of assistant ACTION turns, in chronological order.

    Layout (codebase convention, shared with hca_perstep / progress_credit):
    [instr(user), ack(assistant), obs0(user), action0(assistant), obs1, action1, ...].
    The instruction+ack pair is skipped; action turns sit at conv idx 3, 5, 7, ...
    (assistant, each preceded by a user obs).
    """
    return [i for i in range(3, len(convo), 2)
            if convo[i]['role'] == 'assistant' and convo[i - 1]['role'] == 'user']


_DONE_RE = re.compile(r"\(done\)|\[done\]|\[x\]|✓|✔", re.I)


def parse_plan_subgoals(turn_text: str) -> List[Tuple[str, bool]]:
    """From one assistant turn, return [(subgoal_text, is_done), ...] for the
    numbered lines in the 'Plan:' section (between 'Plan:' and 'Thought:'/'Action:').
    Done markers: '(done)', '[done]', '[x]', '✓'. The marker is stripped from text."""
    lt = turn_text or ""
    low = lt.lower()
    p = low.find("plan:")
    if p < 0:
        return []
    end = len(lt)
    for marker in ("thought:", "action:"):
        m = low.find(marker, p + 5)
        if m >= 0:
            end = min(end, m)
    seg = lt[p:end]
    out: List[Tuple[str, bool]] = []
    for line in seg.splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if not m:
            continue
        item = m.group(1).strip()
        done = bool(_DONE_RE.search(item))
        clean = _DONE_RE.sub("", item).strip().strip("-—:").strip()
        if clean:
            out.append((clean, done))
    return out


def _norm_sub(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def achieved_subgoals(messages) -> List[Tuple[str, int]]:
    """Ordered [(subgoal_text, step_idx), ...] of sub-goals the model MARKED done,
    each recorded at the FIRST action-turn (step_idx) where it appears with '(done)'
    (its text there = the hindsight-confirmed version). Deduped by normalized text.
    A failed trajectory still yields its achieved sub-goals -> usable signal."""
    convo = _to_chat_list(messages)
    action_idxs = _action_turn_indices(convo)
    seen, out = set(), []
    for n, ai in enumerate(action_idxs):
        for text, done in parse_plan_subgoals(convo[ai].get('content', '') or ''):
            if not done:
                continue
            key = _norm_sub(text)
            if key in seen:
                continue
            seen.add(key)
            out.append((text, n))
    return out


# Per-env substrings (lowercased) marking an action that had NO valid effect —
# either the client's parse/illegal guard ("Invalid Action.", common to all three
# envs) or the engine's no-op signal. Used by skip_invalid to drop ineffective
# actions from the forecast target. Unknown envs fall back to the common set.
_INVALID_COMMON = ("invalid action",)
INVALID_OUTCOME_PATTERNS = {
    "alfworld": ("invalid action", "nothing happens"),
    "sciworld": ("invalid action", "no known action matches that input"),
    "webshop": ("invalid action",),
    "babyai": ("invalid action",),
}


def is_invalid_outcome(result_obs: str, env: str = "alfworld") -> bool:
    """True if the env feedback ``result_obs`` indicates the action was invalid /
    had no effect (illegal action, 'Nothing happens.', 'No known action...'). Per-env
    patterns; unknown env uses the common set."""
    low = (result_obs or "").lower()
    for p in INVALID_OUTCOME_PATTERNS.get((env or "").lower(), _INVALID_COMMON):
        if p in low:
            return True
    return False


def build_plan_targets(messages, k: int = 3, skip_invalid: bool = False,
                       env: str = "alfworld") -> List[Dict[str, object]]:
    """For each action turn t, return per-step targets.

    {'prefix_end': idx, 'actions': [a_t..a_{t+K-1}], 'subgoals': [..]}

    ``prefix_end`` is the conversation index of the obs the action responds to
    (= action_turn_index - 1); the SFT prefix is convo[:prefix_end+1].
    ``actions`` are the realized next-K bare action commands (current included).
    ``subgoals`` are the next-K hindsight-confirmed achieved sub-goals (TODO items
    that got marked (done) at or after this step). Steps with no future action are
    skipped.

    ``skip_invalid`` (default off): drop actions whose RESULT observation signals an
    invalid / no-effect outcome (per-env, see is_invalid_outcome) — the forecast
    target then contains only the next-K *effective* actions (looking past the
    skipped ones). ``env`` selects the invalid-outcome patterns. Only affects the
    'actions' target; 'subgoals' (hindsight-achieved) is untouched.
    """
    convo = _to_chat_list(messages)
    action_idxs = _action_turn_indices(convo)
    actions_seq = [extract_action(convo[ai]['content']) for ai in action_idxs]
    ach = achieved_subgoals(messages)   # [(text, step_idx)] hindsight-confirmed milestones

    if skip_invalid:
        # per action turn, the RESULT obs = the next user message after it
        valid_seq = []
        for ai in action_idxs:
            res = (convo[ai + 1]['content'] if (ai + 1 < len(convo)
                   and convo[ai + 1].get('role') == 'user') else '')
            valid_seq.append(not is_invalid_outcome(res, env))
    else:
        valid_seq = [True] * len(action_idxs)

    out: List[Dict[str, object]] = []
    for n, ai in enumerate(action_idxs):
        if skip_invalid:
            # look past invalid actions: next-K *effective* actions from step n on
            fut = [actions_seq[j] for j in range(n, len(actions_seq))
                   if actions_seq[j] and valid_seq[j]][:k]
        else:
            fut = [a for a in actions_seq[n:n + k] if a]
        if not fut:
            continue
        # next-K sub-goals that actually complete at or after this step (hindsight)
        fut_sub = [text for (text, sn) in ach if sn >= n][:k]
        out.append({'prefix_end': ai - 1, 'actions': fut, 'subgoals': fut_sub})
    return out


def plan_block(actions: List[str]) -> str:
    """Render realized actions as an inline Plan block, matching the block-1
    instruction format: 'Plan:\\n1. a\\n2. b\\n...'."""
    return "Plan:\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))


def build_plan_forecast_samples(
    messages,
    tokenizer,
    k: int = 3,
    target: str = "action",
    seq: str = "separate",
    max_length: int = 4096,
    min_target_tokens: int = 1,
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    rng=None,
    skip_invalid: bool = False,
    env: str = "alfworld",
) -> List[Dict[str, "object"]]:
    """Per-step teacher-forced forecast-SFT samples for one trajectory.

    Two orthogonal axes (the only ones after cleanup):

    target ∈ {action, subgoal}
      action  : predict the realized next-K bare action commands (block2-success,
                grounded — does NOT contaminate the no-plan rollout).
      subgoal : predict the next-K hindsight-confirmed achieved sub-goals (the
                TODO items that actually got marked (done) later).

    seq ∈ {separate, inline_consistent}
      separate          : block2 construction. prefix = convo[:obs_t+1] + a
                          synthetic user prompt ("list the next K ..."); target =
                          assistant(bare newline list, +EOS). Standalone — distinct
                          from the rollout turn. Use with inline plan OFF.
      inline_consistent : build the SFT sample to MATCH a real rollout turn so the
                          SFT reinforces (not corrupts) the rollout format. prefix =
                          convo[:obs_t+1] (the obs, NO synthetic prompt); target =
                          a full assistant turn 'Plan:\\n1. ...\\nAction:\\n<a_t>'
                          (Plan block of the realized items + the grounded current
                          action). Use with inline plan ON.

    Horizon: fixed ``k``, or a PER-SAMPLE uniform draw over ``[k_min, k_max]``
    (curriculum horizon growth — see parse_k_schedule/active_k_range). Default is
    fixed (``k_min = k_max = k``). Each per-step sample draws its own k_i and the
    prompt / plan-block are aligned to the REALIZED length after end-of-episode
    clamping (never over-promises). Pass ``rng`` (e.g. random.Random(global_step))
    for reproducible draws; falls back to the module RNG otherwise. Each returned
    sample carries ``k_realized`` (int) for metrics.

    Loss mask covers only the target tokens. Returns dicts with torch tensors.
    """
    import torch
    import random as _random
    if k_min is None or k_max is None:
        k_min = k_max = int(k)
    if rng is None:
        rng = _random
    prompt_tmpl = (DEFAULT_SUBGOAL_PROMPT if target == "subgoal"
                   else DEFAULT_PLAN_PROMPT)

    convo = _to_chat_list(messages)
    samples: List[Dict[str, object]] = []
    for tgt in build_plan_targets(messages, k=k_max, skip_invalid=skip_invalid, env=env):
        pool = (tgt.get('subgoals') if target == "subgoal"
                else tgt.get('actions')) or []
        if not pool:
            continue
        k_i = rng.randint(k_min, k_max)      # per-sample horizon
        items = pool[:k_i]                    # clamp to what's available (<= k_i)
        if not items:
            continue
        realized = len(items)
        if seq == "inline_consistent":
            # SFT sample == a real rollout turn: obs -> assistant(Plan + Action).
            # Plan block lists exactly ``realized`` items (auto-aligned to k_i).
            prefix = list(convo[:tgt['prefix_end'] + 1])
            action = (tgt.get('actions') or [""])[0]
            content = f"{plan_block(items)}\nAction:\n{action}"
            target_msgs = [{'role': 'assistant', 'content': content}]
        else:  # separate — synthetic prompt formatted with the REALIZED count
            prefix = list(convo[:tgt['prefix_end'] + 1])
            prefix.append({'role': 'user', 'content': prompt_tmpl.format(k=realized)})
            target_msgs = [{'role': 'assistant', 'content': "\n".join(items)}]
        s = encode_sft_sample(tokenizer, prefix, target_msgs,
                              max_length=max_length, min_target_tokens=min_target_tokens)
        if s is not None:
            s['k_realized'] = realized
            samples.append(s)

    return samples


def encode_sft_sample(tokenizer, prefix, target_msgs, max_length: int = 4096,
                      min_target_tokens: int = 1):
    """Tokenize a (prefix, target) chat pair into an SFT sample dict whose
    ``loss_mask`` covers ONLY the target (assistant) tokens — obs/prompt in the
    prefix contribute zero loss. Shared by plan-forecast and the sft-ablation
    control so both use byte-identical encoding (clean apples-to-apples). Returns
    None if templating fails or the target is shorter than ``min_target_tokens``."""
    import torch
    try:
        prefix_text = tokenizer.apply_chat_template(
            prefix, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            prefix + target_msgs, tokenize=False, add_generation_prompt=False)
    except Exception:  # pragma: no cover - tokenizer template missing
        return None

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False,
                           return_tensors='pt')['input_ids'][0]
    full_ids = tokenizer(full_text, add_special_tokens=False,
                         return_tensors='pt')['input_ids'][0]
    if full_text.startswith(prefix_text):
        prefix_len = prefix_ids.size(0)
    else:
        common = 0
        for i in range(min(len(prefix_ids), len(full_ids))):
            if prefix_ids[i].item() != full_ids[i].item():
                break
            common = i + 1
        prefix_len = common

    target_len = full_ids.size(0) - prefix_len
    if target_len < min_target_tokens:
        return None

    input_ids = full_ids
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros_like(input_ids)
    loss_mask[prefix_len:] = 1

    if input_ids.size(0) > max_length:
        drop = input_ids.size(0) - max_length
        input_ids = input_ids[drop:]
        attention_mask = attention_mask[drop:]
        loss_mask = loss_mask[drop:]
        if loss_mask.sum().item() < min_target_tokens:
            return None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
    }


def parse_k_schedule(spec: str):
    """Parse a horizon-growth schedule string into sorted stages.

    Format: ``"startStep:kMin:kMax,startStep:kMin:kMax,..."`` (start-step
    semantics: a stage is active for global_step >= startStep and < the next
    stage's startStep; the last stage persists to the end). Returns a sorted list
    of ``(start_step, k_min, k_max)`` tuples, or ``[]`` for empty/None (feature
    off). Raises ValueError on any malformed input so callers can fail fast at init.
    """
    s = (spec or "").strip()
    if not s:
        return []
    stages = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(f"plan_forecast_k_schedule stage '{part}' must be "
                             f"'startStep:kMin:kMax'")
        try:
            st, lo, hi = int(fields[0]), int(fields[1]), int(fields[2])
        except ValueError:
            raise ValueError(f"plan_forecast_k_schedule stage '{part}' has non-integer fields")
        if st < 0:
            raise ValueError(f"plan_forecast_k_schedule start step must be >= 0 (got {st})")
        if lo < 1 or hi < lo:
            raise ValueError(f"plan_forecast_k_schedule stage '{part}' needs 1 <= kMin <= kMax")
        stages.append((st, lo, hi))
    stages.sort(key=lambda x: x[0])
    if stages[0][0] != 0:
        raise ValueError("plan_forecast_k_schedule: first stage must start at step 0")
    for a, b in zip(stages, stages[1:]):
        if b[0] <= a[0]:
            raise ValueError("plan_forecast_k_schedule: start steps must be strictly increasing")
    return stages


def active_k_range(stages, global_step: int):
    """Given parsed stages and the current global_step, return the active
    ``(k_min, k_max)`` — the last stage whose start_step <= global_step."""
    cur = stages[0]
    for st in stages:
        if st[0] <= global_step:
            cur = st
        else:
            break
    return cur[1], cur[2]


def build_plan_forecast_batch(
    messages_list,
    tokenizer,
    rewards: Optional[List[float]] = None,
    k: int = 3,
    gate: str = "wins",
    success_threshold: float = 0.5,
    target: str = "action",
    seq: str = "separate",
    max_length: int = 4096,
    max_samples_per_trajectory: Optional[int] = None,
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    rng=None,
    skip_invalid: bool = False,
    env: str = "alfworld",
    group_ids=None,
    group_gate: str = "off",
    group_low: float = 0.5,
    group_high: float = 1.0,
    group_norm: bool = False,
    group_dedup: bool = True,
):
    """Padded forecast-SFT batch over trajectories, with win/all gating.

    gate='wins' keeps only trajectories whose reward > success_threshold (needs
    ``rewards`` aligned to ``messages_list``); gate='all' keeps everything.
    target ∈ {action, subgoal}; seq ∈ {separate, inline_consistent} — see
    build_plan_forecast_samples. Horizon is a fixed ``k`` unless ``k_min``/``k_max``
    are given (per-sample draw over the active stage; pass ``rng`` for reproducible
    draws). Reuses collate_world_model_samples for padding.

    Two ORTHOGONAL group knobs (both need ``group_ids`` = the GRPO ``uid`` aligned to
    messages_list; both distill successful trajectories only). They compose: gating
    selects WHICH groups' successes to distill, then group_norm reweights those.

    - Group-success GATING (``group_gate``): 'off' (use plain ``gate``), 'low'
      (group success-rate <= group_low), 'low_high' (rate <= group_low OR >= group_high;
      skip the mid-rate groups where GRPO's own signal is strong). A curriculum focus.
    - Group-weight NORMALIZATION (``group_norm``): give every kept GROUP the SAME total
      plan-CE weight (=1 after renorm), so each group's contribution stays constant as
      success-rate rises -> no SFT blow-up. ``group_dedup`` (default True) splits that
      weight over the group's DISTINCT successful action-sequences (each unique seq ->
      1/u_g, further shared among its duplicate copies), so duplicate rollouts don't
      inflate weight -- this matters mid/late training where within-group action
      trajectories become highly repetitive. ``group_dedup=False`` = legacy: split
      evenly per trajectory (1/m_g). Emits per-sample ``loss_weight`` (mean 1; the actor
      applies it scaled by plan_forecast_coef).
    """
    from verl.agent_trainer.ppo.world_model_loss import collate_world_model_samples

    # Per-group success rate (for group_gate).
    group_gate = (group_gate or "off").lower()
    g_rate = {}
    if group_gate != "off" and group_ids is not None and rewards is not None:
        from collections import defaultdict
        _acc = defaultdict(list)
        for gid, r in zip(group_ids, rewards):
            _acc[gid].append(1.0 if (r is not None and float(r) > success_threshold) else 0.0)
        g_rate = {gid: (sum(v) / len(v) if v else 0.0) for gid, v in _acc.items()}

    def _group_ok(i):
        gid = group_ids[i] if (group_ids is not None and i < len(group_ids)) else None
        rate = g_rate.get(gid, 0.0)
        if group_gate == "low":
            return rate <= group_low
        if group_gate == "low_high":
            return rate <= group_low or rate >= group_high
        return True

    all_samples: List[Dict[str, object]] = []
    n_traj_used = 0
    # done/sub-goal monitoring (only meaningful for target='subgoal')
    n_traj_considered = 0
    n_achieved_total = 0
    n_traj_with_done = 0
    traj_records = []   # (group_id, n_samples, start_idx) per distilled traj (group_norm)
    for i, messages in enumerate(messages_list):
        if messages is None:
            continue
        r = rewards[i] if (rewards is not None and i < len(rewards)) else 0.0
        succ = (r is not None and float(r) > success_threshold)
        # keep decision (gating and norm both distill successes only; gating also
        # filters by group success-rate). group_norm then reweights the kept ones.
        if group_gate != "off":
            if not succ or not _group_ok(i):
                continue
        elif group_norm:
            if not succ:
                continue
        elif gate == "wins":
            if not succ:
                continue
        n_traj_considered += 1
        if target == "subgoal":
            ach = achieved_subgoals(messages)
            n_achieved_total += len(ach)
            n_traj_with_done += 1 if ach else 0
        traj_samples = build_plan_forecast_samples(
            messages=messages, tokenizer=tokenizer, k=k,
            target=target, seq=seq, max_length=max_length,
            k_min=k_min, k_max=k_max, rng=rng,
            skip_invalid=skip_invalid, env=env)
        if max_samples_per_trajectory is not None and len(traj_samples) > max_samples_per_trajectory:
            traj_samples = traj_samples[-max_samples_per_trajectory:]
        if traj_samples:
            n_traj_used += 1
        _start = len(all_samples)
        all_samples.extend(traj_samples)
        if group_norm and traj_samples:
            gid = group_ids[i] if (group_ids is not None and i < len(group_ids)) else i
            # dedup key = the trajectory's realized action sequence (full, original)
            _cv = _to_chat_list(messages)
            seq_key = tuple(extract_action(_cv[a].get("content", "")) for a in _action_turn_indices(_cv))
            traj_records.append((gid, len(traj_samples), _start, seq_key))

    # Group-weight normalization: each GROUP contributes equally (total weight 1 after
    # renorm, so every prompt is equal-weight regardless of how many successes it has).
    # group_dedup=True (default): split the group's weight over its DISTINCT successful
    # action-sequences -- each unique seq gets 1/u_g, further split among its duplicate
    # copies (per-copy = 1/(u_g * #copies)). So duplicate rollouts do NOT inflate weight;
    # every distinct successful strategy is equal-weight. group_dedup=False = legacy:
    # split evenly per trajectory (1/m_g). Renormalized to mean 1 so loss scale is fixed.
    if group_norm and traj_records:
        from collections import Counter, defaultdict
        wts = [1.0] * len(all_samples)
        if group_dedup:
            by_g = defaultdict(list)
            for idx, (gid, _cnt, _start, seq) in enumerate(traj_records):
                by_g[gid].append((seq, idx))
            rec_w = [0.0] * len(traj_records)
            for gid, items in by_g.items():
                seqc = Counter(s for s, _ in items)
                u_g = len(seqc)                                  # distinct seqs in group
                for seq, idx in items:
                    rec_w[idx] = 1.0 / (u_g * seqc[seq])         # unique seq equal-weighted
            for idx, (gid, cnt, start, seq) in enumerate(traj_records):
                for j in range(start, start + cnt):
                    wts[j] = rec_w[idx]
        else:
            m_g = Counter(gid for gid, _, _, _ in traj_records)  # group -> #distilled trajs
            for gid, cnt, start, seq in traj_records:
                wt = 1.0 / max(1, m_g[gid])
                for j in range(start, start + cnt):
                    wts[j] = wt
        _mw = (sum(wts) / len(wts)) if wts else 1.0
        if _mw > 0:
            wts = [w / _mw for w in wts]
        for s, w in zip(all_samples, wts):
            s['loss_weight'] = w

    batch = collate_world_model_samples(
        samples=all_samples,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        max_length=max_length,
    )
    _cons = max(1, n_traj_considered)
    # realized-horizon metrics (k_mean = mean realized target length after clamping)
    _eff_min = k_min if k_min is not None else k
    _eff_max = k_max if k_max is not None else k
    _ks = [int(s.get('k_realized', 0)) for s in all_samples]
    k_mean = (sum(_ks) / len(_ks)) if _ks else float(_eff_min)
    meta = {"plan_forecast/n_samples": float(len(all_samples)),
            "plan_forecast/n_traj_used": float(n_traj_used),
            "plan_forecast/n_traj_considered": float(n_traj_considered),
            "plan_forecast/gate_wins": 1.0 if gate == "wins" else 0.0,
            "plan_forecast/seq_inline_consistent": 1.0 if seq == "inline_consistent" else 0.0,
            "plan_forecast/target_subgoal": 1.0 if target == "subgoal" else 0.0,
            "plan_forecast/k_min": float(_eff_min),
            "plan_forecast/k_max": float(_eff_max),
            "plan_forecast/k_mean": float(k_mean),
            "plan_forecast/skip_invalid": 1.0 if skip_invalid else 0.0,
            "plan_forecast/group_gate": {"off": 0.0, "low": 1.0, "low_high": 2.0}.get(group_gate, 0.0),
            "plan_forecast/group_norm": 1.0 if group_norm else 0.0,
            "plan_forecast/k": float(k_mean)}
    if group_gate != "off" and g_rate:
        _kept = sum(1 for gid, rt in g_rate.items()
                    if (rt <= group_low or (group_gate == "low_high" and rt >= group_high)))
        meta.update({
            "plan_forecast/group_low_thresh": float(group_low),
            "plan_forecast/group_high_thresh": float(group_high),
            "plan_forecast/group_n_total": float(len(g_rate)),
            "plan_forecast/group_n_kept": float(_kept),
            "plan_forecast/group_succ_rate_mean": float(sum(g_rate.values()) / len(g_rate)),
        })
    if group_norm and traj_records:
        from collections import Counter as _Counter, defaultdict as _dd
        _mg = _Counter(gid for gid, _, _, _ in traj_records)
        _byg = _dd(list)
        for gid, _c, _s, seq in traj_records:
            _byg[gid].append(seq)
        # per-group distinct-fraction: unique seqs / trajectories (1.0 = no dup in group)
        _ratios = [len(set(v)) / len(v) for v in _byg.values() if v]
        meta.update({
            "plan_forecast/group_n_distilled": float(len(_mg)),
            "plan_forecast/group_succ_traj_per_group_mean": float(sum(_mg.values()) / len(_mg)),
            "plan_forecast/group_dedup": 1.0 if group_dedup else 0.0,
            "plan_forecast/group_unique_frac": float(sum(_ratios) / len(_ratios)) if _ratios else 1.0,
        })
    if target == "subgoal":
        # done-marking health: is the LLM actually checking sub-goals off?
        meta.update({
            "plan_forecast/n_achieved_subgoals": float(n_achieved_total),
            "plan_forecast/achieved_per_traj": float(n_achieved_total) / _cons,
            "plan_forecast/frac_traj_with_done": float(n_traj_with_done) / _cons,
            "plan_forecast/samples_per_traj": float(len(all_samples)) / _cons,
        })
    return batch, meta
