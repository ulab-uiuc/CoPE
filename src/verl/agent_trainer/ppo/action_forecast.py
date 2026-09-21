"""Action-forecast auxiliary loss: at every step t the model predicts the NEXT K
action commands it will take (the "forecast"), supervised by the REALIZED future —
i.e. the actual actions a_t, a_{t+1}, ..., a_{t+K-1} taken in the rollout (the
current action is INCLUDED, so forecast[0] == the action committed this turn).

Post-hoc and teacher-forced: chat-template re-assembly into standalone SFT samples,
collated and scored with CE, targeting the agent's own future ACTION string. It is a
SEPARATE forward pass (``update_action_forecast``) and does NOT touch PG.

Leakage is intentionally ignored (per design): the realized future IS the target.

Gating: ``gate='wins'`` (default) keeps only trajectories with reward above
``success_threshold`` so we never teach the model to foresee a flailing future;
``gate='all'`` uses every trajectory (more data, but pulls forecasts toward bad
futures on losing rollouts — kept for ablation).

PURE logic for sample assembly (stdlib + tokenizer only, CPU-testable). The CE
loss + collate are reused from sft_common, shared with the other auxiliary SFT
objectives.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

DEFAULT_ACTION_FORECAST_PROMPT = (
    "Plan ahead: list the next {k} actions you will take to make progress on the "
    "task, starting with the action you take right now, one action per line."
)

# ``layout='turns'``: the K actions are K assistant turns separated by a fixed user
# prompt, instead of K lines of one assistant message (``layout='list'``, the default
# and the original construction). Under tau2's native protocol the K-line list is
# byte-for-byte Qwen2.5's rendering of several parallel tool calls in ONE turn, so the
# forecast SFT reinforced "continue after </tool_call> with another call" and that bled
# into policy turns: turns with >= 2 <tool_call> blocks went 0.9% -> 3.1% over steps 5-10
# of the v4 run while the client executes only the first, and the reward never sees the
# rest. As K turns every target has the exact shape of a policy turn -- one call or one
# message, then <|im_end|> -- and the loss covers the assistant turns only; the filler
# user turns are constants. What each turn predicts is unchanged: action j conditioned
# on the prefix and actions < j, without their results.
DEFAULT_ACTION_FORECAST_TURNS_PROMPT = (
    "Plan ahead: give the next {k} actions you will take to make progress on the task, "
    "one action per reply, starting with the action you take right now. Reply with the "
    "action only."
)
ACTION_FORECAST_NEXT_PROMPT = "Next action?"
ACTION_FORECAST_LAYOUTS = ("list", "turns")


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


# Code-as-action envs: the action is a fenced markdown code block, with no
# ``Action:`` marker. The default path would pick the last non-empty line, which is
# the closing fence -- on appworld every turn came out as '```', so the forecast
# target became "the next three actions are ```, ```, ```".
_CODE_AS_ACTION_ENVS = ("appworld",)
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)

# Native tool-calling envs (tau2 under the InfoPO-aligned protocol): an action turn is
# either a ``<tool_call>{json}</tool_call>`` block or a plain message to the customer,
# and the conversation is [system, user, assistant, tool|user, assistant, ...] rather
# than the ReAct instr/ack/obs layout. The forecast target for a tool call is the call
# in the exact surface form the policy emits it, ``<tool_call>{json}</tool_call>`` on one
# line; for a message it is the message on one line, prefixed ``say:``. The target must
# be the executed form: with bare JSON as the target, a 0.1-weighted forecast loss taught
# the policy to write bare JSON lines in ordinary turns (19.5% of turns by step 5, tool
# calls down from 29% to 5%), which the client forwards to the customer as text.
_NATIVE_TOOL_ENVS = ("tau2",)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_NATIVE_SAY_MAX_CHARS = 200

# How a native-protocol action is written into the forecast target.
#   'executed' (the original construction): the call RE-SERIALIZED on one line as compact
#     JSON, a message as one ``say:`` line cut at 200 chars. That is a surface form the
#     policy never writes (100% of its calls are the template's newline-wrapped, space-
#     separated JSON), and at the original forecast dose the policy learned it instead:
#     compact-JSON calls went 0% -> 46% -> 96.5% of policy calls over steps 5-7 of the v5
#     run (41% at step 6 of v2), with the entropy spike and the tool-call share falling
#     27% -> 16%.
#   'verbatim': the executed part of the turn as a LITERAL SUBSTRING of what the policy
#     generated, never re-encoded -- the first <tool_call>...</tool_call> block exactly as
#     written (prose around it dropped, as alfworld drops the Thought), or the message
#     text unchanged. The rule the alfworld targets always followed (``go to shelf 1`` is
#     cut out of the turn, not rewritten). Multi-line, so it needs layout='turns'.
ACTION_FORECAST_NATIVE_FORMS = ("executed", "verbatim")
_TOOL_CALL_SPAN_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def _native_action(assistant_text: str, form: str = "executed") -> str:
    import json
    text = assistant_text or ""
    if form == "verbatim":
        if not text.strip():
            return ""
        span = _TOOL_CALL_SPAN_RE.search(text)
        return span.group(0) if span else text
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            call = json.loads(m.group(1))
            body = json.dumps({"name": call.get("name"), "arguments": call.get("arguments", {})},
                              ensure_ascii=False, separators=(",", ":"))
        except Exception:
            body = " ".join(m.group(1).split())
        return f"<tool_call>{body}</tool_call>"
    words = text.split()
    if not words:
        return ""
    line = " ".join(words)
    return "say: " + (line[:_NATIVE_SAY_MAX_CHARS] + "..." if len(line) > _NATIVE_SAY_MAX_CHARS else line)


def _is_native_layout(convo: List[Dict[str, str]]) -> bool:
    """A system-first conversation is the native tool-calling layout; the ReAct layout
    always starts with the user instruction."""
    return bool(convo) and convo[0].get('role') == 'system'


def extract_action(assistant_text: str, env: str = "", form: str = "executed") -> str:
    """The bare action command from an assistant turn (drops the Thought).

    Take the first non-empty line after ``Action:``; fall back to the last
    non-empty line for bare-action envs.
    Returns '' for empty/degenerate turns (e.g. the trailing terminal turn).

    For ``env`` in _CODE_AS_ACTION_ENVS, return the last fenced code block instead,
    matching how AppWorldEnvClient.step extracts the code it executes, so the
    forecast target is exactly the action that ran. ``env=""`` (the default) keeps
    the original behaviour for every other env. ``form`` only applies to the native
    tool-calling envs (see ACTION_FORECAST_NATIVE_FORMS).
    """
    if (env or "").lower() in _CODE_AS_ACTION_ENVS:
        blocks = _FENCE_RE.findall(assistant_text or "")
        return blocks[-1].strip() if blocks else ""
    if (env or "").lower() in _NATIVE_TOOL_ENVS:
        return _native_action(assistant_text, form=form)
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

    Layout (codebase convention):
    [instr(user), ack(assistant), obs0(user), action0(assistant), obs1, action1, ...].
    The instruction+ack pair is skipped; action turns sit at conv idx 3, 5, 7, ...
    (assistant, each preceded by a user obs).
    """
    if _is_native_layout(convo):
        return [i for i in range(1, len(convo)) if convo[i]['role'] == 'assistant']
    return [i for i in range(3, len(convo), 2)
            if convo[i]['role'] == 'assistant' and convo[i - 1]['role'] == 'user']


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
    # appworld is code-as-action, so an action is ineffective only when the code
    # crashed or no executable code was found. Measured over 9344 observations
    # (steps 55-75): "Execution failed. Traceback" 31.7%, "No code available to
    # execute" 1.0%. Do NOT add "login failed" / "status code is 401": those are
    # business-level failures the model printed itself; the code ran and had
    # side effects, unlike alfworld's "nothing happens".
    "appworld": ("invalid action", "execution failed", "no code available to execute",
                 "environment not reset"),
}


# tau2: the client answers an unparseable turn with "Invalid turn ..." and a failed
# tool call with a result that starts with "Error" ("Error: Non-pending order cannot be
# cancelled"); a customer message is never an invalid outcome.
_TAU2_INVALID_PREFIXES = ("error", "invalid turn")


def is_invalid_outcome(result_obs: str, env: str = "alfworld") -> bool:
    """True if the env feedback ``result_obs`` indicates the action was invalid /
    had no effect (illegal action, 'Nothing happens.', 'No known action...'). Per-env
    patterns; unknown env uses the common set."""
    low = (result_obs or "").lower()
    if (env or "").lower() in _NATIVE_TOOL_ENVS:
        return low.lstrip().startswith(_TAU2_INVALID_PREFIXES)
    for p in INVALID_OUTCOME_PATTERNS.get((env or "").lower(), _INVALID_COMMON):
        if p in low:
            return True
    return False


def build_action_targets(messages, k: int = 3, skip_invalid: bool = False,
                       env: str = "alfworld", form: str = "executed") -> List[Dict[str, object]]:
    """For each action turn t, return per-step targets.

    {'prefix_end': idx, 'actions': [a_t..a_{t+K-1}], 'action_turns': [..], 'src_turn': n}

    ``prefix_end`` is the conversation index of the obs the action responds to
    (= action_turn_index - 1); the SFT prefix is convo[:prefix_end+1].
    ``actions`` are the realized next-K bare action commands (current included).
    Steps with no future action are skipped.

    ``skip_invalid`` (default off): drop actions whose RESULT observation signals an
    invalid / no-effect outcome (per-env, see is_invalid_outcome) — the forecast
    target then contains only the next-K *effective* actions (looking past the
    skipped ones). ``env`` selects the invalid-outcome patterns.
    """
    convo = _to_chat_list(messages)
    action_idxs = _action_turn_indices(convo)
    actions_seq = [extract_action(convo[ai]['content'], env=env, form=form) for ai in action_idxs]

    if skip_invalid:
        # per action turn, the RESULT obs = the next user message after it
        valid_seq = []
        for ai in action_idxs:
            res = (convo[ai + 1]['content'] if (ai + 1 < len(convo)
                   and convo[ai + 1].get('role') in ('user', 'tool')) else '')
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
        # Turn index of each forecast slot, for Temporal Ensembling. ``fut`` above is
        # unchanged; this derives the indices of the same elements in parallel and
        # asserts they agree. TE needs slot -> turn to align each forecast with the
        # step it predicts, and with skip_invalid the slot offset is not the turn gap.
        if skip_invalid:
            sel = [j for j in range(n, len(actions_seq))
                   if actions_seq[j] and valid_seq[j]][:k]
        else:
            sel = [n + o for o, a in enumerate(actions_seq[n:n + k]) if a]
        assert [actions_seq[j] for j in sel] == fut, (
            "action_turns diverged from the forecast target; the two derivations "
            f"must stay in sync. n={n} sel={sel} fut={fut}")
        out.append({'prefix_end': ai - 1, 'actions': fut,
                    'action_turns': sel, 'src_turn': n})
    return out


def build_action_forecast_samples(
    messages,
    tokenizer,
    k: int = 3,
    max_length: int = 4096,
    min_target_tokens: int = 1,
    skip_invalid: bool = False,
    env: str = "alfworld",
    layout: str = "list",
    native_form: str = "executed",
) -> List[Dict[str, "object"]]:
    """Per-step teacher-forced forecast-SFT samples for one trajectory.

    Target: the realized next-K bare action commands (grounded — does NOT
    contaminate the rollout).

    Construction (``layout='list'``): prefix = convo[:obs_t+1] + a synthetic user
    prompt ("list the next K ..."); target = assistant(bare newline list, +EOS).
    Standalone — distinct from the rollout turn.
    ``layout='turns'``: the same prefix + a "one action per reply" prompt; target = K
    assistant turns, each one action (+EOS), separated by the fixed user prompt
    ACTION_FORECAST_NEXT_PROMPT; loss on the assistant turns only.

    Horizon: fixed ``k``. The prompt is aligned to the REALIZED
    length after end-of-episode clamping (never over-promises). Each returned
    sample carries ``k_realized`` (int) for metrics.

    Loss mask covers only the target tokens. Returns dicts with torch tensors.
    """
    k = int(k)
    if layout not in ACTION_FORECAST_LAYOUTS:
        raise ValueError(f"action_forecast layout must be one of {ACTION_FORECAST_LAYOUTS}, got {layout!r}")
    if native_form not in ACTION_FORECAST_NATIVE_FORMS:
        raise ValueError(f"action_forecast native_form must be one of {ACTION_FORECAST_NATIVE_FORMS}, got {native_form!r}")
    if native_form == "verbatim" and layout != "turns":
        raise ValueError("native_form='verbatim' targets are multi-line and need layout='turns'")

    convo = _to_chat_list(messages)
    samples: List[Dict[str, object]] = []
    for tgt in build_action_targets(messages, k=k, skip_invalid=skip_invalid, env=env, form=native_form):
        items = (tgt.get('actions') or [])[:k]   # clamp to what's available (<= k)
        if not items:
            continue
        realized = len(items)
        # synthetic prompt formatted with the REALIZED count
        prefix = list(convo[:tgt['prefix_end'] + 1])
        if layout == "turns":
            prefix.append({'role': 'user', 'content': DEFAULT_ACTION_FORECAST_TURNS_PROMPT.format(k=realized)})
            target_msgs = []
            for j, a in enumerate(items):
                if j:
                    target_msgs.append({'role': 'user', 'content': ACTION_FORECAST_NEXT_PROMPT})
                target_msgs.append({'role': 'assistant', 'content': a})
            s = encode_sft_sample(tokenizer, prefix, target_msgs,
                                  max_length=max_length, min_target_tokens=min_target_tokens,
                                  assistant_only=True)
        else:
            prefix.append({'role': 'user', 'content': DEFAULT_ACTION_FORECAST_PROMPT.format(k=realized)})
            target_msgs = [{'role': 'assistant', 'content': "\n".join(items)}]
            s = encode_sft_sample(tokenizer, prefix, target_msgs,
                                  max_length=max_length, min_target_tokens=min_target_tokens)
        if s is not None:
            s['k_realized'] = realized
            samples.append(s)

    return samples


def _prefix_token_len(tokenizer, text: str, full_ids) -> int:
    """Number of leading tokens of ``full_ids`` that render ``text``, a text prefix of
    the sequence ``full_ids`` encodes. Exact when the boundary sits on a special token
    (the chat template's <|im_end|>/<|im_start|> markers); otherwise the longest common
    token prefix, the same fallback encode_sft_sample uses for the prompt boundary."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    n = min(ids.size(0), full_ids.size(0))
    if n == ids.size(0) and bool((full_ids[:n] == ids).all()):
        return n
    common = 0
    for i in range(n):
        if ids[i].item() != full_ids[i].item():
            break
        common = i + 1
    return common


def encode_sft_sample(tokenizer, prefix, target_msgs, max_length: int = 4096,
                      min_target_tokens: int = 1, assistant_only: bool = False):
    """Tokenize a (prefix, target) chat pair into an SFT sample dict whose
    ``loss_mask`` covers ONLY the target (assistant) tokens — obs/prompt in the
    prefix contribute zero loss. Shared by action-forecast and the sft-ablation
    control so both use byte-identical encoding (clean apples-to-apples). Returns
    None if templating fails or the target is shorter than ``min_target_tokens``.

    ``assistant_only``: when ``target_msgs`` holds several turns (assistant, user,
    assistant, ...), put loss on the assistant turns only; the user turns in the target
    are the fixed filler prompts of the ``turns`` layout and are never trained. Each
    assistant span is bounded by the token length of the rendering up to it (with the
    generation prompt) and through it, both text prefixes of the full rendering."""
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

    input_ids = full_ids
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros_like(input_ids)
    if assistant_only and len(target_msgs) > 1:
        for j, msg in enumerate(target_msgs):
            if msg.get('role') != 'assistant':
                continue
            try:
                start_text = tokenizer.apply_chat_template(
                    prefix + list(target_msgs[:j]), tokenize=False, add_generation_prompt=True)
                end_text = tokenizer.apply_chat_template(
                    prefix + list(target_msgs[:j + 1]), tokenize=False, add_generation_prompt=False)
            except Exception:  # pragma: no cover - tokenizer template missing
                return None
            start = _prefix_token_len(tokenizer, start_text, full_ids)
            end = _prefix_token_len(tokenizer, end_text, full_ids)
            loss_mask[start:end] = 1
    else:
        loss_mask[prefix_len:] = 1

    target_len = int(loss_mask.sum().item())
    if target_len < min_target_tokens:
        return None

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


def build_action_forecast_batch(
    messages_list,
    tokenizer,
    rewards: Optional[List[float]] = None,
    k: int = 3,
    gate: str = "wins",
    success_threshold: float = 0.5,
    max_length: int = 4096,
    max_samples_per_trajectory: Optional[int] = None,
    skip_invalid: bool = False,
    env: str = "alfworld",
    group_ids=None,
    group_norm: bool = False,
    layout: str = "list",
    native_form: str = "executed",
):
    """Padded forecast-SFT batch over trajectories, with win/all gating.

    gate='wins' keeps only trajectories whose reward > success_threshold (needs
    ``rewards`` aligned to ``messages_list``); gate='all' keeps everything.
    See build_action_forecast_samples for the sample construction. Horizon is a
    fixed ``k``. Reuses collate_sft_samples for padding.

    Group-weight NORMALIZATION (``group_norm``, needs ``group_ids`` = the GRPO ``uid``
    aligned to messages_list): distill successful trajectories only and give every
    GROUP the SAME total forecast-CE weight, split evenly over its distilled trajectories
    (1/m_g), so each group's contribution stays constant as success-rate rises -> no
    SFT blow-up. Emits per-sample ``loss_weight`` (renormalized to mean 1; the actor
    applies it scaled by action_forecast_coef).
    """
    from verl.agent_trainer.ppo.sft_common import collate_sft_samples

    all_samples: List[Dict[str, object]] = []
    n_traj_used = 0
    n_traj_considered = 0
    traj_records = []   # (group_id, n_samples, start_idx) per distilled traj (group_norm)
    for i, messages in enumerate(messages_list):
        if messages is None:
            continue
        r = rewards[i] if (rewards is not None and i < len(rewards)) else 0.0
        succ = (r is not None and float(r) > success_threshold)
        # keep decision: group_norm distills successes only (like gate='wins');
        # it then reweights the kept ones.
        if (group_norm or gate == "wins") and not succ:
            continue
        n_traj_considered += 1
        traj_samples = build_action_forecast_samples(
            messages=messages, tokenizer=tokenizer, k=k,
            max_length=max_length,
            skip_invalid=skip_invalid, env=env, layout=layout, native_form=native_form)
        if max_samples_per_trajectory is not None and len(traj_samples) > max_samples_per_trajectory:
            traj_samples = traj_samples[-max_samples_per_trajectory:]
        if traj_samples:
            n_traj_used += 1
        _start = len(all_samples)
        all_samples.extend(traj_samples)
        if group_norm and traj_samples:
            gid = group_ids[i] if (group_ids is not None and i < len(group_ids)) else i
            # the trajectory's realized action sequence, for the group_unique_frac metric
            _cv = _to_chat_list(messages)
            seq_key = tuple(extract_action(_cv[a].get("content", ""), env=env, form=native_form) for a in _action_turn_indices(_cv))
            traj_records.append((gid, len(traj_samples), _start, seq_key))

    # Group-weight normalization: each GROUP contributes equally (total weight 1 after
    # renorm, so every prompt is equal-weight regardless of how many successes it has),
    # split evenly per distilled trajectory (1/m_g). Renormalized to mean 1 so loss
    # scale is fixed.
    if group_norm and traj_records:
        from collections import Counter
        wts = [1.0] * len(all_samples)
        m_g = Counter(gid for gid, _, _, _ in traj_records)  # group -> #distilled trajs
        for gid, cnt, start, _seq_key in traj_records:
            wt = 1.0 / max(1, m_g[gid])
            for j in range(start, start + cnt):
                wts[j] = wt
        _mw = (sum(wts) / len(wts)) if wts else 1.0
        if _mw > 0:
            wts = [w / _mw for w in wts]
        for s, w in zip(all_samples, wts):
            s['loss_weight'] = w

    batch = collate_sft_samples(
        samples=all_samples,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        max_length=max_length,
    )
    # realized-horizon metric (k_mean = mean realized target length after clamping)
    _ks = [int(s.get('k_realized', 0)) for s in all_samples]
    k_mean = (sum(_ks) / len(_ks)) if _ks else float(k)
    meta = {"action_forecast/n_samples": float(len(all_samples)),
            "action_forecast/n_traj_used": float(n_traj_used),
            "action_forecast/n_traj_considered": float(n_traj_considered),
            "action_forecast/gate_wins": 1.0 if gate == "wins" else 0.0,
            "action_forecast/k_mean": float(k_mean),
            "action_forecast/skip_invalid": 1.0 if skip_invalid else 0.0,
            "action_forecast/group_norm": 1.0 if group_norm else 0.0,
            "action_forecast/layout_turns": 1.0 if layout == "turns" else 0.0,
            "action_forecast/native_verbatim": 1.0 if native_form == "verbatim" else 0.0,
            "action_forecast/k": float(k_mean)}
    if group_norm and traj_records:
        from collections import Counter as _Counter, defaultdict as _dd
        _mg = _Counter(gid for gid, _, _, _ in traj_records)
        _byg = _dd(list)
        for gid, _c, _s, _seq_key in traj_records:
            _byg[gid].append(_seq_key)
        # per-group distinct-fraction: unique seqs / trajectories (1.0 = no dup in group)
        _ratios = [len(set(v)) / len(v) for v in _byg.values() if v]
        # Batch-level spread of the per-sample weights -- the number that shows whether
        # group_norm separates samples at all (the mean is 1.0 by construction). It must
        # be computed over the assembled batch: the same statistic in the optimizer loop
        # is per micro-batch, and at one sample per micro-batch it is 0 by definition.
        # (Seen on sciworld: loss_weight_std=0.000 for 50 steps while 46 trajectories
        # were spread over 8 groups, so the weights provably differed.)
        _w = [float(x.get("loss_weight", 1.0)) for x in all_samples]
        if len(_w) > 1:
            _wm = sum(_w) / len(_w)
            _wsd = (sum((x - _wm) ** 2 for x in _w) / (len(_w) - 1)) ** 0.5
        else:
            _wm, _wsd = (_w[0] if _w else 1.0), 0.0
        meta.update({
            "action_forecast/group_n_distilled": float(len(_mg)),
            "action_forecast/group_succ_traj_per_group_mean": float(sum(_mg.values()) / len(_mg)),
            "action_forecast/group_unique_frac": float(sum(_ratios) / len(_ratios)) if _ratios else 1.0,
            "action_forecast/batch_loss_weight_mean": float(_wm),
            "action_forecast/batch_loss_weight_std": float(_wsd),
            "action_forecast/batch_loss_weight_min": float(min(_w)) if _w else 1.0,
            "action_forecast/batch_loss_weight_max": float(max(_w)) if _w else 1.0,
        })
    return batch, meta
