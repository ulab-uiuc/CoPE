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
# What a native (tool-calling) forecast predicts. 'all': every action, tool calls and
# customer messages. 'calls': the policy's tool calls only -- anchors are its tool-call
# turns and the target is its next K tool calls, looking past the messages (and results)
# in between, like the ReAct envs, where every action acts on the environment. Messages
# are never trained, so the forecast cannot teach talking instead of acting. In v10
# (targets='all', coef 0.005, balanced decision token) the forecast source -- wins of
# mixed groups -- was announcement-rich ("I will now ...", "Let me proceed ...", up to 30%
# of its messages); from step 6 the policy's own messages announced actions ~1.5x as often
# as GRPO's, and from step 11 its tool calls fell (4.8 -> 2.7 per episode by step 13) while
# its own GRPO signal favoured MORE calls. With 'calls' + balance_calls, every target turn
# starts with <tool_call>, the balance weight of that decision token is 0, and the forecast
# trains only which tool and which arguments.
ACTION_FORECAST_TARGETS = ("all", "calls")
DEFAULT_ACTION_FORECAST_CALLS_PROMPT = (
    "Plan ahead: give the next {k} tool calls you will make to make progress on the task, "
    "one tool call per reply, starting with the one you make right now. Reply with the "
    "tool call only."
)
ACTION_FORECAST_NEXT_CALL_PROMPT = "Next tool call?"


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
# than the ReAct instr/ack/obs layout.
#
# The target is a LITERAL SUBSTRING of the turn the policy generated, never re-encoded:
# the first ``<tool_call>...</tool_call>`` block exactly as written (newlines, spacing,
# prose around it dropped the way alfworld drops the Thought), or the message text
# unchanged. This is the rule every other env's target already follows (``go to shelf 1``
# is cut out of the turn, not rewritten). Two re-serialized targets were tried here and
# both bled into policy turns at the original forecast dose: bare JSON (19.5% of turns by
# step 5, tool calls 29% -> 5%), then compact one-line JSON / ``say:`` one-liners
# (compact-JSON calls 0% -> 46% -> 96.5% of policy calls over steps 5-7, entropy 0.56 ->
# 0.81, tool-call share 27% -> 16%). The policy writes 100% of its calls in the template's
# newline-wrapped, space-separated form; a target in any other form is a second surface
# distribution for it to drift toward. Literal targets are multi-line, so native envs use
# layout='turns'.
_NATIVE_TOOL_ENVS = ("tau2",)
_TOOL_CALL_SPAN_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def _native_action(assistant_text: str) -> str:
    text = assistant_text or ""
    if not text.strip():
        return ""
    span = _TOOL_CALL_SPAN_RE.search(text)
    return span.group(0) if span else text

def _is_native_layout(convo: List[Dict[str, str]]) -> bool:
    """A system-first conversation is the native tool-calling layout; the ReAct layout
    always starts with the user instruction."""
    return bool(convo) and convo[0].get('role') == 'system'


def extract_action(assistant_text: str, env: str = "") -> str:
    """The bare action command from an assistant turn (drops the Thought).

    Take the first non-empty line after ``Action:``; fall back to the last
    non-empty line for bare-action envs.
    Returns '' for empty/degenerate turns (e.g. the trailing terminal turn).

    For ``env`` in _CODE_AS_ACTION_ENVS, return the last fenced code block instead,
    matching how AppWorldEnvClient.step extracts the code it executes, so the
    forecast target is exactly the action that ran. ``env=""`` (the default) keeps
    the original behaviour for every other env. For the native tool-calling envs the
    action is the literal executed span of the turn (see _native_action).
    """
    if (env or "").lower() in _CODE_AS_ACTION_ENVS:
        blocks = _FENCE_RE.findall(assistant_text or "")
        return blocks[-1].strip() if blocks else ""
    if (env or "").lower() in _NATIVE_TOOL_ENVS:
        return _native_action(assistant_text)
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
_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
# A call to a tool the agent does not have ("Error: Tool 'check_status_bar' not found.") --
# in telecom mostly the customer's own device tools. In v11's wins 46% of telecom's call
# targets were such calls, and 47 of its 126 samples with a call had no other call.
_TAU2_TOOL_NOT_FOUND_RE = re.compile(r"^\s*error:\s*tool\s+'[^']*'\s+not found", re.I)


def _native_tool_name(action: str) -> Optional[str]:
    """Tool name of a native call target (a literal ``<tool_call>...</tool_call>`` span),
    None for a message. An unnamed call is identified by its whole text, so only an
    identical later call can count as its redo."""
    if not (action.startswith("<tool_call>") and action.endswith("</tool_call>")):
        return None
    m = _TOOL_NAME_RE.search(action)
    return m.group(1) if m else action


# Tool-error classes, measured over 271 real tau2 tool errors (38.9% of all tool
# results): the class decides whether the action is worth forecasting at all.
#   missing (20.7%)  the tool does not exist -- mostly telecom, where the policy calls
#                    the CUSTOMER's device tools. Nothing correct to learn from it.
#   badargs (1.5%)   right tool, wrong arguments (TypeError, argument constraints).
#                    The call itself is broken, so it is never a target.
#   entity  (69.7%)  "Order not found" -- the call is well formed, the thing is not
#                    there. Usually informative (ask the customer, look it up another
#                    way), so it is kept unless the same tool later succeeded.
#   rule    (6.6%)   "Non-pending order cannot be cancelled" -- the domain refuses.
#                    The highest-information failure there is: it tells the policy the
#                    request is impossible, so it is ALWAYS kept.
_ERR_MISSING_TOOL = re.compile(r"^\s*error:\s*tool\s+'[^']*'\s+not found", re.I)
_ERR_BAD_ARGS = re.compile(r"unexpected keyword argument|missing \d+ required|"
                           r"takes \d+ positional|could not parse arguments|"
                           r"validation error|should match", re.I)
_ERR_ENTITY = re.compile(r"not found", re.I)


def classify_tool_error(result: str, is_error: Optional[bool] = None) -> str:
    """'ok' | 'missing' | 'badargs' | 'entity' | 'rule' for one tool result.

    ``is_error`` is tau2's own ToolMessage.error, carried out of the env server; the
    text is only used to tell the failure classes apart. Without the flag (older
    rollout dumps, other envs) the "Error" prefix stands in for it.
    """
    text = result or ""
    failed = is_error if is_error is not None else text.strip().lower().startswith(_TAU2_INVALID_PREFIXES)
    if not failed:
        return "ok"
    if _ERR_MISSING_TOOL.match(text):
        return "missing"
    if _ERR_BAD_ARGS.search(text):
        return "badargs"
    if _ERR_ENTITY.search(text):
        return "entity"
    return "rule"


def has_executed_action(messages) -> bool:
    """True if the trajectory actually ran a tool call (a tool-role result came back).

    A τ² episode can WIN without doing anything: 17 of the 178 training tasks (13 of
    airline's 30) are "the customer asks for something the policy forbids", their
    ground-truth actions are all read-only, and the reward compares the database
    against that ground truth -- so an agent that only talks matches it, the customer
    says ###STOP###, and the episode scores 1.0. Measured on the v10 rollouts: winning
    episodes averaged 1.3 tool calls against 3.2 for losing ones, and by step 15, 63% of
    the wins had made no call at all. Distilling those wins is what teaches the policy
    to stop acting, so they are dropped from the forecast's source.
    """
    return any((m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)) == 'tool'
               for m in _to_chat_list(messages))


def _turn_results(convo, ai):
    """Every observation message produced by action turn ``ai`` -- a turn with several
    tool calls gets one result per call, so the run of tool/user messages after it is
    what the action actually produced."""
    out = []
    j = ai + 1
    while j < len(convo) and convo[j].get('role') in ('tool', 'user'):
        out.append(convo[j])
        if convo[j].get('role') == 'user':      # the customer's reply ends the run
            break
        j += 1
    return out


def _native_effective(convo, action_idxs, actions_seq, env: str) -> List[bool]:
    """skip_invalid for the native tool-calling protocol: an action is dropped only if it
    was wasted AND redone later -- the meaning skip_invalid has in the ReAct envs
    ("Nothing happens.", then the corrected action).

    * A tool call whose result is an Error is dropped only if the same tool is called
      again later in the trajectory and that call succeeds. A failed call that is not
      redone is kept: in tau2 it is usually informative ("user not found" -> look the user
      up another way; "non-pending order cannot be cancelled" -> tell the customer), and it
      is part of the path that won.
    * A call to a tool the agent does not have ("Error: Tool '...' not found.") is always
      dropped: there is no correct version of it to learn.
    * A message is dropped only if it repeats an earlier message of the trajectory
      word for word (whitespace-normalized).

    The previous rule dropped every call that returned an Error and never a message. In
    winning trajectories 27.9% of calls return an Error but only 37% of those are redone,
    so it cut the tool-call share of the targets to 24.7% against the policy's 31.3%
    (30.3% with no skipping, 29.5% with this rule), biasing the forecast toward talking.
    """
    names, kinds = [], []
    for ai, a in zip(action_idxs, actions_seq):
        name = _native_tool_name(a or "")
        names.append(name)
        # A turn can make several calls, so its outcome is the WORST of its results:
        # judging it by the first one alone made a turn whose second call failed look
        # clean (and the other way round).
        cls = [classify_tool_error(m.get('content', ''), m.get('error'))
               for m in _turn_results(convo, ai) if m.get('role') == 'tool']
        order = {"ok": 0, "entity": 1, "rule": 2, "badargs": 3, "missing": 4}
        kinds.append(max(cls, key=lambda c: order[c]) if cls else "ok")
    valid: List[bool] = []
    seen_messages = set()
    for j, a in enumerate(actions_seq):
        if not a:
            valid.append(True)               # empty turns are filtered by the caller anyway
        elif names[j] is not None:
            k = kinds[j]
            if k in ("missing", "badargs"):
                valid.append(False)          # nothing correct in the call itself
            elif k == "rule":
                valid.append(True)           # the domain's refusal is information
            elif k == "entity":
                # wasted only if the same tool is called again later and works
                valid.append(not any(names[i] == names[j] and kinds[i] == "ok"
                                     for i in range(j + 1, len(actions_seq))))
            else:
                valid.append(True)
        else:
            key = " ".join(a.split())
            valid.append(key not in seen_messages)
            seen_messages.add(key)
    return valid


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
                       env: str = "alfworld", targets: str = "all") -> List[Dict[str, object]]:
    """For each action turn t, return per-step targets.

    {'prefix_end': idx, 'actions': [a_t..a_{t+K-1}], 'action_turns': [..], 'src_turn': n}

    ``prefix_end`` is the conversation index of the obs the action responds to
    (= action_turn_index - 1); the SFT prefix is convo[:prefix_end+1].
    ``actions`` are the realized next-K bare action commands (current included).
    Steps with no future action are skipped.

    ``skip_invalid`` (default off): drop actions whose RESULT observation signals an
    invalid / no-effect outcome (per-env, see is_invalid_outcome) — the forecast
    target then contains only the next-K *effective* actions (looking past the
    skipped ones). ``env`` selects the invalid-outcome patterns. For the native
    tool-calling envs the rule is trajectory-level: see _native_effective.

    ``targets='calls'`` (native tool-calling envs only): anchors are the tool-call turns
    and ``actions`` are the next K tool calls from there on (the current one included,
    unless skip_invalid drops it), looking past messages; see ACTION_FORECAST_TARGETS.
    """
    if targets not in ACTION_FORECAST_TARGETS:
        raise ValueError(f"action_forecast targets must be one of {ACTION_FORECAST_TARGETS}, got {targets!r}")
    calls_only = targets == "calls"
    if calls_only and (env or "").lower() not in _NATIVE_TOOL_ENVS:
        raise ValueError(f"action_forecast targets='calls' needs a native tool-calling env "
                         f"{_NATIVE_TOOL_ENVS}, got env={env!r}")
    convo = _to_chat_list(messages)
    action_idxs = _action_turn_indices(convo)
    actions_seq = [extract_action(convo[ai]['content'], env=env) for ai in action_idxs]

    # A turn with SEVERAL tool calls is not representable as a target: extract_action cuts
    # the first <tool_call> block out of it, while the env server now executes every call
    # in the turn (agentenv_tau2 `native_multicall`, as tau2 does at evaluation). Rather
    # than re-serialise the turn -- which would break the rule that a target is a literal
    # substring of what the policy wrote, the rule three earlier versions were spent
    # establishing -- such a turn is left out of the forecast: it is neither an anchor nor
    # a target item. Measured on the v10 rollouts these are 0.4-1.0% of assistant turns,
    # so the cost is small; `policy/multicall_share` is the metric that says when it stops
    # being small.
    if (env or "").lower() in _NATIVE_TOOL_ENVS:
        for j, ai in enumerate(action_idxs):
            if len(_TOOL_CALL_SPAN_RE.findall(convo[ai].get('content') or '')) >= 2:
                actions_seq[j] = ""

    if skip_invalid and (env or "").lower() in _NATIVE_TOOL_ENVS:
        valid_seq = _native_effective(convo, action_idxs, actions_seq, env)
    elif skip_invalid:
        # per action turn, the RESULT obs = the next user message after it
        valid_seq = []
        for ai in action_idxs:
            res = (convo[ai + 1]['content'] if (ai + 1 < len(convo)
                   and convo[ai + 1].get('role') in ('user', 'tool')) else '')
            valid_seq.append(not is_invalid_outcome(res, env))
    else:
        valid_seq = [True] * len(action_idxs)

    out: List[Dict[str, object]] = []
    if calls_only:
        is_call = [_native_tool_name(a or "") is not None for a in actions_seq]
        for n, ai in enumerate(action_idxs):
            if not is_call[n]:
                continue                      # anchors are the policy's tool-call turns
            sel = [j for j in range(n, len(actions_seq))
                   if actions_seq[j] and valid_seq[j] and is_call[j]][:k]
            if sel:
                out.append({'prefix_end': ai - 1, 'actions': [actions_seq[j] for j in sel],
                            'action_turns': sel, 'src_turn': n})
        return out
    native = (env or "").lower() in _NATIVE_TOOL_ENVS
    for n, ai in enumerate(action_idxs):
        # No anchor on a turn whose own action is not representable (empty turn, or the
        # multi-call turn blanked above): the prompt says "starting with the action you
        # take right now", and the next action is not that.
        if native and not actions_seq[n]:
            continue
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
    targets: str = "all",
    skip_no_call: bool = False,
    stats: Optional[dict] = None,
) -> List[Dict[str, "object"]]:
    """Per-step teacher-forced forecast-SFT samples for one trajectory.

    ``skip_no_call`` (native tool-calling envs): a sample whose K target actions contain no
    tool call is invalid and skipped, so a trajectory without any tool call gives no sample
    and message-only stretches are not forecast; ``stats['skipped_no_call']`` counts them.
    v10/v11 learned from the wins of groups with mixed outcomes, 23-27% of airline and
    44-49% of telecom such wins had no tool call at all (the customer did the device steps,
    or a refusal), and the policy's talk-only conversations grew in every domain -- retail
    from 2% to 18% in v10 although its own forecast wins always called tools.

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
    # 'list' is allowed for the native envs again: K <tool_call> blocks in one assistant
    # message ARE Qwen's rendering of several tool calls in one turn, and since the env
    # server parses a native turn whole (agentenv_tau2 `native_multicall`), such a turn
    # now executes every call -- the same behaviour tau2's own agent has at evaluation.
    # The rest of the sample construction is unchanged; the targets stay literal spans of
    # the policy's turns, joined by a newline the way the template joins parallel calls.

    if skip_no_call and (env or "").lower() not in _NATIVE_TOOL_ENVS:
        raise ValueError(f"action_forecast skip_no_call needs a native tool-calling env "
                         f"{_NATIVE_TOOL_ENVS}, got env={env!r}")

    convo = _to_chat_list(messages)
    samples: List[Dict[str, object]] = []
    turns_prompt, next_prompt = ((DEFAULT_ACTION_FORECAST_CALLS_PROMPT, ACTION_FORECAST_NEXT_CALL_PROMPT)
                                 if targets == "calls" else
                                 (DEFAULT_ACTION_FORECAST_TURNS_PROMPT, ACTION_FORECAST_NEXT_PROMPT))
    for tgt in build_action_targets(messages, k=k, skip_invalid=skip_invalid, env=env, targets=targets):
        items = (tgt.get('actions') or [])[:k]   # clamp to what's available (<= k)
        if not items:
            continue
        if skip_no_call and not any(_native_tool_name(a) is not None for a in items):
            if stats is not None:
                stats['skipped_no_call'] = stats.get('skipped_no_call', 0) + 1
            continue
        realized = len(items)
        # synthetic prompt formatted with the REALIZED count
        prefix = list(convo[:tgt['prefix_end'] + 1])
        if layout == "turns":
            prefix.append({'role': 'user', 'content': turns_prompt.format(k=realized)})
            target_msgs = []
            for j, a in enumerate(items):
                if j:
                    target_msgs.append({'role': 'user', 'content': next_prompt})
                target_msgs.append({'role': 'assistant', 'content': a})
            s = encode_sft_sample(tokenizer, prefix, target_msgs,
                                  max_length=max_length, min_target_tokens=min_target_tokens,
                                  assistant_only=True, return_span_starts=True)
            if s is not None:
                # decision_kind marks the first token of each target turn -- the token that
                # decides "tool call or message": 1 = call (<tool_call>), 2 = message.
                import torch
                kinds = torch.zeros(s['input_ids'].shape, dtype=torch.int8)
                for st, a in zip(s.pop('span_starts'), items):
                    if 0 <= st < kinds.numel():
                        kinds[st] = 1 if _native_tool_name(a) is not None else 2
                s['decision_kind'] = kinds
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
                      min_target_tokens: int = 1, assistant_only: bool = False,
                      return_span_starts: bool = False):
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
    span_starts: List[int] = []   # first token of each assistant target turn, in order
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
            span_starts.append(start)
    else:
        loss_mask[prefix_len:] = 1
        span_starts.append(prefix_len)

    target_len = int(loss_mask.sum().item())
    if target_len < min_target_tokens:
        return None

    drop = 0
    if input_ids.size(0) > max_length:
        drop = input_ids.size(0) - max_length
        input_ids = input_ids[drop:]
        attention_mask = attention_mask[drop:]
        loss_mask = loss_mask[drop:]
        if loss_mask.sum().item() < min_target_tokens:
            return None

    out = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
    }
    if return_span_starts:
        # left truncation shifts every index; a start cut off by it becomes -1
        out['span_starts'] = [s - drop if s - drop >= 0 else -1 for s in span_starts]
    return out


ACTION_FORECAST_GATES = ("wins", "all", "mixed")

# Upper bound on the decision-token weight from call balancing (bounds the variance when
# one kind is rare among the targets; exact neutrality is lost only when it binds).
ACTION_FORECAST_BALANCE_WMAX = 5.0


def call_balance_weights(p: float, q: float, w_max: float = ACTION_FORECAST_BALANCE_WMAX):
    """Weights (w_call, w_msg) for the first token of call / message target turns that make
    the forecast neutral on the call-vs-message decision.

    ``p``: share of the policy's own turns that are tool calls (this step's rollouts);
    ``q``: share of the (sample-weighted) forecast target turns that are tool calls.
    The CE on the decision token pushes P(call) toward 1 on call targets and toward 0 on
    message targets; the pushes cancel at the policy's current rate when
    w_call * q * (1 - p) == w_msg * (1 - q) * p, which w_call = p/q, w_msg = (1-p)/(1-q)
    satisfy. If one kind is absent from the targets the pushes cannot be balanced, and
    the only neutral choice is not to train the decision token at all: (0, 0). The
    content tokens of every target turn are still trained.
    """
    if q <= 0.0 or q >= 1.0:
        return 0.0, 0.0
    return min(p / q, w_max), min((1.0 - p) / (1.0 - q), w_max)


def _policy_call_share(messages_list) -> Optional[float]:
    """Share of assistant turns in this step's rollouts that contain a tool call."""
    turns = calls = 0
    for msgs in messages_list:
        if msgs is None:
            continue
        for m in _to_chat_list(msgs):
            if m.get('role') == 'assistant':
                turns += 1
                calls += _TOOL_CALL_SPAN_RE.search(m.get('content') or '') is not None
    return (calls / turns) if turns else None


def select_forecast_trajectories(n: int, rewards=None, gate: str = "wins",
                                 success_threshold: float = 0.5, group_ids=None,
                                 group_norm: bool = False):
    """Which of ``n`` trajectories feed the forecast SFT, and per-step group counts.

    'wins': reward > success_threshold. 'all': every trajectory (unless group_norm,
    which distils successes only). 'mixed': the wins of groups that contain at least one
    win and one loss -- the groups GRPO learns from. Returns (keep: List[bool], stats).
    """
    from collections import Counter
    if gate not in ACTION_FORECAST_GATES:
        raise ValueError(f"action_forecast gate must be one of {ACTION_FORECAST_GATES}, got {gate!r}")
    succ = []
    for i in range(n):
        r = rewards[i] if (rewards is not None and i < len(rewards)) else 0.0
        succ.append(r is not None and float(r) > success_threshold)
    stats = {}
    if gate == "mixed":
        if group_ids is None:
            raise ValueError("action_forecast gate='mixed' needs group_ids (the GRPO uid)")
        wins, size = Counter(), Counter()
        for i in range(n):
            wins[group_ids[i]] += succ[i]
            size[group_ids[i]] += 1
        mixed = {g for g in size if 0 < wins[g] < size[g]}
        keep = [succ[i] and group_ids[i] in mixed for i in range(n)]
        stats = {"action_forecast/n_groups": float(len(size)),
                 "action_forecast/n_groups_mixed": float(len(mixed)),
                 "action_forecast/n_groups_allwin_skipped": float(sum(wins[g] == size[g] for g in size)),
                 "action_forecast/n_wins_skipped": float(sum(succ) - sum(keep))}
    elif gate == "wins" or group_norm:
        keep = list(succ)
    else:
        keep = [True] * n
    return keep, stats


def _trained_spans(loss_mask) -> List[tuple]:
    """(start, end) of each contiguous run of trained positions -- one per target turn in the
    'turns' layout (the filler prompts between turns are untrained)."""
    idx = [i for i, v in enumerate(loss_mask.tolist()) if v]
    spans = []
    for i in idx:
        if spans and i == spans[-1][1]:
            spans[-1] = (spans[-1][0], i + 1)
        else:
            spans.append((i, i + 1))
    return spans


def action_length_weights(loss_mask):
    """Per-token weights giving every target action the same total weight in its sample's
    loss. The forecast CE is a mean over the sample's N trained tokens; a token of action j
    (a trained span of n_j tokens, K spans) gets N / (K * n_j), so the sample's loss becomes
    the mean over its K actions of each action's mean token CE (exact at one sample per
    micro-batch, the setting every forecast run uses). Untrained positions keep weight 1.

    Why: tool calls are short and messages long -- in v10's forecast data a call target had
    a median of 33 tokens and a message 111 -- so under the plain token mean the 26.7% of
    target turns that are calls carried 9.9% of the trained tokens, and on average 16% of a
    sample's loss. With the 'list' layout the K actions share one span and the weights are 1.
    """
    import torch
    tw = torch.ones(loss_mask.shape, dtype=torch.float32)
    spans = _trained_spans(loss_mask)
    if spans:
        n_tok = sum(b - a for a, b in spans)
        for a, b in spans:
            tw[a:b] = n_tok / (len(spans) * (b - a))
    return tw


def _span_kinds(decision_kind, loss_mask):
    """Per-position kind of the target turn a trained token belongs to (1 = tool call,
    2 = message, 0 = untrained), from the kind marked on each turn's first token."""
    import torch
    kinds = torch.zeros(loss_mask.shape, dtype=torch.int8)
    for a, b in _trained_spans(loss_mask):
        kinds[a:b] = int(decision_kind[a])
    return kinds


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
    balance_calls: bool = False,
    targets: str = "all",
    length_norm: bool = False,
    skip_no_call: bool = False,
    require_action: bool = False,
):
    """Padded forecast-SFT batch over trajectories, with win/all gating.

    ``balance_calls`` (native tool-calling envs, layout='turns'): reweight the first
    token of each target turn -- the token that decides tool call vs message -- so the
    forecast cannot move the policy's call rate (see call_balance_weights); every other
    target token keeps weight 1. The call rate is then set by GRPO alone. Without it, the
    wins that feed the forecast are call-poor in tau2 (refusal and guidance tasks are the
    ones a weak policy wins; inside a mixed group the win calls less than the losses), the
    forecast pulled the policy's call rate from ~30% to ~0 within 15 steps (v8), and once
    no task had a mixed outcome GRPO had no signal left to recover it.

    ``targets='calls'`` forecasts the policy's tool calls only (see ACTION_FORECAST_TARGETS);
    with balance_calls every decision token then has weight 0.

    ``skip_no_call``: skip every sample whose K target actions contain no tool call (see
    build_action_forecast_samples).

    ``length_norm``: every target action gets the same weight in its sample's loss,
    whatever its length (see action_length_weights). With balance_calls, the balance is
    then computed on each decision token's actual weight in the loss, so it stays neutral.

    gate='wins' keeps only trajectories whose reward > success_threshold (needs
    ``rewards`` aligned to ``messages_list``); gate='all' keeps everything.
    gate='mixed' keeps the wins of the GRPO groups that also have a loss (needs
    ``group_ids``), i.e. only where GRPO itself has a learning signal: an all-win group
    has zero advantage, so GRPO leaves it alone, while gate='wins' still distils it. In
    tau2 the all-win groups are the tasks the policy already solves, mostly ones that
    need no tool call (refusals, read-only questions): 8.5-21.5% of their targets are
    calls against the policy's ~30%, and they grew from 23% to 63% of the forecast data
    as the v7 run lost its tool use, down to a step with no mixed group at all where the
    forecast SFT alone kept pushing. With 'mixed' the forecast is empty whenever GRPO is.
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

    keep, group_stats = select_forecast_trajectories(
        n=len(messages_list), rewards=rewards, gate=gate, success_threshold=success_threshold,
        group_ids=group_ids, group_norm=group_norm)

    all_samples: List[Dict[str, object]] = []
    skip_stats: Dict[str, int] = {}
    n_traj_used = 0
    n_traj_considered = 0
    n_traj_no_action = 0
    traj_records = []   # (group_id, n_samples, start_idx) per distilled traj (group_norm)
    for i, messages in enumerate(messages_list):
        if messages is None or not keep[i]:
            continue
        # A win that never ran a tool call is a talk-only win (see has_executed_action):
        # distilling it teaches the policy to stop acting, which is the failure the τ²
        # forecast runs kept reproducing.
        if require_action and (env or "").lower() in _NATIVE_TOOL_ENVS and not has_executed_action(messages):
            n_traj_no_action += 1
            continue
        n_traj_considered += 1
        traj_samples = build_action_forecast_samples(
            messages=messages, tokenizer=tokenizer, k=k,
            max_length=max_length,
            skip_invalid=skip_invalid, env=env, layout=layout, targets=targets,
            skip_no_call=skip_no_call, stats=skip_stats)
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
            seq_key = tuple(extract_action(_cv[a].get("content", ""), env=env) for a in _action_turn_indices(_cv))
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

    import torch
    # Per-action length normalisation, then call/message decision balancing (both after
    # group_norm, so they use the final sample weights).
    balance_meta = {}
    if length_norm:
        for s in all_samples:
            s['token_weight'] = action_length_weights(s['loss_mask'])
        balance_meta["action_forecast/length_norm"] = 1.0
    if balance_calls:
        import torch
        p = _policy_call_share(messages_list)
        n_c = n_m = 0.0
        for s in all_samples:
            dk = s.get('decision_kind')
            if dk is None:
                continue
            lw = float(s.get('loss_weight', 1.0))
            if length_norm:
                # a decision token's actual weight in the loss: loss_weight * token_weight / N
                tw0 = s['token_weight']
                n_tok = max(float(s['loss_mask'][1:].sum()), 1.0)
                n_c += lw * float(tw0[dk == 1].sum()) / n_tok
                n_m += lw * float(tw0[dk == 2].sum()) / n_tok
            else:
                n_c += lw * int((dk == 1).sum())
                n_m += lw * int((dk == 2).sum())
        balance_meta["action_forecast/balance_calls"] = 1.0
        if p is not None and (n_c + n_m) > 0:
            q = n_c / (n_c + n_m)
            w_c, w_m = call_balance_weights(p, q)
            for s in all_samples:
                dk = s.get('decision_kind')
                if dk is None:
                    continue
                tw = s['token_weight'].clone() if 'token_weight' in s else torch.ones(dk.shape, dtype=torch.float32)
                tw[dk == 1] *= w_c
                tw[dk == 2] *= w_m
                s['token_weight'] = tw
            eff = (w_c * n_c) / (w_c * n_c + w_m * n_m) if (w_c * n_c + w_m * n_m) > 0 else float('nan')
            balance_meta.update({
                "action_forecast/policy_call_share": float(p),
                "action_forecast/target_call_share": float(q),
                "action_forecast/balanced_call_share": float(eff),
                "action_forecast/w_call": float(w_c),
                "action_forecast/w_msg": float(w_m),
            })
    # Where the forecast's weight goes: tool-call tokens as a share of the trained tokens,
    # and of the loss weight once token/sample weights apply (native 'turns' samples only).
    if any(s.get('decision_kind') is not None for s in all_samples):
        raw_c = raw_all = eff_c = eff_all = 0.0
        for s in all_samples:
            dk = s.get('decision_kind')
            if dk is None:
                continue
            lm = s['loss_mask'].to(bool)
            kinds = _span_kinds(dk, s['loss_mask'])
            tw = s.get('token_weight')
            tw = tw if tw is not None else torch.ones(lm.shape, dtype=torch.float32)
            lw = float(s.get('loss_weight', 1.0))
            n_tok = max(float(s['loss_mask'][1:].sum()), 1.0)
            raw_c += float(((kinds == 1) & lm).sum()); raw_all += float(lm.sum())
            eff_c += lw * float(tw[(kinds == 1) & lm].sum()) / n_tok
            eff_all += lw * float(tw[lm].sum()) / n_tok
        if raw_all > 0:
            balance_meta["action_forecast/call_token_share"] = raw_c / raw_all
        if eff_all > 0:
            balance_meta["action_forecast/call_weight_share"] = eff_c / eff_all
    for s in all_samples:
        s.pop('decision_kind', None)

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
            "action_forecast/gate_mixed": 1.0 if gate == "mixed" else 0.0,
            **group_stats,
            "action_forecast/k_mean": float(k_mean),
            "action_forecast/skip_invalid": 1.0 if skip_invalid else 0.0,
            "action_forecast/group_norm": 1.0 if group_norm else 0.0,
            "action_forecast/layout_turns": 1.0 if layout == "turns" else 0.0,
            "action_forecast/targets_calls": 1.0 if targets == "calls" else 0.0,
            "action_forecast/skip_no_call": 1.0 if skip_no_call else 0.0,
            "action_forecast/n_skipped_no_call": float(skip_stats.get('skipped_no_call', 0)),
            "action_forecast/require_action": 1.0 if require_action else 0.0,
            "action_forecast/n_traj_dropped_no_action": float(n_traj_no_action),
            **balance_meta,
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


# ---- policy-format monitoring ---------------------------------------------------------
# What the forecast can break, watched per step. The τ² forecast runs all failed the same
# way: the policy drifted toward a surface form the target used and the environment does
# not execute (bare JSON in v2, compact JSON in v5, several calls in one turn in v4), and
# that showed in the rollouts long before it showed in the reward. sciworld's own forecast
# run does the same thing at 27x the control's rate (bare commands 0.22% -> 5.84% of turns,
# every one of them a wasted turn), so this is not a τ²-only risk.
_BARE_JSON_CALL_RE = re.compile(r'^\s*\{\s*"(?:name|arguments)"\s*:', re.M)
_OPEN_TAG_RE = re.compile(r"<tool_call>")
_CLOSE_TAG_RE = re.compile(r"</tool_call>")


def policy_format_metrics(messages_list, prefix: str = "policy") -> Dict[str, float]:
    """Per-step shares of the policy's own turns, for the native tool-calling protocol.

    ``<prefix>/call_turn_share``   turns that carry at least one tool call
    ``<prefix>/multicall_share``   turns with 2+ calls (the env now runs them all, but a
                                   turn the reward never asked for is still a drift signal)
    ``<prefix>/malformed_share``   turns whose <tool_call> tags do not pair up -- seen in
                                   the v10 rollouts, where three opening tags and one
                                   closing tag sent the whole turn to the customer as text
    ``<prefix>/bare_json_share``   a call written without the tags (the v2 failure)
    ``<prefix>/no_call_episodes``  episodes that never called a tool
    """
    turns = calls = multi = malformed = bare = 0
    eps = nocall_eps = 0
    for msgs in messages_list or []:
        if msgs is None:
            continue
        eps += 1
        ep_calls = 0
        for m in _to_chat_list(msgs):
            if m.get('role') != 'assistant':
                continue
            c = m.get('content') or ''
            turns += 1
            n_open, n_close = len(_OPEN_TAG_RE.findall(c)), len(_CLOSE_TAG_RE.findall(c))
            n_span = len(_TOOL_CALL_SPAN_RE.findall(c))
            if n_span:
                calls += 1
                ep_calls += n_span
            if n_span >= 2:
                multi += 1
            if n_open != n_close:
                malformed += 1
            if _BARE_JSON_CALL_RE.search(_TOOL_CALL_SPAN_RE.sub('', c)):
                bare += 1
        nocall_eps += (ep_calls == 0)
    t = max(turns, 1)
    return {f"{prefix}/call_turn_share": calls / t,
            f"{prefix}/multicall_share": multi / t,
            f"{prefix}/malformed_share": malformed / t,
            f"{prefix}/bare_json_share": bare / t,
            f"{prefix}/no_call_episodes": nocall_eps / max(eps, 1),
            f"{prefix}/assistant_turns": float(turns)}
