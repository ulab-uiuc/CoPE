"""Action forecasting under τ²-bench's native tool-calling protocol.

The ReAct helpers assume [instr, ack, obs, action, obs, action, ...] with ``Action:``
lines; a native conversation is [system, user, assistant, tool|user, assistant, ...]
and an action is a <tool_call> block or a customer message. These tests pin the
native path and check the ReAct path is untouched.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verl.agent_trainer.ppo.action_forecast import (  # noqa: E402
    _action_turn_indices, build_action_targets, extract_action, is_invalid_outcome)

NATIVE = [
    {"role": "system", "content": "You are a customer service agent.\n<policy>P</policy>"},
    {"role": "user", "content": "Hi, I want to cancel my order."},
    {"role": "assistant", "content": "Sure. <tool_call>\n{\"name\": \"find_user_id_by_email\", \"arguments\": {\"email\": \"a@b.c\"}}\n</tool_call>"},
    {"role": "tool", "content": "yara_muller_8652"},
    {"role": "assistant", "content": "Found you.\nWhich order?\n"},
    {"role": "user", "content": "#W1"},
    {"role": "assistant", "content": "<tool_call>{\"name\": \"cancel_pending_order\", \"arguments\": {\"order_id\": \"#W1\", \"reason\": \"no longer needed\"}}</tool_call>"},
    {"role": "tool", "content": "Error: Non-pending order cannot be cancelled"},
    {"role": "assistant", "content": "<tool_call>{\"name\": \"get_order_details\", \"arguments\": {\"order_id\": \"#W1\"}}</tool_call>"},
    {"role": "tool", "content": "{\"order_id\": \"#W1\", \"status\": \"delivered\"}"},
    {"role": "assistant", "content": "That order was delivered, so it cannot be cancelled."},
    {"role": "user", "content": "###STOP###"},
]

REACT = [
    {"role": "user", "content": "instruction"}, {"role": "assistant", "content": "Ok."},
    {"role": "user", "content": "obs0"}, {"role": "assistant", "content": "Thought: x\nAction: go north"},
    {"role": "user", "content": "obs1"}, {"role": "assistant", "content": "Action: take key"},
]


def test_native_layout_every_assistant_turn_is_an_action():
    assert _action_turn_indices(NATIVE) == [2, 4, 6, 8, 10]


def test_react_layout_unchanged():
    assert _action_turn_indices(REACT) == [3, 5]
    assert extract_action(REACT[3]["content"]) == "go north"


def _call(action):
    assert action.startswith("<tool_call>") and action.endswith("</tool_call>") and "\n" not in action
    return json.loads(action[len("<tool_call>"):-len("</tool_call>")])


def test_extract_native_tool_call_is_one_line_in_the_executed_form():
    # the target is exactly what the policy must emit for the call to run, so a forecast
    # that bleeds into an ordinary turn is still a valid tool call rather than bare JSON
    out = extract_action(NATIVE[2]["content"], env="tau2")
    assert _call(out) == {"name": "find_user_id_by_email", "arguments": {"email": "a@b.c"}}


def test_extract_native_message_is_one_line_with_say_prefix():
    assert extract_action(NATIVE[4]["content"], env="tau2") == "say: Found you. Which order?"
    assert extract_action("", env="tau2") == ""


def test_tau2_invalid_outcomes():
    assert is_invalid_outcome("Error: Non-pending order cannot be cancelled", env="tau2")
    assert is_invalid_outcome("Invalid turn: reply with a <tool_call> or a message", env="tau2")
    assert not is_invalid_outcome("{\"status\": \"ok\", \"error_count\": 0}", env="tau2")
    assert not is_invalid_outcome("I don't have my order id.", env="tau2")


def test_native_targets_skip_invalid_looks_past_the_failed_call():
    tg = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2")
    by_turn = {t["prefix_end"] + 1: t["actions"] for t in tg}
    # from the message turn (idx 4): next effective actions skip the failed cancel (idx 6)
    names = [a.split(":", 1)[0] if a.startswith("say") else _call(a)["name"] for a in by_turn[4]]
    assert names == ["say", "get_order_details", "say"]
    # without skip_invalid the failed call is part of the target
    tg2 = build_action_targets(NATIVE, k=3, skip_invalid=False, env="tau2")
    by2 = {t["prefix_end"] + 1: t["actions"] for t in tg2}
    assert _call(by2[6][0])["name"] == "cancel_pending_order"
    # every target's prefix ends on the observation the action answers
    assert all(NATIVE[t["prefix_end"]]["role"] in ("user", "tool") for t in tg)


# ---- layout='turns': K assistant turns, loss on the assistant turns only ----------------

import os  # noqa: E402

import pytest  # noqa: E402

from verl.agent_trainer.ppo.action_forecast import (  # noqa: E402
    ACTION_FORECAST_NEXT_PROMPT, build_action_forecast_samples)


@pytest.fixture(scope="module")
def tok():
    path = os.environ.get("COPE_TEST_TOKENIZER")
    if not path:
        pytest.skip("set COPE_TEST_TOKENIZER to a local Qwen2.5 snapshot")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(path)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        build_action_forecast_samples(NATIVE, tokenizer=None, k=3, env="tau2", layout="lines")


def _masked_spans(sample):
    spans, cur = [], []
    for t, m in zip(sample["input_ids"].tolist(), sample["loss_mask"].tolist()):
        if m:
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def test_turns_layout_is_k_policy_shaped_turns(tok):
    samples = build_action_forecast_samples(NATIVE, tok, k=3, skip_invalid=True, env="tau2", layout="turns")
    assert samples
    s = samples[0]                       # forecast from the first action turn
    spans = _masked_spans(s)
    assert len(spans) == s["k_realized"] == 3
    expected = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2")[0]["actions"]
    for span, action in zip(spans, expected):
        text = tok.decode(span)
        # each trained span is exactly one action, closed like a policy turn
        assert text.startswith(action), text
        assert text.rstrip("\n").endswith("<|im_end|>"), text
        assert "\n" not in text.rstrip("\n").replace("<|im_end|>", "")
    # the filler prompt between turns is never trained, and the K actions are separate turns
    ids = s["input_ids"].tolist()
    filler = tok.encode(ACTION_FORECAST_NEXT_PROMPT, add_special_tokens=False)
    hits = [i for i in range(len(ids) - len(filler) + 1) if ids[i:i + len(filler)] == filler]
    assert len(hits) == 2
    for i in hits:
        assert not any(s["loss_mask"][i:i + len(filler)].tolist())
    full = tok.decode(ids)
    assert full.count("<|im_start|>assistant") == 3 + sum(1 for m in NATIVE[:2] if m["role"] == "assistant")


def test_list_layout_is_unchanged(tok):
    s = build_action_forecast_samples(NATIVE, tok, k=3, skip_invalid=True, env="tau2")[0]
    spans = _masked_spans(s)
    assert len(spans) == 1                # one assistant turn, K lines
    text = tok.decode(spans[0])
    # skip_invalid looks past the failed cancel: find_user, say, get_order_details
    assert text.count("<tool_call>") == 2 and text.count("say:") == 1
    assert text.rstrip("\n").endswith("<|im_end|>")


# ---- native_form='verbatim': the executed span, cut out of the turn unchanged -----------

def test_verbatim_form_is_a_literal_substring_of_the_turn():
    # tool-call turn: the <tool_call> block exactly as written (newlines, spaced JSON),
    # the prose before it dropped like alfworld drops the Thought
    out = extract_action(NATIVE[2]["content"], env="tau2", form="verbatim")
    assert out == "<tool_call>\n{\"name\": \"find_user_id_by_email\", \"arguments\": {\"email\": \"a@b.c\"}}\n</tool_call>"
    assert out in NATIVE[2]["content"]
    # message turn: the text unchanged -- no say: prefix, no truncation, no reflow
    assert extract_action(NATIVE[4]["content"], env="tau2", form="verbatim") == NATIVE[4]["content"]
    assert extract_action("  \n", env="tau2", form="verbatim") == ""
    # the executed form is untouched
    assert _call(extract_action(NATIVE[2]["content"], env="tau2"))["name"] == "find_user_id_by_email"


def test_verbatim_form_needs_the_turns_layout():
    with pytest.raises(ValueError):
        build_action_forecast_samples(NATIVE, tokenizer=None, k=3, env="tau2", layout="list", native_form="verbatim")


def test_verbatim_turns_train_exactly_the_policy_spans(tok):
    samples = build_action_forecast_samples(NATIVE, tok, k=3, skip_invalid=True, env="tau2",
                                            layout="turns", native_form="verbatim")
    s = samples[0]
    spans = _masked_spans(s)
    expected = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2", form="verbatim")[0]["actions"]
    assert len(spans) == len(expected) == 3
    for span, action in zip(spans, expected):
        text = tok.decode(span)
        assert text == action + "<|im_end|>\n", (text, action)
    # the target tokens are the policy's own tokens: encoding the literal span gives the same ids
    for span, action in zip(spans, expected):
        assert span[:-2] == tok.encode(action, add_special_tokens=False)
