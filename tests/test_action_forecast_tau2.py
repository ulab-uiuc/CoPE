"""Action forecasting under τ²-bench's native tool-calling protocol.

The ReAct helpers assume [instr, ack, obs, action, obs, action, ...] with ``Action:``
lines; a native conversation is [system, user, assistant, tool|user, assistant, ...]
and an action is a <tool_call> block or a customer message. These tests pin the
native path and check the ReAct path is untouched.

The invariant that matters: a forecast target is a LITERAL SUBSTRING of the turn the
policy generated -- never re-encoded -- and the trained tokens are that substring's
tokens. Everything else (turns layout, filler prompts never trained) follows from it.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verl.agent_trainer.ppo.action_forecast import (  # noqa: E402
    ACTION_FORECAST_NEXT_PROMPT, _action_turn_indices, build_action_forecast_samples,
    build_action_targets, extract_action, is_invalid_outcome)

CALL_1 = "<tool_call>\n{\"name\": \"find_user_id_by_email\", \"arguments\": {\"email\": \"a@b.c\"}}\n</tool_call>"
CALL_CANCEL = "<tool_call>{\"name\": \"cancel_pending_order\", \"arguments\": {\"order_id\": \"#W1\", \"reason\": \"no longer needed\"}}</tool_call>"
CALL_DETAILS = "<tool_call>{\"name\": \"get_order_details\", \"arguments\": {\"order_id\": \"#W1\"}}</tool_call>"

NATIVE = [
    {"role": "system", "content": "You are a customer service agent.\n<policy>P</policy>"},
    {"role": "user", "content": "Hi, I want to cancel my order."},
    {"role": "assistant", "content": "Sure. " + CALL_1},
    {"role": "tool", "content": "yara_muller_8652"},
    {"role": "assistant", "content": "Found you.\nWhich order?\n"},
    {"role": "user", "content": "#W1"},
    {"role": "assistant", "content": CALL_CANCEL},
    {"role": "tool", "content": "Error: Non-pending order cannot be cancelled"},
    {"role": "assistant", "content": CALL_DETAILS},
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


# ---- the target is a literal substring of the policy's turn --------------------------------

def test_tool_call_target_is_the_block_exactly_as_written():
    # newlines and spacing kept, the prose before it dropped like alfworld drops the Thought
    out = extract_action(NATIVE[2]["content"], env="tau2")
    assert out == CALL_1
    assert out in NATIVE[2]["content"]


def test_message_target_is_the_text_unchanged():
    # no prefix, no truncation, no reflow -- not even the trailing newline
    assert extract_action(NATIVE[4]["content"], env="tau2") == "Found you.\nWhich order?\n"
    long = "x" * 5000
    assert extract_action(long, env="tau2") == long
    assert extract_action("", env="tau2") == ""
    assert extract_action("  \n", env="tau2") == ""


def test_first_of_several_calls_is_the_target():
    # the client executes only the first call of a turn; so does the target
    out = extract_action(CALL_DETAILS + "\n" + CALL_CANCEL, env="tau2")
    assert out == CALL_DETAILS


def test_tau2_invalid_outcomes():
    assert is_invalid_outcome("Error: Non-pending order cannot be cancelled", env="tau2")
    assert is_invalid_outcome("Invalid turn: reply with a <tool_call> or a message", env="tau2")
    assert not is_invalid_outcome("{\"status\": \"ok\", \"error_count\": 0}", env="tau2")
    assert not is_invalid_outcome("I don't have my order id.", env="tau2")


def test_native_targets_skip_invalid_looks_past_the_failed_call():
    tg = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2")
    by_turn = {t["prefix_end"] + 1: t["actions"] for t in tg}
    # from the message turn (idx 4): next effective actions skip the failed cancel (idx 6)
    assert by_turn[4] == ["Found you.\nWhich order?\n", CALL_DETAILS,
                          "That order was delivered, so it cannot be cancelled."]
    # without skip_invalid the failed call is part of the target
    tg2 = build_action_targets(NATIVE, k=3, skip_invalid=False, env="tau2")
    by2 = {t["prefix_end"] + 1: t["actions"] for t in tg2}
    assert by2[6][0] == CALL_CANCEL
    # every target's prefix ends on the observation the action answers
    assert all(NATIVE[t["prefix_end"]]["role"] in ("user", "tool") for t in tg)


def test_native_targets_need_the_turns_layout():
    # literal targets are multi-line; the K-line list would blur their boundaries
    with pytest.raises(ValueError):
        build_action_forecast_samples(NATIVE, tokenizer=None, k=3, env="tau2", layout="list")
    with pytest.raises(ValueError):
        build_action_forecast_samples(NATIVE, tokenizer=None, k=3, env="tau2", layout="lines")


# ---- turns layout: K assistant turns, loss on the assistant turns only ---------------------

@pytest.fixture(scope="module")
def tok():
    path = os.environ.get("COPE_TEST_TOKENIZER")
    if not path:
        pytest.skip("set COPE_TEST_TOKENIZER to a local Qwen2.5 snapshot")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(path)


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


def test_turns_train_exactly_the_policy_spans(tok):
    samples = build_action_forecast_samples(NATIVE, tok, k=3, skip_invalid=True, env="tau2", layout="turns")
    assert samples
    s = samples[0]                       # forecast from the first action turn
    spans = _masked_spans(s)
    expected = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2")[0]["actions"]
    assert len(spans) == s["k_realized"] == len(expected) == 3
    for span, action in zip(spans, expected):
        # each trained span is one literal action closed like a policy turn ...
        assert tok.decode(span) == action + "<|im_end|>\n"
        # ... and its tokens are the tokens of that literal text
        assert span[:-2] == tok.encode(action, add_special_tokens=False)
    # the filler prompt between turns is never trained, and the K actions are separate turns
    ids = s["input_ids"].tolist()
    filler = tok.encode(ACTION_FORECAST_NEXT_PROMPT, add_special_tokens=False)
    hits = [i for i in range(len(ids) - len(filler) + 1) if ids[i:i + len(filler)] == filler]
    assert len(hits) == 2
    for i in hits:
        assert not any(s["loss_mask"][i:i + len(filler)].tolist())
    assert tok.decode(ids).count("<|im_start|>assistant") == 3


def test_react_list_layout_is_unchanged(tok):
    # the ReAct envs keep the original K-line construction
    s = build_action_forecast_samples(REACT, tok, k=2, env="alfworld")[0]
    spans = _masked_spans(s)
    assert len(spans) == 1
    assert tok.decode(spans[0]) == "go north\ntake key<|im_end|>\n"
