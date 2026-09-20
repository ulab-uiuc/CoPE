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
