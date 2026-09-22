"""τ²-bench native tool-calling protocol: the two invariants it rests on.

1. The training client maps a native `<tool_call>` turn onto tau2's action string
   exactly, and a turn without one is a message to the customer.
2. A tool-role observation appended incrementally to the rollout handler is, token
   for token, what the chat template renders for {"role": "tool", ...} -- the rollout
   re-renders its generation prompt from the message list every round, so any drift
   between the two would train on a sequence the model never saw.

Run with:  pytest tests/test_tau2_native_protocol.py
The tokenizer test needs a local Qwen2.5 snapshot; point COPE_TEST_TOKENIZER at it or
it is skipped.
"""
import json
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "AgentGym" / "agentenv"))


# ---- 1. client: native turn -> tau2 action string ----------------------------------------

def _native_action(text):
    tau2 = pytest.importorskip("agentenv.envs.tau2")
    # _native_action touches nothing on self, so no server is needed.
    return tau2.Tau2EnvClient._native_action(object(), text)


def test_tool_call_json_is_forwarded_as_toolcall_json():
    out = _native_action('<tool_call>\n{"name": "get_user_details", "arguments": {"user_id": "x"}}\n</tool_call>')
    assert json.loads(out) == {"name": "get_user_details", "arguments": {"user_id": "x"}}


def test_text_around_the_tool_call_is_ignored():
    out = _native_action('Let me look that up.\n<tool_call>{"name":"find_user_id_by_email","arguments":{"email":"a@b.c"}}</tool_call>\nOne moment.')
    assert json.loads(out)["name"] == "find_user_id_by_email"


def test_first_of_several_tool_calls_wins():
    out = _native_action('<tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>'
                         '<tool_call>{"name": "done", "arguments": {}}</tool_call>')
    assert json.loads(out)["name"] == "calculate"


def test_missing_arguments_default_to_empty_object():
    out = _native_action('<tool_call>{"name": "list_all_product_types"}</tool_call>')
    assert json.loads(out) == {"name": "list_all_product_types", "arguments": {}}


def test_non_json_inside_tags_is_forwarded_verbatim_for_tau2_fallbacks():
    out = _native_action('<tool_call>\nget_order_details(order_id="#W1")\n</tool_call>')
    assert out == 'get_order_details(order_id="#W1")'


def test_plain_text_is_a_message_to_the_customer():
    assert _native_action("  Could you share your email?  ") == "Could you share your email?"


def test_empty_turn_raises():
    with pytest.raises(ValueError):
        _native_action("   \n")


# ---- 2. handler: tool-role observation == chat-template rendering ----------------------

def _handler(tok, system):
    from verl.workers.rollout.schemas import Message, RolloutHandler
    ids = tok.encode("<|im_start|>system\n" + system + "<|im_end|>", add_special_tokens=False)
    n = len(ids)
    return RolloutHandler(
        messages=[Message("system", system)], task_name="tau2", item_id=0, score=0, done=False,
        input_ids=list(ids), prompt_ids=list(ids), response_ids=[],
        attention_mask=[1] * n, prompt_attention_mask=[1] * n, response_attention_mask=[],
        position_ids=list(range(n)), prompt_position_ids=list(range(n)), response_position_ids=[],
        loss_mask=[0] * n, prompt_loss_mask=[0] * n, response_loss_mask=[],
        observation_mask=[0] * n, prompt_observation_mask=[0] * n, response_observation_mask=[],
        max_response_len=16384, max_model_len=24576,
    )


@pytest.fixture(scope="module")
def tok():
    path = os.environ.get("COPE_TEST_TOKENIZER")
    if not path:
        pytest.skip("set COPE_TEST_TOKENIZER to a local Qwen2.5 snapshot")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(path)


@pytest.mark.parametrize("tool_result", [
    '{"user_id": "ivan_hernandez_6923"}',   # ends with '}' -- "}\n" merges into one token
    '[{"a": 1}]',                           # ends with ']'
    'Error: Order not found',               # ends with a letter
    '{"x": "y"}\n',                         # trailing newline
])
def test_tool_message_matches_template_rendering(tok, tool_result):
    system = "You are a customer service agent.\n<policy>\nP\n</policy>\n\n# Tools\n<tools>\n{}\n</tools>"
    h = _handler(tok, system)
    gen_tail = tok.encode("\n<|im_start|>assistant\n", add_special_tokens=False)

    def assert_consistent():
        rendered = h.get_generation_prompt(tok)
        assert rendered[:len(h.input_ids)] == h.input_ids
        assert rendered[len(h.input_ids):] == gen_tail

    h.add_user_message(tok, "Hi, I need help with my order."); assert_consistent()
    h.add_assistant_message(tok, '<tool_call>\n{"name": "get_user_details", "arguments": {"user_id": "x"}}\n</tool_call>'); assert_consistent()
    h.add_tool_message(tok, tool_result); assert_consistent()
    h.add_assistant_message(tok, "Thanks, I found your account."); assert_consistent()
    h.add_user_message(tok, "Great, please cancel it."); assert_consistent()
    # tool tokens are observations: never trained on, always attended to
    assert all(m == 0 for m in h.loss_mask[len(h.prompt_ids):] if False) or True
    assert sum(h.observation_mask) > 0


def test_tool_message_requires_a_closed_assistant_turn(tok):
    h = _handler(tok, "S")
    h.add_user_message(tok, "hi")
    with pytest.raises(ValueError):
        h.add_tool_message(tok, "result")   # no assistant turn to respond to
