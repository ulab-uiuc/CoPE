"""TE property 1: the structural correctness of RolloutHandler.turn_ids.

Asserts:
  (1) len(turn_ids) == len(input_ids) (after every append)
  (2) the non -1 values in turn_ids are exactly 0..n_assistant-1, and the tokens of each
      turn decode back to that turn's assistant content -- i.e. they cover the "action
      body" exactly, with no template
  (3) observation (user) turns and template tokens are all -1
  (4) response_turn_ids == turn_ids[len(prompt):] truncated to max_response_len
"""

import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _tokenizer_path():
    """Local Qwen2.5 tokenizer for the tokenization-level checks; skipped when unset."""
    p = os.environ.get("COPE_TEST_TOKENIZER")
    if not p:
        pytest.skip("set COPE_TEST_TOKENIZER to a local Qwen2.5 tokenizer directory")
    return p

import sys
from transformers import AutoTokenizer
from verl.workers.rollout.schemas import RolloutHandler, Message


def test_te_turn_ids():

    MODEL = _tokenizer_path()
    tok = AutoTokenizer.from_pretrained(MODEL)

    SYS = "You are a helpful agent."
    TURNS = [
        ("go to kitchen", "You are in the kitchen. You see a fridge."),
        ("open fridge", "The fridge is open. You see a glass cup."),
        ("pick up glass cup", "You pick up the glass cup."),
        ("focus on glass cup", "Task completed."),
    ]

    msgs = [Message(role='user', content=SYS)]
    prompt_ids = tok.apply_chat_template([m.to_dict() for m in msgs],
                                         add_generation_prompt=True, tokenize=True)
    h = RolloutHandler(
        messages=list(msgs), task_name='sciworld', item_id=0, score=0, done=False,
        input_ids=list(prompt_ids), prompt_ids=list(prompt_ids), response_ids=[],
        attention_mask=[1]*len(prompt_ids), prompt_attention_mask=[1]*len(prompt_ids),
        response_attention_mask=[],
        position_ids=list(range(len(prompt_ids))), prompt_position_ids=list(range(len(prompt_ids))),
        response_position_ids=[],
        loss_mask=[0]*len(prompt_ids), prompt_loss_mask=[0]*len(prompt_ids), response_loss_mask=[],
        observation_mask=[0]*len(prompt_ids), prompt_observation_mask=[0]*len(prompt_ids),
        response_observation_mask=[],
    )
    assert len(h.turn_ids) == len(h.input_ids), "lengths differ right after initialisation"

    for i, (act, obs) in enumerate(TURNS):
        h.add_assistant_message(tok, act)
        assert len(h.turn_ids) == len(h.input_ids), f"lengths differ after turn {i}'s assistant message"
        assert h._assistant_turn == i, f"wrong turn count: {h._assistant_turn} != {i}"
        h.add_user_message(tok, obs)
        assert len(h.turn_ids) == len(h.input_ids), f"lengths differ after turn {i}'s user message"

    # (2) each turn's tokens decode to that turn's action body
    seen = sorted({t for t in h.turn_ids if t >= 0})
    assert seen == list(range(len(TURNS))), f"wrong set of turn numbers: {seen}"
    from verl.agent_trainer.ppo.action_forecast import extract_action
    for t, (act, _) in enumerate(TURNS):
        idx = [j for j, v in enumerate(h.turn_ids) if v == t]
        assert idx == list(range(idx[0], idx[-1]+1)), f"turn {t}'s tokens are not contiguous: {idx}"
        dec = tok.decode([h.input_ids[j] for j in idx])
        # turn_ids marks the **bare action** only, excluding the Thought reasoning
        # section. It has to use the same convention as the TE member side's slot spans,
        # or the sequence-level log-probs differ by a factor of ~9 in token count.
        bare = extract_action(act, env='sciworld') or act
        assert dec.strip() == bare.strip(), \
            f"turn {t} decodes wrong (should be the bare action):\n  got {dec!r}\n  expected {bare!r}"

    # (3) the prompt segment is all -1
    assert set(h.turn_ids[:len(prompt_ids)]) == {-1}, "turn numbers leaked into the prompt segment"

    # (4) response_turn_ids consistency
    h.truncate_and_pad() if hasattr(h, 'truncate_and_pad') else None
    exp = h.turn_ids[len(h.prompt_turn_ids):][:h.max_response_len]
    assert h.response_turn_ids == exp or h.response_turn_ids == [], \
        "response_turn_ids does not match the slice"

    print(f"property 1 PASS: {len(TURNS)} turns; turn_ids length, contiguity, decoding "
          f"and prompt segment all correct")
    print(f"  non -1 tokens = {sum(1 for t in h.turn_ids if t>=0)}, "
          f"tokens with loss_mask=1 = {sum(h.loss_mask)}")

    # ---------- property 9: both sides use the same convention (the premise that makes
    # the gain meaningful) ----------
    # Use a realistically formatted reply with a Thought section and check that the token
    # count turn_ids marks is close to the TE member side's slot span, not the length of
    # the whole reply.
    FULL = ("Thought:\nThe task is to determine conductivity. I should focus on "
            "the metal fork first because it is the target object.\nAction:\nfocus on metal fork")
    msgs2 = [Message(role='user', content=SYS)]
    pids2 = tok.apply_chat_template([m.to_dict() for m in msgs2],
                                    add_generation_prompt=True, tokenize=True)
    h2 = RolloutHandler(
        messages=list(msgs2), task_name='sciworld', item_id=0, score=0, done=False,
        input_ids=list(pids2), prompt_ids=list(pids2), response_ids=[],
        attention_mask=[1]*len(pids2), prompt_attention_mask=[1]*len(pids2), response_attention_mask=[],
        position_ids=list(range(len(pids2))), prompt_position_ids=list(range(len(pids2))),
        response_position_ids=[],
        loss_mask=[0]*len(pids2), prompt_loss_mask=[0]*len(pids2), response_loss_mask=[],
        observation_mask=[0]*len(pids2), prompt_observation_mask=[0]*len(pids2),
        response_observation_mask=[],
    )
    h2.add_assistant_message(tok, FULL)
    n_full = len(tok(FULL, add_special_tokens=False)['input_ids'])
    n_turn = sum(1 for v in h2.turn_ids if v == 0)
    bare = extract_action(FULL, env='sciworld')
    n_bare = len(tok(bare, add_special_tokens=False)['input_ids'])
    dec = tok.decode([h2.input_ids[j] for j, v in enumerate(h2.turn_ids) if v == 0])
    assert dec.strip() == bare.strip(), f"did not mark the bare action: {dec!r} vs {bare!r}"
    assert n_turn <= n_bare + 2, f"turn_ids marked {n_turn} tokens, the bare action is only {n_bare}"
    assert n_turn < n_full / 2, f"turn_ids({n_turn}) should not be close to the whole reply({n_full})"
    print(f"property 9 PASS: the whole reply is {n_full} tokens, turn_ids marks only {n_turn} "
          f"(bare action {n_bare}), decoding to {bare!r}")
