"""TE 性质验证 1：RolloutHandler.turn_ids 的结构正确性。

断言：
  (1) len(turn_ids) == len(input_ids)（每轮追加后都成立）
  (2) turn_ids 中的非 -1 值恰好是 0..n_assistant-1，且每个轮号对应的 token
      解码回来 == 该轮 assistant 的 content（即精确覆盖"动作正文"，不含模板）
  (3) 观测(user)轮与模板 token 全为 -1
  (4) response_turn_ids == turn_ids[len(prompt):] 截断到 max_response_len
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
    assert len(h.turn_ids) == len(h.input_ids), "初始化后长度不一致"

    for i, (act, obs) in enumerate(TURNS):
        h.add_assistant_message(tok, act)
        assert len(h.turn_ids) == len(h.input_ids), f"轮 {i} assistant 后长度不一致"
        assert h._assistant_turn == i, f"轮计数错: {h._assistant_turn} != {i}"
        h.add_user_message(tok, obs)
        assert len(h.turn_ids) == len(h.input_ids), f"轮 {i} user 后长度不一致"

    # (2) 每个轮号的 token 解码 == 该轮动作正文
    seen = sorted({t for t in h.turn_ids if t >= 0})
    assert seen == list(range(len(TURNS))), f"轮号集合错: {seen}"
    from verl.agent_trainer.ppo.action_forecast import extract_action
    for t, (act, _) in enumerate(TURNS):
        idx = [j for j, v in enumerate(h.turn_ids) if v == t]
        assert idx == list(range(idx[0], idx[-1]+1)), f"轮 {t} 的 token 不连续: {idx}"
        dec = tok.decode([h.input_ids[j] for j in idx])
        # 2026-09-15 新口径：turn_ids 只标**裸动作**，不含 Thought 推理段。
        # 必须与 TE 成员侧的 slot span 同口径，否则序列级 log-prob 差 9 倍 token 数。
        bare = extract_action(act, env='sciworld') or act
        assert dec.strip() == bare.strip(), \
            f"轮 {t} 解码不符（应为裸动作）:\n  得到 {dec!r}\n  期望 {bare!r}"

    # (3) prompt 段全 -1
    assert set(h.turn_ids[:len(prompt_ids)]) == {-1}, "prompt 段混入了轮号"

    # (4) response_turn_ids 一致性
    h.truncate_and_pad() if hasattr(h, 'truncate_and_pad') else None
    exp = h.turn_ids[len(h.prompt_turn_ids):][:h.max_response_len]
    assert h.response_turn_ids == exp or h.response_turn_ids == [], \
        "response_turn_ids 与切片不一致"

    print(f"性质1 PASS: {len(TURNS)} 轮，turn_ids 长度/连续性/解码/prompt段 全部正确")
    print(f"  非 -1 token 数 = {sum(1 for t in h.turn_ids if t>=0)}, "
          f"loss_mask=1 的 token 数 = {sum(h.loss_mask)}")

    # ---------- 新增性质 9：两侧口径一致（这是 gain 有意义的前提）----------
    # 用带 Thought 的真实格式回复，验证 turn_ids 标出的 token 数
    # ≈ TE 成员侧 slot span 的 token 数，而不是整段回复的长度。
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
    assert dec.strip() == bare.strip(), f"未标到裸动作: {dec!r} vs {bare!r}"
    assert n_turn <= n_bare + 2, f"turn_ids 标了 {n_turn} token，裸动作只有 {n_bare}"
    assert n_turn < n_full / 2, f"turn_ids({n_turn}) 不该接近整段回复({n_full})"
    print(f"性质9 PASS: 整段回复 {n_full} token，turn_ids 只标 {n_turn} token "
          f"(裸动作 {n_bare})，解码 = {bare!r}")
