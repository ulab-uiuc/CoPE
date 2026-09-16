"""TE 性质 6+8：build_plan_targets 的 action_turns 正确 + fut 无回归。

函数内部已有 assert 保证 [actions_seq[j] for j in sel] == fut，所以只要能跑通，
fut 就没被改坏。这里额外验证：
  (6) 随机轨迹 × {skip_invalid T/F} 下都能跑通（内部 assert 不触发）
  (8) skip_invalid=True 时 action_turns 会跳过无效动作的轮号
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

import sys, random
from verl.agent_trainer.ppo.plan_forecast import build_plan_targets


def test_te_action_turns():

    def mk(acts, results):
        """acts[i] 是第 i 轮动作，results[i] 是它的结果观测。"""
        m = [{'role':'user','content':'sys'},{'role':'assistant','content':'Ok.'},
             {'role':'user','content':'task'}]
        for a, r in zip(acts, results):
            m.append({'role':'assistant','content':a})
            m.append({'role':'user','content':r})
        return m

    # --- 性质 8：第 1 轮(0-based)动作无效，skip_invalid 应跳过它 ---
    acts = ['go to kitchen','open nothing','open fridge','pick up cup']
    res  = ['You are in the kitchen','No known action matches that input.',
            'The fridge is open','You pick up the cup']
    for k in (2,3,4):
        for si, tl in ((True, build_plan_targets(mk(acts,res), k=k, skip_invalid=True,  env='sciworld')),
                       (False, build_plan_targets(mk(acts,res), k=k, skip_invalid=False, env='sciworld'))):
            t0 = tl[0]['action_turns']
            assert t0[0] == 0, f"k={k} si={si}: 轮0 的 slot1 必须是轮0, 实际 {t0}"
            if si:
                assert 1 not in t0, f"k={k}: skip_invalid=True 却没跳过无效轮1: {t0}"
            else:
                assert len(t0) < 2 or 1 in t0, f"k={k}: skip_invalid=False 却跳了轮1: {t0}"
            for tg in tl:
                at = tg['action_turns']
                assert len(at) == len(tg['actions']), (len(at), len(tg['actions']))
                assert all(at[i] < at[i+1] for i in range(len(at)-1)), f"未严格递增: {at}"
                # 注意：slot1 的轮号**不一定**等于发出轮！skip_invalid=True 时，若发出轮
                # 自己的动作无效，会被自己的过滤器滤掉，此时 at[0] > src_turn。
                # 这意味着 TE 的成员筛选必须按真实轮号 (t > s)，不能按 slot 下标 (j >= 1)。
                assert at[0] >= tg['src_turn'], f"slot1 轮号不应早于发出轮: {at[0]} < {tg['src_turn']}"
                if not si:
                    assert at[0] == tg['src_turn'], "skip_invalid=False 时 slot1 必须就是发出轮"

    # 显式验证那个坑：发出轮自身动作无效时，slot1 的 k 已经 >= 1
    t_skip = build_plan_targets(mk(acts,res), k=3, skip_invalid=True, env='sciworld')
    bysrc = {tg['src_turn']: tg for tg in t_skip}
    assert 1 in bysrc, "轮1 应该仍然产出 target（它往后看仍有有效动作）"
    tg1 = bysrc[1]
    assert tg1['action_turns'][0] == 2, f"轮1 的 slot1 应指向轮2, 实际 {tg1['action_turns']}"
    n_k0 = sum(1 for tg in t_skip for t in tg['action_turns'] if t == tg['src_turn'])
    n_k1 = sum(1 for tg in t_skip for t in tg['action_turns'] if t >  tg['src_turn'])
    print(f"  性质8 PASS: skip_invalid 跳号正确（k=2,3,4）")
    print(f"  关键发现: k=0 的 slot 有 {n_k0} 个, k>=1 的成员有 {n_k1} 个 —— "
          f"成员筛选必须按 t>s，不能按 slot 下标")

    # --- 性质 6：随机轨迹压测，内部 assert 不触发即说明 fut 无回归 ---
    rng = random.Random(0)
    BAD = ['No known action matches that input.','The fridge is open','Nothing happens.','You see a cup']
    n_tg = 0
    for trial in range(200):
        T = rng.randint(1, 12)
        a = [f'action_{i}_{rng.randint(0,99)}' for i in range(T)]
        r = [rng.choice(BAD) for _ in range(T)]
        for si in (True, False):
            for k in (1,2,3,4,5):
                tg = build_plan_targets(mk(a,r), k=k, skip_invalid=si, env='sciworld')
                n_tg += len(tg)
    print(f"  性质6 PASS: 200 条随机轨迹 × {{skip T/F}} × k∈1..5 共 {n_tg} 个 target，内部断言全未触发")
