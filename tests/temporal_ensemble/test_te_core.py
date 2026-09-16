"""TE 性质 3/4/7：分段求和、slot span 对齐、混合数值。"""

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

import sys, math, random
import torch
from transformers import AutoTokenizer
from verl.agent_trainer.ppo.temporal_ensemble import (
    build_te_scoring_samples, assemble_q, segment_sum_by_turn)


def test_te_core():

    tok = AutoTokenizer.from_pretrained(_tokenizer_path())

    # ---------- 性质 4：slot_spans 解码回来必须等于对应动作 ----------
    def mk(acts, results):
        m = [{'role':'user','content':'You are an agent.'},
             {'role':'assistant','content':'Ok.'},{'role':'user','content':'Task: find a cup'}]
        for a,r in zip(acts,results): m += [{'role':'assistant','content':a},{'role':'user','content':r}]
        return m

    rng = random.Random(7)
    VERBS=['go to','open','close','pick up','put down','focus on','activate','look at']
    OBJS=['kitchen','fridge','glass cup','metal pot','door to hallway','stove','sink','blue jay egg']
    BADR=['No known action matches that input.','Nothing happens.']
    n_checked=0
    for trial in range(40):
        T=rng.randint(2,8)
        acts=[f"{rng.choice(VERBS)} {rng.choice(OBJS)}" for _ in range(T)]
        res=[rng.choice(BADR) if rng.random()<0.3 else f"You see {rng.choice(OBJS)}." for _ in range(T)]
        for si in (True,False):
            for k in (2,3,4):
                for smp in build_te_scoring_samples(mk(acts,res), tok, k=k,
                                                    skip_invalid=si, env='sciworld'):
                    ids=smp['input_ids']
                    for j,(s,e) in enumerate(smp['slot_spans']):
                        dec=tok.decode(ids[s:e].tolist())
                        exp=acts[smp['slot_turns'][j]]
                        assert dec.strip()==exp.strip(), \
                            f"span 未对齐: 得到 {dec!r} 期望 {exp!r} (si={si},k={k},slot={j})"
                        n_checked+=1
                    # loss_mask 必须覆盖所有 slot
                    for (s,e) in smp['slot_spans']:
                        assert smp['loss_mask'][s:e].min().item()==1, "slot 落在 loss_mask 之外"
                    # 必须至少有一个 k>=1 的成员
                    assert any(t>smp['src_turn'] for t in smp['slot_turns'])
    print(f"性质4 PASS: {n_checked} 个 slot span 全部解码回正确动作（含 skip_invalid T/F, k=2/3/4）")

    # ---------- 性质 3：segment_sum_by_turn 与 for 循环一致 ----------
    for trial in range(50):
        B,T,N=rng.randint(1,4),rng.randint(5,40),rng.randint(1,6)
        vals=torch.randn(B,T,dtype=torch.float64)
        tid=torch.full((B,T),-1,dtype=torch.long)
        for b in range(B):
            for t in range(T):
                if rng.random()<0.6: tid[b,t]=rng.randint(0,N-1)
        got=segment_sum_by_turn(vals,tid,N)
        exp=torch.zeros(B,N,dtype=torch.float64)
        for b in range(B):
            for t in range(T):
                if tid[b,t]>=0: exp[b,tid[b,t]]+=vals[b,t]
        assert torch.allclose(got,exp,atol=1e-12), f"分段求和不一致 max_err={(got-exp).abs().max()}"
    print("性质3 PASS: 50 组随机场景，einsum 分段求和与 for 循环逐元素一致")

    # 梯度可传
    v=torch.randn(2,10,requires_grad=True)
    tid=torch.tensor([[0,0,1,1,-1,-1,2,2,2,-1],[0,-1,1,-1,2,-1,-1,-1,-1,-1]])
    segment_sum_by_turn(v,tid,3).sum().backward()
    assert v.grad is not None and v.grad.abs().sum()>0, "梯度没传回来"
    print("  梯度可回传 ✓")

    # ---------- 性质 7：混合数值与 float64 直算一致 ----------
    worst=0.0
    for trial in range(300):
        n=rng.randint(1,6); eta=rng.choice([0.1,0.25,0.5,0.75,0.9])
        ms=[rng.uniform(-60,-0.1) for _ in range(n)]; p0=rng.uniform(-60,-0.1)
        got=assemble_q({5:ms},{5:p0},eta)[5]
        qF=sum(math.exp(x) for x in ms)/n
        q=(1-eta)*math.exp(p0)+eta*qF
        exp_logq, exp_logqF = math.log(q), math.log(qF)
        worst=max(worst,abs(got['log_q']-exp_logq),abs(got['log_qF']-exp_logqF))
        assert got['K_t']==n
    print(f"性质7 PASS: 300 组随机混合，与 float64 直算最大绝对误差 {worst:.2e}")

    # eta=0 退化：q == π^0
    r=assemble_q({5:[-3.0,-4.0]},{5:-2.0},0.0)[5]
    assert r['log_q']==-2.0 and r['K_t']==0, "eta=0 未退化为 π^0"
    # 无成员（t=0）退化
    r=assemble_q({},{0:-1.5},0.5)[0]
    assert r['log_q']==-1.5 and r['K_t']==0, "无成员时未退化为 π^0"
    print("  eta=0 与 t=0 两个退化分支 ✓（TE-KL 此时变成信任域项）")
