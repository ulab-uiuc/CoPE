"""逐 token 混合（te_mix='token'）的性质测试。

性质10: assemble_te_tensors_token 的混合结果与 float64 直算逐位置一致
性质11: 长度对不上的成员被丢弃并计数，绝不截断错位
性质12: te_log_q 与 log_prob 同形 -> dp_actor 走逐 token 分支，梯度可回传
性质13: 无成员 / eta=0 时 te_log_q == old_log_probs -> KL 恒为 0（惰性退化）
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

import sys, math, random
import torch
from verl.agent_trainer.ppo.temporal_ensemble import assemble_te_tensors_token


def test_te_token_mix():

    random.seed(0); torch.manual_seed(0)
    B, T = 6, 40

    def make_case(n_members=3, bad_len=False):
        turn_ids = torch.full((B, T), -1, dtype=torch.long)
        old_lp = torch.randn(B, T) * 1.5 - 1.0
        index, slot_tok_lp = [], []
        spans = {}
        for i in range(B):
            p = 3
            for t in range(3):
                n = random.randint(2, 5)
                turn_ids[i, p:p + n] = t
                spans[(i, t)] = (p, n)
                p += n + random.randint(1, 3)
        for i in range(B):
            for src in range(2):
                turns, rows = [], []
                for t in range(src + 1, 3):
                    n = spans[(i, t)][1]
                    if bad_len and t == 2:
                        n += 1                       # 故意长度不匹配
                    turns.append(t)
                    rows.append([random.uniform(-4, -0.1) for _ in range(n)])
                if turns:
                    index.append((i, turns, src))
                    slot_tok_lp.append(rows)
        return turn_ids, old_lp, index, slot_tok_lp, spans

    # ---------- 性质 10 ----------
    eta = 0.5
    turn_ids, old_lp, index, slot_tok_lp, spans = make_case()
    logq, valid, meta = assemble_te_tensors_token(index, slot_tok_lp, old_lp, turn_ids,
                                                  eta=eta, traj_return=None)
    # 独立 float64 重算
    members = {}
    for (i, turns, src), rows in zip(index, slot_tok_lp):
        for t, toks in zip(turns, rows):
            if t > src:
                members.setdefault((i, t), []).append(toks)
    maxerr = 0.0; nchk = 0
    for (i, t), ms in members.items():
        p0, n = spans[(i, t)]
        good = [m for m in ms if len(m) == n]
        if not good: continue
        for j in range(n):
            vals = [m[j] for m in good]
            mx = max(vals)
            qF = mx + math.log(sum(math.exp(v - mx) for v in vals) / len(vals))
            a = math.log1p(-eta) + float(old_lp[i, p0 + j])
            b = math.log(eta) + qF
            m_ = max(a, b)
            ref = m_ + math.log(math.exp(a - m_) + math.exp(b - m_))
            maxerr = max(maxerr, abs(ref - float(logq[i, p0 + j]))); nchk += 1
    assert maxerr < 1e-5, maxerr
    assert nchk > 0
    print(f"性质10 PASS: {nchk} 个 token 位置，与 float64 直算最大误差 {maxerr:.2e}")

    # ---------- 性质 11 ----------
    turn_ids2, old_lp2, index2, slot_tok2, spans2 = make_case(bad_len=True)
    logq2, valid2, meta2 = assemble_te_tensors_token(index2, slot_tok2, old_lp2,
                                                     turn_ids2, eta=0.5)
    assert meta2['te/token_len_mismatch'] > 0, meta2
    # turn 2 的成员全长度不符 -> 该轮 token 应保持 = old_log_probs
    bad = 0
    for i in range(B):
        p0, n = spans2[(i, 2)]
        if not bool(valid2[i, p0]):
            assert torch.allclose(logq2[i, p0:p0 + n], old_lp2[i, p0:p0 + n]); bad += 1
    assert bad > 0
    print(f"性质11 PASS: {int(meta2['te/token_len_mismatch'])} 个长度不符成员被丢弃，"
          f"{bad} 条轨迹的该轮回退为 π⁰（未截断错位）")

    # ---------- 性质 12 ----------
    assert logq.shape == old_lp.shape, (logq.shape, old_lp.shape)
    lp = old_lp.clone().requires_grad_(True)
    kl = logq.detach() - lp
    kld = torch.exp(kl) - kl - 1
    loss = (kld * valid.to(kld.dtype)).sum() / valid.sum().clamp(min=1)
    loss.backward()
    g = lp.grad
    assert torch.isfinite(g).all()
    assert (g[valid] != 0).any() and (g[~valid] == 0).all()
    print(f"性质12 PASS: te_log_q 与 log_prob 同形 {tuple(logq.shape)}；"
          f"梯度只落在 {int(valid.sum())} 个被覆盖 token 上，其余恒 0")

    # ---------- 性质 13 ----------
    logq0, valid0, _ = assemble_te_tensors_token(index, slot_tok_lp, old_lp, turn_ids, eta=0.0)
    assert torch.allclose(logq0, old_lp.float()), "eta=0 应完全退化为 π⁰"
    assert int(valid0.sum()) == 0
    logqE, validE, _ = assemble_te_tensors_token([], [], old_lp, turn_ids, eta=0.5)
    assert torch.allclose(logqE, old_lp.float()) and int(validE.sum()) == 0
    print("性质13 PASS: eta=0 与无成员两个退化分支都令 te_log_q == π⁰ (KL 恒 0)")
    print("\n=== 逐 token 混合测试全部通过 ===")
