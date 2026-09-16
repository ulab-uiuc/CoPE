"""性质15/16: 扣共模后，TE 项不再提供"一律压低动作 token"的共模力。

背景（2026-09-15，λ=0.1 首次训练 20 步后叫停）：
  gain 恒为负 -> q < π⁰ -> k3 梯度 g = 1-exp(kl) 恒为正 -> 所有动作 token 一律被压低。
  实测共模 ≈0.20/token，而 win/fail 的差只有 0.13/token —— 共模比信号还大。
  TE 又只覆盖裸动作 token，模型把概率质量挪到非动作文本上即可规避：
  response_length 931 -> 3264（逼近 4096 上限），成功率归零，te/kl 自我满足地跌到 0。

性质15: 中心化权重 w = g - g_bar 在被覆盖 token 上均值为 0（无共模）
性质16: w 仍保留 win/fail 的方向性（难预测的被压、好预测的被抬）
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
import torch
from verl.agent_trainer.ppo.temporal_ensemble import assemble_te_tensors_token


def test_te_centering():

    torch.manual_seed(0)
    B, T = 8, 60
    turn_ids = torch.full((B, T), -1, dtype=torch.long)
    old_lp = torch.zeros(B, T)
    spans = {}
    for i in range(B):
        p = 2
        for t in range(4):
            n = 3
            turn_ids[i, p:p+n] = t
            spans[(i, t)] = (p, n)
            old_lp[i, p:p+n] = -0.5           # π⁰ 固定，便于构造已知 gain
            p += n + 2

    # 前 4 条轨迹 = "win"（forecast 较准, gain≈-0.2），后 4 条 = "fail"（gain≈-0.9）
    index, slot_tok = [], []
    for i in range(B):
        g = -0.2 if i < 4 else -0.9
        for src in range(3):
            turns, rows = [], []
            for t in range(src+1, 4):
                turns.append(t); rows.append([-0.5 + g] * 3)
            if turns:
                index.append((i, turns, src)); slot_tok.append(rows)

    logq, valid, meta = assemble_te_tensors_token(index, slot_tok, old_lp, turn_ids, eta=0.5)
    gbar = meta['te/g_bar']
    assert gbar == gbar and gbar > 0, f"g_bar 应为正（共模压制存在）: {gbar}"

    g = 1.0 - torch.exp(logq - old_lp)
    w = (g - gbar) * valid

    # --- 性质15 ---
    mean_w = float(w.sum() / valid.sum())
    assert abs(mean_w) < 1e-5, f"中心化后共模应为 0，实际 {mean_w:.2e}"
    mean_g = float((g * valid).sum() / valid.sum())
    print(f"性质15 PASS: 中心化前共模 {mean_g:+.4f}/token（恒为正=一律压低），"
          f"中心化后 {mean_w:+.2e}")

    # --- 性质16 ---
    win_mask = valid.clone(); win_mask[4:] = False
    fail_mask = valid.clone(); fail_mask[:4] = False
    w_win = float((w * win_mask).sum() / win_mask.sum())
    w_fail = float((w * fail_mask).sum() / fail_mask.sum())
    assert w_win < 0 < w_fail, f"win 应被抬(w<0)、fail 应被压(w>0)，实际 win={w_win:+.4f} fail={w_fail:+.4f}"
    assert abs(w_fail - w_win) > 0.05, "win/fail 的方向性力度过小"
    print(f"性质16 PASS: win 侧 w={w_win:+.4f}（被抬），fail 侧 w={w_fail:+.4f}（被压），"
          f"差 {w_fail-w_win:.4f}/token")

    # --- 损失对 log π 的导数确实等于 w ---
    lp = old_lp.clone().requires_grad_(True)
    loss = ((g - gbar).detach() * valid * lp).sum() / valid.sum()
    loss.backward()
    expect = w / valid.sum()
    assert torch.allclose(lp.grad, expect, atol=1e-6), "损失对 log π 的导数应恰为 w/N"
    print("  损失梯度 = w/N 已验证：写废话无法降低该项（共模为 0，无逃逸收益）")
    print("\n=== 中心化测试通过 ===")
