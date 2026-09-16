"""全词表逐 token KL（te_mix='fullvocab'）的性质测试。

前两次失败的根因都是"只在实际发生的 token 上算 log q - log π"得到无归一化目标：
  A 版(原始KL)  : 有不动点，但只覆盖动作 token -> 模型写废话规避 -> 长度 931->3264
  B 版(扣共模)  : 逃逸路径堵死，但损失退化成线性项 w·log π，两个方向都无界、
                  没有不动点 -> 熵 0.37->6.72，输出变词沙拉
本测试立的就是这两条各自缺的性质。性质17/18/19 是 B 版翻车时我漏测的那部分。
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

import sys, math
import torch
from verl.agent_trainer.ppo.temporal_ensemble import fullvocab_te_kl


def test_te_fullvocab():

    torch.manual_seed(0)
    B, T, V, C, M = 3, 12, 50, 4, 8
    ETA = 0.5

    def mk(logits=None, qf_from=None, npos=4):
        """构造 cov_* 张量；qf_from 给定时用它作为 qF 的来源分布。"""
        cov_pos = torch.full((B, C), -1, dtype=torch.int32)
        cov_ids = torch.full((B, C, M), -1, dtype=torch.int32)
        cov_prs = torch.zeros((B, C, M))
        for b in range(B):
            for c in range(npos):
                cov_pos[b, c] = c + 1
                src = qf_from[b, c + 1] if qf_from is not None else torch.softmax(torch.randn(V), -1)
                v, i = torch.topk(src, M)
                cov_ids[b, c] = i.to(torch.int32); cov_prs[b, c] = v
        return cov_pos, cov_ids, cov_prs

    logits = torch.randn(B, T, V) * 1.5

    # ---------- 性质17: qF = p0 时该项恒为 0、梯度恒为 0 ----------
    p_all = torch.softmax(logits, -1)
    cov_pos, cov_ids, cov_prs = mk(qf_from=p_all)
    lg = logits.clone().requires_grad_(True)
    kl, n = fullvocab_te_kl(lg, cov_pos, cov_ids, cov_prs, ETA)
    # top-M 截断使 qF 只覆盖部分质量，理论 0 会有截断残差；先量化它
    kl.backward()
    gmax = float(lg.grad.abs().max())
    print(f"性质17 PASS: qF=p0 时 KL={float(kl):.3e}（top-{M}/{V} 截断残差），"
          f"梯度最大分量 {gmax:.3e}")
    assert abs(float(kl)) < 0.05, float(kl)
    assert gmax < 0.05, gmax

    # ---------- 性质25: KL >= 0 ----------
    neg = 0
    for trial in range(200):
        lg2 = torch.randn(B, T, V) * 2.0
        cp, ci, cr = mk()
        k, _ = fullvocab_te_kl(lg2, cp, ci, cr, ETA)
        if float(k) < -1e-6: neg += 1
    assert neg == 0, f"{neg}/200 次出现负 KL"
    print("性质25 PASS: 200 组随机构造，KL 恒 >= 0（B 版违反的正是这条）")

    # ---------- 性质20: 与稠密 float64 直算逐位置一致（M=V，无截断） ----------
    Mfull = V
    cov_pos_f = torch.full((B, C), -1, dtype=torch.int32)
    cov_ids_f = torch.full((B, C, Mfull), -1, dtype=torch.int32)
    cov_prs_f = torch.zeros((B, C, Mfull))
    qf_dense = torch.zeros(B, C, V, dtype=torch.float64)
    for b in range(B):
        for c in range(4):
            cov_pos_f[b, c] = c + 1
            d = torch.softmax(torch.randn(V) * 1.2, -1).double()
            qf_dense[b, c] = d
            v, i = torch.topk(d.float(), Mfull)
            cov_ids_f[b, c] = i.to(torch.int32); cov_prs_f[b, c] = v
    kl_impl, _ = fullvocab_te_kl(logits, cov_pos_f, cov_ids_f, cov_prs_f, ETA)
    ref = []
    for b in range(B):
        for c in range(4):
            pth = torch.softmax(logits[b, c + 1].double(), -1)
            p0 = pth.clone()
            q = (1 - ETA) * p0 + ETA * qf_dense[b, c]
            q = q / q.sum()
            ref.append(float((pth * (pth.log() - q.log())).sum()))
    err = abs(float(kl_impl) - sum(ref) / len(ref))
    assert err < 1e-6, err
    print(f"性质20 PASS: M=V 时与 float64 稠密直算一致，误差 {err:.2e}")

    # ---------- 性质18/19: 不动点在 π=qF；偏离必上升 ----------
    qf = torch.softmax(torch.randn(V) * 1.2, -1)
    cov_pos1 = torch.full((1, 1), 1, dtype=torch.int32)
    v, i = torch.topk(qf, V)
    cov_ids1 = i.view(1, 1, V).to(torch.int32); cov_prs1 = v.view(1, 1, V)
    base = torch.zeros(1, T, V)
    base[0, 1] = qf.log()                      # π = qF
    lg3 = base.clone().requires_grad_(True)
    kl3, _ = fullvocab_te_kl(lg3, cov_pos1, cov_ids1, cov_prs1, ETA)
    kl3.backward()
    g_at_fp = float(lg3.grad.abs().max())
    assert g_at_fp < 1e-5, g_at_fp
    print(f"性质18 PASS: π=qF 处梯度 {g_at_fp:.2e}（不动点存在且可达）")

    vals = []
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        mixed = (1 - a) * qf + a * torch.full((V,), 1.0 / V)   # 向均匀分布扰动
        b2 = torch.zeros(1, T, V); b2[0, 1] = mixed.log()
        k, _ = fullvocab_te_kl(b2, cov_pos1, cov_ids1, cov_prs1, ETA)
        vals.append(float(k))
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), vals
    print(f"性质19 PASS: 向均匀分布扰动时 KL 单调上升 "
          f"{' -> '.join(f'{x:.4f}' for x in vals)}")
    print("  （A 版正是靠把质量挪到别处规避惩罚；这条封死了那条路）")

    # ---------- 性质24: 数值稳健 ----------
    lg4 = torch.randn(B, T, V) * 30.0          # 极端 logits -> p0 有极小值
    cp, ci, cr = mk()
    k4, _ = fullvocab_te_kl(lg4, cp, ci, cr, ETA)
    assert torch.isfinite(k4), k4
    k5, n5 = fullvocab_te_kl(logits, torch.full((B, C), -1, dtype=torch.int32), ci, cr, ETA)
    assert n5 == 0 and float(k5) == 0.0
    k6, n6 = fullvocab_te_kl(logits, cp, ci, cr, 0.0)
    assert n6 == 0 and float(k6) == 0.0
    print("性质24 PASS: 极端 logits 不出 NaN/Inf；无覆盖位置与 eta=0 两条退化路径返回 0")
    print("\n=== 全词表 KL 测试通过 ===")
