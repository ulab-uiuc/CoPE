"""性质14: λ_TE 必须真的抵达损失函数。

背景（2026-09-15，λ=0.1 首次正式启动时抓到）：
  update_policy 里 `batch = data.select(batch_keys=select_keys).batch` 丢掉 meta_info，
  紧接着 `for batch_idx, data in enumerate(dataloader)` 又把 data 重新绑定成 TensorDict。
  于是循环体内 `data.meta_info.get('te_lambda')` 恒为 0，TE-KL 一次都不执行；
  同时 te_log_q/te_valid/turn_ids 不在 select_keys 里，也会被整个丢掉。
  两者都是"配置齐全、日志正常、梯度里啥也没有"的静默失效，λ=0 的 dry-run 无法暴露。

本测试用 AST 静态检查这两条不变量。
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

import ast, sys


def test_te_lambda_reaches_loss():

    SRC = str(_SRC / 'verl/workers/agent_actor/dp_actor.py')
    tree = ast.parse(open(SRC).read())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == 'update_policy'), None)
    assert fn is not None, "未找到 update_policy"

    # --- 1) te_lambda 必须在函数顶层(非循环内)从 meta_info 读取 ---
    top_assign_line = None
    for node in fn.body:                       # 只看顶层语句，不进循环
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'te_lambda':
                    src = ast.dump(node.value)
                    assert 'meta_info' in src, "te_lambda 必须来自 meta_info"
                    top_assign_line = node.lineno
    assert top_assign_line, \
        "te_lambda 必须在 update_policy 顶层赋值（不能在 for 循环里读 data.meta_info）"

    # --- 2) 必须早于 dataloader 的那次 rebind ---
    rebind_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
            names = [e.id for e in node.target.elts if isinstance(e, ast.Name)]
            if 'data' in names:
                rebind_line = node.lineno
    assert rebind_line, "未找到 `for batch_idx, data in enumerate(dataloader)`"
    assert top_assign_line < rebind_line, \
        f"te_lambda 在第 {top_assign_line} 行赋值，晚于第 {rebind_line} 行的 data 重绑定"

    # --- 3) nothing inside the loop may read meta_info: `data` there is a bare TensorDict ---
    loop = next(n for n in ast.walk(fn)
                if isinstance(n, ast.For) and n.lineno == rebind_line)
    meta_reads = [n.lineno for n in ast.walk(loop)
                  if isinstance(n, ast.Attribute) and n.attr == 'meta_info']
    assert not meta_reads, f"meta_info read inside the rebinding loop at lines {meta_reads}"

    # --- 4) TE 张量必须进 select_keys ---
    src_txt = open(SRC).read()
    i_sel = src_txt.index('select_keys = [')
    i_loop = src_txt.index('for batch_idx, data in enumerate(dataloader)')
    window = src_txt[i_sel:i_loop]
    for k in ('te_log_q', 'te_valid', 'turn_ids'):
        assert k in window, f"{k} 未加入 select_keys，会在 select 时被丢弃"
    assert ('te_lambda > 0' in window or "te_mix == 'fullvocab'" in window), \
        "select_keys 的追加必须由 TE 相关条件门控（保证关闭时逐字不变）"

    # --- 5) TE-KL 门控仍是双重的 ---
    assert "te_lambda > 0.0 and 'te_log_q' in data.keys()" in src_txt, "legacy TE-KL double gate is missing"

    print(f"性质14 PASS: te_lambda 在第 {top_assign_line} 行从 meta_info 读取，"
          f"早于第 {rebind_line} 行的 data 重绑定")
    print("  te_log_q/te_valid/turn_ids 均已进 select_keys，且由 te_lambda>0 门控")
    print("  TE-KL 双重门控完好（λ>0 且 batch 里确有 te_log_q）")
