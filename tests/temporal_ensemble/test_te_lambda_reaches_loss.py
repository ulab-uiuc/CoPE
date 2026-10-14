"""Property 14: lambda_TE must actually reach the loss function.

Background (caught the first time lambda=0.1 was launched for real):
  in update_policy, `batch = data.select(batch_keys=select_keys).batch` drops meta_info,
  and the immediately following `for batch_idx, data in enumerate(dataloader)` rebinds
  `data` to a bare TensorDict. So inside the loop `data.meta_info.get('te_lambda')` was
  always 0 and the TE-KL never executed once; at the same time te_log_q/te_valid/turn_ids
  were not in select_keys and got dropped entirely.
  Both are silent failures -- configuration complete, logs normal, nothing in the
  gradient -- and a lambda=0 dry run cannot expose either.

This test checks both invariants statically with the AST.
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
    assert fn is not None, "update_policy not found"

    # --- 1) te_lambda must be read from meta_info at the function's top level ---
    top_assign_line = None
    for node in fn.body:                       # top-level statements only, not the loops
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'te_lambda':
                    src = ast.dump(node.value)
                    assert 'meta_info' in src, "te_lambda must come from meta_info"
                    top_assign_line = node.lineno
    assert top_assign_line, \
        "te_lambda must be assigned at update_policy's top level (not read from data.meta_info inside the for loop)"

    # --- 2) and it must come before the dataloader rebind ---
    rebind_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
            names = [e.id for e in node.target.elts if isinstance(e, ast.Name)]
            if 'data' in names:
                rebind_line = node.lineno
    assert rebind_line, "`for batch_idx, data in enumerate(dataloader)` not found"
    assert top_assign_line < rebind_line, \
        f"te_lambda is assigned on line {top_assign_line}, after the data rebind on line {rebind_line}"

    # --- 3) nothing inside the loop may read meta_info: `data` there is a bare TensorDict ---
    loop = next(n for n in ast.walk(fn)
                if isinstance(n, ast.For) and n.lineno == rebind_line)
    meta_reads = [n.lineno for n in ast.walk(loop)
                  if isinstance(n, ast.Attribute) and n.attr == 'meta_info']
    assert not meta_reads, f"meta_info read inside the rebinding loop at lines {meta_reads}"

    # --- 4) the TE tensors must be in select_keys ---
    src_txt = open(SRC).read()
    i_sel = src_txt.index('select_keys = [')
    i_loop = src_txt.index('for batch_idx, data in enumerate(dataloader)')
    window = src_txt[i_sel:i_loop]
    for k in ('te_log_q', 'te_valid', 'turn_ids'):
        assert k in window, f"{k} is not in select_keys and would be dropped by the select"
    assert ('te_lambda > 0' in window or "te_mix == 'fullvocab'" in window), \
        "the select_keys additions must be gated on a TE condition (so the path is verbatim unchanged when TE is off)"

    # --- 5) the TE-KL gate is still a double gate ---
    assert "te_lambda > 0.0 and 'te_log_q' in data.keys()" in src_txt, "legacy TE-KL double gate is missing"

    print(f"property 14 PASS: te_lambda is read from meta_info on line {top_assign_line}, "
          f"before the data rebind on line {rebind_line}")
    print("  te_log_q/te_valid/turn_ids are all in select_keys, gated on te_lambda>0")
    print("  the TE-KL double gate is intact (lambda>0 and te_log_q actually in the batch)")
