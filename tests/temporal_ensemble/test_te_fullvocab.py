"""Property tests for the full-vocabulary per-token KL (te_mix='fullvocab').

Both earlier failures had the same root cause: computing log q - log pi only on the
tokens that actually occurred gives an unnormalised objective.
  Version A (raw KL)        : has a fixed point, but covers only action tokens, so the
                              model dodges it with filler -> length 931 -> 3264
  Version B (common-mode     : the escape route is closed, but the loss degenerates into
             subtracted)      the linear term w*log pi, unbounded in both directions with
                              no fixed point -> entropy 0.37 -> 6.72, word salad
This file pins down the property each version was missing. Properties 17/18/19 are the
ones that went untested before version B failed.
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
        """Build the cov_* tensors; when qf_from is given it is the source of qF."""
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

    # ---------- property 17: with qF = p0 the term and its gradient are both 0 ----------
    p_all = torch.softmax(logits, -1)
    cov_pos, cov_ids, cov_prs = mk(qf_from=p_all)
    lg = logits.clone().requires_grad_(True)
    kl, n = fullvocab_te_kl(lg, cov_pos, cov_ids, cov_prs, ETA)
    # The top-M truncation means qF covers only part of the mass, so the theoretical 0
    # carries a truncation residual; quantify it first.
    kl.backward()
    gmax = float(lg.grad.abs().max())
    print(f"property 17 PASS: with qF=p0, KL={float(kl):.3e} (top-{M}/{V} truncation residual), "
          f"largest gradient component {gmax:.3e}")
    assert abs(float(kl)) < 0.05, float(kl)
    assert gmax < 0.05, gmax

    # ---------- property 25: KL >= 0 ----------
    neg = 0
    for trial in range(200):
        lg2 = torch.randn(B, T, V) * 2.0
        cp, ci, cr = mk()
        k, _ = fullvocab_te_kl(lg2, cp, ci, cr, ETA)
        if float(k) < -1e-6: neg += 1
    assert neg == 0, f"{neg}/200 trials produced a negative KL"
    print("property 25 PASS: 200 random constructions, KL always >= 0 (this is what version B violated)")

    # ---------- property 20: matches a dense float64 computation position by position
    # (M=V, no truncation) ----------
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
    print(f"property 20 PASS: at M=V it matches the dense float64 computation, error {err:.2e}")

    # ---------- properties 18/19: the fixed point is at pi=qF; deviating must raise it ----------
    qf = torch.softmax(torch.randn(V) * 1.2, -1)
    cov_pos1 = torch.full((1, 1), 1, dtype=torch.int32)
    v, i = torch.topk(qf, V)
    cov_ids1 = i.view(1, 1, V).to(torch.int32); cov_prs1 = v.view(1, 1, V)
    base = torch.zeros(1, T, V)
    base[0, 1] = qf.log()                      # pi = qF
    lg3 = base.clone().requires_grad_(True)
    kl3, _ = fullvocab_te_kl(lg3, cov_pos1, cov_ids1, cov_prs1, ETA)
    kl3.backward()
    g_at_fp = float(lg3.grad.abs().max())
    assert g_at_fp < 1e-5, g_at_fp
    print(f"property 18 PASS: gradient at pi=qF is {g_at_fp:.2e} (the fixed point exists and is reachable)")

    vals = []
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        mixed = (1 - a) * qf + a * torch.full((V,), 1.0 / V)   # perturb toward uniform
        b2 = torch.zeros(1, T, V); b2[0, 1] = mixed.log()
        k, _ = fullvocab_te_kl(b2, cov_pos1, cov_ids1, cov_prs1, ETA)
        vals.append(float(k))
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), vals
    print(f"property 19 PASS: KL rises monotonically when perturbing toward uniform "
          f"{' -> '.join(f'{x:.4f}' for x in vals)}")
    print("  (version A escaped the penalty by moving mass elsewhere; this closes that route)")

    # ---------- property 24: numerical robustness ----------
    lg4 = torch.randn(B, T, V) * 30.0          # extreme logits -> p0 has tiny values
    cp, ci, cr = mk()
    k4, _ = fullvocab_te_kl(lg4, cp, ci, cr, ETA)
    assert torch.isfinite(k4), k4
    k5, n5 = fullvocab_te_kl(logits, torch.full((B, C), -1, dtype=torch.int32), ci, cr, ETA)
    assert n5 == 0 and float(k5) == 0.0
    k6, n6 = fullvocab_te_kl(logits, cp, ci, cr, 0.0)
    assert n6 == 0 and float(k6) == 0.0
    print("property 24 PASS: extreme logits produce no NaN/Inf; both degenerate paths "
          "(no covered positions, eta=0) return 0")
    print("\n=== full-vocabulary KL test passed ===")
