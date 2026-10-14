"""Property tests for per-token mixing (te_mix='token').

Property 10: assemble_te_tensors_token's mixture matches a direct float64 computation
             position by position
Property 11: members whose length does not match are dropped and counted, never
             truncated into a misalignment
Property 12: te_log_q has the same shape as log_prob -> dp_actor takes the per-token
             branch and the gradient flows
Property 13: with no members, or eta=0, te_log_q == old_log_probs -> the KL is
             identically 0 (lazy degeneration)
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
                        n += 1                       # deliberate length mismatch
                    turns.append(t)
                    rows.append([random.uniform(-4, -0.1) for _ in range(n)])
                if turns:
                    index.append((i, turns, src))
                    slot_tok_lp.append(rows)
        return turn_ids, old_lp, index, slot_tok_lp, spans

    # ---------- property 10 ----------
    eta = 0.5
    turn_ids, old_lp, index, slot_tok_lp, spans = make_case()
    logq, valid, meta = assemble_te_tensors_token(index, slot_tok_lp, old_lp, turn_ids,
                                                  eta=eta, traj_return=None)
    # independent float64 recomputation
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
    print(f"property 10 PASS: {nchk} token positions, max error vs direct float64 {maxerr:.2e}")

    # ---------- property 11 ----------
    turn_ids2, old_lp2, index2, slot_tok2, spans2 = make_case(bad_len=True)
    logq2, valid2, meta2 = assemble_te_tensors_token(index2, slot_tok2, old_lp2,
                                                     turn_ids2, eta=0.5)
    assert meta2['te/token_len_mismatch'] > 0, meta2
    # every member of turn 2 has a mismatched length -> those tokens must stay = old_log_probs
    bad = 0
    for i in range(B):
        p0, n = spans2[(i, 2)]
        if not bool(valid2[i, p0]):
            assert torch.allclose(logq2[i, p0:p0 + n], old_lp2[i, p0:p0 + n]); bad += 1
    assert bad > 0
    print(f"property 11 PASS: {int(meta2['te/token_len_mismatch'])} length-mismatched members dropped, "
          f"that turn falls back to pi^0 in {bad} trajectories (no truncated misalignment)")

    # ---------- property 12 ----------
    assert logq.shape == old_lp.shape, (logq.shape, old_lp.shape)
    lp = old_lp.clone().requires_grad_(True)
    kl = logq.detach() - lp
    kld = torch.exp(kl) - kl - 1
    loss = (kld * valid.to(kld.dtype)).sum() / valid.sum().clamp(min=1)
    loss.backward()
    g = lp.grad
    assert torch.isfinite(g).all()
    assert (g[valid] != 0).any() and (g[~valid] == 0).all()
    print(f"property 12 PASS: te_log_q has the same shape as log_prob {tuple(logq.shape)}; "
          f"the gradient lands only on the {int(valid.sum())} covered tokens, 0 everywhere else")

    # ---------- property 13 ----------
    logq0, valid0, _ = assemble_te_tensors_token(index, slot_tok_lp, old_lp, turn_ids, eta=0.0)
    assert torch.allclose(logq0, old_lp.float()), "eta=0 should degenerate fully to pi^0"
    assert int(valid0.sum()) == 0
    logqE, validE, _ = assemble_te_tensors_token([], [], old_lp, turn_ids, eta=0.5)
    assert torch.allclose(logqE, old_lp.float()) and int(validE.sum()) == 0
    print("property 13 PASS: both degenerate branches (eta=0, no members) give te_log_q == pi^0, KL identically 0")
    print("\n=== per-token mixing tests all passed ===")
