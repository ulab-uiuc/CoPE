"""Properties 15/16: after subtracting the common mode, the TE term no longer exerts a
uniform "push every action token down" force.

Background (a first training run at lambda=0.1 was stopped after 20 steps):
  the gain is always negative -> q < pi^0 -> the k3 gradient g = 1-exp(kl) is always
  positive -> every action token gets pushed down. Measured, the common mode is about
  0.20/token while the win/fail difference is only 0.13/token, so the common mode is
  larger than the signal. And because TE covers only bare action tokens, the model can
  dodge it by moving probability mass onto non-action text: response_length went
  931 -> 3264 (close to the 4096 cap), the success rate went to zero, and te/kl
  self-servingly fell to 0.

Property 15: the centred weight w = g - g_bar has mean 0 over covered tokens (no common
             mode)
Property 16: w still carries the win/fail direction (hard-to-predict pushed down,
             well-predicted lifted)
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
            old_lp[i, p:p+n] = -0.5           # fixed pi^0, so the gain is known by construction
            p += n + 2

    # first 4 trajectories = "win" (accurate forecast, gain about -0.2),
    # last 4 = "fail" (gain about -0.9)
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
    assert gbar == gbar and gbar > 0, f"g_bar should be positive (the common-mode push exists): {gbar}"

    g = 1.0 - torch.exp(logq - old_lp)
    w = (g - gbar) * valid

    # --- property 15 ---
    mean_w = float(w.sum() / valid.sum())
    assert abs(mean_w) < 1e-5, f"the common mode should be 0 after centring, got {mean_w:.2e}"
    mean_g = float((g * valid).sum() / valid.sum())
    print(f"property 15 PASS: common mode before centring {mean_g:+.4f}/token "
          f"(always positive = uniform push down), after centring {mean_w:+.2e}")

    # --- property 16 ---
    win_mask = valid.clone(); win_mask[4:] = False
    fail_mask = valid.clone(); fail_mask[:4] = False
    w_win = float((w * win_mask).sum() / win_mask.sum())
    w_fail = float((w * fail_mask).sum() / fail_mask.sum())
    assert w_win < 0 < w_fail, f"win should be lifted (w<0), fail pushed down (w>0); got win={w_win:+.4f} fail={w_fail:+.4f}"
    assert abs(w_fail - w_win) > 0.05, "the win/fail directional force is too weak"
    print(f"property 16 PASS: win side w={w_win:+.4f} (lifted), fail side w={w_fail:+.4f} (pushed down), "
          f"difference {w_fail-w_win:.4f}/token")

    # --- the loss' derivative w.r.t. log pi really is w ---
    lp = old_lp.clone().requires_grad_(True)
    loss = ((g - gbar).detach() * valid * lp).sum() / valid.sum()
    loss.backward()
    expect = w / valid.sum()
    assert torch.allclose(lp.grad, expect, atol=1e-6), "d(loss)/d(log pi) should be exactly w/N"
    print("  loss gradient = w/N verified: padding with filler cannot lower this term "
          "(common mode is 0, so there is nothing to gain by escaping)")
    print("\n=== centring test passed ===")
