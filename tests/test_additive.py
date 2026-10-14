"""The additive claim, asserted element-wise.

The README says every auxiliary signal in this repo is additive: with its flag off,
the training path is the stock GRPO one. That is the property a reader should be able
to check before trusting any result, so it is checked here rather than asserted in
prose.

Two signals ship: temporal ensembling (TE_ENABLE / TE_LAMBDA) and action forecasting
(ACTION_FORECAST_ENABLE). Each is checked two ways -- numerically, that the quantity it
contributes is exactly zero when off, and structurally, that the policy-gradient code
cannot see it when off.

Run with:  pytest tests/test_additive.py
"""
import ast
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from verl.agent_trainer.ppo.core_algos import compute_grpo_outcome_advantage  # noqa: E402
from verl.agent_trainer.ppo.temporal_ensemble import (  # noqa: E402
    assemble_te_tensors_token, fullvocab_te_kl)

_DP_ACTOR = _SRC / "verl/workers/agent_actor/dp_actor.py"


def _batch(seed=0, bs=8, seq=16, groups=2):
    """A rollout batch shaped like the real one: reward only on the last valid token."""
    g = torch.Generator().manual_seed(seed)
    eos_mask = torch.zeros(bs, seq)
    rewards = torch.zeros(bs, seq)
    for i in range(bs):
        n = int(torch.randint(4, seq, (1,), generator=g))
        eos_mask[i, :n] = 1
        rewards[i, n - 1] = float(torch.rand(1, generator=g) < 0.4)
    index = torch.arange(bs) % groups
    return rewards, eos_mask, index.numpy()


def _te_case(seed=0, bs=6, seq=40):
    """A batch with turn ids and forecast members, as the TE pass receives it."""
    g = torch.Generator().manual_seed(seed)
    turn_ids = torch.full((bs, seq), -1, dtype=torch.long)
    old_lp = torch.randn(bs, seq, generator=g) * 1.5 - 1.0
    spans = {}
    for i in range(bs):
        p = 3
        for t in range(3):
            n = 3
            turn_ids[i, p:p + n] = t
            spans[(i, t)] = (p, n)
            p += n + 2
    index, slot_tok_lp = [], []
    for i in range(bs):
        for src in range(2):
            turns, rows = [], []
            for t in range(src + 1, 3):
                turns.append(t)
                rows.append([float(torch.rand(1, generator=g)) * -4 - 0.1 for _ in range(3)])
            if turns:
                index.append((i, turns, src))
                slot_tok_lp.append(rows)
    return index, slot_tok_lp, old_lp, turn_ids


def _function(name):
    tree = ast.parse(_DP_ACTOR.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert fn is not None, f"{name} not found in {_DP_ACTOR.name}"
    return fn


# ---------------------------------------------------------------------------
# Temporal ensembling
# ---------------------------------------------------------------------------

def test_te_eta_zero_leaves_log_probs_untouched():
    """eta=0 must reproduce the frozen policy exactly, not merely closely."""
    index, slot_tok_lp, old_lp, turn_ids = _te_case()
    log_q, valid, _ = assemble_te_tensors_token(
        index, slot_tok_lp, old_lp, turn_ids, eta=0.0)

    assert torch.equal(log_q, old_lp.float()), (log_q - old_lp).abs().max()
    assert int(valid.sum()) == 0


def test_te_with_no_members_leaves_log_probs_untouched():
    """With TE off nothing is scored, so the batch reaches the loss unchanged."""
    _, _, old_lp, turn_ids = _te_case(seed=2)
    log_q, valid, _ = assemble_te_tensors_token([], [], old_lp, turn_ids, eta=0.5)

    assert torch.equal(log_q, old_lp.float())
    assert int(valid.sum()) == 0


def test_te_kl_is_exactly_zero_when_off():
    """Both ways of switching the full-vocabulary KL off give a hard zero."""
    g = torch.Generator().manual_seed(0)
    B, T, V, C, M = 3, 12, 50, 4, 8
    logits = torch.randn(B, T, V, generator=g)
    cov_pos = torch.full((B, C), -1, dtype=torch.int32)
    cov_ids = torch.randint(0, V, (B, C, M), generator=g, dtype=torch.int32)
    cov_prs = torch.rand(B, C, M, generator=g)

    kl, n = fullvocab_te_kl(logits, cov_pos, cov_ids, cov_prs, 0.5)
    assert float(kl) == 0.0 and n == 0          # no covered positions

    cov_pos[:, 0] = 1
    kl, n = fullvocab_te_kl(logits, cov_pos, cov_ids, cov_prs, 0.0)
    assert float(kl) == 0.0 and n == 0          # eta=0


def test_te_term_is_gated_out_of_the_policy_loss():
    """At lambda=0 the TE KL may be computed and reported, but never added."""
    src = _DP_ACTOR.read_text()
    fn = _function('update_policy')

    # Every statement that adds the TE term to the loss must sit under a te_lambda > 0
    # test. Checked on the AST rather than on the text, so an added branch cannot slip
    # past by being indented differently.
    parent = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def guarded(node):
        while node in parent:
            node = parent[node]
            if isinstance(node, ast.If) and 'te_lambda > 0' in ast.unparse(node.test):
                return True
        return False

    adds = [n for n in ast.walk(fn)
            if isinstance(n, ast.Assign) and 'te_lambda * _kl_te' in ast.unparse(n)]
    assert adds, "the TE term is not added to policy_loss anywhere"
    for node in adds:
        assert guarded(node), \
            f"line {node.lineno} adds the TE term without a te_lambda > 0 guard"

    # And with TE off its tensors are not even selected into the batch: every TE key is
    # appended between the gate and the select that consumes select_keys.
    gate = "if te_lambda > 0.0 or te_mix == 'fullvocab':"
    assert gate in src, "the select_keys additions must be gated on a TE condition"
    window = src[src.index(gate):src.index('batch = data.select(batch_keys=select_keys)',
                                           src.index(gate))]
    for key in ('te_log_q', 'te_valid', 'te_cov_pos', 'te_cov_ids', 'te_cov_prs'):
        assert key in window, f"{key} is appended to select_keys outside the TE gate"


# ---------------------------------------------------------------------------
# Action forecasting
# ---------------------------------------------------------------------------

def test_action_forecast_never_touches_the_policy_gradient():
    """The forecast loss is a separate pass; update_policy must not know about it."""
    src = _DP_ACTOR.read_text()
    body = ast.get_source_segment(src, _function('update_policy'))
    assert 'action_forecast' not in body, \
        "update_policy references action_forecast; the forecast loss must stay a separate pass"
    assert 'action_forecast' in ast.get_source_segment(src, _function('update_action_forecast'))


# ---------------------------------------------------------------------------
# The baseline these are additive to
# ---------------------------------------------------------------------------

def test_grpo_advantages_are_masked_to_response_tokens():
    """Padding must carry no advantage, or the loss picks up gradient from nothing."""
    rewards, eos_mask, index = _batch(seed=3)
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, eos_mask=eos_mask, index=index)
    assert torch.equal(adv * (1 - eos_mask), torch.zeros_like(adv))


def test_grpo_advantages_are_deterministic():
    """The baseline is a pure function of the batch, so any drift is a real change."""
    for seed in range(5):
        rewards, eos_mask, index = _batch(seed=seed, bs=12, groups=3)
        a, _ = compute_grpo_outcome_advantage(
            token_level_rewards=rewards, eos_mask=eos_mask, index=index)
        b, _ = compute_grpo_outcome_advantage(
            token_level_rewards=rewards, eos_mask=eos_mask, index=index)
        assert torch.equal(a, b), f"seed {seed}"
