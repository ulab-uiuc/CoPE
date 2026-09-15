"""The additive claim, asserted element-wise.

The README says every auxiliary signal in this repo is additive: with its flag off,
the training path is the stock GRPO one. That is the property a reader should be able
to check before trusting any result, so it is checked here rather than asserted in
prose.

Run with:  pytest tests/test_additive.py
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verl.agent_trainer.ppo.core_algos import compute_grpo_outcome_advantage  # noqa: E402
from verl.trainer.ppo.info_grpo import compute_info_grpo_advantage  # noqa: E402


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


def test_info_grpo_with_zero_intrinsic_equals_grpo():
    """intrinsic_weight=0 must reproduce plain GRPO exactly, not merely closely."""
    rewards, eos_mask, index = _batch()
    zero = torch.zeros_like(rewards)

    base_adv, base_ret = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, eos_mask=eos_mask, index=index)
    info_adv, info_ret = compute_info_grpo_advantage(
        token_level_rewards=rewards, token_level_intrinsic_rewards=zero,
        eos_mask=eos_mask, index=index, intrinsic_weight=0.0)

    assert torch.equal(base_adv, info_adv), (base_adv - info_adv).abs().max()
    assert torch.equal(base_ret, info_ret)


def test_info_grpo_zero_intrinsic_holds_across_seeds():
    """The equality is structural, so it cannot depend on the draw."""
    for seed in range(5):
        rewards, eos_mask, index = _batch(seed=seed, bs=12, groups=3)
        base, _ = compute_grpo_outcome_advantage(
            token_level_rewards=rewards, eos_mask=eos_mask, index=index)
        info, _ = compute_info_grpo_advantage(
            token_level_rewards=rewards,
            token_level_intrinsic_rewards=torch.zeros_like(rewards),
            eos_mask=eos_mask, index=index, intrinsic_weight=0.0)
        assert torch.equal(base, info), f"seed {seed}"


def test_advantages_are_masked_to_response_tokens():
    """Padding must carry no advantage, or the loss picks up gradient from nothing."""
    rewards, eos_mask, index = _batch(seed=3)
    for adv in (
        compute_grpo_outcome_advantage(
            token_level_rewards=rewards, eos_mask=eos_mask, index=index)[0],
        compute_info_grpo_advantage(
            token_level_rewards=rewards,
            token_level_intrinsic_rewards=torch.rand_like(rewards) * eos_mask,
            eos_mask=eos_mask, index=index, intrinsic_weight=0.1)[0],
    ):
        assert torch.equal(adv * (1 - eos_mask), torch.zeros_like(adv))


def test_nonzero_intrinsic_actually_changes_the_advantage():
    """Guards the other direction: the zero-weight test must not pass vacuously."""
    rewards, eos_mask, index = _batch(seed=7)
    intrinsic = torch.rand_like(rewards) * eos_mask

    off, _ = compute_info_grpo_advantage(
        token_level_rewards=rewards, token_level_intrinsic_rewards=intrinsic,
        eos_mask=eos_mask, index=index, intrinsic_weight=0.0)
    on, _ = compute_info_grpo_advantage(
        token_level_rewards=rewards, token_level_intrinsic_rewards=intrinsic,
        eos_mask=eos_mask, index=index, intrinsic_weight=0.5)

    assert not torch.equal(off, on)
