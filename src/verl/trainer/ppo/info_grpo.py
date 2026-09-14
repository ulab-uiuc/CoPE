"""InfoPO-style advantage: GRPO outcome signal plus a turn-level information gain.

Motivation, measured on this benchmark rather than assumed: with tau2's binary reward
78.7% of GRPO groups have no within-group spread, so the outcome term carries gradient
for only a fifth of the batch. Four configurations of plain GRPO (binary/dense reward x
local/hosted customer) each lifted exactly one domain at the other two's expense, and
telecom regressed in all four. Restoring within-group variance via dense partial credit
was necessary but not sufficient — it only moved which domain won. What is missing is a
signal that is non-degenerate *per turn* even when the whole trajectory fails.

That is the term implemented here. Following InfoPO (arXiv 2603.00656), each turn whose
action followed an environment observation earns an intrinsic reward equal to how much
seeing that observation shifted the policy's next-action distribution:

    r_info(t) = KL( pi(a_t | context, obs) || pi(a_t | context, <masked>) )

It is deliberately *not* a reshaped outcome reward: the task's definition of success is
untouched, so the resulting numbers stay comparable to a binary-reward baseline. The two
advantages are normalised separately and combined with a variance gate, so the intrinsic
term dominates only where the outcome term has nothing to say.
"""

from collections import defaultdict

import torch


def compute_info_grpo_advantage(
    token_level_rewards: torch.Tensor,
    token_level_intrinsic_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index,
    epsilon: float = 1e-6,
    intrinsic_weight: float = 0.1,
    gate_temperature: float = 0.05,
):
    """Orthogonal fusion of outcome and information-gain advantages.

    Args:
        token_level_rewards: (bs, seq) outcome reward, nonzero at the final token.
        token_level_intrinsic_rewards: (bs, seq) per-token information gain.
        eos_mask: (bs, seq) 1 on response tokens.
        index: (bs,) group id per sample; GRPO normalises within a group.
        intrinsic_weight: beta_0, the intrinsic term's ceiling (paper uses 0.1 on tau2).
        gate_temperature: T in the variance gate (paper uses 0.05 on tau2).

    Returns:
        (advantages, returns), both (bs, seq).
    """
    with torch.no_grad():
        bsz = token_level_rewards.shape[0]
        scores = (token_level_rewards * eos_mask).sum(dim=-1)

        # --- outcome advantage: standard GRPO, unchanged ---------------------------
        id2score = defaultdict(list)
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        id2mean, id2std = {}, {}
        for idx, vals in id2score.items():
            if len(vals) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                v = torch.stack(vals)
                id2mean[idx] = v.mean()
                id2std[idx] = v.std()
        adv_outcome = torch.zeros_like(scores)
        for i in range(bsz):
            adv_outcome[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        # --- intrinsic advantage: normalised per token, keeping turn granularity ---
        # Averaging it to a per-trajectory scalar first would throw away exactly the
        # information that makes it useful, namely that turns differ.
        intr = token_level_intrinsic_rewards * eos_mask
        n_tok = eos_mask.sum().clamp(min=1)
        intr_mean = intr.sum() / n_tok
        intr_var = ((intr - intr_mean) ** 2 * eos_mask).sum() / n_tok
        adv_intrinsic = (intr - intr_mean) / (intr_var.sqrt() + epsilon) * eos_mask

        # --- variance gate ---------------------------------------------------------
        # Per group, how much the outcome disagrees within the group. When a group is
        # degenerate (all rollouts scored the same) sigma is 0, the gate opens, and the
        # intrinsic term supplies the gradient that the outcome term cannot.
        gate = torch.zeros_like(scores)
        for i in range(bsz):
            sigma = id2std[index[i]]
            gate[i] = torch.exp(-sigma / gate_temperature)
        beta = intrinsic_weight * gate

        advantages = adv_outcome.unsqueeze(-1) * eos_mask + beta.unsqueeze(-1) * adv_intrinsic

    return advantages, advantages
