"""Progress/Exploration credit: add an all-POSITIVE advantage bonus on the action
tokens of model-classified PROGRESS steps (omega_p) and EXPLORATION steps (omega_e),
with omega_p > omega_e. A step judged BOTH P and E counts as P. Like HCAPO's additive
micro advantage but a fixed positive bonus per category (no group-norm, no negatives).

Default applied only on WINNING trajectories (A_GRPO > threshold), so we never push
up progress/exploration of failed rollouts. Gated by a switch in ray_trainer.

This module holds the PURE tensor application (CPU-testable). The per-turn P/E masks
are produced by the worker (text classifiers) as (B,T) tensors aligned to action
tokens; this function consumes them.
"""
import torch
from typing import Dict, Tuple


def apply_pe_credit(advantages: torch.Tensor,
                    response_mask: torch.Tensor,
                    progress_mask: torch.Tensor,
                    explore_mask: torch.Tensor,
                    omega_p: float,
                    omega_e: float,
                    wins_only: bool = True,
                    success_threshold: float = 0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """advantages, *_mask : (B, T). progress_mask / explore_mask are 1 on the action
    tokens of classified progress / exploration turns. Returns (new_advantages, metrics).

      bonus = omega_p * progress + omega_e * explore_only_not_progress      (per token)
      A' = A + bonus * win_gate
    """
    adv = advantages
    pm = (progress_mask > 0.5).to(adv.dtype)
    em = ((explore_mask > 0.5) & (progress_mask <= 0.5)).to(adv.dtype)   # P takes priority
    bonus = omega_p * pm + omega_e * em                                  # (B,T), all >= 0

    if wins_only:
        rm = response_mask.bool()
        B = adv.shape[0]
        win = torch.zeros(B, 1, dtype=adv.dtype, device=adv.device)
        for i in range(B):
            m = rm[i]
            if m.any() and float(adv[i][m].mean().item()) > success_threshold:
                win[i, 0] = 1.0
        bonus = bonus * win

    new_adv = adv + bonus
    added = bonus[bonus > 0]
    metrics = {
        "pe_credit/active": 1.0,
        "pe_credit/omega_p": float(omega_p),
        "pe_credit/omega_e": float(omega_e),
        "pe_credit/frac_progress_tok": float(pm.mean().item()),
        "pe_credit/frac_explore_tok": float(em.mean().item()),
        "pe_credit/bonus_mean_over_added": float(added.mean().item()) if added.numel() else 0.0,
        "pe_credit/n_tokens_boosted": float((bonus > 0).sum().item()),
    }
    return new_adv, metrics
