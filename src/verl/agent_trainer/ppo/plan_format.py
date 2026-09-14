"""Plan FORMAT reward: a per-turn shaping signal that rewards turns which emit a
well-formed K-action Plan block, and penalizes turns that drop it. Plan-only —
deliberately simple (nothing about Thought/Action/ordering).

Motivation: under RL the inline plan behaviour decays (the policy drops the Plan
block once it is "good enough" on task reward). A format reward applied per-turn
on the advantage directly counter-pressures that decay and does NOT vanish when
the group's task reward has no variance (unlike adding to the GRPO return).

Two pure pieces (CPU-testable, stdlib + torch only):
  - score_turn_format(text, k)        : graded [0,1] format compliance of one turn
  - apply_plan_format_advantage(...)  : add coef*(score - baseline) on each turn's
                                        response tokens (pure tensor)
The per-turn -> (B,T) scatter lives in ray_trainer (driver-side, from
rollout_messages + turn boundaries).
"""
import re
from typing import Dict, Tuple

_NUM_LINE = re.compile(r"(?m)^\s*(\d+)\.\s+\S")


def count_plan_lines(text: str) -> int:
    """Number of 'N. <something>' numbered lines in the Plan section.

    Counts numbered lines that appear AFTER 'Plan:' and BEFORE 'Action:' (if
    present), so the plan list is isolated from any incidental numbering.
    """
    lt = text.lower()
    p = lt.find("plan:")
    if p < 0:
        return 0
    a = lt.find("action:", p)
    segment = text[p: a if a >= 0 else len(text)]
    return len(_NUM_LINE.findall(segment))


def score_turn_format(text: str, k: int = 3) -> float:
    """Plan presence reward for one generated turn: the fraction of the expected
    K plan lines present (0 if there is no 'Plan:' section). Pure plan signal —
    nothing about Thought/Action/order. Range [0, 1]."""
    if "plan:" not in (text or "").lower():
        return 0.0
    return min(count_plan_lines(text), k) / max(1, k)


def apply_plan_format_advantage(advantages, response_mask, format_tok,
                                coef: float, baseline: float = 0.5,
                                clip: float = 0.0,
                                penalty_only: bool = True) -> Tuple["object", Dict[str, float]]:
    """A'_t = A_t + bonus_t on response tokens, bonus_t = coef*(format_tok-baseline).

    advantages / response_mask / format_tok : (B, T). ``format_tok`` carries each
    turn's [0,1] plan score broadcast over that turn's response tokens (0 off the
    response region).

    penalty_only=True (default, recommended): clamp bonus to <= 0, so the reward
    ONLY penalizes turns that drop the plan (score < baseline) and NEVER rewards
    turns that keep it. This keeps the signal dormant while compliance is high
    (no per-token positive bonus -> no length-inflation incentive, no positive
    advantage bias) and active only when the plan degenerates. With baseline as
    the threshold, a turn scoring >= baseline gets 0.

    penalty_only=False: symmetric (+coef/2 for full plan, -coef/2 for none).
    Optional ``clip`` (>0) caps the per-token bonus magnitude.
    """
    import torch
    rm = response_mask.to(advantages.dtype)
    bonus = coef * (format_tok.to(advantages.dtype) - baseline) * rm
    if penalty_only:
        bonus = bonus.clamp(max=0.0)
    if clip and clip > 0:
        bonus = bonus.clamp(-clip, clip)
    new_adv = advantages + bonus

    nz = rm.sum().clamp(min=1.0)
    mean_score = float((format_tok.to(advantages.dtype) * rm).sum().item() / nz.item())
    pen_tok = (bonus < 0).to(advantages.dtype)
    metrics = {
        "plan_format/active": 1.0,
        "plan_format/coef": float(coef),
        "plan_format/baseline": float(baseline),
        "plan_format/penalty_only": 1.0 if penalty_only else 0.0,
        "plan_format/mean_score": mean_score,
        "plan_format/bonus_abs_mean": float((bonus.abs().sum().item()) / nz.item()),
        "plan_format/frac_tokens_penalized": float((pen_tok * rm).sum().item() / nz.item()),
    }
    return new_adv, metrics
