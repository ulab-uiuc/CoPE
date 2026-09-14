"""World Model-Conditioned Entropy Regularized Clipping utilities.

This module adapts the WMC-ERC behavior used in OpenTinker to the
AgentGym-RL training loop. It uses per-token policy entropy as a world-model
uncertainty signal on observation tokens and applies a turn-level mask or
soft clipping coefficient to actor advantages.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist


def _epistemic_shape_fn(x: float, shape: str) -> float:
    """Concave (or linear) shaping of the normalized epistemic signal.

    `x = U_t / u_scale` (>= 0). The U -> weight relationship should be
    'on-vs-off sensitive, more-vs-much-more blunt': steep near 0 (cleanly
    separate epistemic-bearing turns from calibrated/aleatoric ones), flat for
    large x (importance saturates with surprise; robust to single-model proxy
    outliers). Hence a concave default.
    """
    if shape == "linear":
        return x
    if shape == "sqrt":
        return math.sqrt(x)
    if shape == "log":
        return math.log1p(x)
    raise ValueError(f"epistemic_shape must be 'linear' | 'sqrt' | 'log', got {shape!r}")


def _turn_spans(mask_row: torch.Tensor):
    """Return [start, end) token spans of contiguous action-token runs in a
    1-D response mask. Each run is one agent action (one turn)."""
    idx = mask_row.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return []
    breaks = (idx[1:] - idx[:-1] > 1).nonzero(as_tuple=True)[0]
    bounds = [0] + (breaks + 1).tolist() + [idx.numel()]
    return [(int(idx[a].item()), int(idx[b - 1].item()) + 1)
            for a, b in zip(bounds[:-1], bounds[1:])]


def apply_hca_advantage(
    data,
    ratio_clip_min: float = 0.8,
    ratio_clip_max: float = 1.2,
    temp: float = 5.0,
    omega: float = 1.0,
    gamma: float = 0.95,
    smooth_alpha: float = 0.5,
    success_threshold: float = 0.5,
) -> Dict[str, float]:
    """HCAPO multi-scale advantage (arXiv:2603.08754, training-free).

    Reads:
        data.batch['advantages']     : (B, T) — GRPO scalar Â broadcast (macro)
        data.batch['h_log_probs']    : (B, T) — log π(a_t | s_t, s_final)
        data.batch['response_mask']  : (B, T) — 1 on action tokens
        data.batch['traj_return']    : (B,)   — per-trajectory return R
        data.non_tensor_batch['uid'] : (B,)   — group id per sample

    Writes (in place):
        data.batch['advantages'] = A_GRPO + ω · A_micro

    Per action a_t (a contiguous action-token run = one turn):
        π_hind(a_t) = exp( mean_j log π(y_j | …, s_final) / T_temp )      (Eq.6)
        ρ_t = clip( π_hind(a_t) / mean_k π_hind(a_k), C_min, C_max )       (Eq.7)
        Q^H_t = ρ_t · γ^{T−1−t} · R                                       (Eq.5)
        (optional) Q̃^H_t = α·Q^H_t + (1−α)·Q^H_{t+1}     temporal smoothing
        A_micro_t = (Q^H_t − μ_H) / σ_H    cross-state group-normalized     (Eq.8)
    do-no-harm: zero out negative A_micro on successful trials.
    """
    import numpy as np
    advantages = data.batch['advantages']                       # (B, T)
    response_mask = data.batch['response_mask']                 # (B, T)
    h_log_probs = data.batch['h_log_probs'].to(advantages.dtype)
    R = data.batch['traj_return'].to(advantages.dtype)          # (B,)
    uid = data.non_tensor_batch['uid']
    B, T = advantages.shape
    device = advantages.device
    dt = advantages.dtype

    _, inverse = np.unique(uid, return_inverse=True)
    grp = torch.as_tensor(inverse, dtype=torch.long)            # (B,) group idx
    rm_bool = response_mask.to(torch.bool)

    # ---- Pass 1: per-turn ρ and Q^H, scatter Q into a per-token tensor ----
    Q_tok = torch.zeros(B, T, dtype=dt, device=device)
    turn_q = []          # flat per-turn Q^H values
    turn_grp = []        # group index per turn
    turn_loc = []        # (sample_i, start, end) for scatter-back of A_micro

    rho_all = []
    n_clipped = 0
    n_turns_total = 0

    for i in range(B):
        spans = _turn_spans(rm_bool[i])
        if not spans:
            continue
        Ti = len(spans)
        # π_hind per turn (sharpened geometric mean of token probs)
        pi_hind = []
        for (s, e) in spans:
            # π_hind over the whole assistant turn (Thought+Action). Action-only
            # scoring was removed: in CoT agents the action string is a near-
            # deterministic readout of the Thought (per-token logp≈0), so
            # action-only flattens ρ — the hindsight signal lives in the Thought.
            mean_logp = h_log_probs[i, s:e].mean()
            pi_hind.append(torch.exp(mean_logp / max(temp, 1e-6)))
        pi_hind = torch.stack(pi_hind)                          # (Ti,)
        pi_bar = pi_hind.mean().clamp(min=1e-8)
        rho = torch.clamp(pi_hind / pi_bar, ratio_clip_min, ratio_clip_max)
        n_clipped += int(((pi_hind / pi_bar < ratio_clip_min) |
                          (pi_hind / pi_bar > ratio_clip_max)).sum().item())
        n_turns_total += Ti
        rho_all.append(rho.detach())

        # discounted return G_t = γ^{T-1-t} · R  (t 0-based, last turn → γ^0)
        t_idx = torch.arange(Ti, device=device, dtype=dt)
        G = (gamma ** (Ti - 1 - t_idx)) * R[i]
        Q = rho.to(dt) * G                                      # (Ti,)

        # temporal smoothing: blend with the next turn's Q (last turn → itself)
        if smooth_alpha < 1.0 and Ti > 1:
            Q_next = torch.cat([Q[1:], Q[-1:]])
            Q = smooth_alpha * Q + (1.0 - smooth_alpha) * Q_next

        for ti, (s, e) in enumerate(spans):
            turn_q.append(Q[ti])
            turn_grp.append(int(grp[i].item()))
            turn_loc.append((i, s, e))

    if not turn_q:
        # no action tokens anywhere — leave A_GRPO untouched
        return {'hca/num_turns': 0.0}

    turn_q_t = torch.stack(turn_q)                              # (N,)
    turn_grp_t = torch.tensor(turn_grp, dtype=torch.long, device=device)

    # ---- Pass 2: cross-state group normalization of Q^H (μ_H, σ_H per group),
    #             do-no-harm mask, scatter A_micro back to action tokens ----
    n_g = int(turn_grp_t.max().item()) + 1
    g_sum = torch.zeros(n_g, dtype=dt, device=device).scatter_add_(0, turn_grp_t, turn_q_t)
    g_cnt = torch.zeros(n_g, dtype=dt, device=device).scatter_add_(
        0, turn_grp_t, torch.ones_like(turn_q_t))
    g_mean = g_sum / g_cnt.clamp(min=1.0)
    g_sqsum = torch.zeros(n_g, dtype=dt, device=device).scatter_add_(
        0, turn_grp_t, turn_q_t * turn_q_t)
    g_var = (g_sqsum / g_cnt.clamp(min=1.0)) - g_mean * g_mean
    g_std = g_var.clamp(min=0.0).sqrt()

    A_micro_tok = torch.zeros(B, T, dtype=dt, device=device)
    a_micro_vals = []
    for n, (i, s, e) in enumerate(turn_loc):
        g = int(turn_grp_t[n].item())
        a_micro = (turn_q_t[n] - g_mean[g]) / (g_std[g] + 1e-6)
        if R[i] > success_threshold and a_micro < 0:   # do-no-harm on wins
            a_micro = torch.zeros((), dtype=dt, device=device)
        A_micro_tok[i, s:e] = a_micro
        a_micro_vals.append(a_micro.detach())

    # ---- Multi-scale combine: A = A_GRPO + ω · A_micro (action tokens) ----
    data.batch['advantages'] = advantages + omega * A_micro_tok * response_mask.to(dt)

    rho_cat = torch.cat(rho_all) if rho_all else torch.zeros(1, device=device)
    a_micro_cat = torch.stack(a_micro_vals)
    rm_sum = response_mask.sum().clamp(min=1.0)
    metrics = {
        'hca/rho_mean': float(rho_cat.mean().item()),
        'hca/rho_min': float(rho_cat.min().item()),
        'hca/rho_max': float(rho_cat.max().item()),
        'hca/rho_clip_frac': float(n_clipped / max(1, n_turns_total)),
        'hca/Q_mean': float(turn_q_t.mean().item()),
        'hca/Q_abs_mean': float(turn_q_t.abs().mean().item()),
        'hca/A_micro_mean': float(a_micro_cat.mean().item()),
        'hca/A_micro_abs_mean': float(a_micro_cat.abs().mean().item()),
        'hca/A_micro_nonzero_frac': float((a_micro_cat.abs() > 1e-6).float().mean().item()),
        'hca/num_turns': float(n_turns_total),
        'hca/turns_per_traj': float(n_turns_total / max(1, B)),
        'hca/z_success_frac': float((R > success_threshold).float().mean().item()),
    }
    return metrics


def compute_turn_boundaries(response_mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """Extract contiguous assistant-token spans from response_mask.

    Each span corresponds to one action turn in the rollout response region.
    """
    boundaries_per_sample: List[List[Tuple[int, int]]] = []

    for sample_mask in response_mask.bool():
        sample_boundaries: List[Tuple[int, int]] = []
        start = None
        for idx, flag in enumerate(sample_mask.tolist()):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                sample_boundaries.append((start, idx))
                start = None
        if start is not None:
            sample_boundaries.append((start, len(sample_mask)))
        boundaries_per_sample.append(sample_boundaries)

    return boundaries_per_sample


def compute_s_star(
    old_log_probs: torch.Tensor,
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn policy blind confidence S_*.

    S_*^t = mean over action tokens in turn t of: p_k * (H + log p_k)
    """
    batch_size = old_log_probs.shape[0]
    device = old_log_probs.device
    s_star_per_sample = []

    for i in range(batch_size):
        s_star_turns = []
        for start, end in turn_boundaries[i]:
            log_p = old_log_probs[i, start:end]
            H = entropys[i, start:end]
            mask = response_mask[i, start:end]
            count = mask.sum()

            if count > 0:
                p_k = torch.exp(log_p)
                s_token = p_k * (H + log_p)
                s_token = torch.nan_to_num(s_token, nan=0.0)
                s_t = (s_token * mask).sum() / count
            else:
                s_t = torch.tensor(0.0, device=device)

            s_star_turns.append(s_t)
        s_star_per_sample.append(s_star_turns)

    return s_star_per_sample


def compute_h_wm(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask_response: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn World Model Loss (NLL).

    WM_Loss^t = mean negative log-likelihood at env token positions following action turn t.
    """
    batch_size = old_log_probs.shape[0]
    seq_len = old_log_probs.shape[1]
    device = old_log_probs.device
    env_mask = attention_mask_response * (1.0 - response_mask)

    h_wm_per_sample = []

    for i in range(batch_size):
        boundaries = turn_boundaries[i]
        h_wm_turns = []

        for t, (start, end) in enumerate(boundaries):
            if t + 1 < len(boundaries):
                env_end = boundaries[t + 1][0]
            else:
                env_end = seq_len

            region_mask = env_mask[i, end:env_end]
            region_log_prob = old_log_probs[i, end:env_end]
            count = region_mask.sum()

            if count > 0:
                # WM Loss is -log_prob
                h_wm_t = -(region_log_prob * region_mask).sum() / count
            else:
                h_wm_t = torch.tensor(0.0, device=device)

            h_wm_turns.append(h_wm_t)

        h_wm_per_sample.append(h_wm_turns)

    return h_wm_per_sample


def compute_h_wm_entropy(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask_response: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn World Model Entropy.

    H_WM^t = mean prediction entropy at env token positions following action turn t.
    """
    batch_size = entropys.shape[0]
    seq_len = entropys.shape[1]
    device = entropys.device
    env_mask = attention_mask_response * (1.0 - response_mask)

    h_wm_per_sample = []

    for i in range(batch_size):
        boundaries = turn_boundaries[i]
        h_wm_turns = []

        for t, (start, end) in enumerate(boundaries):
            if t + 1 < len(boundaries):
                env_end = boundaries[t + 1][0]
            else:
                env_end = seq_len

            region_mask = env_mask[i, end:env_end]
            region_entropy = entropys[i, end:env_end]
            count = region_mask.sum()

            if count > 0:
                h_wm_t = (region_entropy * region_mask).sum() / count
            else:
                h_wm_t = torch.tensor(0.0, device=device)

            h_wm_turns.append(h_wm_t)

        h_wm_per_sample.append(h_wm_turns)

    return h_wm_per_sample


def compute_h_action(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn mean action entropy.

    H_action^t = mean per-token entropy over action tokens in turn t.
    """
    batch_size = entropys.shape[0]
    device = entropys.device
    h_action_per_sample = []

    for i in range(batch_size):
        h_action_turns = []
        for start, end in turn_boundaries[i]:
            H = entropys[i, start:end]
            mask = response_mask[i, start:end]
            count = mask.sum()

            if count > 0:
                h_t = (H * mask).sum() / count
            else:
                h_t = torch.tensor(0.0, device=device)

            h_action_turns.append(h_t)
        h_action_per_sample.append(h_action_turns)

    return h_action_per_sample


def compute_pi_per_turn(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn action probability p(a|s).

    p(a|s)^t = exp(sum_{k in turn t} log p_k)
    """
    batch_size = old_log_probs.shape[0]
    device = old_log_probs.device
    pi_per_turn_per_sample = []

    for i in range(batch_size):
        pi_turns = []
        for start, end in turn_boundaries[i]:
            log_p = old_log_probs[i, start:end]
            mask = response_mask[i, start:end]

            if mask.sum() > 0:
                # Sum log probs of action tokens in this turn
                turn_log_prob = (log_p * mask).sum()
                pi_t = torch.exp(turn_log_prob)
            else:
                pi_t = torch.tensor(1.0, device=device)

            pi_turns.append(pi_t)
        pi_per_turn_per_sample.append(pi_turns)

    return pi_per_turn_per_sample


def _prepare_wmc_erc_quantities(
    batch,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
    current_step: int,
):
    """Compute the shared WMC-ERC quantities used by both the
    post-advantage `apply_wmc_erc` and the pre-advantage
    `apply_wmc_erc_to_reward` entry points.

    Returns a dict with: turn_boundaries, s_star, h_wm_nll, h_wm_entropy,
    h_action, pi_per_turn, h_wm_ref, target_h, traj_failed, batch_s_bar,
    batch_s_std, batch_h_bar, response_mask, attention_mask_response,
    response_length, batch_size, device. Side-effect: updates running_stats.
    Returns None if the batch has no turns at all.
    """
    response_mask = batch.batch["response_mask"]
    old_log_probs = batch.batch["old_log_probs"]
    attention_mask = batch.batch["attention_mask"]
    batch_size = response_mask.shape[0]
    response_length = response_mask.shape[1]
    device = response_mask.device
    attention_mask_response = attention_mask[:, -response_length:]

    # Success/Failure info
    wmloss_add_only_failed = bool(wmc_erc_config.get("wmloss_add_only_failed", False))
    if wmloss_add_only_failed and 'task_scores' in batch.batch.keys():
        traj_scores = batch.batch['task_scores'].sum(dim=-1)
        traj_failed = (traj_scores <= 0.0)
    else:
        traj_failed = torch.ones(batch_size, dtype=torch.bool, device=device)

    turn_boundaries = compute_turn_boundaries(response_mask)

    s_star = compute_s_star(old_log_probs, entropys, response_mask, turn_boundaries)
    h_wm_nll = compute_h_wm(old_log_probs, response_mask, attention_mask_response, turn_boundaries)
    h_wm_entropy = compute_h_wm_entropy(entropys, response_mask, attention_mask_response, turn_boundaries)
    h_action = compute_h_action(entropys, response_mask, turn_boundaries)
    pi_per_turn = compute_pi_per_turn(old_log_probs, response_mask, turn_boundaries)

    ref_entropy = batch.batch.get("ref_entropy", None)
    if ref_entropy is not None:
        h_wm_ref = compute_h_wm_entropy(ref_entropy, response_mask, attention_mask_response, turn_boundaries)
    else:
        h_wm_ref = None

    use_entropy = bool(wmc_erc_config.get("wmloss_add_use_entropy", False))
    target_h = h_wm_entropy if use_entropy else h_wm_nll

    all_s = [s.item() for turns in s_star for s in turns]
    all_h_nll = [h.item() for turns in h_wm_nll for h in turns]
    all_h_entropy = [h.item() for turns in h_wm_entropy for h in turns]
    if not all_s:
        return None

    all_s_tensor = torch.tensor(all_s, device=device, dtype=torch.float32)
    all_h_nll_tensor = torch.tensor(all_h_nll, device=device, dtype=torch.float32)
    all_h_entropy_tensor = torch.tensor(all_h_entropy, device=device, dtype=torch.float32)
    batch_s_bar_t = all_s_tensor.mean()
    batch_s_std_t = all_s_tensor.std(correction=0) if len(all_s) > 1 else torch.tensor(0.0, device=device)
    batch_h_bar_nll_t = all_h_nll_tensor.mean()
    batch_h_bar_entropy_t = all_h_entropy_tensor.mean()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch_s_bar_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_s_std_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_h_bar_nll_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_h_bar_entropy_t, op=dist.ReduceOp.AVG)

    batch_s_bar = batch_s_bar_t.item()
    batch_s_std = batch_s_std_t.item() + 1e-8
    batch_h_bar_nll = batch_h_bar_nll_t.item() + 1e-8
    batch_h_bar_entropy = batch_h_bar_entropy_t.item() + 1e-8
    # The "active" running baseline used by the add-mode offset formula must
    # match whichever quantity is used as the add term (see use_entropy).
    batch_h_bar = batch_h_bar_entropy if use_entropy else batch_h_bar_nll

    momentum = wmc_erc_config.get("momentum", 0.9) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "momentum", 0.9)
    if 's_bar' not in running_stats:
        running_stats["s_bar"] = batch_s_bar
        running_stats["s_std"] = batch_s_std
        running_stats["h_bar_nll"] = batch_h_bar_nll
        running_stats["h_bar_entropy"] = batch_h_bar_entropy
    else:
        running_stats["s_bar"] = (1 - momentum) * batch_s_bar + momentum * running_stats["s_bar"]
        running_stats["s_std"] = (1 - momentum) * batch_s_std + momentum * running_stats["s_std"]
        prev_nll = running_stats.get("h_bar_nll", running_stats.get("h_bar", batch_h_bar_nll))
        prev_ent = running_stats.get("h_bar_entropy", running_stats.get("h_bar", batch_h_bar_entropy))
        running_stats["h_bar_nll"] = (1 - momentum) * batch_h_bar_nll + momentum * prev_nll
        running_stats["h_bar_entropy"] = (1 - momentum) * batch_h_bar_entropy + momentum * prev_ent
    # `h_bar` mirrors whichever baseline matches the configured add term, so
    # downstream consumers (offset formula, legacy code paths) read the
    # right running average without branching on use_entropy themselves.
    running_stats["h_bar"] = (
        running_stats["h_bar_entropy"] if use_entropy else running_stats["h_bar_nll"]
    )

    return {
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "attention_mask_response": attention_mask_response,
        "response_length": response_length,
        "batch_size": batch_size,
        "device": device,
        "turn_boundaries": turn_boundaries,
        "s_star": s_star,
        "h_wm_nll": h_wm_nll,
        "h_wm_entropy": h_wm_entropy,
        "h_action": h_action,
        "pi_per_turn": pi_per_turn,
        "h_wm_ref": h_wm_ref,
        "target_h": target_h,
        "use_entropy": use_entropy,
        "traj_failed": traj_failed,
        "batch_s_bar": batch_s_bar,
        "batch_s_std": batch_s_std,
        "batch_h_bar": batch_h_bar,
        "batch_h_bar_nll": batch_h_bar_nll,
        "batch_h_bar_entropy": batch_h_bar_entropy,
    }


def _compute_add_offsets_and_metrics(
    batch,
    quantities: Dict,
    wmc_erc_config,
    running_stats: Dict[str, float],
    current_step: int,
) -> Tuple[List[Tuple[int, int, torch.Tensor]], Dict[str, float]]:
    """Compute per-(sample, turn) add-mode offsets and accompanying metrics.

    Returns:
        offsets: list of (sample_idx, turn_idx, offset_tensor) tuples
        add_metrics: dict of metrics (may be empty)
    """
    target_h = quantities["target_h"]
    h_wm_ref = quantities["h_wm_ref"]
    turn_boundaries = quantities["turn_boundaries"]
    traj_failed = quantities["traj_failed"]
    batch_size = quantities["batch_size"]

    # Linear decay for coef
    wmloss_add_coef_start = float(wmc_erc_config.get("wmloss_add_coef", 0.1))
    wmloss_add_coef_end = float(wmc_erc_config.get("wmloss_add_coef_end", wmloss_add_coef_start))
    wmloss_add_horizon = int(wmc_erc_config.get("wmloss_add_horizon", 1))

    if wmloss_add_horizon > 0:
        alpha_decay = min(current_step / wmloss_add_horizon, 1.0)
        wmloss_add_coef = wmloss_add_coef_start + alpha_decay * (wmloss_add_coef_end - wmloss_add_coef_start)
    else:
        wmloss_add_coef = wmloss_add_coef_start

    wmloss_add_use_grouped = bool(wmc_erc_config.get("wmloss_add_use_grouped", False))
    wmloss_add_use_ref_baseline = bool(wmc_erc_config.get("wmloss_add_use_ref_baseline", False))
    has_group_id = 'group_id' in batch.batch.keys()

    offsets: List[Tuple[int, int, torch.Tensor]] = []
    add_metrics: Dict[str, float] = {}
    all_offset_vals: List[float] = []

    if wmloss_add_use_ref_baseline and h_wm_ref is not None:
        for i in range(batch_size):
            if not traj_failed[i]:
                continue
            for t in range(len(target_h[i])):
                h_val = target_h[i][t]
                h_ref_val = h_wm_ref[i][t]
                factor_t = torch.clamp(h_val - h_ref_val, min=0.0, max=0.5)
                offset_t = wmloss_add_coef * factor_t
                offsets.append((i, t, offset_t))
                all_offset_vals.append(offset_t.item())

        all_h_ref = [h.item() for turns in h_wm_ref for h in turns]
        if all_h_ref:
            add_metrics.update({
                "wmc_erc/h_wm_ref_mean": float(np.mean(all_h_ref)),
                "wmc_erc/h_wm_ref_std": float(np.std(all_h_ref)),
            })

    elif wmloss_add_use_grouped and has_group_id:
        gids = batch.batch['group_id'].long()
        g_max_local = gids.max() if gids.numel() > 0 else torch.tensor(-1, device=gids.device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(g_max_local, op=dist.ReduceOp.MAX)
        G = int(g_max_local.item()) + 1

        flat_h = []
        flat_gids = []
        for i in range(batch_size):
            if not traj_failed[i]:
                continue
            for h_val in target_h[i]:
                flat_h.append(h_val)
                flat_gids.append(gids[i])

        if flat_h:
            flat_h_tensor = torch.stack(flat_h)
            flat_gids_tensor = torch.stack(flat_gids)

            sum_per_g = torch.zeros(G, dtype=flat_h_tensor.dtype, device=flat_h_tensor.device)
            cnt_per_g = torch.zeros(G, dtype=flat_h_tensor.dtype, device=flat_h_tensor.device)
            sum_per_g.scatter_add_(0, flat_gids_tensor, flat_h_tensor)
            cnt_per_g.scatter_add_(0, flat_gids_tensor, torch.ones_like(flat_h_tensor))

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_per_g, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt_per_g, op=dist.ReduceOp.SUM)

            mu_per_g = sum_per_g / cnt_per_g.clamp(min=1.0)

            factor_per_sample = []
            all_factors = []
            for i in range(batch_size):
                sample_factors = []
                if not traj_failed[i]:
                    factor_per_sample.append(sample_factors)
                    continue
                gid = gids[i].item()
                mu_h_g = mu_per_g[gid]
                for h_val in target_h[i]:
                    factor_t = torch.clamp(h_val - mu_h_g, min=0.0, max=0.5)
                    sample_factors.append(factor_t)
                    all_factors.append(factor_t)
                factor_per_sample.append(sample_factors)

            if all_factors:
                all_factors_tensor = torch.stack(all_factors)
                sum_fac_per_g = torch.zeros(G, dtype=all_factors_tensor.dtype, device=all_factors_tensor.device)
                sum_fac_per_g.scatter_add_(0, flat_gids_tensor, all_factors_tensor)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(sum_fac_per_g, op=dist.ReduceOp.SUM)
                mu_fac_per_g = sum_fac_per_g / cnt_per_g.clamp(min=1.0)

                for i in range(batch_size):
                    if not traj_failed[i]:
                        continue
                    gid = gids[i].item()
                    mu_fac_g = mu_fac_per_g[gid]
                    for t in range(len(factor_per_sample[i])):
                        offset_t = wmloss_add_coef * (factor_per_sample[i][t] - mu_fac_g)
                        offsets.append((i, t, offset_t))
                        all_offset_vals.append(offset_t.item())

            add_metrics.update({
                "wmc_erc/wmloss_mu_per_group_mean": mu_per_g[cnt_per_g > 0].mean().item() if (cnt_per_g > 0).any() else 0.0,
                "wmc_erc/wmloss_mu_per_group_std": mu_per_g[cnt_per_g > 0].std(unbiased=False).item() if (cnt_per_g > 0).sum() > 1 else 0.0,
            })
    else:
        # Fallback: EMA baseline. Clamp factor to [0, 0.5] for parity with the
        # grouped / ref_baseline branches above. Without the clamp this branch
        # produces NEGATIVE offsets whenever a turn's target_h is below the
        # running EMA — which, with use_entropy=True, is the typical case for
        # successful trajectories (clean env templates → low env entropy).
        # That makes curiosity anti-correlated with task reward and erodes
        # the GRPO signal-to-noise ratio (penalises success, rewards
        # confusion). Single-sided positive bonus is the intended semantic.
        mu_h = running_stats["h_bar"]
        for i in range(batch_size):
            if not traj_failed[i]:
                continue
            for t in range(len(target_h[i])):
                h_val = target_h[i][t]
                factor_t = torch.clamp(h_val - mu_h, min=0.0, max=0.5)
                offset_t = wmloss_add_coef * factor_t
                offsets.append((i, t, offset_t))
                all_offset_vals.append(offset_t.item())

    if all_offset_vals:
        add_metrics.update({
            "wmc_erc/wmloss_offset_mean": float(np.mean(all_offset_vals)),
            "wmc_erc/wmloss_offset_std": float(np.std(all_offset_vals)),
            "wmc_erc/wmloss_offset_max": float(np.max(all_offset_vals)),
            "wmc_erc/wmloss_offset_min": float(np.min(all_offset_vals)),
            "wmc_erc/wmloss_coef": float(wmloss_add_coef),
        })

    add_metrics["wmc_erc/num_failed_trajs"] = int(traj_failed.sum().item())
    return offsets, add_metrics


def apply_wmc_erc_to_reward(
    batch,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
    step: int = None,
):
    """Inject the WMC-ERC `add`-mode curiosity bonus into `token_level_rewards`
    BEFORE `compute_advantage`, so it rides through GRPO group normalization
    along with the task reward. With this routing, `wmloss_add_coef` controls
    the *relative* weight of the curiosity signal rather than its absolute
    scale, which restores the unit-variance property GRPO relies on.

    No-op unless `wmc_erc.enable`, `clipping_method == "add"`, and
    `wmloss_add_to_reward` are all set. When active, the caller is expected
    to skip the post-advantage `apply_wmc_erc` invocation to avoid
    double-counting. The bonus is deposited on the last response token of
    each turn — for outcome-based estimators (GRPO/RLOO/REMAX) only the per-
    trajectory sum matters; for position-sensitive estimators this credits
    end-of-turn.
    """
    enable = wmc_erc_config.get("enable", True) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "enable", True)
    if not enable:
        return batch, {}
    clipping_method = wmc_erc_config.get("clipping_method", "mask") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_method", "mask")
    if clipping_method != "add":
        return batch, {}
    if not bool(wmc_erc_config.get("wmloss_add_to_reward", False)):
        return batch, {}
    if "token_level_rewards" not in batch.batch.keys():
        return batch, {}

    if step is not None:
        running_stats['step'] = step
    else:
        running_stats['step'] = running_stats.get('step', 0) + 1
    current_step = running_stats['step']

    q = _prepare_wmc_erc_quantities(batch, entropys, wmc_erc_config, running_stats, current_step)
    if q is None:
        return batch, {}

    offsets, add_metrics = _compute_add_offsets_and_metrics(
        batch, q, wmc_erc_config, running_stats, current_step,
    )

    # Aggregate per-turn offsets per trajectory before injecting. GRPO's
    # outcome-based estimator does sum(token_level_rewards, dim=-1) per
    # trajectory; depositing one offset per turn at end-of-turn means a
    # T-turn trajectory contributes T*offset to the trajectory return,
    # which makes curiosity scale roughly T times the task reward and
    # dominates the GRPO group-normalized advantage. Aggregating per
    # trajectory keeps the bonus on the same scale as the task reward
    # (~[0, 1]). Default 'mean'; 'sum' kept as an opt-in for parity with
    # the legacy advantage-injection path.
    aggregate = wmc_erc_config.get("wmloss_add_to_reward_aggregate", "mean")
    if aggregate not in ("mean", "sum"):
        raise ValueError(
            f"wmloss_add_to_reward_aggregate must be 'mean' or 'sum', got {aggregate!r}"
        )

    per_traj: Dict[int, List[torch.Tensor]] = {}
    for i, _t, offset_t in offsets:
        per_traj.setdefault(i, []).append(offset_t)

    token_level_rewards = batch.batch["token_level_rewards"]
    turn_boundaries = q["turn_boundaries"]
    per_traj_offset_vals: List[float] = []
    for i, offset_list in per_traj.items():
        if not turn_boundaries[i]:
            continue
        stacked = torch.stack(offset_list)
        traj_offset = stacked.mean() if aggregate == "mean" else stacked.sum()
        # Drop the trajectory-level bonus on the last response token of the
        # whole trajectory. For outcome-based estimators only the per-traj
        # sum matters; for position-sensitive ones this credits trajectory
        # end (after the final action).
        last_end = turn_boundaries[i][-1][1] - 1
        token_level_rewards[i, last_end] = token_level_rewards[i, last_end] + traj_offset
        per_traj_offset_vals.append(traj_offset.item())
    batch.batch["token_level_rewards"] = token_level_rewards

    if per_traj_offset_vals:
        add_metrics.update({
            "wmc_erc/wmloss_traj_offset_mean": float(np.mean(per_traj_offset_vals)),
            "wmc_erc/wmloss_traj_offset_std": float(np.std(per_traj_offset_vals)),
            "wmc_erc/wmloss_traj_offset_max": float(np.max(per_traj_offset_vals)),
            "wmc_erc/wmloss_traj_offset_min": float(np.min(per_traj_offset_vals)),
            "wmc_erc/wmloss_traj_offset_n": len(per_traj_offset_vals),
            "wmc_erc/wmloss_aggregate_is_mean": 1.0 if aggregate == "mean" else 0.0,
        })

    env_mask = q["attention_mask_response"] * (1.0 - q["response_mask"])
    env_count = env_mask.sum()
    wm_nll = (-(q["old_log_probs"] * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0
    wm_entropy = ((entropys * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    metrics = {
        "wmc_erc/batch_s_bar": float(q["batch_s_bar"]),
        "wmc_erc/batch_s_std": float(q["batch_s_std"]),
        "wmc_erc/batch_h_bar": float(q["batch_h_bar"]),
        "wmc_erc/batch_h_bar_nll": float(q["batch_h_bar_nll"]),
        "wmc_erc/batch_h_bar_entropy": float(q["batch_h_bar_entropy"]),
        "wmc_erc/running_s_bar": float(running_stats["s_bar"]),
        "wmc_erc/running_s_std": float(running_stats["s_std"]),
        "wmc_erc/running_h_bar": float(running_stats["h_bar"]),
        "wmc_erc/running_h_bar_nll": float(running_stats["h_bar_nll"]),
        "wmc_erc/running_h_bar_entropy": float(running_stats["h_bar_entropy"]),
        "wmc_erc/use_entropy": 1.0 if q["use_entropy"] else 0.0,
        "wmc_erc/total_turns": sum(len(b) for b in turn_boundaries),
        "wmc_erc/wm_nll": wm_nll,
        "wmc_erc/wm_entropy": wm_entropy,
        "wmc_erc/add_to_reward": 1.0,
    }
    metrics.update(add_metrics)
    return batch, metrics


def compute_epistemic_signal(
    h_wm_nll: List[List[torch.Tensor]],
    h_wm_entropy: List[List[torch.Tensor]],
    h_wm_ref,
    use_ref: bool,
) -> List[List[torch.Tensor]]:
    """Per-turn epistemic ('learnable surprise') signal U_t.

    Calibration gap:  U_t = max(0, NLL_t - H_t).
        NLL_t = realized world-model surprise at the env tokens after turn t
                (depends on what the env actually returned).
        H_t   = the model's own anticipated uncertainty there
                (entropy of its env-token distribution, independent of the
                 realised env output).
    U_t > 0 means the model was surprised *beyond* its own stated uncertainty
    -> a confident-but-wrong prediction about the transition -> reducible /
    epistemic -> worth learning from. U_t ~ 0 means realized surprise matches
    anticipated uncertainty (well calibrated -> mostly aleatoric / irreducible
    env noise -> NOT worth chasing). This is the handle that avoids the
    prediction-error curiosity 'noisy-TV' failure.

    If use_ref and a reference world model is available, compare the current
    NLL against the reference's anticipated uncertainty instead, which is a
    stronger (cross-model) epistemic proxy: how surprised the *current* world
    model still is relative to the reference baseline.
    """
    out: List[List[torch.Tensor]] = []
    for i in range(len(h_wm_nll)):
        turns: List[torch.Tensor] = []
        for t in range(len(h_wm_nll[i])):
            nll = h_wm_nll[i][t]
            if use_ref and h_wm_ref is not None:
                ref = h_wm_ref[i][t]
            else:
                ref = h_wm_entropy[i][t]
            turns.append(torch.clamp(nll - ref, min=0.0))
        out.append(turns)
    return out


def apply_epistemic_advantage_scale(
    batch,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
    h_wm_nll: List[List[torch.Tensor]],
    h_wm_entropy: List[List[torch.Tensor]],
    h_wm_ref,
    attention_mask_response: torch.Tensor,
    old_log_probs: torch.Tensor,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
):
    """Sign-preserving, trajectory-normalized advantage credit redistribution
    driven by the per-turn world-model epistemic signal.

    GRPO broadcasts one outcome advantage to every response token, which is a
    coarse credit assignment. Here we redistribute that fixed per-trajectory
    credit across turns by an estimate of how much each turn led to a genuinely
    *informative* (epistemic-surprising) world transition:

      w_t  ~  base + clamp(g(U_t / u_scale), 0, s_max)      (per turn, >= 0)
      w    normalized within each trajectory so the token-weighted mean is 1
      A   *=  w                                             (per token)

    where g() is a concave shaping (sqrt by default) so the response is sharp
    at the low end (epistemic vs calibrated/aleatoric) and saturating at the
    high end (importance saturates with surprise; robust to proxy outliers).

    Properties that make this a credit *redistribution* rather than reward
    shaping or amplification:
      * Sign preserved: w >= 0, so success stays success and failure stays
        failure. (This is why it does not reproduce the curiosity-as-reward
        failure mode where confused failures got rewarded.)
      * Trajectory-level scale preserved: token-weighted mean(w) == 1, so the
        whole trajectory's gradient magnitude is unchanged; only the
        *within-trajectory* distribution of credit shifts toward consequential
        turns. This keeps trajectories' relative weight in the batch intact
        (no length / entropy bias across trajectories).
      * `base` is the contrast knob in units of the (EMA-smoothed) mean signal:
        large base -> w -> 1 everywhere (near-uniform); small base -> w tracks
        U_t (low-signal turns strongly compressed).

    Sign asymmetry (base_pos vs base_neg): in GRPO outcome mode the advantage
    sign is constant within a trajectory, so 'concentrate credit onto high-U
    turns' *rewards* exploration on winning trajectories (A>0) but *punishes*
    exploration on losing ones (A<0) — the same mechanism flips incentive with
    the sign. On the failure side U is also a worse culprit-proxy (the real
    cause may be a confident-but-correct-world action with low U) and the WM-SFT
    loss already absorbs the surprising transition regardless of outcome. So we
    allow a gentler (more uniform) redistribution on losing trajectories via
    base_neg >= base_pos. base_neg == base_pos recovers the symmetric design;
    base_neg -> inf makes losing trajectories revert to vanilla uniform GRPO.
    """
    use_ref = bool(wmc_erc_config.get("epistemic_use_ref", False))
    base_pos = float(wmc_erc_config.get("epistemic_base", 1.0))
    base_neg = float(wmc_erc_config.get("epistemic_base_neg", base_pos))
    s_max = float(wmc_erc_config.get("epistemic_s_max", 3.0))
    shape = wmc_erc_config.get("epistemic_shape", "sqrt")
    momentum = float(wmc_erc_config.get("momentum", 0.9)) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "momentum", 0.9)
    # Optional: small additive raw-entropy bonus injected BEFORE the
    # multiplicative scaling — mimics the baseline's net exploration signal
    # (which redistribution alone cannot supply). Off by default.
    pre_add_coef = float(wmc_erc_config.get("epistemic_pre_add_coef", 0.0))
    # Optional: on FAILURE trajectories (A_grpo < 0), INVERT the per-turn
    # signal s_t -> (s_max - s_t) before the additive `base +` step. Rationale:
    # with the default (success-direction) scaling, failure trajectories get
    # |A_t| ~ w_t -- i.e. HIGH on exploratory turns (high U) and LOW on
    # routine/loop turns (low U). That's the wrong direction on failures: it
    # punishes exploration and preserves stuck loops. Inverting on neg flips
    # this so on failures the gradient hits routine turns hardest (penalises
    # loops) and exploratory turns lightest (protects exploration). Off by
    # default; recovers prior behaviour when off. Implemented at the s_t
    # level (NOT touching `base`), so the overall r_t range stays in
    # [base, base + s_max] either way.
    invert_on_neg = bool(wmc_erc_config.get("epistemic_invert_on_neg", False))

    epi = compute_epistemic_signal(h_wm_nll, h_wm_entropy, h_wm_ref, use_ref)

    if pre_add_coef != 0.0:
        # offset_t = pre_add_coef * (H_wm_entropy_t - batch_mean(H_wm_entropy)),
        # all-reduced batch mean; broadcast across the turn's response tokens.
        all_h_ent = [h.item() for turns in h_wm_entropy for h in turns]
        if all_h_ent:
            mean_h_t = torch.tensor(all_h_ent, device=advantages.device, dtype=torch.float32).mean()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(mean_h_t, op=dist.ReduceOp.AVG)
            mean_h = mean_h_t.item()
            for i in range(advantages.shape[0]):
                for t, (start, end) in enumerate(turn_boundaries[i]):
                    offset = pre_add_coef * (h_wm_entropy[i][t].item() - mean_h)
                    advantages[i, start:end] = advantages[i, start:end] + offset

    all_u = [u.item() for turns in epi for u in turns]
    if not all_u:
        return batch, {}

    device = advantages.device
    batch_size = advantages.shape[0]

    # Batch mean of the signal, all-reduced, for a stable `base` unit.
    # Also track the fraction of turns with nonzero U: the signal is
    # zero-inflated (well-calibrated turns -> U==0 exactly), so this quantifies
    # how much the all-turns mean is dragged toward 0 by the zeros. Low
    # frac_nonzero => the all-turns mean is a poor scale and a nonzero-turns
    # mean normalizer would discriminate better at the low end.
    u_arr = torch.tensor(all_u, device=device, dtype=torch.float32)
    u_bar_t = u_arr.mean()
    nz_count_t = (u_arr > 0).sum().to(torch.float32)
    all_count_t = torch.tensor(float(len(all_u)), device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(u_bar_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(nz_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_count_t, op=dist.ReduceOp.SUM)
    batch_u_bar = u_bar_t.item()
    frac_nonzero = (nz_count_t.item() / all_count_t.item()) if all_count_t.item() > 0 else 0.0

    if "epi_u_bar" not in running_stats:
        running_stats["epi_u_bar"] = batch_u_bar
    else:
        running_stats["epi_u_bar"] = (1 - momentum) * batch_u_bar + momentum * running_stats["epi_u_bar"]
    u_scale = running_stats["epi_u_bar"] + 1e-8

    weight_vals: List[float] = []
    s_vals: List[float] = []
    n_amplified = 0
    n_compressed = 0
    n_pos_traj = 0
    n_neg_traj = 0
    n_inverted_traj = 0

    for i in range(batch_size):
        turns = turn_boundaries[i]
        if not turns:
            continue
        # Sign of this trajectory's (constant) advantage -> pick the base.
        first_start = turns[0][0]
        traj_adv = advantages[i, first_start].item()
        if traj_adv >= 0.0:
            base = base_pos
            n_pos_traj += 1
            do_invert = False
        else:
            base = base_neg
            n_neg_traj += 1
            do_invert = invert_on_neg
            if do_invert:
                n_inverted_traj += 1

        r_list: List[float] = []
        tok_counts: List[int] = []
        for t, (start, end) in enumerate(turns):
            s_t = min(_epistemic_shape_fn(epi[i][t].item() / u_scale, shape), s_max)
            s_vals.append(s_t)
            # On failures with invert_on_neg, flip s within [0, s_max] so
            # low-U (routine/loop) turns get the high r and high-U (exploratory)
            # turns get the low r — opposite of the success-direction default.
            s_for_r = (s_max - s_t) if do_invert else s_t
            r_list.append(base + s_for_r)
            tok_counts.append(end - start)
        denom = sum(r * c for r, c in zip(r_list, tok_counts))
        n_resp = sum(tok_counts)
        if denom <= 0.0 or n_resp == 0:
            continue
        # token-weighted mean(w) == 1  =>  scale = N_resp / sum_t(r_t * c_t)
        scale = n_resp / denom
        for t, (start, end) in enumerate(turns):
            w = r_list[t] * scale
            advantages[i, start:end] *= w
            weight_vals.append(w)
            if w > 1.0:
                n_amplified += 1
            elif w < 1.0:
                n_compressed += 1

    batch.batch["advantages"] = advantages

    env_mask = attention_mask_response * (1.0 - response_mask)
    env_count = env_mask.sum()
    wm_nll = (-(old_log_probs * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0
    wm_entropy = ((entropys * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    total_turns = len(weight_vals)
    shape_code = {"linear": 0.0, "sqrt": 1.0, "log": 2.0}.get(shape, -1.0)
    metrics = {
        "wmc_erc/epi_use_ref": 1.0 if use_ref else 0.0,
        "wmc_erc/epi_base_pos": base_pos,
        "wmc_erc/epi_base_neg": base_neg,
        "wmc_erc/epi_pre_add_coef": pre_add_coef,
        "wmc_erc/epi_invert_on_neg": 1.0 if invert_on_neg else 0.0,
        "wmc_erc/epi_n_inverted_traj": n_inverted_traj,
        "wmc_erc/epi_shape": shape_code,
        "wmc_erc/epi_n_pos_traj": n_pos_traj,
        "wmc_erc/epi_n_neg_traj": n_neg_traj,
        "wmc_erc/epi_signal_batch_mean": float(batch_u_bar),
        "wmc_erc/epi_signal_running": float(running_stats["epi_u_bar"]),
        "wmc_erc/epi_frac_nonzero": float(frac_nonzero),
        "wmc_erc/epi_signal_norm_mean": float(np.mean(s_vals)) if s_vals else 0.0,
        "wmc_erc/epi_signal_norm_max": float(np.max(s_vals)) if s_vals else 0.0,
        "wmc_erc/epi_weight_mean": float(np.mean(weight_vals)) if weight_vals else 1.0,
        "wmc_erc/epi_weight_std": float(np.std(weight_vals)) if weight_vals else 0.0,
        "wmc_erc/epi_weight_max": float(np.max(weight_vals)) if weight_vals else 1.0,
        "wmc_erc/epi_weight_min": float(np.min(weight_vals)) if weight_vals else 1.0,
        "wmc_erc/epi_frac_amplified": float(n_amplified / total_turns) if total_turns else 0.0,
        "wmc_erc/epi_frac_compressed": float(n_compressed / total_turns) if total_turns else 0.0,
        "wmc_erc/total_turns": total_turns,
        "wmc_erc/wm_nll": wm_nll,
        "wmc_erc/wm_entropy": wm_entropy,
    }
    return batch, metrics


def apply_epistemic_intrinsic_add(
    batch,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
    h_wm_nll: List[List[torch.Tensor]],
    h_wm_entropy: List[List[torch.Tensor]],
    h_wm_ref,
    attention_mask_response: torch.Tensor,
    old_log_probs: torch.Tensor,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
):
    """Design A: Pure non-negative additive intrinsic reward on advantage.

      A_t  +=  coef * min(U_t, cap)        where  U_t = max(0, NLL_t - H_t)

    Differs from clipping_method='add' (which recenters by batch mean) and from
    'epistemic_scale' (which multiplies and per-trajectory-normalizes):
      * No recentering — pure positive bonus on epistemically-informative turns
        and zero on routine/calibrated turns. Routine turns are NOT penalised.
      * Applied AFTER GRPO normalisation (advantage layer), so it is a fixed
        per-turn perturbation independent of group reward scale.
      * Sign-preserving in the strong sense: on success trajectories it boosts
        the positive advantage; on failure trajectories it softens (makes less
        negative) the advantage on exploratory turns — without flipping signs.

    Semantics: per-turn intrinsic reward for actions whose env consequence the
    world model confidently mispredicted. This is the calibration-gap analog of
    ICM-style curiosity (noisy-TV-robust thanks to the max(0, NLL - H) form),
    injected at the advantage layer so it parallels and amplifies the WM-SFT
    training signal (high-U transitions are exactly what WM-SFT needs to learn
    from).
    """
    use_ref = bool(wmc_erc_config.get("epi_intrinsic_use_ref", False))
    coef = float(wmc_erc_config.get("epi_intrinsic_coef", 0.3))
    cap = float(wmc_erc_config.get("epi_intrinsic_cap", 0.5))

    epi = compute_epistemic_signal(h_wm_nll, h_wm_entropy, h_wm_ref, use_ref)

    bonus_vals: List[float] = []
    u_vals: List[float] = []
    n_pos_bonus = 0
    n_capped = 0
    n_pos_traj = 0
    n_neg_traj = 0

    batch_size = advantages.shape[0]
    for i in range(batch_size):
        turns = turn_boundaries[i]
        if not turns:
            continue
        # Track trajectory sign just for diagnostics; the bonus itself is
        # sign-agnostic (purely added).
        first_start = turns[0][0]
        traj_adv = advantages[i, first_start].item()
        if traj_adv >= 0.0:
            n_pos_traj += 1
        else:
            n_neg_traj += 1
        for t, (start, end) in enumerate(turns):
            u = epi[i][t].item()
            u_vals.append(u)
            if u > cap:
                n_capped += 1
            bonus = coef * min(u, cap)
            advantages[i, start:end] += bonus
            bonus_vals.append(bonus)
            if bonus > 0:
                n_pos_bonus += 1
    batch.batch["advantages"] = advantages

    env_mask = attention_mask_response * (1.0 - response_mask)
    env_count = env_mask.sum()
    wm_nll = (-(old_log_probs * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0
    wm_entropy = ((entropys * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    total = len(bonus_vals)
    metrics = {
        "wmc_erc/epi_intrinsic_coef": coef,
        "wmc_erc/epi_intrinsic_cap": cap,
        "wmc_erc/epi_intrinsic_use_ref": 1.0 if use_ref else 0.0,
        "wmc_erc/epi_intrinsic_bonus_mean": float(np.mean(bonus_vals)) if bonus_vals else 0.0,
        "wmc_erc/epi_intrinsic_bonus_max": float(np.max(bonus_vals)) if bonus_vals else 0.0,
        "wmc_erc/epi_intrinsic_bonus_sum_avg_per_traj": (
            float(np.sum(bonus_vals)) / max(1, n_pos_traj + n_neg_traj)
        ),
        "wmc_erc/epi_intrinsic_u_mean": float(np.mean(u_vals)) if u_vals else 0.0,
        "wmc_erc/epi_intrinsic_u_max": float(np.max(u_vals)) if u_vals else 0.0,
        "wmc_erc/epi_intrinsic_frac_pos_bonus": float(n_pos_bonus / total) if total else 0.0,
        "wmc_erc/epi_intrinsic_frac_capped": float(n_capped / total) if total else 0.0,
        "wmc_erc/epi_intrinsic_n_turns": total,
        "wmc_erc/epi_intrinsic_n_pos_traj": n_pos_traj,
        "wmc_erc/epi_intrinsic_n_neg_traj": n_neg_traj,
        "wmc_erc/wm_nll": wm_nll,
        "wmc_erc/wm_entropy": wm_entropy,
    }
    return batch, metrics


def apply_ref_nll_add_advantage(batch, wmc_erc_config) -> Tuple["object", Dict[str, float]]:
    """Direct additive shaping from the FROZEN initial ref model's env-token NLL.

        A'_{i,t} = A_{i,t} + coef * ref_NLL_env(turn t)

    For each action turn t, take the reference model's mean negative
    log-likelihood over the environment (observation) tokens that follow that
    turn, multiply by ``ref_nll_coef``, and add it to every action token's
    advantage in that turn.

    Rationale: because the ref model is fixed at the initial checkpoint, this
    per-state signal does NOT decay as the actor's own world model improves —
    it is a stationary measure of how surprising/informative each env
    transition is under the base model. This deliberately overrides every other
    wmc_erc path (clipping_method, add-to-reward, epistemic_*); it is gated
    solely by ``wmc_erc.ref_nll_add`` at the trainer level.

    Requires ``ref_log_prob`` in the batch (i.e. the reference policy was run,
    which is the case whenever KL-to-ref is enabled). No-op + flag metric if
    absent.
    """
    coef = float(wmc_erc_config.get("ref_nll_coef", 0.0)
                 if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "ref_nll_coef", 0.0))
    advantages = batch.batch["advantages"]
    if "ref_log_prob" not in batch.batch.keys():
        return batch, {"wmc_erc/ref_nll_add_active": 0.0, "wmc_erc/ref_nll_missing": 1.0}

    response_mask = batch.batch["response_mask"]
    ref_log_prob = batch.batch["ref_log_prob"].to(advantages.dtype)
    response_length = advantages.shape[1]
    attention_mask = batch.batch["attention_mask"]
    attention_mask_response = attention_mask[:, -response_length:]
    turn_boundaries = compute_turn_boundaries(response_mask)

    # Per-turn mean ref NLL over the env tokens following each action turn.
    # compute_h_wm returns -mean(ref_log_prob) on env-token positions per turn.
    ref_nll = compute_h_wm(ref_log_prob, response_mask, attention_mask_response, turn_boundaries)

    all_off: List[float] = []
    all_nll: List[float] = []
    for i in range(advantages.shape[0]):
        for t, (start, end) in enumerate(turn_boundaries[i]):
            nll_t = ref_nll[i][t]
            off = coef * nll_t
            advantages[i, start:end] += off
            all_off.append(off.item())
            all_nll.append(nll_t.item())
    batch.batch["advantages"] = advantages

    metrics: Dict[str, float] = {
        "wmc_erc/ref_nll_add_active": 1.0,
        "wmc_erc/ref_nll_coef": coef,
    }
    if all_off:
        metrics.update({
            "wmc_erc/ref_nll_offset_mean": float(np.mean(all_off)),
            "wmc_erc/ref_nll_offset_max": float(np.max(all_off)),
            "wmc_erc/ref_nll_offset_min": float(np.min(all_off)),
            "wmc_erc/ref_nll_mean": float(np.mean(all_nll)),
        })
    return batch, metrics


def apply_uncertainty_scale_advantage(
    batch, advantages, response_mask, turn_boundaries, h_wm_entropy,
    s_star, wmc_erc_config, running_stats,
):
    """Scale advantage to suppress entropy collapse on uncertain-outcome turns.

    Theory anchor: a policy-gradient step that reinforces action a_t changes the
    policy entropy by  dH ≈ −A_t · s★_t , where s★ = compute_s_star is the
    per-turn entropy-collapse rate (>0 ⟺ reinforcing concentrates probability;
    <0 ⟺ exploring the tail raises entropy). We down-weight A_t in proportion to
    how much the update would collapse entropy AND how epistemically uncertain
    the env consequence is, via precision / inverse-variance weighting:

        c_t      = relu(s★_t) / s★⁺_group               # normalized collapse rate ≥0
        u_t      = max(0, U_t / U_group − 1)            # env-uncertainty excess ≥0
        scale_t  = clamp( 1 / (1 + κ · c_t · u_t), floor, 1 )
        A'_{i,t} = scale_t · A_{i,t}                    # symmetric over sign

    The two factors enter as a PRODUCT: scaling only bites where the update both
    collapses entropy (c>0) AND the outcome is unpredictable (u>0); either ≈0 ⇒
    scale≈1. U_t = mean env-token entropy of the observation FOLLOWING a_t.
    Normalizers U_group / s★⁺_group are means over the GRPO GROUP (all turns of
    all trajectories with the same uid = same prompt = same env) — this removes
    between-task entropy-scale, not the within-task turn-type signal we want.

    If uncertainty_scale_renormalize (default True), each group's total |A| is
    restored to its pre-scaling value AFTER scaling, turning this into a pure
    REDISTRIBUTION (uncertain turns down-weighted, the rest proportionally
    up-weighted) so the overall advantage magnitude — and thus commit speed —
    is unchanged. Gated by clipping_method == "uncertainty_scale".
    """
    kappa = float(wmc_erc_config.get("uncertainty_scale_kappa", 1.0))
    floor = float(wmc_erc_config.get("uncertainty_scale_min", 0.5))
    renormalize = bool(wmc_erc_config.get("uncertainty_scale_renormalize", True))

    # GROUP normalization: U_t and s★_t are normalized by the mean over ALL turns
    # of ALL trajectories sharing the same prompt (GRPO group = uid). Same prompt
    # ⇒ same env ⇒ same entropy scale, so this removes pure between-TASK scale
    # (the real confound) using G× more samples than per-trajectory, while
    # keeping the within-task turn-type signal. (Measured: between-task ≈ all of
    # the cross-traj scale; within-group between-rollout ≈ 0.)
    B = advantages.shape[0]
    uid = batch.non_tensor_batch['uid']
    _, ginv = np.unique(uid, return_inverse=True)          # group index per traj

    # collect per-turn values, bucket positive U / positive s★ by group
    turn_info = []                                         # (i, s, e, g, U_t, s_t)
    grp_u = defaultdict(list)
    grp_sp = defaultdict(list)
    glob_u, glob_sp = [], []
    for i in range(B):
        g = int(ginv[i])
        for t, (start, end) in enumerate(turn_boundaries[i]):
            U_t = h_wm_entropy[i][t].item()
            s_t = s_star[i][t].item()
            turn_info.append((i, start, end, g, U_t, s_t))
            if U_t > 0.0:
                grp_u[g].append(U_t); glob_u.append(U_t)
            if s_t > 0.0:
                grp_sp[g].append(s_t); glob_sp.append(s_t)

    glob_u_mean = float(np.mean(glob_u)) if glob_u else 1.0
    glob_sp_mean = float(np.mean(glob_sp)) if glob_sp else 1.0
    mu_u = {g: float(np.mean(v)) for g, v in grp_u.items()}     # per-group env-entropy mean
    mu_sp = {g: float(np.mean(v)) for g, v in grp_sp.items()}   # per-group positive-s★ mean

    all_scale, all_x = [], []
    grp_orig_mass = defaultdict(float)   # Σ|A| per group BEFORE scaling
    grp_new_mass = defaultdict(float)    # Σ|A| per group AFTER scaling
    for (i, start, end, g, U_t, s_t) in turn_info:
        denom_u = max(mu_u.get(g, glob_u_mean), 1e-6)          # fallback: batch-global
        denom_s = max(mu_sp.get(g, glob_sp_mean), 1e-6)
        u = max(0.0, U_t / denom_u - 1.0)
        c = max(0.0, s_t) / denom_s
        x = c * u
        scale = 1.0 / (1.0 + kappa * x)
        scale = min(1.0, max(floor, scale))
        seg = advantages[i, start:end]
        orig_abs = float(seg.abs().sum().item())
        grp_orig_mass[g] += orig_abs
        grp_new_mass[g] += orig_abs * scale                    # scale>0 ⇒ |scaled|=|orig|·scale
        advantages[i, start:end] = seg * scale
        all_scale.append(scale)
        all_x.append(x)

    # renormalize: restore each GROUP's total |advantage| to its pre-scaling value
    # so uncertainty_scale becomes a pure REDISTRIBUTION (down-weight uncertain
    # turns, proportionally up-weight the rest) and does NOT shrink the overall
    # advantage magnitude — keeping the commit speed unchanged. Ratio ≥ 1.
    renorm_ratios = []
    if renormalize:
        grp_ratio = {}
        for g in grp_orig_mass:
            nm = grp_new_mass[g]
            grp_ratio[g] = (grp_orig_mass[g] / nm) if nm > 1e-8 else 1.0
            renorm_ratios.append(grp_ratio[g])
        for (i, start, end, g, U_t, s_t) in turn_info:
            r = grp_ratio.get(g, 1.0)
            if r != 1.0:
                advantages[i, start:end] = advantages[i, start:end] * r
    batch.batch["advantages"] = advantages

    metrics: Dict[str, float] = {
        "wmc_erc/uscale_active": 1.0,
        "wmc_erc/uscale_kappa": kappa,
        "wmc_erc/uscale_renormalize": 1.0 if renormalize else 0.0,
        "wmc_erc/uscale_n_groups": float(len(mu_u)),
        "wmc_erc/uscale_u_group_mean": float(np.mean(list(mu_u.values()))) if mu_u else 0.0,
        "wmc_erc/uscale_sp_group_mean": float(np.mean(list(mu_sp.values()))) if mu_sp else 0.0,
    }
    if renorm_ratios:
        metrics["wmc_erc/uscale_renorm_ratio_mean"] = float(np.mean(renorm_ratios))
    if all_scale:
        arr = np.array(all_scale)
        metrics.update({
            "wmc_erc/uscale_mean": float(arr.mean()),
            "wmc_erc/uscale_min": float(arr.min()),
            "wmc_erc/uscale_collapse_unc_mean": float(np.mean(all_x)),  # mean of c·u driving the scale
            "wmc_erc/frac_scaled": float((arr < 0.99).mean()),
        })
    return batch, metrics


def apply_safe_commit_advantage(
    batch, advantages, response_mask, turn_boundaries, h_wm_entropy,
    s_star, wmc_erc_config, running_stats,
):
    """Safe-Commit Sharpener: boost advantage on turns that are SAFE to commit
    (env outcome already determined), to accelerate commit speed — the opposite
    sign of uncertainty_scale (which SHRINKS advantage on uncertain turns).
    Empirically the curve gains come from committing FASTER where it is safe.

    Theory anchor: a policy-gradient step reinforcing a_t changes entropy by
    dH ≈ −A_t · s★_t, so a larger |A_t| ⇒ faster entropy descent (faster commit)
    on turn t. We boost |A_t| on turns whose ENV OUTCOME is already determined.

    Safe-to-commit weight w_t uses an EXOGENOUS gate (env determinism, NOT the
    policy's own confidence), so there is no confidence→boost→more-confidence
    feedback loop (the loop that makes s★/action-entropy boosting collapse — the
    HCA ω=1.0 over-fit):

        w_t = clip(1 − U_t / U_group, 0, 1)              # low env-entropy ⇒ safe ⇒ w→1

    U_t = mean env-token entropy of the obs FOLLOWING a_t; U_group / w̄_group /
    σ_w_group are statistics over the GRPO GROUP (uid = same prompt = same env),
    removing between-task scale.

    Two modes (safe_commit_mode):

    • "add" (DEFAULT) — additive, structurally identical to HCAPO's multi-scale
      advantage so its magnitude matches HCA at ω=1.0 (HCA logged A_micro_abs_mean
      ≈0.49, mean≈+0.24-0.37). Replaces HCA's noise-dominated hindsight ρ with the
      exogenous safe weight:
          A_safe_t = (w_t − w̄_group) / (σ_w_group + ε)    # group-norm ⇒ ~unit std
          do-no-harm: zero A_safe on negatives, applied only to WINNING trajectories
          A'_{i,t} = A_GRPO_{i,t} + ω · A_safe_t           # action tokens
      ω = safe_commit_omega is the direct magnitude dial (ω=1.0 ≈ HCA scale; lower
      ω ⇒ smaller added advantage). "Win" = trajectory A_GRPO > success_threshold.
      Gating to wins mirrors HCA's Q^H ∝ R (failures get no micro signal).

    • "redistribute" — multiplicative, mean-preserving sharpener (trajectory
      UNBIASED, adds NO net magnitude):
          g_t = clip(1 + κ·(w_t − w̄_group), g_min, g_max) ; A'_{i,t} = g_t · A_{i,t}
      with safe_commit_renormalize (default True) restoring each group's Σ|A|.

    Semantic-label injection (both modes): if batch.non_tensor_batch carries
    'safe_commit_w' (object array; per-traj list of per-turn weights in [0,1],
    1.0=text-classified DETERMINISTIC, 0.0=REVEALING) it OVERRIDES the U-derived
    w_t. Turns with no valid signal (U_t<=0, no label) are neutral. Gated by
    clipping_method == "safe_commit".
    """
    mode = str(wmc_erc_config.get("safe_commit_mode", "add"))
    omega = float(wmc_erc_config.get("safe_commit_omega", 1.0))
    success_threshold = float(wmc_erc_config.get("safe_commit_success_threshold", 0.0))
    # γ-recency tilt (add mode; default 0 = OFF = pure content targeting). The case
    # study showed the OLD HCA's speed came from a content-blind γ^{T-1-t} recency
    # ramp (corr 0.87 with turn position). This optionally re-introduces that
    # "reward the successful end-game" prior WHILE keeping the low-U content target:
    #   A_safe_t *= (1 + λ · γ^{T-1-t})    (last turn → ×(1+λ); earlier decays)
    recency = float(wmc_erc_config.get("safe_commit_recency", 0.0))
    recency_gamma = float(wmc_erc_config.get("safe_commit_recency_gamma", 0.95))
    kappa = float(wmc_erc_config.get("safe_commit_kappa", 1.0))
    g_max = float(wmc_erc_config.get("safe_commit_gmax", 2.0))
    g_min = float(wmc_erc_config.get("safe_commit_gmin", 0.5))
    renormalize = bool(wmc_erc_config.get("safe_commit_renormalize", True))

    B = advantages.shape[0]
    uid = batch.non_tensor_batch['uid']
    _, ginv = np.unique(uid, return_inverse=True)                 # group index per traj
    # optional semantic safe-weight injection (text classifier): per-traj list of
    # per-turn weights in [0,1]; overrides the U-derived weight when present.
    label_w = None
    if hasattr(batch, "non_tensor_batch"):
        label_w = batch.non_tensor_batch.get("safe_commit_w", None)

    # pass 1: gather U_t / optional label per turn; per-group positive-U means
    raw = []                                                      # (i,start,end,g,U_t,lab_w,t,Ti)
    grp_u = defaultdict(list); glob_u = []
    for i in range(B):
        g = int(ginv[i])
        Ti = len(turn_boundaries[i])
        for t, (start, end) in enumerate(turn_boundaries[i]):
            U_t = h_wm_entropy[i][t].item()
            lw = None
            if label_w is not None:
                try:
                    lw = float(label_w[i][t])
                except Exception:
                    lw = None
            raw.append((i, start, end, g, U_t, lw, t, Ti))
            if U_t > 0.0:
                grp_u[g].append(U_t); glob_u.append(U_t)
    glob_u_mean = float(np.mean(glob_u)) if glob_u else 1.0
    mu_u = {g: float(np.mean(v)) for g, v in grp_u.items()}       # per-group env-entropy mean

    # pass 2: per-turn safe-commit weight w_t ∈ [0,1] (label overrides U); group means of w
    turn_info = []                                                # (i,start,end,g,w_or_None,t,Ti)
    grp_w = defaultdict(list)
    for (i, start, end, g, U_t, lw, t, Ti) in raw:
        if lw is not None:
            w = min(1.0, max(0.0, lw))
        elif U_t > 0.0:
            denom_u = max(mu_u.get(g, glob_u_mean), 1e-6)         # fallback: batch-global
            w = min(1.0, max(0.0, 1.0 - U_t / denom_u))
        else:
            w = None                                             # no signal → neutral (g=1)
        turn_info.append((i, start, end, g, w, t, Ti))
        if w is not None:
            grp_w[g].append(w)
    mu_w = {g: float(np.mean(v)) for g, v in grp_w.items()}
    sigma_w = {g: float(np.std(v)) for g, v in grp_w.items()}     # per-group w spread (add mode)

    # per-trajectory "win" flag = trajectory A_GRPO > threshold (advantages are the
    # GRPO scalar broadcast at entry, constant over a trajectory's action tokens).
    rm_bool = response_mask.to(torch.bool)
    win = {}
    for i in range(B):
        m = rm_bool[i]
        win[i] = (float(advantages[i][m].mean().item()) > success_threshold) if m.any() else False

    metrics: Dict[str, float] = {
        "wmc_erc/safecommit_active": 1.0,
        "wmc_erc/safecommit_mode_add": 1.0 if mode == "add" else 0.0,
        "wmc_erc/safecommit_used_label": 1.0 if label_w is not None else 0.0,
        "wmc_erc/safecommit_n_groups": float(len(mu_w)),
        "wmc_erc/safecommit_w_mean": (
            float(np.mean([w for v in grp_w.values() for w in v])) if grp_w else 0.0
        ),
    }

    if mode == "add":
        # HCA-structured additive boost: A' = A_GRPO + ω · A_safe, where A_safe is
        # the GROUP-normalized safe weight (so it is ~unit std like HCA's A_micro,
        # making ω the magnitude dial; ω=1.0 ≈ HCA's 0.49 abs / +0.3 mean). Apply
        # only on winning trajectories, do-no-harm (no negative add). ω = the knob.
        # pass A: per-turn base boost a_safe (do-no-harm, wins only) + recency multiplier
        boost = []                                                # (i,start,end, a_safe, mult)
        per_i_num = defaultdict(float)                            # Σ a_safe·mult  (per traj)
        per_i_den = defaultdict(float)                            # Σ a_safe       (per traj)
        for (i, start, end, g, w, t_idx, Ti) in turn_info:
            if w is None or not win[i]:
                continue
            sd = sigma_w.get(g, 0.0)
            if sd < 1e-6:
                continue                                          # no within-group spread
            a_safe = (w - mu_w.get(g, w)) / (sd + 1e-6)
            if a_safe < 0.0:
                a_safe = 0.0                                      # do-no-harm
            mult = (1.0 + recency * (recency_gamma ** max(0, Ti - 1 - t_idx))) if recency > 0.0 else 1.0
            boost.append((i, start, end, a_safe, mult))
            per_i_num[i] += a_safe * mult
            per_i_den[i] += a_safe
        # pass B: γ-recency is a PURE TILT — normalize the multiplier per trajectory so
        # the total added boost per trajectory is UNCHANGED by λ (Σ a_safe·mult_norm =
        # Σ a_safe). This decouples recency (where the boost goes) from ω (how much is
        # added), so raising λ no longer inflates magnitude / collapses entropy.
        added = []
        for (i, start, end, a_safe, mult) in boost:
            if recency > 0.0 and per_i_num[i] > 1e-8:
                Z = per_i_num[i] / max(per_i_den[i], 1e-8)        # a_safe-weighted mean mult
                a_eff = a_safe * (mult / Z)
            else:
                a_eff = a_safe
            delta = omega * a_eff
            advantages[i, start:end] = advantages[i, start:end] + delta
            added.append(delta)
        batch.batch["advantages"] = advantages
        metrics["wmc_erc/safecommit_omega"] = omega
        metrics["wmc_erc/safecommit_recency"] = recency
        if added:
            arr = np.array(added)
            metrics.update({
                "wmc_erc/safecommit_add_abs_mean": float(np.abs(arr).mean()),
                "wmc_erc/safecommit_add_mean": float(arr.mean()),
                "wmc_erc/safecommit_add_max": float(arr.max()),
                "wmc_erc/safecommit_add_nonzero_frac": float((arr > 1e-6).mean()),
            })
        return batch, metrics

    # mode == "redistribute": multiplicative, mean-preserving sharpener.
    all_g = []
    grp_orig_mass = defaultdict(float)                           # Σ|A| per group BEFORE
    grp_new_mass = defaultdict(float)                            # Σ|A| per group AFTER
    for (i, start, end, g, w, _t, _Ti) in turn_info:
        if w is None:
            gt = 1.0
        else:
            wbar = mu_w.get(g, w)
            gt = 1.0 + kappa * (w - wbar)
            gt = min(g_max, max(g_min, gt))
        seg = advantages[i, start:end]
        orig_abs = float(seg.abs().sum().item())
        grp_orig_mass[g] += orig_abs
        grp_new_mass[g] += orig_abs * gt                        # gt>0 ⇒ |scaled|=|orig|·gt
        advantages[i, start:end] = seg * gt
        all_g.append(gt)

    # renormalize: restore each GROUP's total |advantage| so the booster is a pure
    # mean-preserving REDISTRIBUTION toward safe turns (trajectory-level unbiased).
    renorm_ratios = []
    if renormalize:
        grp_ratio = {}
        for g in grp_orig_mass:
            nm = grp_new_mass[g]
            grp_ratio[g] = (grp_orig_mass[g] / nm) if nm > 1e-8 else 1.0
            renorm_ratios.append(grp_ratio[g])
        for (i, start, end, g, w, _t, _Ti) in turn_info:
            r = grp_ratio.get(g, 1.0)
            if r != 1.0:
                advantages[i, start:end] = advantages[i, start:end] * r
    batch.batch["advantages"] = advantages

    metrics.update({
        "wmc_erc/safecommit_kappa": kappa,
        "wmc_erc/safecommit_gmax": g_max,
        "wmc_erc/safecommit_gmin": g_min,
        "wmc_erc/safecommit_renormalize": 1.0 if renormalize else 0.0,
    })
    if renorm_ratios:
        metrics["wmc_erc/safecommit_renorm_ratio_mean"] = float(np.mean(renorm_ratios))
    if all_g:
        arr = np.array(all_g)
        metrics.update({
            "wmc_erc/safecommit_g_mean": float(arr.mean()),
            "wmc_erc/safecommit_g_max": float(arr.max()),
            "wmc_erc/safecommit_g_min": float(arr.min()),
            "wmc_erc/safecommit_frac_boosted": float((arr > 1.01).mean()),
        })
    return batch, metrics


def apply_wmc_erc(
    batch,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
    step: int = None,
):
    """Apply WMC-ERC dynamic entropy clipping to batch advantages."""
    enable = wmc_erc_config.get("enable", True) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "enable", True)
    if not enable:
        return batch, {}

    # If the bonus was already injected into token_level_rewards before
    # compute_advantage, skip the post-advantage `add` write to avoid
    # double-counting. Other (masking) methods still go through.
    clipping_method_check = wmc_erc_config.get("clipping_method", "mask") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_method", "mask")
    if clipping_method_check == "add" and bool(wmc_erc_config.get("wmloss_add_to_reward", False)):
        return batch, {}

    if step is not None:
        running_stats['step'] = step
    else:
        running_stats['step'] = running_stats.get('step', 0) + 1
    current_step = running_stats['step']

    clipping_method = wmc_erc_config.get("clipping_method", "mask") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_method", "mask")

    # Success/Failure info
    wmloss_add_only_failed = bool(wmc_erc_config.get("wmloss_add_only_failed", False))
    if wmloss_add_only_failed and 'task_scores' in batch.batch.keys():
        # Sum scores over tokens/turns to get trajectory success
        traj_scores = batch.batch['task_scores'].sum(dim=-1)  # [B]
        traj_failed = (traj_scores <= 0.0)
    else:
        traj_failed = torch.ones(batch.batch['advantages'].shape[0], dtype=torch.bool, device=batch.batch['advantages'].device)

    response_mask = batch.batch["response_mask"]
    old_log_probs = batch.batch["old_log_probs"]
    advantages = batch.batch["advantages"]
    batch_size = advantages.shape[0]
    response_length = advantages.shape[1]
    attention_mask = batch.batch["attention_mask"]
    attention_mask_response = attention_mask[:, -response_length:]

    turn_boundaries = compute_turn_boundaries(response_mask)

    s_star = compute_s_star(old_log_probs, entropys, response_mask, turn_boundaries)
    h_wm_nll = compute_h_wm(old_log_probs, response_mask, attention_mask_response, turn_boundaries)
    h_wm_entropy = compute_h_wm_entropy(entropys, response_mask, attention_mask_response, turn_boundaries)

    ref_entropy = batch.batch.get("ref_entropy", None)
    if ref_entropy is not None:
        h_wm_ref = compute_h_wm_entropy(ref_entropy, response_mask, attention_mask_response, turn_boundaries)
    else:
        h_wm_ref = None

    # Epistemic credit redistribution on advantages (sign-preserving,
    # trajectory-normalized). Uses only the per-turn WM quantities computed
    # above and returns early — none of the s_star / add / mask machinery runs.
    if clipping_method == "epistemic_scale":
        return apply_epistemic_advantage_scale(
            batch=batch,
            advantages=advantages,
            response_mask=response_mask,
            turn_boundaries=turn_boundaries,
            h_wm_nll=h_wm_nll,
            h_wm_entropy=h_wm_entropy,
            h_wm_ref=h_wm_ref,
            attention_mask_response=attention_mask_response,
            old_log_probs=old_log_probs,
            entropys=entropys,
            wmc_erc_config=wmc_erc_config,
            running_stats=running_stats,
        )

    # Uncertainty-scaling: shrink advantage on turns where the model is unsure
    # what the env will do next, to prevent action-entropy collapse on
    # unpredictable-outcome turns. Scales symmetrically; no renormalization.
    if clipping_method == "uncertainty_scale":
        return apply_uncertainty_scale_advantage(
            batch=batch,
            advantages=advantages,
            response_mask=response_mask,
            turn_boundaries=turn_boundaries,
            h_wm_entropy=h_wm_entropy,
            s_star=s_star,
            wmc_erc_config=wmc_erc_config,
            running_stats=running_stats,
        )

    # Safe-Commit Sharpener: AMPLIFY advantage on safe-to-commit (env-deterministic
    # / low next-obs entropy) turns to accelerate commit there. Opposite sign of
    # uncertainty_scale; mean-preserving (renormalized) ⇒ trajectory-unbiased.
    if clipping_method == "safe_commit":
        return apply_safe_commit_advantage(
            batch=batch,
            advantages=advantages,
            response_mask=response_mask,
            turn_boundaries=turn_boundaries,
            h_wm_entropy=h_wm_entropy,
            s_star=s_star,
            wmc_erc_config=wmc_erc_config,
            running_stats=running_stats,
        )

    # Pure non-negative additive intrinsic bonus on advantage (Design A,
    # calibration-gap ICM variant). No recentering, no per-turn redistribution.
    if clipping_method == "epi_intrinsic_add":
        return apply_epistemic_intrinsic_add(
            batch=batch,
            advantages=advantages,
            response_mask=response_mask,
            turn_boundaries=turn_boundaries,
            h_wm_nll=h_wm_nll,
            h_wm_entropy=h_wm_entropy,
            h_wm_ref=h_wm_ref,
            attention_mask_response=attention_mask_response,
            old_log_probs=old_log_probs,
            entropys=entropys,
            wmc_erc_config=wmc_erc_config,
            running_stats=running_stats,
        )

    all_s = [s.item() for turns in s_star for s in turns]

    # Per-turn signal that the additive bonus is built from. Options:
    #   use_gap     -> epistemic calibration gap max(0, NLL - H): noisy-TV-robust
    #                  "learnable surprise". Baseline mechanism, better signal.
    #   use_entropy -> raw env-token entropy H (the original baseline signal).
    #   else        -> env-token NLL.
    use_gap = bool(wmc_erc_config.get("wmloss_add_use_gap", False))
    use_entropy = bool(wmc_erc_config.get("wmloss_add_use_entropy", False))
    if use_gap:
        target_h = [
            [torch.clamp(nll - h, min=0.0) for nll, h in zip(h_wm_nll[i], h_wm_entropy[i])]
            for i in range(len(h_wm_nll))
        ]
    elif use_entropy:
        target_h = h_wm_entropy
    else:
        target_h = h_wm_nll

    all_h = [h.item() for turns in target_h for h in turns]

    if not all_s:
        return batch, {}

    # Sync stats across processes
    all_s_tensor = torch.tensor(all_s, device=advantages.device, dtype=torch.float32)
    all_h_tensor = torch.tensor(all_h, device=advantages.device, dtype=torch.float32)

    batch_s_bar_t = all_s_tensor.mean()
    batch_s_std_t = all_s_tensor.std(correction=0) if len(all_s) > 1 else torch.tensor(0.0, device=advantages.device)
    batch_h_bar_t = all_h_tensor.mean()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch_s_bar_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_s_std_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_h_bar_t, op=dist.ReduceOp.AVG)

    batch_s_bar = batch_s_bar_t.item()
    batch_s_std = batch_s_std_t.item() + 1e-8
    batch_h_bar = batch_h_bar_t.item() + 1e-8

    momentum = wmc_erc_config.get("momentum", 0.9) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "momentum", 0.9)
    if 's_bar' not in running_stats:
        running_stats["s_bar"] = batch_s_bar
        running_stats["s_std"] = batch_s_std
        running_stats["h_bar"] = batch_h_bar
    else:
        running_stats["s_bar"] = (1 - momentum) * batch_s_bar + momentum * running_stats["s_bar"]
        running_stats["s_std"] = (1 - momentum) * batch_s_std + momentum * running_stats["s_std"]
        running_stats["h_bar"] = (1 - momentum) * batch_h_bar + momentum * running_stats["h_bar"]

    if clipping_method == "add":
        # Additive curiosity bonus: A' = A + coef * (factor - mean(factor))
        # factor = clip(0.5, max(0, H_wm - H_wm_mean))
        
        # Linear decay for coef
        wmloss_add_coef_start = float(wmc_erc_config.get("wmloss_add_coef", 0.1))
        wmloss_add_coef_end = float(wmc_erc_config.get("wmloss_add_coef_end", wmloss_add_coef_start))
        wmloss_add_horizon = int(wmc_erc_config.get("wmloss_add_horizon", 1))
        
        if wmloss_add_horizon > 0:
            alpha_decay = min(current_step / wmloss_add_horizon, 1.0)
            wmloss_add_coef = wmloss_add_coef_start + alpha_decay * (wmloss_add_coef_end - wmloss_add_coef_start)
        else:
            wmloss_add_coef = wmloss_add_coef_start

        wmloss_add_use_grouped = bool(wmc_erc_config.get("wmloss_add_use_grouped", False))
        wmloss_add_use_ref_baseline = bool(wmc_erc_config.get("wmloss_add_use_ref_baseline", False))
        has_group_id = 'group_id' in batch.batch.keys()

        all_offsets = []
        add_metrics = {}

        if wmloss_add_use_ref_baseline and h_wm_ref is not None:
            # Formula: offset_t = alpha * 1[failed] * clamp(H_wm - H_ref, 0, 0.5)
            # No recentering as per request
            for i in range(batch_size):
                if not traj_failed[i]:
                    continue
                for t in range(len(target_h[i])):
                    h_val = target_h[i][t]
                    h_ref_val = h_wm_ref[i][t]
                    
                    factor_t = torch.clamp(h_val - h_ref_val, min=0.0, max=0.5)
                    offset_t = wmloss_add_coef * factor_t
                    
                    start, end = turn_boundaries[i][t]
                    advantages[i, start:end] += offset_t
                    all_offsets.append(offset_t.item())
            
            # Metrics for Ref Baseline
            all_h_ref = [h.item() for turns in h_wm_ref for h in turns]
            if all_h_ref:
                add_metrics.update({
                    "wmc_erc/h_wm_ref_mean": float(np.mean(all_h_ref)),
                    "wmc_erc/h_wm_ref_std": float(np.std(all_h_ref)),
                })

        elif wmloss_add_use_grouped and has_group_id:
            gids = batch.batch['group_id'].long()
            g_max_local = gids.max() if gids.numel() > 0 else torch.tensor(-1, device=gids.device, dtype=torch.long)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(g_max_local, op=dist.ReduceOp.MAX)
            G = int(g_max_local.item()) + 1
            
            flat_h = []
            flat_gids = []
            for i in range(batch_size):
                if not traj_failed[i]:
                    continue
                for h_val in target_h[i]:
                    flat_h.append(h_val)
                    flat_gids.append(gids[i])
            
            if flat_h:
                flat_h_tensor = torch.stack(flat_h)
                flat_gids_tensor = torch.stack(flat_gids)
                
                sum_per_g = torch.zeros(G, dtype=flat_h_tensor.dtype, device=flat_h_tensor.device)
                cnt_per_g = torch.zeros(G, dtype=flat_h_tensor.dtype, device=flat_h_tensor.device)
                sum_per_g.scatter_add_(0, flat_gids_tensor, flat_h_tensor)
                cnt_per_g.scatter_add_(0, flat_gids_tensor, torch.ones_like(flat_h_tensor))
                
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(sum_per_g, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cnt_per_g, op=dist.ReduceOp.SUM)
                
                mu_per_g = sum_per_g / cnt_per_g.clamp(min=1.0)
                
                # Compute factors
                factor_per_sample = []
                all_factors = []
                for i in range(batch_size):
                    sample_factors = []
                    if not traj_failed[i]:
                        factor_per_sample.append(sample_factors)
                        continue
                    gid = gids[i].item()
                    mu_h_g = mu_per_g[gid]
                    for h_val in target_h[i]:
                        factor_t = torch.clamp(h_val - mu_h_g, min=0.0, max=0.5)
                        sample_factors.append(factor_t)
                        all_factors.append(factor_t)
                    factor_per_sample.append(sample_factors)
                
                # Compute grouped mean of factors
                if all_factors:
                    all_factors_tensor = torch.stack(all_factors)
                    sum_fac_per_g = torch.zeros(G, dtype=all_factors_tensor.dtype, device=all_factors_tensor.device)
                    sum_fac_per_g.scatter_add_(0, flat_gids_tensor, all_factors_tensor)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(sum_fac_per_g, op=dist.ReduceOp.SUM)
                    mu_fac_per_g = sum_fac_per_g / cnt_per_g.clamp(min=1.0)
                    
                    # Apply grouped offset
                    for i in range(batch_size):
                        if not traj_failed[i]:
                            continue
                        gid = gids[i].item()
                        mu_fac_g = mu_fac_per_g[gid]
                        for t in range(len(factor_per_sample[i])):
                            offset_t = wmloss_add_coef * (factor_per_sample[i][t] - mu_fac_g)
                            start, end = turn_boundaries[i][t]
                            advantages[i, start:end] += offset_t
                            all_offsets.append(offset_t.item())
                
                add_metrics.update({
                    "wmc_erc/wmloss_mu_per_group_mean": mu_per_g[cnt_per_g > 0].mean().item() if (cnt_per_g > 0).any() else 0.0,
                    "wmc_erc/wmloss_mu_per_group_std": mu_per_g[cnt_per_g > 0].std(unbiased=False).item() if (cnt_per_g > 0).sum() > 1 else 0.0,
                })
        else:
            # Fallback: EMA baseline. offset_t = coef * (target_h_t - h_bar_EMA)
            mu_h = running_stats["h_bar"]

            for i in range(batch_size):
                if not traj_failed[i]:
                    continue
                for t in range(len(target_h[i])):
                    h_val = target_h[i][t]

                    offset_t = wmloss_add_coef * (h_val - mu_h)

                    start, end = turn_boundaries[i][t]
                    advantages[i, start:end] += offset_t
                    all_offsets.append(offset_t.item())

        if all_offsets:
            batch.batch["advantages"] = advantages
            add_metrics.update({
                "wmc_erc/wmloss_offset_mean": float(np.mean(all_offsets)),
                "wmc_erc/wmloss_offset_std": float(np.std(all_offsets)),
                "wmc_erc/wmloss_offset_max": float(np.max(all_offsets)),
                "wmc_erc/wmloss_offset_min": float(np.min(all_offsets)),
                "wmc_erc/wmloss_coef": float(wmloss_add_coef),
            })
        else:
            add_metrics = {}
        
        # Track number of failed trajectories
        add_metrics["wmc_erc/num_failed_trajs"] = int(traj_failed.sum().item())
    else:
        # Unknown clipping method — keep advantages unchanged. The supported
        # methods are: 'add', 'epistemic_scale', 'epi_intrinsic_add'. The
        # latter two are routed via early returns above.
        add_metrics = {}

    env_mask = attention_mask_response * (1.0 - response_mask)
    env_count = env_mask.sum()
    wm_nll = (-(old_log_probs * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    metrics = {
        "wmc_erc/batch_s_bar": float(batch_s_bar),
        "wmc_erc/batch_s_std": float(batch_s_std),
        "wmc_erc/batch_h_bar": float(batch_h_bar),
        "wmc_erc/running_s_bar": float(running_stats["s_bar"]),
        "wmc_erc/running_s_std": float(running_stats["s_std"]),
        "wmc_erc/running_h_bar": float(running_stats["h_bar"]),
        "wmc_erc/total_turns": len(all_h),
        "wmc_erc/wm_nll": wm_nll,
    }
    metrics.update(add_metrics)

    return batch, metrics
