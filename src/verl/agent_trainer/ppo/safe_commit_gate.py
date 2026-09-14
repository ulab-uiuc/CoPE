"""Text-level safe-commit gate (default OFF; enabled by safe_commit_text_gate).

Per scored turn, ask the policy whether the action's OUTCOME is predictable
(predict-first classifier validated in analyze_epistemic_multi.py). Combine that
SEMANTIC predictability with a low-ACTION-ENTROPY metric: commit (boost advantage)
iff the turn is BOTH model-judged predictable AND low-entropy. The semantic gate
(exogenous) contains the action-entropy feedback loop — a confident action whose
outcome is NOT predictable is never boosted.

The resulting per-turn weight w_t ∈ [0,1] is written to
batch.non_tensor_batch['safe_commit_w'] and consumed by
apply_safe_commit_advantage (clipping_method='safe_commit', mode='add'): it boosts
A_GRPO only on w_t>0 turns of winning trajectories, at HCA-matched magnitude (ω).

This module is PURE (no torch.distributed / no model). Generation is injected.
"""
import re
from collections import defaultdict
from typing import List, Tuple, Callable, Optional

DOMAIN = {
    "alfworld": "an ALFWorld embodied household environment",
    "sciworld": "a ScienceWorld interactive science-experiment environment",
    "webshop":  "a WebShop e-commerce web-browsing environment",
    "babyai":   "a BabyAI grid-world instruction-following environment",
}

SYS_TMPL = (
    "You are analyzing an autonomous agent operating in {domain}. Decide whether "
    "the OUTCOME of a proposed action is already determined by what the agent has "
    "ALREADY observed (DETERMINISTIC), or whether executing it REVEALS hidden "
    "information the agent could not have predicted beforehand (REVEALING).")

USER_TMPL = (
    "{history}\nCURRENT OBSERVATION:\n{cur_obs}\n\nPROPOSED ACTION: {action}\n\n"
    "Step 1. PREDICT the exact resulting observation in ONE short sentence.\n"
    "Step 2. Could you have stated that EXACT result from prior observations "
    "(including the HISTORY above) alone, BEFORE acting? If yes -> DETERMINISTIC. "
    "If the action uncovers contents/results/state you had not yet seen -> REVEALING.\n"
    "Answer in exactly this format:\nPREDICT: <one sentence>\n"
    "LABEL: <DETERMINISTIC or REVEALING>\nHIDDEN: <hidden variable in <=8 words, or 'none'>")

_NONE_RE = re.compile(r"^\s*(none|n/?a|nothing|no hidden)\b", re.I)


def parse_action(assistant_text: str) -> str:
    """Extract the action command from an assistant turn ('...Action:\\n<cmd>')."""
    m = re.search(r"Action:\s*(.+)", assistant_text or "", re.S)
    if not m:
        return ""
    for ln in m.group(1).strip().splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def parse_predictable(text: str) -> int:
    """1 if the classifier judged the outcome DETERMINISTIC (hidden=none), else 0.

    Label derived from the HIDDEN field (robust): 'none' -> DETERMINISTIC; a named
    hidden variable -> REVEALING. (Direct binary LABEL collapses to all-REVEALING.)
    """
    hm = re.search(r"HIDDEN:\s*(.+)", text or "")
    hid = hm.group(1).strip() if hm else ""
    if hid == "" or _NONE_RE.match(hid):
        # fall back to explicit LABEL only when HIDDEN is missing/empty
        if re.search(r"LABEL:\s*REVEAL", text or "", re.I):
            return 0
        return 1
    return 0


def iter_action_turns(messages: List[dict], history_len: int = 3,
                      obs_clip: int = 800, hist_obs_clip: int = 300):
    """Yield (history_str, cur_obs, action) for each scored assistant action turn,
    in chronological order (aligned with response-mask turn_boundaries). The first
    assistant message (the 'OK' ack, part of the prompt seed) is skipped.
    """
    conv = [c for c in messages if c.get("role") in ("user", "assistant")]
    out = []
    for ai in range(3, len(conv), 2):                 # 0=instr,1=ack,2=obs,3=action,...
        if conv[ai].get("role") != "assistant":
            continue
        action = parse_action(conv[ai].get("content", ""))
        cur_obs = (conv[ai - 1].get("content", "") or "")[:obs_clip]
        hlines = []
        for j in range(2, ai - 1, 2):
            if j < len(conv) and conv[j].get("role") == "user":
                a = parse_action(conv[j + 1].get("content", "")) if j + 1 < len(conv) else ""
                hlines.append(f"obs: {(conv[j].get('content','') or '')[:hist_obs_clip]}\nyou did: {a}")
        hist = "HISTORY:\n" + ("\n".join(hlines[-history_len:]) if hlines else "(start)")
        out.append((hist, cur_obs, action))
    return out


def build_messages(env: str, history: str, cur_obs: str, action: str) -> List[dict]:
    domain = DOMAIN.get(env, "an interactive environment")
    return [
        {"role": "system", "content": SYS_TMPL.format(domain=domain)},
        {"role": "user", "content": USER_TMPL.format(history=history, cur_obs=cur_obs, action=action)},
    ]


def combine_w(predictable_per_traj: List[List[int]],
              action_ent_per_traj: List[List[float]],
              ginv: List[int],
              gate: str = "and",
              ent_quantile: float = 0.5) -> List[List[float]]:
    """Combine semantic predictability with low-action-entropy → per-turn w_t∈[0,1].

    low_entropy_t := action entropy below its GROUP (uid) reference (mean for 'and';
    used as a soft factor otherwise). 'and' (default, the strict "iff predictable
    AND low-entropy"): w=1.0 iff predictable and low-entropy else 0.0. 'soft':
    w = predictable · clip(1 − ent_t/ent_group, 0, 1).
    """
    # per-group action-entropy reference (mean over that group's turns)
    grp_ent = defaultdict(list)
    for i, ents in enumerate(action_ent_per_traj):
        g = int(ginv[i])
        for e in ents:
            if e is not None:
                grp_ent[g].append(float(e))
    grp_mean = {g: (sum(v) / len(v)) for g, v in grp_ent.items() if v}
    glob = [e for v in grp_ent.values() for e in v]
    glob_mean = (sum(glob) / len(glob)) if glob else 0.0

    w_out = []
    for i, preds in enumerate(predictable_per_traj):
        g = int(ginv[i])
        ref = grp_mean.get(g, glob_mean)
        ents = action_ent_per_traj[i]
        row = []
        n = min(len(preds), len(ents))
        for t in range(n):
            p = float(preds[t]) if preds[t] is not None else 0.0
            e = ents[t]
            if e is None:
                row.append(0.0); continue
            if gate == "soft":
                low = max(0.0, 1.0 - (float(e) / (ref + 1e-6)))
                row.append(p * low)
            else:  # 'and'
                low = 1.0 if float(e) < ref else 0.0
                row.append(1.0 if (p > 0.5 and low > 0.5) else 0.0)
        w_out.append(row)
    return w_out
