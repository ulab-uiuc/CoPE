"""Text-level PROGRESS-credit reasoning probe (validation, pre-training).

feature.md frames credit assignment as value increments: a "progress" step is one
that materially advanced expected task completion (V(τ_t) > V(τ_{t-1})); credit for
it should flow to the EARLIER steps that made it possible (HCA-style prerequisites).

Before building a value-free, text-based version of that, we validate two abilities
of the FROZEN policy, purely at the text level, on real trajectories:

  (A) PROGRESS identification — which steps advanced the task (vs redundant / failed
      / no-op / exploratory-without-gain)?
  (B) PREREQUISITE attribution — for a progress step, which EARLIER steps were
      necessary for it to happen / succeed?

This module is PURE (stdlib only; no torch/model). The model call is injected by the
GPU runner (probe_progress_credit.py). It provides: trajectory→numbered-steps,
prompt builders, output parsers, per-env weak structural gold, and scoring.
"""
import re
from typing import List, Tuple, Dict, Optional, Set

DOMAIN = {
    "alfworld": "an ALFWorld embodied household environment",
    "sciworld": "a ScienceWorld interactive science-experiment environment",
    "webshop":  "a WebShop e-commerce web-browsing environment",
    "babyai":   "a BabyAI grid-world instruction-following environment",
}

# ---------------------------------------------------------------- parsing helpers
def parse_action(assistant_text: str) -> str:
    m = re.search(r"Action:\s*(.+)", assistant_text or "", re.S)
    if not m:
        # webshop/sciworld sometimes emit the action bare; take last non-empty line
        lines = [l.strip() for l in (assistant_text or "").splitlines() if l.strip()]
        return lines[-1] if lines else ""
    for ln in m.group(1).strip().splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def extract_goal(messages: List[dict]) -> str:
    """Best-effort task goal: first user message that states the task."""
    for c in messages:
        if c.get("role") == "user":
            t = c.get("content", "") or ""
            m = re.search(r"(?:Your task is to:?|Your question:|Instruction:\s*\[SEP\])\s*(.+?)(?:\n|\[SEP\]|$)",
                          t, re.S)
            if m:
                return m.group(1).strip()[:300]
    # fallback: first user message head
    for c in messages:
        if c.get("role") == "user":
            return (c.get("content", "") or "")[:300]
    return ""


def trajectory_steps(messages: List[dict], obs_clip: int = 320) -> List[Dict]:
    """Return ordered scored steps: [{idx, action, result}] where result = the env
    observation FOLLOWING the action. Skips the initial instruction + ack."""
    conv = [c for c in messages if c.get("role") in ("user", "assistant")]
    steps = []
    for ai in range(3, len(conv), 2):                  # 0=instr,1=ack,2=obs,3=action,...
        if conv[ai].get("role") != "assistant":
            continue
        action = parse_action(conv[ai].get("content", ""))
        result = ""
        if ai + 1 < len(conv) and conv[ai + 1].get("role") == "user":
            result = (conv[ai + 1].get("content", "") or "")[:obs_clip].replace("\n", " ")
        steps.append({"idx": len(steps), "action": action, "result": result})
    return steps


def render_steps(steps: List[Dict]) -> str:
    return "\n".join(f"{s['idx']}: {s['action']}  ->  {s['result']}" for s in steps)


# ---------------------------------------------------------------- prompt builders
def build_progress_messages(env: str, goal: str, steps: List[Dict]) -> List[dict]:
    domain = DOMAIN.get(env, "an interactive environment")
    sys = (f"You are analyzing an agent's trajectory in {domain}. A PROGRESS step is one "
           "whose action materially advanced the task toward its GOAL — it changed the "
           "world or the agent's knowledge in a way that was actually needed for completion. "
           "Steps that leave the state effectively unchanged, repeat an earlier step, fail to "
           "take effect, or explore without obtaining anything useful are NOT progress.")
    user = (f"GOAL: {goal}\n\nSTEPS (idx: action -> result):\n{render_steps(steps)}\n\n"
            "List ONLY the step indices that are PROGRESS steps, most-load-bearing first.\n"
            "Answer in exactly this format:\nPROGRESS: <comma-separated indices>\n"
            "REASONS: <idx=short reason; ...>")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def build_prereq_messages(env: str, goal: str, steps: List[Dict], p_idx: int) -> List[dict]:
    domain = DOMAIN.get(env, "an interactive environment")
    p_action = next((s["action"] for s in steps if s["idx"] == p_idx), "")
    sys = (f"You are analyzing an agent's trajectory in {domain}. A PREREQUISITE of a step "
           "is an EARLIER step that was NECESSARY for it to be possible or to succeed — "
           "remove the prerequisite and the target step could not have happened.")
    user = (f"GOAL: {goal}\n\nSTEPS (idx: action -> result):\n{render_steps(steps)}\n\n"
            f"For PROGRESS step {p_idx} ({p_action}), list the EARLIER step indices that were "
            f"necessary PREREQUISITES (must be < {p_idx}). Exclude redundant/unrelated steps.\n"
            "Answer in exactly this format:\nPREREQ: <comma-separated indices>\n"
            "WHY: <short causal chain>")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- prompt VARIANTS
# Each variant returns chat messages; all parse with parse_progress / parse_prereq
# (they must all end with "PROGRESS: <idx,..>" / "PREREQ: <idx,..>"). Used by the
# sweep runner to compare prompts and let us pick the best.

def _steps_block(goal: str, steps: List[Dict]) -> str:
    return f"GOAL: {goal}\n\nSTEPS (idx: action -> result):\n{render_steps(steps)}"

def progress_variants(env: str, goal: str, steps: List[Dict]) -> Dict[str, List[dict]]:
    domain = DOMAIN.get(env, "an interactive environment")
    blk = _steps_block(goal, steps)
    fmt = ("\nAnswer in exactly this format:\nPROGRESS: <comma-separated indices>\n"
           "REASONS: <idx=short reason; ...>")
    V = {}
    # v1 direct (the original)
    V["direct"] = [
        {"role": "system", "content":
         (f"You are analyzing an agent's trajectory in {domain}. A PROGRESS step is one whose "
          "action materially advanced the task toward its GOAL — it changed the world or the "
          "agent's knowledge in a way actually needed for completion. Steps that leave the "
          "state effectively unchanged, repeat an earlier step, fail to take effect, or "
          "explore without obtaining anything useful are NOT progress.")},
        {"role": "user", "content": f"{blk}\n\nList ONLY the step indices that are PROGRESS steps, "
         f"most-load-bearing first.{fmt}"}]
    # v2 chain-of-thought first
    V["cot"] = [
        {"role": "system", "content":
         f"You are analyzing an agent's trajectory in {domain}. Reason carefully, then answer."},
        {"role": "user", "content":
         f"{blk}\n\nFirst THINK step-by-step: for each step, did it move the task closer to the "
         "GOAL (changed state/knowledge needed for success) or was it redundant / failed / a "
         "no-op / fruitless exploration? Then list ONLY the genuine PROGRESS step indices."
         f"{fmt}"}]
    # v3 value-increment framing (feature.md)
    V["value"] = [
        {"role": "system", "content":
         (f"You are estimating progress in an agent's trajectory in {domain}. Think of a hidden "
          "'closeness-to-success' value that rises when a step brings the goal nearer. A PROGRESS "
          "step is one AFTER WHICH success became more certain or closer; non-progress steps "
          "leave it unchanged (redundant, failed, no-op, fruitless).")},
        {"role": "user", "content":
         f"{blk}\n\nList ONLY the indices of steps that RAISED closeness-to-success.{fmt}"}]
    # v4 counterfactual / load-bearing
    V["counterfactual"] = [
        {"role": "system", "content":
         (f"You are analyzing an agent's trajectory in {domain}. A step is LOAD-BEARING if "
          "removing it would make the task fail or force the agent to redo it; non-load-bearing "
          "steps could be deleted with the task still succeeding.")},
        {"role": "user", "content":
         f"{blk}\n\nList ONLY the indices of LOAD-BEARING steps (those whose removal breaks the "
         f"task).{fmt}"}]
    # v5 few-shot (one worked example before the real one)
    V["fewshot"] = [
        {"role": "system", "content":
         f"You are analyzing an agent's trajectory in {domain}. A PROGRESS step materially "
         "advanced the task; repeats, ineffective actions, and fruitless exploration are NOT "
         "progress."},
        {"role": "user", "content":
         ("EXAMPLE — GOAL: put a cooled apple on the table\nSTEPS:\n0: go to fridge -> closed\n"
          "1: open fridge -> you see an apple\n2: take apple -> picked up\n3: take apple -> "
          "no change (already held)\n4: go to table -> arrived\n5: put apple on table -> done\n"
          "PROGRESS: 1, 2, 5\nREASONS: 1=revealed apple; 2=obtained it; 5=completed goal "
          "(3 had no effect, 0/4 are mere navigation)\n\n"
          f"NOW THE REAL ONE.\n{blk}\n\nList the PROGRESS step indices.{fmt}")}]
    # v6 per-step forced decision (like the predict-first that fixed epistemic collapse)
    V["perstep"] = [
        {"role": "system", "content":
         f"You are analyzing an agent's trajectory in {domain}. Judge EACH step independently."},
        {"role": "user", "content":
         f"{blk}\n\nGo through the steps in order. For each, decide PROGRESS (it advanced the "
         "task toward the GOAL) or SKIP (repeat / failed / no-op / fruitless). Then collect the "
         f"PROGRESS ones.{fmt}"}]
    return V

def prereq_variants(env: str, goal: str, steps: List[Dict], p_idx: int) -> Dict[str, List[dict]]:
    domain = DOMAIN.get(env, "an interactive environment")
    p_action = next((s["action"] for s in steps if s["idx"] == p_idx), "")
    blk = _steps_block(goal, steps)
    fmt = ("\nAnswer in exactly this format:\nPREREQ: <comma-separated indices>\nWHY: <short chain>")
    head = f"{blk}\n\nTarget PROGRESS step {p_idx}: {p_action}  (prerequisites must be < {p_idx})"
    V = {}
    V["direct"] = [
        {"role": "system", "content":
         (f"You are analyzing an agent's trajectory in {domain}. A PREREQUISITE of a step is an "
          "EARLIER step that was NECESSARY for it to be possible or to succeed.")},
        {"role": "user", "content":
         f"{head}\n\nList the earlier step indices that were necessary PREREQUISITES. Exclude "
         f"redundant/unrelated steps.{fmt}"}]
    V["counterfactual"] = [
        {"role": "system", "content":
         f"You are analyzing an agent's trajectory in {domain}. Reason counterfactually."},
        {"role": "user", "content":
         f"{head}\n\nFor each earlier step, ask: 'WITHOUT this step, could step {p_idx} still have "
         f"happened and succeeded?' List ONLY the indices where the answer is NO (truly "
         f"necessary).{fmt}"}]
    V["chain"] = [
        {"role": "system", "content":
         f"You are analyzing an agent's trajectory in {domain}."},
        {"role": "user", "content":
         f"{head}\n\nTrace the MINIMAL causal chain of earlier steps that enabled step {p_idx} "
         f"(e.g. you must possess/locate/open something first). List those indices.{fmt}"}]
    V["rating"] = [
        {"role": "system", "content":
         f"You are analyzing an agent's trajectory in {domain}. Rate necessity strictly."},
        {"role": "user", "content":
         f"{head}\n\nFor each earlier step, rate its necessity for step {p_idx}: 2=strictly "
         "required, 1=helpful, 0=irrelevant/redundant. Then list ONLY the indices rated 2."
         f"{fmt}"}]
    return V


EXPLORE_SYS = (
    "You are analyzing an agent acting in {domain}, where it can only PARTIALLY observe its "
    "surroundings. An EXPLORATION step is one the agent took to obtain information it was still "
    "missing — for example searching for an object whose location it did not know, moving around "
    "to find a room or a path, or looking/reading/checking to learn the state of things — so that "
    "it could then act correctly. Judge a step by what the agent knew AT THE TIME, not in "
    "hindsight: a search that happened to find nothing is still exploration, because the agent "
    "could not have known that beforehand — do not call it wasted just because the target was "
    "ultimately somewhere else. For contrast, steps that directly carry out the task — obtaining, "
    "using, or placing the right thing, or going somewhere specifically in order to act there — "
    "are PROGRESS, not exploration; and purely repeated, failed, or irrelevant actions are "
    "neither. Identify the EXPLORATION steps.")

def build_explore_messages(env: str, goal: str, steps: List[Dict]) -> List[dict]:
    """EXPLORATION probe — steps the agent took to gather information it lacked under
    partial observability (search for an object/place, find a path, check state),
    judged from the agent's ex-ante viewpoint so empty/failed searches still count;
    distinct from PROGRESS (directly carrying out the task) and from useless."""
    domain = DOMAIN.get(env, "an interactive environment")
    user = (f"GOAL: {goal}\n\nSTEPS (idx: action -> result):\n{render_steps(steps)}\n\n"
            "List the EXPLORATION steps — the ones the agent took to gather information it did not "
            "yet have (searching for something, finding a way or place, checking state), including "
            "searches that came up empty. Exclude steps that directly perform the task (those are "
            "PROGRESS) and any merely repeated, failed, or irrelevant ones. If the agent never had "
            "to gather information, return none.\n"
            "Answer in exactly this format:\nINFO: <what information was being gathered, short>\n"
            "EXPLORE: <comma-separated indices, or 'none'>")
    return [{"role": "system", "content": EXPLORE_SYS.format(domain=domain)},
            {"role": "user", "content": user}]


def build_infodep_messages(env: str, goal: str, steps: List[Dict], p_idx: int) -> List[dict]:
    """INFORMATION-dependency (reverse dependency) probe. Distinct from the physical
    prerequisite: an INFORMATION prerequisite of a step is an earlier step that
    supplied knowledge the agent needed in order to KNOW to take this step, or to
    take it correctly — typically a search/look/check that revealed where the target
    was, what state it was in, or which choice was right. (A physical prerequisite,
    by contrast, is an earlier action mechanically required, e.g. you must be holding
    an object before you can use it.) Here we ask for the INFORMATION prerequisites."""
    domain = DOMAIN.get(env, "an interactive environment")
    p_action = next((s["action"] for s in steps if s["idx"] == p_idx), "")
    sys = (f"You are analyzing an agent acting in {domain} under partial observation. Before the "
           "agent can correctly perform a key action, it often must first LEARN things it could "
           "not know at the start — where an object is, what state something is in, which option "
           "is right. An INFORMATION PREREQUISITE of a step is an EARLIER step that contributed "
           "information the agent needed to know to take this step (or to take it correctly). This "
           "includes BOTH (a) steps that DIRECTLY revealed the needed fact (e.g. seeing the target, "
           "learning its state), AND (b) steps that contributed INDIRECTLY by RULING OUT "
           "alternatives or narrowing the possibilities. In a search, every place the agent "
           "checked and found empty still gave real information — 'the target is NOT here' — which "
           "was a necessary part of eventually locating it. So when a step depended on a search, "
           "its information prerequisites are the WHOLE search chain (every check that narrowed it "
           "down), not only the final step that happened to find the answer. (A physical "
           "prerequisite, by contrast, is an earlier action that was mechanically required.) "
           "Identify the INFORMATION prerequisites.")
    user = (f"GOAL: {goal}\n\nSTEPS (idx: action -> result):\n{render_steps(steps)}\n\n"
            f"Step {p_idx} ({p_action}) relied on something the agent had to LEARN first because "
            "it could not see it at the start (e.g. WHERE the target was). The agent learned it by "
            "SEARCHING. List EVERY step that was part of that search — the whole sequence of checks "
            "the agent made to pin the answer down, NOT just the final check that happened to "
            "reveal it. Include the checks that came up empty: each one ruled out a possibility "
            "and was a necessary part of narrowing down to the answer (e.g. checking 8 shelves to "
            "discover the item is on the 8th means ALL 8 checks were part of finding it). Exclude "
            "only steps that were merely physically required or truly unrelated to the search. If "
            f"step {p_idx} needed no prior search, return none.\n"
            "Answer in exactly this format:\nINFO_DEP: <comma-separated indices, or 'none'>\n"
            "WHY: <short>")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- output parsers
def _int_list(text: str, tag: str, hi: Optional[int] = None) -> List[int]:
    m = re.search(rf"{tag}:\s*([0-9,\s]+)", text or "")
    if not m:
        return []
    out = []
    for tok in re.split(r"[,\s]+", m.group(1).strip()):
        if tok.isdigit():
            v = int(tok)
            if hi is None or v <= hi:
                out.append(v)
    return sorted(set(out))


def parse_progress(text: str, n_steps: int) -> List[int]:
    return _int_list(text, "PROGRESS", hi=n_steps - 1)


def parse_prereq(text: str, p_idx: int) -> List[int]:
    return [i for i in _int_list(text, "PREREQ", hi=p_idx - 1) if i < p_idx]


def parse_explore(text: str, n_steps: int) -> List[int]:
    return _int_list(text, "EXPLORE", hi=n_steps - 1)


def parse_infodep(text: str, p_idx: int) -> List[int]:
    return [i for i in _int_list(text, "INFO_DEP", hi=p_idx - 1) if i < p_idx]


def parse_target(text: str) -> str:
    m = re.search(r"(?:INFO|TARGET):\s*(.+)", text or "")
    return m.group(1).strip()[:80] if m else ""


# ---------------------------------------------------------------- weak structural gold
_OBJ_RE = re.compile(r"\b([a-z]+ \d+)\b")          # "tomato 3", "microwave 1", "B07PM8MJKZ"-ish
_WS_ID_RE = re.compile(r"\b([bB][0-9A-Z]{9})\b")   # webshop ASIN

def _nouns(action: str) -> Set[str]:
    s = set(m.group(1) for m in _OBJ_RE.finditer(action.lower()))
    s |= set(m.group(1) for m in _WS_ID_RE.finditer(action))
    return s

_FAIL_RE = re.compile(r"no known action|nothing happens|not open|is closed|can't|cannot|"
                      r"invalid|no matches that input|don't|do not", re.I)

def terminal_progress_idx(steps: List[Dict]) -> int:
    """Weak gold for the FINAL progress step of a SUCCESSFUL trajectory: the last
    action that actually DID something — skip trailing help/look/inventory and any
    action whose result signals failure/no-op (early-phase 'successes' are often a
    single real action followed by many failed repeats)."""
    for s in reversed(steps):
        a = s["action"].lower()
        if (a and not a.startswith(("help", "look", "inventory"))
                and not _FAIL_RE.search(s["result"])):
            return s["idx"]
    return steps[-1]["idx"] if steps else -1

def weak_prereq_gold(steps: List[Dict], p_idx: int) -> Set[int]:
    """Earlier steps sharing a key noun (object / receptacle / product id) with the
    progress action — a crude necessary-condition proxy (e.g. 'heat tomato 3' shares
    'tomato 3' with the earlier 'take tomato 3')."""
    p = next((s for s in steps if s["idx"] == p_idx), None)
    if not p:
        return set()
    keys = _nouns(p["action"])
    gold = set()
    for s in steps:
        if s["idx"] >= p_idx:
            continue
        if _nouns(s["action"]) & keys:
            gold.add(s["idx"])
    return gold


# ---------------------------------------------------------------- scoring
def prf(pred: Set[int], gold: Set[int]) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f
