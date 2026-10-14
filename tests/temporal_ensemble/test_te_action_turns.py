"""TE properties 6+8: build_action_targets' action_turns are correct, and fut has not
regressed.

The function already asserts internally that [actions_seq[j] for j in sel] == fut, so if
it runs at all, fut is intact. This adds:
  (6) random trajectories x {skip_invalid T/F} all run through (no internal assert fires)
  (8) with skip_invalid=True, action_turns skips the turn number of an invalid action
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

import sys, random
from verl.agent_trainer.ppo.action_forecast import build_action_targets


def test_te_action_turns():

    def mk(acts, results):
        """acts[i] is turn i's action, results[i] is the observation it produced."""
        m = [{'role':'user','content':'sys'},{'role':'assistant','content':'Ok.'},
             {'role':'user','content':'task'}]
        for a, r in zip(acts, results):
            m.append({'role':'assistant','content':a})
            m.append({'role':'user','content':r})
        return m

    # --- property 8: turn 1 (0-based) is invalid, skip_invalid should skip it ---
    acts = ['go to kitchen','open nothing','open fridge','pick up cup']
    res  = ['You are in the kitchen','No known action matches that input.',
            'The fridge is open','You pick up the cup']
    for k in (2,3,4):
        for si, tl in ((True, build_action_targets(mk(acts,res), k=k, skip_invalid=True,  env='sciworld')),
                       (False, build_action_targets(mk(acts,res), k=k, skip_invalid=False, env='sciworld'))):
            t0 = tl[0]['action_turns']
            assert t0[0] == 0, f"k={k} si={si}: turn 0's slot 1 must be turn 0, got {t0}"
            if si:
                assert 1 not in t0, f"k={k}: skip_invalid=True did not skip invalid turn 1: {t0}"
            else:
                assert len(t0) < 2 or 1 in t0, f"k={k}: skip_invalid=False skipped turn 1: {t0}"
            for tg in tl:
                at = tg['action_turns']
                assert len(at) == len(tg['actions']), (len(at), len(tg['actions']))
                assert all(at[i] < at[i+1] for i in range(len(at)-1)), f"not strictly increasing: {at}"
                # Note: slot 1's turn number is **not necessarily** the emitting turn. With
                # skip_invalid=True, if the emitting turn's own action is invalid it is
                # dropped by its own filter, so at[0] > src_turn.
                # This is why TE's member selection must go by real turn number (t > s),
                # never by slot index (j >= 1).
                assert at[0] >= tg['src_turn'], f"slot 1 cannot precede the emitting turn: {at[0]} < {tg['src_turn']}"
                if not si:
                    assert at[0] == tg['src_turn'], "with skip_invalid=False, slot 1 must be the emitting turn"

    # Pin down that trap explicitly: when the emitting turn's own action is invalid,
    # slot 1 is already a k >= 1 member.
    t_skip = build_action_targets(mk(acts,res), k=3, skip_invalid=True, env='sciworld')
    bysrc = {tg['src_turn']: tg for tg in t_skip}
    assert 1 in bysrc, "turn 1 should still produce a target (it still sees valid actions ahead)"
    tg1 = bysrc[1]
    assert tg1['action_turns'][0] == 2, f"turn 1's slot 1 should point at turn 2, got {tg1['action_turns']}"
    n_k0 = sum(1 for tg in t_skip for t in tg['action_turns'] if t == tg['src_turn'])
    n_k1 = sum(1 for tg in t_skip for t in tg['action_turns'] if t >  tg['src_turn'])
    print(f"  property 8 PASS: skip_invalid renumbers correctly (k=2,3,4)")
    print(f"  key finding: {n_k0} slots at k=0, {n_k1} members at k>=1 -- "
          f"member selection must go by t>s, not by slot index")

    # --- property 6: random stress test; no internal assert firing means fut is intact ---
    rng = random.Random(0)
    BAD = ['No known action matches that input.','The fridge is open','Nothing happens.','You see a cup']
    n_tg = 0
    for trial in range(200):
        T = rng.randint(1, 12)
        a = [f'action_{i}_{rng.randint(0,99)}' for i in range(T)]
        r = [rng.choice(BAD) for _ in range(T)]
        for si in (True, False):
            for k in (1,2,3,4,5):
                tg = build_action_targets(mk(a,r), k=k, skip_invalid=si, env='sciworld')
                n_tg += len(tg)
    print(f"  property 6 PASS: 200 random trajectories x {{skip T/F}} x k in 1..5, "
          f"{n_tg} targets, no internal assertion fired")
