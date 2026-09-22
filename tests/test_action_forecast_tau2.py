"""Action forecasting under τ²-bench's native tool-calling protocol.

The ReAct helpers assume [instr, ack, obs, action, obs, action, ...] with ``Action:``
lines; a native conversation is [system, user, assistant, tool|user, assistant, ...]
and an action is a <tool_call> block or a customer message. These tests pin the
native path and check the ReAct path is untouched.

The invariant that matters: a forecast target is a LITERAL SUBSTRING of the turn the
policy generated -- never re-encoded -- and the trained tokens are that substring's
tokens. Everything else (turns layout, filler prompts never trained) follows from it.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verl.agent_trainer.ppo.action_forecast import (  # noqa: E402
    ACTION_FORECAST_NEXT_PROMPT, _action_turn_indices, build_action_forecast_samples,
    build_action_targets, extract_action, is_invalid_outcome)

CALL_1 = "<tool_call>\n{\"name\": \"find_user_id_by_email\", \"arguments\": {\"email\": \"a@b.c\"}}\n</tool_call>"
CALL_CANCEL = "<tool_call>{\"name\": \"cancel_pending_order\", \"arguments\": {\"order_id\": \"#W1\", \"reason\": \"no longer needed\"}}</tool_call>"
CALL_DETAILS = "<tool_call>{\"name\": \"get_order_details\", \"arguments\": {\"order_id\": \"#W1\"}}</tool_call>"

NATIVE = [
    {"role": "system", "content": "You are a customer service agent.\n<policy>P</policy>"},
    {"role": "user", "content": "Hi, I want to cancel my order."},
    {"role": "assistant", "content": "Sure. " + CALL_1},
    {"role": "tool", "content": "yara_muller_8652"},
    {"role": "assistant", "content": "Found you.\nWhich order?\n"},
    {"role": "user", "content": "#W1"},
    {"role": "assistant", "content": CALL_CANCEL},
    {"role": "tool", "content": "Error: Non-pending order cannot be cancelled"},
    {"role": "assistant", "content": CALL_DETAILS},
    {"role": "tool", "content": "{\"order_id\": \"#W1\", \"status\": \"delivered\"}"},
    {"role": "assistant", "content": "That order was delivered, so it cannot be cancelled."},
    {"role": "user", "content": "###STOP###"},
]

REACT = [
    {"role": "user", "content": "instruction"}, {"role": "assistant", "content": "Ok."},
    {"role": "user", "content": "obs0"}, {"role": "assistant", "content": "Thought: x\nAction: go north"},
    {"role": "user", "content": "obs1"}, {"role": "assistant", "content": "Action: take key"},
]


def test_native_layout_every_assistant_turn_is_an_action():
    assert _action_turn_indices(NATIVE) == [2, 4, 6, 8, 10]


def test_react_layout_unchanged():
    assert _action_turn_indices(REACT) == [3, 5]
    assert extract_action(REACT[3]["content"]) == "go north"


# ---- the target is a literal substring of the policy's turn --------------------------------

def test_tool_call_target_is_the_block_exactly_as_written():
    # newlines and spacing kept, the prose before it dropped like alfworld drops the Thought
    out = extract_action(NATIVE[2]["content"], env="tau2")
    assert out == CALL_1
    assert out in NATIVE[2]["content"]


def test_message_target_is_the_text_unchanged():
    # no prefix, no truncation, no reflow -- not even the trailing newline
    assert extract_action(NATIVE[4]["content"], env="tau2") == "Found you.\nWhich order?\n"
    long = "x" * 5000
    assert extract_action(long, env="tau2") == long
    assert extract_action("", env="tau2") == ""
    assert extract_action("  \n", env="tau2") == ""


def test_first_of_several_calls_is_the_target():
    # the client executes only the first call of a turn; so does the target
    out = extract_action(CALL_DETAILS + "\n" + CALL_CANCEL, env="tau2")
    assert out == CALL_DETAILS


def test_tau2_invalid_outcomes():
    assert is_invalid_outcome("Error: Non-pending order cannot be cancelled", env="tau2")
    assert is_invalid_outcome("Invalid turn: reply with a <tool_call> or a message", env="tau2")
    assert not is_invalid_outcome("{\"status\": \"ok\", \"error_count\": 0}", env="tau2")
    assert not is_invalid_outcome("I don't have my order id.", env="tau2")


def test_failed_call_that_is_not_redone_is_kept():
    # the cancel fails ("non-pending order") and is never retried: it told the agent the
    # request is impossible, it is part of the winning path, so it stays in the target
    tg = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2")
    by_turn = {t["prefix_end"] + 1: t["actions"] for t in tg}
    assert by_turn[4] == ["Found you.\nWhich order?\n", CALL_CANCEL, CALL_DETAILS]
    # every target's prefix ends on the observation the action answers
    assert all(NATIVE[t["prefix_end"]]["role"] in ("user", "tool") for t in tg)


CALL_EMAIL_BAD = "<tool_call>\n{\"name\": \"find_user_id_by_email\", \"arguments\": {\"email\": \"a@b.c\"}}\n</tool_call>"
CALL_EMAIL_OK = "<tool_call>\n{\"name\": \"find_user_id_by_email\", \"arguments\": {\"email\": \"a@b.co\"}}\n</tool_call>"
GREETING = "Found you. What can I help you with today?"

RETRY = [
    {"role": "system", "content": "You are a customer service agent."},
    {"role": "user", "content": "Hi, my email is a@b.c"},
    {"role": "assistant", "content": CALL_EMAIL_BAD},                    # 2: fails, redone at 6
    {"role": "tool", "content": "Error: User not found"},
    {"role": "assistant", "content": "I could not find that email. Could you check it?"},
    {"role": "user", "content": "Sorry, it is a@b.co"},
    {"role": "assistant", "content": CALL_EMAIL_OK},                     # 6: succeeds
    {"role": "tool", "content": "yara_muller_8652"},
    {"role": "assistant", "content": GREETING},                          # 8
    {"role": "user", "content": "Hm?"},
    {"role": "assistant", "content": "  " + GREETING.replace(" ", "  ")},  # 10: repeat of 8
    {"role": "user", "content": "Cancel #W1 please."},
    {"role": "assistant", "content": CALL_CANCEL},                       # 12
    {"role": "tool", "content": "{\"order_id\": \"#W1\", \"status\": \"cancelled\"}"},
]


def test_failed_call_redone_later_is_dropped():
    tg = build_action_targets(RETRY, k=3, skip_invalid=True, env="tau2")
    first = {t["prefix_end"] + 1: t for t in tg}[2]
    # the failed lookup is skipped: the plan from turn 2 starts with the message, then
    # the lookup that worked
    assert first["actions"] == ["I could not find that email. Could you check it?", CALL_EMAIL_OK, GREETING]


def test_repeated_message_is_dropped():
    tg = build_action_targets(RETRY, k=3, skip_invalid=True, env="tau2")
    from_greeting = {t["prefix_end"] + 1: t["actions"] for t in tg}[8]
    # the word-for-word repeat at turn 10 is not a step of the plan
    assert from_greeting == [GREETING, CALL_CANCEL]


def test_without_skip_invalid_everything_stays():
    tg = build_action_targets(RETRY, k=3, skip_invalid=False, env="tau2")
    by = {t["prefix_end"] + 1: t["actions"] for t in tg}
    assert by[2][0] == CALL_EMAIL_BAD
    assert by[8] == [GREETING, "  " + GREETING.replace(" ", "  "), CALL_CANCEL]


# ---- gate='mixed': only the wins of groups that also have a loss ---------------------------

def test_mixed_gate_keeps_only_wins_of_mixed_groups():
    from verl.agent_trainer.ppo.action_forecast import select_forecast_trajectories
    group_ids = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    rewards = [1, 1, 1, 1, 0, 0, 0, 0, 0]       # a all-win, b mixed, c all-fail
    keep, stats = select_forecast_trajectories(9, rewards, gate="mixed", group_ids=group_ids, group_norm=True)
    assert keep == [False, False, False, True, False, False, False, False, False]
    assert stats["action_forecast/n_groups_mixed"] == 1
    assert stats["action_forecast/n_groups_allwin_skipped"] == 1
    assert stats["action_forecast/n_wins_skipped"] == 3
    # no mixed group -> nothing to distil, like GRPO has nothing to learn
    keep2, _ = select_forecast_trajectories(6, [1, 1, 1, 0, 0, 0], gate="mixed", group_ids=list("aaabbb"))
    assert not any(keep2)
    # 'wins' is unchanged: every win
    keep3, _ = select_forecast_trajectories(9, rewards, gate="wins", group_ids=group_ids, group_norm=True)
    assert keep3 == [True, True, True, True, False, False, False, False, False]


def test_mixed_gate_needs_group_ids_and_known_gates_only():
    from verl.agent_trainer.ppo.action_forecast import select_forecast_trajectories
    with pytest.raises(ValueError):
        select_forecast_trajectories(3, [1, 0, 1], gate="mixed")
    with pytest.raises(ValueError):
        select_forecast_trajectories(3, [1, 0, 1], gate="best")


def test_native_targets_need_the_turns_layout():
    # literal targets are multi-line; the K-line list would blur their boundaries
    with pytest.raises(ValueError):
        build_action_forecast_samples(NATIVE, tokenizer=None, k=3, env="tau2", layout="list")
    with pytest.raises(ValueError):
        build_action_forecast_samples(NATIVE, tokenizer=None, k=3, env="tau2", layout="lines")


# ---- turns layout: K assistant turns, loss on the assistant turns only ---------------------

@pytest.fixture(scope="module")
def tok():
    path = os.environ.get("COPE_TEST_TOKENIZER")
    if not path:
        pytest.skip("set COPE_TEST_TOKENIZER to a local Qwen2.5 snapshot")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(path)


def _masked_spans(sample):
    spans, cur = [], []
    for t, m in zip(sample["input_ids"].tolist(), sample["loss_mask"].tolist()):
        if m:
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def test_turns_train_exactly_the_policy_spans(tok):
    samples = build_action_forecast_samples(NATIVE, tok, k=3, skip_invalid=True, env="tau2", layout="turns")
    assert samples
    s = samples[0]                       # forecast from the first action turn
    spans = _masked_spans(s)
    expected = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2")[0]["actions"]
    assert len(spans) == s["k_realized"] == len(expected) == 3
    for span, action in zip(spans, expected):
        # each trained span is one literal action closed like a policy turn ...
        assert tok.decode(span) == action + "<|im_end|>\n"
        # ... and its tokens are the tokens of that literal text
        assert span[:-2] == tok.encode(action, add_special_tokens=False)
    # the filler prompt between turns is never trained, and the K actions are separate turns
    ids = s["input_ids"].tolist()
    filler = tok.encode(ACTION_FORECAST_NEXT_PROMPT, add_special_tokens=False)
    hits = [i for i in range(len(ids) - len(filler) + 1) if ids[i:i + len(filler)] == filler]
    assert len(hits) == 2
    for i in hits:
        assert not any(s["loss_mask"][i:i + len(filler)].tolist())
    assert tok.decode(ids).count("<|im_start|>assistant") == 3


def test_react_list_layout_is_unchanged(tok):
    # the ReAct envs keep the original K-line construction
    s = build_action_forecast_samples(REACT, tok, k=2, env="alfworld")[0]
    spans = _masked_spans(s)
    assert len(spans) == 1
    assert tok.decode(spans[0]) == "go north\ntake key<|im_end|>\n"


# ---- balance_calls: the forecast must not move the policy's call-vs-message rate ----------

def test_call_balance_weights_are_neutral():
    from verl.agent_trainer.ppo.action_forecast import call_balance_weights
    for p, q in [(0.3, 0.14), (0.3, 0.5), (0.1, 0.4), (0.5, 0.5)]:
        w_c, w_m = call_balance_weights(p, q, w_max=100)
        # the decision-token pushes cancel at the policy's own rate
        assert abs(w_c * q * (1 - p) - w_m * (1 - q) * p) < 1e-12
        assert abs(w_c * q / (w_c * q + w_m * (1 - q)) - p) < 1e-12
    # one kind absent from the targets: no weighting is neutral, so the decision is untrained
    assert call_balance_weights(0.3, 0.0) == (0.0, 0.0)
    assert call_balance_weights(0.3, 1.0) == (0.0, 0.0)
    # a policy that never calls: call targets' decision untrained, messages' pushes are 0 anyway
    w_c, w_m = call_balance_weights(0.0, 0.2)
    assert w_c == 0.0 and abs(w_c * 0.2 * 1.0 - w_m * 0.8 * 0.0) < 1e-12
    # weights are bounded
    assert call_balance_weights(0.9, 0.01)[0] <= 5.0


def test_decision_tokens_mark_the_first_token_of_each_target_turn(tok):
    samples = build_action_forecast_samples(NATIVE, tok, k=3, skip_invalid=True, env="tau2", layout="turns")
    call_id = tok.convert_tokens_to_ids("<tool_call>")
    for s in samples:
        dk = s["decision_kind"]
        marks = [i for i in range(dk.numel()) if dk[i] > 0]
        assert len(marks) == s["k_realized"]
        for i in marks:
            assert s["loss_mask"][i] == 1                         # a trained token ...
            assert i == 0 or s["loss_mask"][i - 1] == 0           # ... that opens its span
            assert (s["input_ids"][i].item() == call_id) == (dk[i].item() == 1)


def test_balanced_batch_carries_the_policy_call_share(tok):
    from verl.agent_trainer.ppo.action_forecast import (
        ACTION_FORECAST_BALANCE_WMAX, build_action_forecast_batch)
    msgs = [NATIVE, NATIVE, RETRY, RETRY]
    batch, meta = build_action_forecast_batch(
        msgs, tok, rewards=[1, 0, 1, 0], k=3, gate="mixed", skip_invalid=True, env="tau2",
        group_ids=["a", "a", "b", "b"], group_norm=True, layout="turns", balance_calls=True)
    p, q = meta["action_forecast/policy_call_share"], meta["action_forecast/target_call_share"]
    assert 0 < p < 1 and 0 < q < 1
    if max(meta["action_forecast/w_call"], meta["action_forecast/w_msg"]) < ACTION_FORECAST_BALANCE_WMAX:
        assert abs(meta["action_forecast/balanced_call_share"] - p) < 1e-9
    tw = batch["token_weight"]
    assert tw.shape == batch["input_ids"].shape
    # only decision tokens deviate from 1
    off = (tw != 1.0) & (batch["loss_mask"] == 1)
    assert off.sum() > 0 and off.sum() <= 3 * batch["input_ids"].size(0)
    # without the flag nothing changes
    batch2, meta2 = build_action_forecast_batch(
        msgs, tok, rewards=[1, 0, 1, 0], k=3, gate="mixed", skip_invalid=True, env="tau2",
        group_ids=["a", "a", "b", "b"], group_norm=True, layout="turns")
    assert "token_weight" not in batch2 and "action_forecast/w_call" not in meta2


def test_token_weight_scales_the_numerator_only():
    import torch
    from verl.agent_trainer.ppo.sft_common import compute_sft_loss_from_logits
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0]])
    tw = torch.ones(2, 6)
    tw[0, 2] = 3.0
    tw[1, 1] = 0.0
    base = compute_sft_loss_from_logits(logits, labels, mask)
    same = compute_sft_loss_from_logits(logits, labels, mask, token_weight=torch.ones(2, 6))
    assert torch.allclose(base, same)
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), reduction="none").view(2, 5)
    m = mask[:, 1:].bool()
    want = (ce * tw[:, 1:] * m).sum() / m.sum()
    got = compute_sft_loss_from_logits(logits, labels, mask, token_weight=tw)
    assert torch.allclose(got, want)


# ---- targets='calls': forecast the policy's tool calls only, never its messages ------------

def test_calls_targets_are_the_next_tool_calls_from_each_call_turn():
    tg = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2", targets="calls")
    by = {t["prefix_end"] + 1: t for t in tg}
    # anchors are the tool-call turns only (2, 6, 8); the messages at 4 and 10 are skipped
    assert sorted(by) == [2, 6, 8]
    assert by[2]["actions"] == [CALL_1, CALL_CANCEL, CALL_DETAILS]
    assert by[6]["actions"] == [CALL_CANCEL, CALL_DETAILS]      # the failed cancel is not redone: kept
    assert by[8]["actions"] == [CALL_DETAILS]
    # action_turns index the action turns the targets come from, as for 'all'
    assert by[2]["action_turns"] == [0, 2, 3]


def test_calls_targets_keep_the_redo_rule():
    tg = build_action_targets(RETRY, k=3, skip_invalid=True, env="tau2", targets="calls")
    by = {t["prefix_end"] + 1: t["actions"] for t in tg}
    assert sorted(by) == [2, 6, 12]
    # the failed lookup redone at turn 6 is dropped, messages never appear
    assert by[2] == [CALL_EMAIL_OK, CALL_CANCEL]
    assert by[12] == [CALL_CANCEL]
    raw = {t["prefix_end"] + 1: t["actions"] for t in
           build_action_targets(RETRY, k=3, skip_invalid=False, env="tau2", targets="calls")}
    assert raw[2] == [CALL_EMAIL_BAD, CALL_EMAIL_OK, CALL_CANCEL]
    # 'all' is unchanged by the new option
    assert build_action_targets(RETRY, k=3, skip_invalid=True, env="tau2") == \
        build_action_targets(RETRY, k=3, skip_invalid=True, env="tau2", targets="all")


def test_calls_targets_need_a_native_env_and_known_values():
    with pytest.raises(ValueError):
        build_action_targets(REACT, k=2, env="alfworld", targets="calls")
    with pytest.raises(ValueError):
        build_action_targets(NATIVE, k=2, env="tau2", targets="messages")


def test_calls_samples_train_only_call_spans(tok):
    from verl.agent_trainer.ppo.action_forecast import (
        ACTION_FORECAST_NEXT_CALL_PROMPT, DEFAULT_ACTION_FORECAST_CALLS_PROMPT)
    samples = build_action_forecast_samples(NATIVE, tok, k=3, skip_invalid=True, env="tau2",
                                            layout="turns", targets="calls")
    expected = [t["actions"] for t in build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2", targets="calls")]
    assert len(samples) == len(expected) == 3
    call_id = tok.convert_tokens_to_ids("<tool_call>")
    for s, acts in zip(samples, expected):
        spans = _masked_spans(s)
        assert [tok.decode(sp) for sp in spans] == [a + "<|im_end|>\n" for a in acts]
        text = tok.decode(s["input_ids"].tolist())
        assert DEFAULT_ACTION_FORECAST_CALLS_PROMPT.format(k=len(acts)) in text
        assert text.count(ACTION_FORECAST_NEXT_CALL_PROMPT) == len(acts) - 1
        dk = s["decision_kind"]
        marks = [i for i in range(dk.numel()) if dk[i] > 0]
        assert len(marks) == len(acts)
        assert all(dk[i].item() == 1 and s["input_ids"][i].item() == call_id for i in marks)


def test_calls_batch_leaves_the_decision_token_untrained(tok):
    from verl.agent_trainer.ppo.action_forecast import build_action_forecast_batch
    msgs = [NATIVE, NATIVE, RETRY, RETRY]
    kw = dict(rewards=[1, 0, 1, 0], k=3, gate="mixed", skip_invalid=True, env="tau2",
              group_ids=["a", "a", "b", "b"], group_norm=True, layout="turns", balance_calls=True)
    batch, meta = build_action_forecast_batch(msgs, tok, targets="calls", **kw)
    assert meta["action_forecast/targets_calls"] == 1.0
    assert meta["action_forecast/target_call_share"] == 1.0
    assert meta["action_forecast/w_call"] == 0.0 and meta["action_forecast/w_msg"] == 0.0
    tw, lm, ids = batch["token_weight"], batch["loss_mask"], batch["input_ids"]
    call_id = tok.convert_tokens_to_ids("<tool_call>")
    opens = (ids == call_id) & (lm == 1)
    assert opens.sum() > 0 and (tw[opens] == 0).all()           # the call decision is never trained
    assert ((tw != 1.0) & (lm == 1)).sum() == opens.sum()        # ... and nothing else is reweighted
    _, meta_all = build_action_forecast_batch(msgs, tok, **kw)
    assert meta_all["action_forecast/targets_calls"] == 0.0


# ---- length_norm: every target action weighs the same in its sample's loss ----------------

SPANS = [(1, 3), (5, 10), (11, 14)]            # trained spans of 2, 5 and 3 tokens
LM = [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]


def test_action_length_weights_give_each_action_the_same_weight():
    import torch
    from verl.agent_trainer.ppo.action_forecast import action_length_weights
    lm = torch.tensor(LM)
    tw = action_length_weights(lm)
    n_tok = int(lm.sum())
    for a, b in SPANS:
        assert float(tw[a:b].sum()) == pytest.approx(n_tok / len(SPANS))
    assert float(tw[lm.bool()].sum()) == pytest.approx(n_tok)      # the token-mean scale is kept
    assert (tw[~lm.bool()] == 1).all()


def test_length_normalised_loss_is_the_mean_over_actions():
    import torch
    from verl.agent_trainer.ppo.action_forecast import action_length_weights
    from verl.agent_trainer.ppo.sft_common import compute_sft_loss_from_logits
    torch.manual_seed(0)
    T, V = len(LM), 13
    logits, labels = torch.randn(1, T, V), torch.randint(0, V, (1, T))
    lm = torch.tensor([LM])
    got = compute_sft_loss_from_logits(logits, labels, lm, token_weight=action_length_weights(lm[0]).unsqueeze(0))
    ce = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:], reduction="none")   # ce[t-1] scores token t
    want = torch.stack([ce[a - 1:b - 1].mean() for a, b in SPANS]).mean()
    assert torch.allclose(got, want)


def _turn_spans(batch, i, tok):
    from verl.agent_trainer.ppo.action_forecast import _trained_spans
    call_id = tok.convert_tokens_to_ids("<tool_call>")
    return [(a, b, batch["input_ids"][i, a].item() == call_id) for a, b in _trained_spans(batch["loss_mask"][i])]


def test_length_norm_batch_weighs_every_turn_the_same(tok):
    from verl.agent_trainer.ppo.action_forecast import build_action_forecast_batch
    msgs = [NATIVE, NATIVE, RETRY, RETRY]
    kw = dict(rewards=[1, 0, 1, 0], k=3, gate="mixed", skip_invalid=True, env="tau2",
              group_ids=["a", "a", "b", "b"], group_norm=True, layout="turns")
    b0, m0 = build_action_forecast_batch(msgs, tok, **kw)
    b1, m1 = build_action_forecast_batch(msgs, tok, length_norm=True, **kw)
    assert m1["action_forecast/length_norm"] == 1.0 and "action_forecast/length_norm" not in m0
    assert "token_weight" not in b0
    want_num = want_den = 0.0
    for i in range(b1["input_ids"].size(0)):
        spans, n_tok = _turn_spans(b1, i, tok), int(b1["loss_mask"][i].sum())
        for a, b, _ in spans:
            assert float(b1["token_weight"][i, a:b].sum()) == pytest.approx(n_tok / len(spans), rel=1e-5)
        lw = float(b1["loss_weight"][i])
        want_num += lw * sum(c for _, _, c in spans) / len(spans); want_den += lw
    # with every turn weighing the same, the calls' share of the loss is their share of the turns
    assert m1["action_forecast/call_weight_share"] == pytest.approx(want_num / want_den, rel=1e-4)
    assert m1["action_forecast/call_token_share"] == pytest.approx(m0["action_forecast/call_token_share"])


def test_length_norm_keeps_the_call_balance_neutral(tok):
    from verl.agent_trainer.ppo.action_forecast import ACTION_FORECAST_BALANCE_WMAX, build_action_forecast_batch
    msgs = [NATIVE, NATIVE, RETRY, RETRY]
    batch, meta = build_action_forecast_batch(
        msgs, tok, rewards=[1, 0, 1, 0], k=3, gate="mixed", skip_invalid=True, env="tau2",
        group_ids=["a", "a", "b", "b"], group_norm=True, layout="turns", balance_calls=True, length_norm=True)
    p = meta["action_forecast/policy_call_share"]
    num = den = 0.0
    for i in range(batch["input_ids"].size(0)):
        lw, n_tok = float(batch["loss_weight"][i]), float(batch["loss_mask"][i, 1:].sum())
        for a, _, is_call in _turn_spans(batch, i, tok):
            w = lw * float(batch["token_weight"][i, a]) / n_tok       # the decision token's weight in the loss
            den += w; num += w * is_call
    assert max(meta["action_forecast/w_call"], meta["action_forecast/w_msg"]) < ACTION_FORECAST_BALANCE_WMAX
    assert num / den == pytest.approx(p, rel=1e-6)


# ---- skip_no_call: a forecast sample whose targets hold no tool call is skipped -------------

TALK = [
    {"role": "system", "content": "You are a customer service agent."},
    {"role": "user", "content": "My phone has no signal."},
    {"role": "assistant", "content": "Please turn airplane mode off and tell me what the status bar shows."},
    {"role": "user", "content": "Airplane mode is off now and I have signal."},
    {"role": "assistant", "content": "Great, you are all set."},
    {"role": "user", "content": "Thanks! ###STOP###"},
]


def test_skip_no_call_drops_message_only_targets(tok):
    from verl.agent_trainer.ppo.action_forecast import _native_tool_name
    kw = dict(k=3, skip_invalid=True, env="tau2", layout="turns")
    full = build_action_forecast_samples(NATIVE, tok, **kw)
    stats = {}
    kept = build_action_forecast_samples(NATIVE, tok, skip_no_call=True, stats=stats, **kw)
    targets = build_action_targets(NATIVE, k=3, skip_invalid=True, env="tau2")
    n_without_call = sum(not any(_native_tool_name(a) is not None for a in t["actions"]) for t in targets)
    assert n_without_call >= 1                                   # the closing message of NATIVE
    assert len(kept) == len(full) - n_without_call and stats["skipped_no_call"] == n_without_call
    call_id = tok.convert_tokens_to_ids("<tool_call>")
    for s in kept:                                              # every kept sample trains a call
        assert any(s["input_ids"][i].item() == call_id for i in range(s["decision_kind"].numel()) if s["decision_kind"][i] == 1)


def test_skip_no_call_drops_a_talk_only_trajectory(tok):
    kw = dict(k=3, skip_invalid=True, env="tau2", layout="turns")
    assert build_action_forecast_samples(TALK, tok, **kw)
    assert build_action_forecast_samples(TALK, tok, skip_no_call=True, **kw) == []


def test_skip_no_call_needs_a_native_env(tok):
    with pytest.raises(ValueError):
        build_action_forecast_samples(REACT, tok, k=2, env="alfworld", skip_no_call=True)


def test_skip_no_call_batch_counts_and_keeps_the_balance_neutral(tok):
    from verl.agent_trainer.ppo.action_forecast import ACTION_FORECAST_BALANCE_WMAX, build_action_forecast_batch
    msgs = [NATIVE, TALK, RETRY, TALK]
    kw = dict(rewards=[1, 1, 1, 1], k=3, gate="wins", skip_invalid=True, env="tau2",
              group_ids=["a", "b", "c", "d"], group_norm=True, layout="turns", balance_calls=True, length_norm=True)
    b0, m0 = build_action_forecast_batch(msgs, tok, **kw)
    b1, m1 = build_action_forecast_batch(msgs, tok, skip_no_call=True, **kw)
    assert m0["action_forecast/skip_no_call"] == 0.0 and m0["action_forecast/n_skipped_no_call"] == 0.0
    assert m1["action_forecast/skip_no_call"] == 1.0 and m1["action_forecast/n_skipped_no_call"] > 0
    assert m1["action_forecast/n_samples"] == m0["action_forecast/n_samples"] - m1["action_forecast/n_skipped_no_call"]
    assert m1["action_forecast/n_traj_used"] == 2                 # the talk-only trajectories give nothing
    p = m1["action_forecast/policy_call_share"]
    num = den = 0.0
    for i in range(b1["input_ids"].size(0)):
        lw, n_tok = float(b1["loss_weight"][i]), float(b1["loss_mask"][i, 1:].sum())
        for a, _, is_call in _turn_spans(b1, i, tok):
            w = lw * float(b1["token_weight"][i, a]) / n_tok
            den += w; num += w * is_call
    assert max(m1["action_forecast/w_call"], m1["action_forecast/w_msg"]) < ACTION_FORECAST_BALANCE_WMAX
    assert num / den == pytest.approx(p, rel=1e-6)


# ---- a call to a tool the agent does not have is never a target, nor counts as a call -------

CALL_STATUS = "<tool_call>\n{\"name\": \"check_status_bar\", \"arguments\": {}}\n</tool_call>"
CALL_LOOKUP = "<tool_call>\n{\"name\": \"get_customer_by_phone\", \"arguments\": {\"phone_number\": \"555-123-2002\"}}\n</tool_call>"
DEVICE = [
    {"role": "system", "content": "You are a telecom support agent."},
    {"role": "user", "content": "My phone shows no service."},
    {"role": "assistant", "content": CALL_STATUS},                                  # 2: the customer's tool
    {"role": "tool", "content": "Error: Tool 'check_status_bar' not found."},
    {"role": "assistant", "content": "Could you tell me what your status bar shows?"},
    {"role": "user", "content": "No signal and airplane mode is on. My number is 555-123-2002."},
    {"role": "assistant", "content": CALL_LOOKUP},                                  # 6
    {"role": "tool", "content": "{\"customer_id\": \"C1001\"}"},
    {"role": "assistant", "content": "Please turn airplane mode off."},
    {"role": "user", "content": "Done, it works now. ###STOP###"},
]
DEVICE_ONLY = DEVICE[:5] + [{"role": "user", "content": "It works now. ###STOP###"}]


def test_calls_to_missing_tools_are_never_targets():
    tg = build_action_targets(DEVICE, k=3, skip_invalid=True, env="tau2")
    assert all(CALL_STATUS not in t["actions"] for t in tg)
    by = {t["prefix_end"] + 1: t["actions"] for t in tg}
    assert by[2] == ["Could you tell me what your status bar shows?", CALL_LOOKUP, "Please turn airplane mode off."]
    # the rule belongs to skip_invalid: without it the call stays
    assert build_action_targets(DEVICE, k=3, skip_invalid=False, env="tau2")[0]["actions"][0] == CALL_STATUS


def test_missing_tool_calls_do_not_count_as_calls(tok):
    kw = dict(k=3, skip_invalid=True, env="tau2", layout="turns", skip_no_call=True)
    assert build_action_forecast_samples(DEVICE_ONLY, tok, **kw) == []      # its only call was not the agent's
    assert build_action_forecast_samples(DEVICE, tok, **kw)                 # the real lookup keeps samples
