#!/usr/bin/env python
"""Smoke test for Tau2EnvClient (Stage 2).

The server-side smoke test (smoke_tau2_env.py) posts bare JSON tool calls. This one
goes through the actual training-side client, so it covers what that test does not:
`conversation_start` fetched from /system_prompt, ReAct Thought/Action parsing, the
code-fence / multi-line repairs in `_clean_action`, tool-call vs message routing, and
the invalid-action path returning done=False instead of raising.

Run with the *training* interpreter, with the cope AgentGym checkout on PYTHONPATH:

    PYTHONNOUSERSITE=1 PYTHONPATH=<repo>/AgentGym/agentenv \\
      <agentgym-rl>/bin/python scripts/smoke_tau2_client.py --addr http://127.0.0.1:36202
"""

import argparse
import sys

from agentenv.envs.tau2 import Tau2EnvClient, _clean_action


def check_clean_action() -> list[str]:
    """`_clean_action` guards the two ways a valid tool call silently degrades into a
    chat message inside tau2's `^\\w+\\s*\\(.*\\)$` check."""
    cases = [
        ("get_user_details(user_id='x')", "get_user_details(user_id='x')"),
        ("```python\nget_order_details(order_id='#W1')\n```", "get_order_details(order_id='#W1')"),
        ("```\ndone()\n```", "done()"),
        ("find_user_id_by_name_zip(\n  first_name='Yusuf',\n  zip='19122'\n)",
         "find_user_id_by_name_zip( first_name='Yusuf', zip='19122' )"),
        ("Sure, I can help with that.", "Sure, I can help with that."),
    ]
    fails = []
    for raw, want in cases:
        got = _clean_action(raw)
        status = "ok " if got == want else "FAIL"
        print(f"  [{status}] {raw!r:<70} -> {got!r}")
        if got != want:
            fails.append(f"_clean_action({raw!r}) == {got!r}, want {want!r}")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="http://127.0.0.1:36202")
    parser.add_argument("--item-id", type=int, default=0)
    args = parser.parse_args()

    fails = []

    print("[client] _clean_action:")
    fails += check_clean_action()

    print("\n[client] constructing Tau2EnvClient ...")
    client = Tau2EnvClient(env_server_base=args.addr, data_len=1, timeout=600)
    prompt = client.conversation_start[0]["value"]
    print(f"  system prompt: {len(prompt)} chars, ack={client.conversation_start[1]['value']!r}")
    if "# Policy" not in prompt or "# Tools" not in prompt:
        fails.append("conversation_start was not populated from /system_prompt")

    client.reset(args.item_id)
    print(f"  reset obs: {client.observe()[:100]!r}")

    # 1) ReAct-wrapped tool call -> must come back as a tool result.
    out = client.step(
        "Thought:\nI need to identify the customer first.\n\n"
        "Action:\nfind_user_id_by_name_zip(first_name='Yusuf', last_name='Rossi', zip='19122')"
    )
    print(f"\n  [tool call ] done={out.done} reward={out.reward} state={out.state[:90]!r}")
    if not out.state.startswith("tool:"):
        fails.append(f"ReAct tool call was not routed as a tool call: {out.state[:120]!r}")

    # 2) ReAct-wrapped plain message -> must reach the user simulator.
    out = client.step("Thought:\nConfirm with the customer.\n\nAction:\nI found your account. How can I help?")
    print(f"  [message   ] done={out.done} reward={out.reward} state={out.state[:90]!r}")
    if out.state.startswith("tool:"):
        fails.append("plain message was misrouted as a tool call")

    # 3) Garbage -> corrective observation, episode must stay alive.
    out = client.step("")
    print(f"  [invalid   ] done={out.done} reward={out.reward} state={out.state[:60]!r}")
    if out.done:
        fails.append("invalid action ended the episode; it must return done=False")

    client.close()

    print()
    if fails:
        for f in fails:
            print(f"[client] FAIL: {f}")
        return 1
    print("[client] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
