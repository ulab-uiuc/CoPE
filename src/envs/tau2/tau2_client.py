"""tau2-bench env client for AgentGym. Talks to the agentenv-tau2 FastAPI server.

ReAct wrapper over tau2's action string protocol: each turn the model emits Thought +
Action, where the Action is either a functional tool call `get_order_details(order_id='#W1')`
or a plain message to the customer. The client strips the Thought and forwards the bare
action; the server hands it to tau2's `AgentGymEnv.step`, whose `parse_action_string`
decides tool-call vs message.

The system prompt (domain policy + tool signatures) is fetched from the server at
construction time rather than hardcoded here, so it always matches the domain the
server was actually launched with.
"""

import re
from typing import Any, Optional

import requests

from agentenv.controller import BaseAdapter, BaseEnvClient, BaseTask
from agentenv.controller.types import (
    ActionFormat,
    ConversationMessage,
    StepOutput,
)

# Mirrors tau2.utils.tools.is_functional_tool_call, which is what the server ultimately
# uses to decide tool-call vs message.
_TOOL_CALL_RE = re.compile(r"^\w+\s*\(.*\)$", re.DOTALL)
_TOOL_CALL_HEAD_RE = re.compile(r"^\w+\s*\(")

_FALLBACK_SYSTEM = (
    "You are a customer service agent. Reply every turn with a Thought and an Action."
)


def _clean_action(action: str) -> str:
    """Normalise a model-emitted action into something tau2 will parse as intended.

    Two fixes matter here:

    1. Code fences. The prompt forbids them, but models emit them anyway, and a fenced
       tool call fails tau2's `^\\w+\\s*\\(.*\\)$` check and silently becomes a *message
       to the customer* instead of a tool call.
    2. Multi-line tool calls. tau2 matches that pattern without re.DOTALL, so a call
       wrapped across lines also degrades into a message. If the action opens like a
       call and closes with ')', collapse its whitespace onto one line.
    """
    action = action.strip()
    if action.startswith("```"):
        action = re.sub(r"^```[a-zA-Z]*\n?", "", action)
        action = re.sub(r"\n?```$", "", action).strip()
    if (
        "\n" in action
        and _TOOL_CALL_HEAD_RE.match(action)
        and action.endswith(")")
    ):
        action = re.sub(r"\s+", " ", action)
    return action


class Tau2Adapter(BaseAdapter):
    # Placeholder only. Tau2EnvClient replaces `conversation_start` per instance with
    # the server's /system_prompt, which carries the real domain policy and tool list.
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage({"from": "human", "loss": None, "value": _FALLBACK_SYSTEM}),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }


class Tau2EnvClient(BaseEnvClient):
    adapter_cls = Tau2Adapter

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 2400, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {ok}")
        self.env_id = ok.json()

        # One prompt per configured domain. Mixed-domain training puts retail and
        # airline tasks in the same batch, and their policies and tool lists are
        # different documents -- a single shared prompt would advertise the wrong tools
        # for most of the batch. `conversation_start` stays bound to the first domain so
        # single-domain callers are unaffected.
        self.domains = requests.get(
            f"{self.env_server_base}/domains", timeout=self.timeout
        ).json()
        self.conversation_start_by_domain = {
            d: (
                ConversationMessage({
                    "from": "human", "loss": None,
                    "value": requests.get(
                        f"{self.env_server_base}/system_prompt",
                        params={"domain": d}, timeout=self.timeout).json(),
                }),
                ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
            )
            for d in self.domains
        }
        self.conversation_start = self.conversation_start_by_domain[self.domains[0]]

    def conversation_start_for(self, domain: Optional[str] = None):
        """Prompt pair for `domain`, falling back to the first configured domain."""
        if domain and domain in self.conversation_start_by_domain:
            return self.conversation_start_by_domain[domain]
        return self.conversation_start

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict) -> Any:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}", json=data, timeout=self.timeout
        )
        assert res.status_code == 200, (res.status_code, res.text[:300])
        return res.json()

    def _get(self, path: str) -> Any:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}", timeout=self.timeout
        )
        assert res.status_code == 200, (res.status_code, res.text[:300])
        return res.json()

    def observe(self) -> str:
        return self._get("observation")

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        try:
            parsed = Tau2Adapter.action_parser(action, self.action_format)
            parsed = _clean_action(parsed)
            if not parsed:
                raise ValueError("empty action")
        except Exception as e:
            # Never raise: the rollout loop's except-handler marks the trajectory done
            # with score 0, which silently burns a rollout and skews the GRPO group
            # baseline. Hand back a corrective observation and let the episode run on.
            print(f"tau2 action parse error: {e}")
            return StepOutput(
                state="Invalid Action. Reply with a Thought and an Action.\n\n"
                + self.observe(),
                reward=0.0,
                done=False,
            )
        response = self._post("step", {"action": parsed})
        return StepOutput(
            state=response["state"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int) -> str:
        return self._post("reset", {"session_id": idx})

    def close(self):
        try:
            self._post("close", {})
        except Exception:
            pass


class Tau2Task(BaseTask):
    env_client_cls = Tau2EnvClient
    env_name = "Tau2"

    def __init__(self, client_args: dict, n_clients: int = 1, *args, **kwargs) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
