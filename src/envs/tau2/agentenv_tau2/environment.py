"""
Tau2EnvServer -- wraps tau2-bench's gym `AgentGymEnv` into the AgentGym env-server
contract (create / reset / step / observation), mirroring AppWorldEnvServer.

Each env slot holds one live `AgentGymEnv`, which itself drives a tau2 `Orchestrator`
(agent <-> user simulator <-> domain tools) on a background daemon thread. The agent's
action is a string: either a functional tool call `get_user_details(user_id='x')`, a
ToolCall JSON object, or plain text (which is delivered to the user simulator).

Reward is sparse and terminal: 0.0 until the simulation ends, then tau2's evaluator
score for the run.

Config via env vars:
  TAU2_DOMAIN         domain name, e.g. retail / airline / telecom (default: retail)
  TAU2_TASK_SPLIT     split used for integer session_id -> task_id mapping (default: train)
  TAU2_MAX_STEPS      max orchestrator steps per episode (default: 60)
  TAU2_SOLO_MODE      1 => no user simulator, agent works a ticket alone (default: 0)
  TAU2_USER_LLM       litellm model string for the user simulator
                      (default: openai/user-sim, i.e. a local OpenAI-compatible server)
  TAU2_USER_API_BASE  api_base forwarded to litellm, e.g. http://127.0.0.1:38001/v1
  TAU2_USER_API_KEY   api_key forwarded to litellm (default: "EMPTY", what vLLM expects)
  TAU2_USER_TEMPERATURE  user simulator sampling temperature (default: 0.0)
  TAU2_REWARD_BASIS   tau2 EvaluationType for the reward (default: env)
                      `env` = DB/env-assertion only: deterministic and free.
                      `all` = the official basis, but ~112/114 retail tasks include
                      NL_ASSERTION, so every episode costs one judge-LLM call.
  TAU2_PROMPT_VARIANT base | strict (default: strict) -- see _ENDINGS below.
  TAU2_REWARD_SHAPE   binary | dense (default: binary). `dense` adds graded partial
                      credit from per-action checks so GRPO groups are not all-zero.
  TAU2_DENSE_WEIGHT   ceiling for partial credit (default: 0.5), < 1 so a partial
                      never outscores a real solve.
  TAU2_FORCE_DONE_AFTER  agent steps after which the server ends the episode itself
                      (default: 0 = off). tau2 only scores a run once its orchestrator
                      terminates, so without this every episode that merely runs out of
                      the caller's turn budget is returned as an unevaluated 0. Set it
                      to the caller's MAX_ROUNDS.
"""

import os
import threading
from typing import Optional, Tuple

from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.gym.gym_agent import AgentGymEnv, GymAgent
from tau2.registry import registry


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


class _Tau2GymEnv(AgentGymEnv):
    """`AgentGymEnv` with a configurable reward basis.

    Upstream `AgentGymEnv._get_reward` hardcodes `EvaluationType.ALL`, which pulls in
    the NL-assertion judge LLM for any task whose `reward_basis` contains NL_ASSERTION
    (112 of retail's 114 tasks). For RL that means one extra judge call per rollout,
    plus judge noise in the GRPO group baseline -- so the basis is made configurable
    and defaults to ENV (pure DB / env-assertion state check).
    """

    def __init__(self, *args, reward_eval_type: EvaluationType,
                 reward_shape: str = "binary", dense_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_eval_type = reward_eval_type
        self.reward_shape = reward_shape
        self.dense_weight = dense_weight

    def _get_reward(self) -> tuple[float, str]:
        if self._simulation_run is None:
            return 0.0, "{}"
        task = self._get_task()
        result = evaluate_simulation(
            simulation=self._simulation_run,
            task=task,
            evaluation_type=self.reward_eval_type,
            solo_mode=self.solo_mode,
            domain=self.domain,
        )
        reward = result.reward
        if self.reward_shape == "dense" and reward < 1.0:
            # Binary DB-state reward leaves ~88% of retail tasks at exactly 0 for a 7B
            # policy, so a GRPO group of 8 is almost always all-zero -> zero advantage
            # -> no gradient. tau2's ACTION evaluator checks each ground-truth action
            # independently and costs no LLM call, so partial progress can be graded.
            #
            # Two tiers, because exact matching alone is still too sparse here: a weak
            # policy calls the right tool with the wrong arguments far more often than
            # it lands an exact ground-truth call, and scoring those identically to
            # doing nothing is what collapses the group. Exact match dominates so the
            # ordering (exact > right tool > nothing) is preserved, and the whole thing
            # is scaled by dense_weight (< 1) so a partial never beats a real solve.
            try:
                act = evaluate_simulation(
                    simulation=self._simulation_run,
                    task=task,
                    evaluation_type=EvaluationType.ACTION,
                    solo_mode=self.solo_mode,
                    domain=self.domain,
                )
                checks = act.action_checks or []
                if checks:
                    exact = sum(1 for c in checks if c.action_match) / len(checks)
                    called = {
                        tc.name
                        for m in (self._simulation_run.messages or [])
                        for tc in (getattr(m, "tool_calls", None) or [])
                    }
                    by_name = sum(1 for c in checks if c.action.name in called) / len(checks)
                    reward = self.dense_weight * (0.7 * exact + 0.3 * by_name)
            except Exception as e:
                print(f"[tau2] dense reward failed, falling back to binary: {e}")
        return reward, result.model_dump_json()


class Tau2EnvServer:
    def __init__(self) -> None:
        self._max_id = 0
        self._lock = threading.Lock()
        self.env: dict[int, Optional[_Tau2GymEnv]] = {}
        self.last_obs: dict[int, str] = {}
        self.steps: dict[int, int] = {}

        # Comma-separated for mixed-domain training. Single-domain training on retail
        # measurably degrades the policy on airline (28.1% -> 11.9% on the held-out
        # split), so the ability to train on several domains at once is what decides
        # whether the recipe is usable rather than just locally effective.
        # "+" and "," both separate domains. Prefer "+" from a shell: a comma inside
        # `sbatch --export` has to be escaped and the backslash survives into the value,
        # which shows up as KeyError: Task Set 'retail\\' not found.
        _dom_raw = os.environ.get("TAU2_DOMAIN", "retail").replace("+", ",")
        self.domains = [d.strip().strip("\\") for d in _dom_raw.split(",") if d.strip().strip("\\")]
        self.domain = self.domains[0]
        self.split = os.environ.get("TAU2_TASK_SPLIT", "train")
        self.max_steps = int(os.environ.get("TAU2_MAX_STEPS", "60"))
        self.solo_mode = _env_flag("TAU2_SOLO_MODE")
        self.user_llm = os.environ.get("TAU2_USER_LLM", "openai/user-sim")
        self.reward_eval_type = EvaluationType(
            os.environ.get("TAU2_REWARD_BASIS", "env").strip().lower()
        )
        self.prompt_variant = os.environ.get("TAU2_PROMPT_VARIANT", "strict").strip().lower()
        if self.prompt_variant not in _ENDINGS:
            raise ValueError(
                f"TAU2_PROMPT_VARIANT must be one of {sorted(_ENDINGS)}, "
                f"got {self.prompt_variant!r}"
            )
        # binary: tau2's own pass/fail. dense: partial credit from per-action checks,
        # which is what keeps GRPO groups from being uniformly zero.
        self.reward_shape = os.environ.get("TAU2_REWARD_SHAPE", "binary").strip().lower()
        if self.reward_shape not in ("binary", "dense"):
            raise ValueError(f"TAU2_REWARD_SHAPE must be binary|dense, got {self.reward_shape!r}")
        self.dense_weight = float(os.environ.get("TAU2_DENSE_WEIGHT", "0.5"))
        # Agent steps after which the server ends the episode itself so it gets scored.
        # Set this to the caller's per-episode turn budget (verl's MAX_ROUNDS).
        self.force_done_after = int(os.environ.get("TAU2_FORCE_DONE_AFTER", "0"))

        self.user_llm_args = {
            "temperature": float(os.environ.get("TAU2_USER_TEMPERATURE", "0.0"))
        }
        # Forwarded verbatim into litellm.completion(**kwargs) by tau2's
        # utils/llm_utils.py::generate. Two deployments are supported:
        #
        #   local vLLM      TAU2_USER_LLM=openai/user-sim
        #                   TAU2_USER_API_BASE=http://host:port/v1
        #   hosted OpenAI   TAU2_USER_LLM=openai/gpt-4o-mini   (no api_base)
        #
        # api_key is set independently of api_base: the hosted case needs a real key
        # and no api_base, and the two used to be coupled so the key was silently
        # dropped whenever api_base was absent. Prefer TAU2_USER_API_KEY_FILE over
        # TAU2_USER_API_KEY under slurm -- exported variables are visible to anyone
        # who can run `scontrol show job`.
        api_base = os.environ.get("TAU2_USER_API_BASE")
        if api_base:
            self.user_llm_args["api_base"] = api_base
        key_file = os.environ.get("TAU2_USER_API_KEY_FILE")
        api_key = None
        if key_file:
            path = os.path.expanduser(key_file)
            try:
                with open(path) as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                # Only the hosted path actually needs a credential; with a local
                # endpoint configured, fall through to the EMPTY key below rather
                # than taking the whole env-server cluster down at startup.
                if not api_base:
                    raise
                print(f"[tau2] TAU2_USER_API_KEY_FILE={path} not found; using a "
                      f"local endpoint, so continuing without a key")
        if api_key is None and not key_file:
            api_key = os.environ.get("TAU2_USER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.user_llm_args["api_key"] = api_key
        elif api_base:
            # vLLM accepts any non-empty key; only the hosted path really needs one.
            self.user_llm_args["api_key"] = "EMPTY"

        # Ordered task-id list: verl hands us an integer item id, tau2 task ids are
        # opaque strings, so the index -> id mapping has to live here.
        # Entries are (domain, task_id). Domains are concatenated in the order given,
        # which must match the order the item-id dataset was generated in.
        self._task_ids: list[tuple[str, str]] = []
        for dom in self.domains:
            for t in registry.get_tasks_loader(dom)(self.split):
                self._task_ids.append((dom, t.id))
        if not self._task_ids:
            raise RuntimeError(
                f"tau2: no tasks for domains {self.domains} split '{self.split}'"
            )

        # Domain policy + tool schemas are static per domain, so build them once and
        # serve them over /system_prompt. The client fetches this at construction time
        # to populate `conversation_start` (which is what becomes the RL prompt).
        self._system_prompts = {d: self._build_system_prompt(d) for d in self.domains}

        print(
            f"[tau2] domains={','.join(self.domains)} split={self.split} "
            f"tasks={len(self._task_ids)} max_steps={self.max_steps} "
            f"solo={self.solo_mode} user_llm={self.user_llm} "
            f"reward_basis={self.reward_eval_type.value} prompt={self.prompt_variant} "
            f"reward_shape={self.reward_shape} force_done={self.force_done_after}"
        )

    # ---- prompt -------------------------------------------------------------
    def _build_system_prompt(self, domain: str) -> str:
        environment = registry.get_env_constructor(domain)(
            solo_mode=self.solo_mode
        )
        policy = environment.get_policy()
        # Go through GymAgent rather than environment.get_tools() directly: its
        # __init__ appends the `done` stop tool, so this is the exact tool set the
        # running agent will accept. Building the prompt from the environment alone
        # would advertise a tool list the runtime doesn't match.
        tools = sorted(
            GymAgent(tools=environment.get_tools(), domain_policy=policy).tools,
            key=lambda t: t.name,
        )

        lines = []
        for tool in tools:
            schema = tool.openai_schema
            fn = schema.get("function", schema)
            params = fn.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = set(params.get("required", []) or [])
            sig = ", ".join(
                f"{n}: {p.get('type', 'any')}" + ("" if n in required else " = None")
                for n, p in props.items()
            )
            lines.append(f"- {tool.name}({sig})\n    {fn.get('description', '').strip()}")
        tools_desc = "\n".join(lines)

        return TAU2_SYSTEM_TEMPLATE.format(
            policy=policy.strip(),
            tools=tools_desc,
            ending=_ENDINGS[self.prompt_variant],
        )

    def get_system_prompt(self, domain: Optional[str] = None) -> str:
        """Prompt for `domain`, or the first configured domain. Under mixed-domain
        training each task needs its own policy and tool list, so the caller has to ask
        per domain -- a single shared prompt would describe the wrong tools for most of
        the batch."""
        return self._system_prompts[domain or self.domain]

    def get_domains(self) -> list[str]:
        return list(self.domains)

    # ---- lifecycle ----------------------------------------------------------
    def create(self) -> int:
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1
            self.env[env_idx] = None  # not yet bound to a task
        return env_idx

    def _task_id_for(self, session_id: Optional[int]) -> tuple[str, str]:
        sid = 0 if session_id is None else int(session_id)
        return self._task_ids[sid % len(self._task_ids)]

    def _close_slot(self, env_idx: int) -> None:
        env = self.env.get(env_idx)
        if env is not None:
            # Drain the orchestrator thread before dropping the env. verl closes every
            # env client at the end of each rollout, which for a trajectory that ran
            # out of max_rounds happens while tau2's simulation is still mid-episode --
            # its daemon thread is parked on a threading.Event and would never exit,
            # leaking one thread (and one live domain DB) per abandoned episode.
            # Stepping `done()` drives the orchestrator to its normal termination.
            try:
                if (
                    getattr(env, "_orchestrator", None) is not None
                    and not env._simulation_done.is_set()
                ):
                    env.step("done()")
            except Exception as e:
                print(f"[tau2] draining env {env_idx} failed: {e}")
            try:
                env.close()
            except Exception:
                pass
        self.env.pop(env_idx, None)
        self.last_obs.pop(env_idx, None)
        self.steps.pop(env_idx, None)

    def reset(self, env_idx: int, session_id: Optional[int]) -> str:
        """(Re)bind this slot to a task; return the opening observation."""
        domain, task_id = self._task_id_for(session_id)
        env = self.env.get(env_idx)
        # A slot can only be reused for the same domain: the orchestrator, tool set and
        # DB are all built per domain at construction time.
        if env is not None and env.domain != domain:
            self._close_slot(env_idx)
            env = None
        if env is None:
            env = _Tau2GymEnv(
                domain=domain,
                task_id=task_id,
                max_steps=self.max_steps,
                solo_mode=self.solo_mode,
                user_llm=self.user_llm,
                user_llm_args=dict(self.user_llm_args),
                reward_eval_type=self.reward_eval_type,
                reward_shape=self.reward_shape,
                dense_weight=self.dense_weight,
                # Keep False: it makes _format_observation return only the messages
                # after the last assistant message, i.e. the delta. verl's
                # RolloutHandler already accumulates the conversation itself, so
                # returning full history here would grow the context quadratically.
                all_messages_as_observation=False,
            )
            self.env[env_idx] = env
        else:
            # Reuse the slot's env rather than building a new one per episode.
            # Each AgentGymEnv owns a daemon orchestrator thread parked on a
            # threading.Event; a fresh object every reset would orphan the old thread,
            # and over a training run those accumulate. AgentGymEnv.reset() joins its
            # own previous thread, and _get_task() re-reads task_id at reset time, so
            # re-pointing the slot is enough.
            env.task_id = task_id

        obs, _info = env.reset()
        self.steps[env_idx] = 0
        self.last_obs[env_idx] = obs or ""
        return self.last_obs[env_idx]

    # ---- interaction --------------------------------------------------------
    def step(self, env_idx: int, action: str) -> Tuple[str, float, bool, None]:
        env = self.env.get(env_idx)
        if env is None:
            return "Environment not reset. Call /reset first.", 0.0, False, None
        obs, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
        self.steps[env_idx] = self.steps.get(env_idx, 0) + 1

        # tau2 only scores a run once its orchestrator terminates -- until then
        # _simulation_run is None and _get_reward returns 0.0 unconditionally. Roughly
        # half of a weak policy's episodes simply run out of the caller's turn budget
        # without the customer ever saying ###STOP###, so they were being handed back as
        # reward 0 with no evaluation at all, no matter how much of the task they did.
        # Ending the episode ourselves at the budget makes those trajectories scoreable.
        if not done and self.force_done_after and self.steps[env_idx] >= self.force_done_after:
            try:
                obs2, reward, terminated, truncated, _ = env.step("done()")
                obs = obs2 or obs
                done = True
            except Exception as e:
                print(f"[tau2] force-done on env {env_idx} failed: {e}")

        self.last_obs[env_idx] = obs or ""
        return self.last_obs[env_idx], float(reward), done, None

    def observation(self, env_idx: int) -> str:
        return self.last_obs.get(env_idx, "")

    def close(self, env_idx: int) -> None:
        self._close_slot(env_idx)

    def __del__(self):
        for idx in list(self.env.keys()):
            self._close_slot(idx)


_TAU2_FORMAT_BLOCK = """# Response format

EVERY turn you MUST reply with exactly a Thought and an Action:

Thought:
<your reasoning about what to do next>

Action:
<a single tool call, OR a message to the customer>

The Action is interpreted as follows:
- If it looks like `tool_name(arg='value', other=123)` it is executed as a tool call,
  and the next observation is the tool's result.
- Otherwise it is sent verbatim to the customer as your message, and the next
  observation is their reply.

Rules for the Action line:
- Write the tool call on ONE line. Use only the parameters listed above.
- Never put a tool call and a message in the same Action -- do one or the other.
- Never wrap the Action in quotes, backticks or a code block.
"""

# variant `base`: the original ending. Measured on retail train with Qwen2.5-7B, 76% of
# all failures were the agent calling transfer_to_human_agents -- the "or you have told
# them you cannot help" clause plus the transfer tool's own description read as
# permission to bail whenever the task got hard, and only 2 of 74 tasks actually want a
# transfer.
_ENDING_BASE = """- The customer ends the conversation when they are satisfied. Only call `done()`
  yourself once everything the customer asked for is resolved or you have told them
  you cannot help -- calling it early ends the episode and scores zero.
"""

# variant `strict`: same information, no escape hatch.
_ENDING_STRICT = """- The customer ends the conversation when they are satisfied. Do not end it yourself
  while anything they asked for is still unresolved.

# Before you act

- Authenticate FIRST. You cannot use a user id until a tool returned it to you. Call
  find_user_id_by_email, or find_user_id_by_name_zip, and use the id it returns.
  Never invent, guess, or reuse an id from an example.
- Never call transfer_to_human_agents unless the customer explicitly asks for a human.
  Being unsure, missing a detail, or finding the task hard is NOT a reason to transfer
  -- ask the customer for what you are missing, or look it up with a tool. A transfer
  leaves the task unfinished and scores zero.
- Finish the job. Reading with get_* tools changes nothing on its own; almost every
  request needs a write tool (cancel_, modify_, return_, exchange_) before it is done.
  Confirm the details with the customer, then make that call.
- Only call `done()` after the write tool has succeeded, or after the customer has
  confirmed they need nothing else.
"""

_ENDINGS = {"base": _ENDING_BASE, "strict": _ENDING_STRICT}

TAU2_SYSTEM_TEMPLATE = """You are a customer service agent. You talk to a customer and use tools to act on their behalf, and you must follow the policy below exactly.

# Policy

{policy}

# Tools

{tools}

""" + _TAU2_FORMAT_BLOCK + "{ending}"


tau2_env_server = Tau2EnvServer()
