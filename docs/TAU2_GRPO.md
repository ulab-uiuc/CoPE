# GRPO on tau²-bench

Runs AgentGym-RL's existing GRPO stack against
[tau2-bench](https://github.com/sierra-research/tau2-bench) (the repo is now τ³-bench
v1.0.1). No algorithm changes — tau²-bench is wired in as one more AgentGym
environment, so every knob in `verl/agent_trainer/config/ppo_trainer.yaml` applies
unchanged.

## Why it is split across processes

tau2 requires Python `>=3.12,<3.14`; the training env (`agentgym-rl`) is Python 3.10
with vllm 0.6.3. They cannot share an interpreter. This uses the same shape AgentGym
already uses for webshop (whose server runs on Python 3.8): the environment runs as its
own HTTP service, and the training side talks to it through a thin `requests` client.

```
examples/train/AgentGym-RL/tau2_grpo_train.sh            plain GRPO
examples/train/AgentGym-RL/tau2_grpo_forecast_train.sh   GRPO + action forecasting
scripts/launch_tau2_grpo_tmux.sh / scripts/launch_tau2_cope_tmux.sh   the same two, detached in tmux
└─ scripts/run_tau2_pipeline.sh            one process tree, torn down on exit
   ├─ customer                             USERSIM_MODE=hosted: gpt-4o-mini via litellm
   │                                       USERSIM_MODE=local:  scripts/run_tau2_usersim_server.sh,
   │                                         a vLLM OpenAI server on its own GPU
   ├─ scripts/run_tau2_env_service.sh      N x `tau2-env` FastAPI processes (envs/tau2, py3.12),
   │                                         each holding {env_idx: AgentGymEnv}; health-gated
   └─ scripts/run_tau2_grpo_train.sh       python -m verl.agent_trainer.main_ppo \
                                             algorithm.adv_estimator=grpo \
                                             actor_rollout_ref.agentgym.task_name=tau2
```

## One-time setup

```bash
# 1. AgentGym submodule (SSH is broken on this host; HTTPS resolves the pinned commit)
git config submodule.AgentGym.url https://github.com/PolarisDane/Agentgym.git
git submodule update --init AgentGym

# 2. tau2 env-server conda env (Python 3.12). ./envs/tau2 is what the launchers expect
#    by default (TAU2_ENV overrides).
TAU2_ENV_DEFAULT=./envs/tau2
conda create -y -p ${TAU2_ENV_DEFAULT} python=3.12
P=${TAU2_ENV_DEFAULT}/bin/pip
$P install -e tau2-bench
$P install -e AgentGym/agentenv-tau2
$P install gymnasium          # tau2's gym module needs it; not a tau2 core dep

# 3. item-id dataset
${TAU2_ENV_DEFAULT}/bin/python - <<'PY'
import json
from tau2.registry import registry
for split in ["train", "test"]:
    tasks = registry.get_tasks_loader("retail")(split)
    rows = [{"item_id": f"tau2_{i}", "task_type": "retail", "task_id": t.id}
            for i, t in enumerate(tasks)]
    with open(f"data/tau2_retail_{split}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(split, len(rows))
PY
```

The training env is **not** modified. `run_tau2_grpo_train.sh` sets
`PYTHONPATH=<repo>/AgentGym/agentenv`, which wins over the `agentenv` editable install
already in `agentgym-rl` (that one points at a different AgentGym checkout and has no
`Tau2EnvClient`). `PYTHONNOUSERSITE=1` keeps a broken `~/.local` transformers out of
the import path.

## Running

### In one command (no scheduler)

```bash
# InfoPO's training protocol -- three domains, tau2's native tool calling, gpt-4o-mini
# customer -- with this repo's plain GRPO. Needs an OpenAI key in .secrets/openai_api_key.
TRAIN_ENV=/path/to/conda/env MODEL_PATH=/path/to/Qwen2.5-7B-Instruct CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash examples/train/AgentGym-RL/tau2_grpo_train.sh

# the same, detached in tmux
bash scripts/launch_tau2_grpo_tmux.sh

# this repo's original retail / ReAct / dense-reward setup
PRESET=repo bash examples/train/AgentGym-RL/tau2_grpo_train.sh

# no API credit: a local Qwen customer on its own GPU (deltas only, not absolute scores)
USERSIM_MODE=local USERSIM_GPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/train/AgentGym-RL/tau2_grpo_train.sh

# check prerequisites and print the resolved configuration without launching anything
DRY_RUN=1 bash examples/train/AgentGym-RL/tau2_grpo_train.sh

# GRPO + the action-forecast auxiliary loss (same protocol; adds a forecast-SFT pass per step)
bash examples/train/AgentGym-RL/tau2_grpo_forecast_train.sh
```

Both entry points run `scripts/run_tau2_pipeline.sh`: it applies the tau2-bench patch,
starts the customer (if local), brings up the env cluster behind a health gate, runs
`scripts/run_tau2_grpo_train.sh`, and tears everything down on exit. Every knob of the
training script is a plain environment variable; the `PRESET` fills the rest. Logs land
in `runlogs/<exp>/` (`pipeline.log`, `train.log`, `env_cluster/`, and `main_task.out`,
a link to the Ray task that prints the step metrics -- see "Things that bite"),
checkpoints in `CKPT_DIR` -- point it at a big disk, one FSDP checkpoint of a 7B is ~86GB.

### On slurm (what this cluster has)

```bash
sbatch scripts/sbatch_tau2_grpo.sh                                  # full run, 2 nodes
sbatch --nodes=1 --export=ALL,TINY=1 scripts/sbatch_tau2_grpo.sh    # smoke run, 1 node
```

These are **40GB** A100s, and a 7B actor plus a co-resident vLLM rollout engine does not
fit on 4 of them — a 4-GPU run reaches step 2 and then OOMs in
`actor_rollout_update_actor`. The default allocation is therefore 2 nodes: the user
simulator gets its own node, and all 8 GPUs on the batch node train. The script falls
back to user-sim on GPU 0 + training on GPUs 1–7 when given `--nodes=1`.

Memory settings that matter on 40GB cards, all overridable:
`ROLLOUT_GPU_MEMORY_UTILIZATION=0.30`, `PARAM_OFFLOAD=True`, `OPTIMIZER_OFFLOAD=True`,
`USE_REMOVE_PADDING=True`. On 80GB cards raise the first toward 0.8.

### Directly (no scheduler)

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
USERSIM_MODEL=/path/to/user-sim-model \
CUDA_VISIBLE_DEVICES=1,2,3 USERSIM_GPU=0 \
  bash launch_tau2_grpo_tmux.sh
```

`USERSIM_GPU` must not appear in `CUDA_VISIBLE_DEVICES`. Omit `USERSIM_MODEL` to reuse a
user-sim server that is already up.

Useful overrides: `TAU2_DOMAIN` (retail / airline / telecom), `TAU2_TASK_SPLIT`,
`TAU2_REWARD_BASIS`, `ROLLOUT_N`, `TRAIN_BATCH_SIZE`, `MAX_ROUNDS`, `TRAIN_FILE`.

**`TAU2_TASK_SPLIT` must match the split `TRAIN_FILE` was generated from.** verl passes
an integer item id and the server resolves it as `task_ids[item_id % len(task_ids)]`
over its own split — a mismatch silently trains on the wrong tasks.

## Reward

Two settings do the heavy lifting. Without them GRPO runs but does not learn.

**`TAU2_FORCE_DONE_AFTER`** (set to the caller's `MAX_ROUNDS`). tau2 only scores a run
once its orchestrator terminates — until then `_simulation_run` is None and the reward
is 0 regardless of what the agent achieved. About 55% of a 7B policy's episodes simply
exhaust the turn budget without the customer ever saying `###STOP###`, so they were all
being returned as unevaluated zeros. Ending the episode server-side at the budget took
the measured solve rate from 3.7% to 10.1% on its own.

**`TAU2_REWARD_SHAPE=dense`** (default). Binary DB-state reward leaves 64 of 74 retail
tasks at a uniform zero, so only 4/74 GRPO groups have any within-group spread and 75%
of optimizer steps have `pg_loss` exactly 0 — three quarters of training is a no-op.
Dense adds graded partial credit from tau2's per-action checks, in two tiers: exact
ground-truth match (weight 0.7) and right-tool-wrong-arguments (weight 0.3), the whole
thing scaled by `TAU2_DENSE_WEIGHT=0.5` so a partial can never outscore a real solve.
Eval A/B at identical 10.1% solve rate: informative groups 5.4% → 73%.

`TAU2_REWARD_BASIS=env` scores from DB / env-assertion state only. `all` is tau2's
`EvaluationType.ALL`: DB/env-state, ACTION and COMMUNICATE checks multiplied together,
which is what `tau2 run` scores at evaluation time -- so training on `env` optimises a
looser criterion than the one you are measured on (retail and airline tasks can pass
the DB check and still fail for not telling the customer something). At the pinned
commit `c5b2d22` `all` costs **no LLM call**: none of the 292 tasks carries an
`NL_ASSERTION` in its reward basis, and plain `ALL` never invokes the NL evaluator
(only the WIP `ALL_WITH_NL_ASSERTIONS` does). An earlier version of this note claimed a
judge call on 112 of 114 retail tasks; that describes later tau2 trees, which do add
LLM-judge reviewers, not this one. Use `all` for numbers comparable to InfoPO.

`_shape_task_reward` in `vllm_rollout.py` passes tau2's score through unchanged; do not
re-binarise it or the partial credit is discarded.

## Native tool-calling protocol (InfoPO alignment)

The harness originally spoke a ReAct text protocol: the policy document, a rendered
tool-signature table and a `Thought:/Action:` format block went in as a *user* turn
followed by an `Ok.` assistant turn, actions were pulled out of the text with a regex,
and every observation came back as a user turn. That is not how InfoPO trains and not
how `tau2 run` evaluates. Checked against InfoPO's released `train.parquet` and
`examples/tau2/train.sh`:

| | InfoPO training | `TAU2_PROMPT_VARIANT=native` + `NATIVE_TOOLS=True` |
|---|---|---|
| prompt | tau2's stock agent prompt as the **system** message: six lines and `<policy>…</policy>`, nothing else | byte-identical (verified for all three domains) |
| tools | the domain's tool schemas, natively (`extra_info.tool_schemas`: retail 15, airline 14, telecom 13, no `done`) | the same schemas, rendered as text exactly as Qwen2.5's template renders `tools=` (token-identical, verified) |
| action | `<tool_call>{"name":…,"arguments":{…}}</tool_call>`; plain text = message to the customer | same; forwarded to tau2 as ToolCall JSON / text |
| observation | tool results as `tool`-role messages, customer replies as user turns | same (`RolloutHandler.add_tool_message`) |
| customer | gpt-4o-mini @ temperature 0.7 | `TAU2_USER_TEMPERATURE=0.7` in the `infopo` preset |

Rendering the tools block as text rather than passing `tools=` keeps the dataset's
tokenized prompt and the rollout's re-rendered generation prompt identical, which the
handler relies on. One tokenization subtlety decides that equality for tool results:
the content and the closing `</tool_response>` must be encoded together, because
Qwen's pre-tokenizer merges a closing `}` with the newline after it.
`tests/test_tau2_native_protocol.py` pins both invariants.

What could not be recovered from InfoPO's public code: the `Tau2Env` wrapper their
verl fork instantiates is not in the repository, so how the first customer message is
injected and how their `use_nl_assertions` flags map to a reward are inferred, not
read. The algorithm is also different by design: their published numbers are
`info_grpo` with the intrinsic reward; the `infopo` preset here is plain GRPO with
their hyperparameters and interface, i.e. the baseline their method is compared to.

## Model size: this rollout is TP=1 only

`Qwen2.5-7B-Instruct` at `tensor_model_parallel_size=1` is the validated setup. Do not
raise TP without doing the work below first.

`agent_vllm_rollout/vllm_rollout.py` sets tensor parallelism when it builds the engine
and then ignores it: `generate_sequences` has every rank create its own env clients and
drive its own conversations. Ranks inside one TP group must run identical forward passes
in lockstep, so with TP>1 they feed unrelated prompts through a tensor-parallel forward.
Observed on Qwen2.5-14B: TP=2 and TP=4 both produce fluent-looking garbage from the
policy (the same 14B served as an ordinary vLLM user simulator at TP=4 was fine), and
the differing rollout lengths blow the NCCL collective timeout. Making it TP-aware means
having TP-rank-0 drive the env and broadcasting actions/observations to its peers.

That caps usable model size: a 14B is 28GB of bf16 weights, which on a 40GB card leaves
too little KV cache at TP=1. LoRA does not help here — the base weights still occupy the
rollout GPU, and the training-side memory it would save is already offloaded to host.

Two fixes made while establishing this are kept because they are correct regardless:

- **`max_model_len` is now passed to the vLLM engine.** It never was, so the engine used
  the model config's own limit (32768 for Qwen2.5) when sizing and validating the KV
  cache. It now uses `min(max_model_len, prompt_length + response_length)` — the exact
  value `preprocess_prompt_to_rollout_handler` already caps trajectories at, so no
  sequence that used to fit is truncated; the engine simply stops reserving for lengths
  the rollout can never produce. The 7B numbers below were measured before this change
  and are unaffected by it.
- **The default NCCL process-group timeout is raised to 3600s** in
  `agent_fsdp_workers.py`. verl set that on the main group only; sub-groups created by
  `new_group()` fell back to NCCL's compiled-in 10-minute default, which a slow agent
  rollout can exceed.

## vLLM >= 0.6.6 and Blackwell GPUs

`agent_vllm_rollout` was written against verl's vendored vLLM 0.6.3 (`offload_model_weights`,
`init/free_cache_engine`), and its version gate compared version strings, so a newer
vLLM was rejected at import with `cannot import name 'vLLMRollout'`. Blackwell (sm_120)
cards need CUDA 12.8 / torch 2.7 / vLLM 0.9, and 0.6.3 has no kernels for them, so
this was not a downgrade situation. The rollout now drives either engine:

- **>= 0.6.6**: the stock `LLM` through `distributed_executor_backend="external_launcher"`
  with `enable_sleep_mode`, `load_format=dummy` (weights arrive from FSDP through the
  sharding manager, which already handled this path with `sleep()`/`wake_up()`), an
  explicit `seed`, and a shim that reproduces the vendored engine's padded-tensor
  return contract. The agent loop is untouched.
- The training script pins **`VLLM_USE_V1=0`**: the sharding manager loads dtensor
  weights through `llm_engine.model_executor.driver_worker`, a V0 path.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set only for <= 0.6.3: sleep
  mode allocates through a memory pool, and torch refuses to combine the two
  (pytorch#147851, "Expandable segments are not compatible with memory pool").

Validated: 44 GRPO steps of Qwen2.5-7B-Instruct on RTX PRO 6000 Blackwell with vLLM
0.9.2, solve rates on retail matching the numbers above.

## Comparing against published numbers (InfoPO)

Numbers from the harness above are **not** comparable to the τ²-bench columns in papers
like InfoPO (arXiv 2603.00656). To compare, evaluate through tau2's own CLI with
`scripts/sbatch_tau2_align.sh`, which reproduces InfoPO's `eval/tau2bench/eval_vllm.sh`:

```bash
sbatch scripts/sbatch_tau2_align.sh                                    # base model
sbatch --export=ALL,MODEL_PATH=<hf-ckpt>,TAG=trained scripts/sbatch_tau2_align.sh
```

Measured for Qwen2.5-7B-Instruct against the paper's prompting row, Avg@4 on the test
splits. Each column removes one more difference from our harness:

| domain | our harness | CLI, tau2 v1.0.1 | CLI, tau2 @ 2026-02 | paper |
|---|---|---|---|---|
| airline | 18.8 | 13.8 | **8.8** | 7.5 |
| retail | — | 6.9 | **9.4** | 13.1 |
| telecom | — | 5.6 | **8.1** | 14.4 |
| mean | — | 8.8 | **8.8** | 11.7 |

1 se is about 2.3 points here. Airline reproduces exactly; retail is 1.6 se low; telecom
is still 2.7 se low and **does not reproduce**. What the right-hand column does settle is
the *version*: v1.0.1's errors are 6–9 points in inconsistent directions (airline high,
retail and telecom low), while the 2026-02 tree is consistently close, cutting mean
absolute error from 7.1 to 3.7 points.

Telecom's residual gap is **policy degeneration in long tool-calling conversations**,
not a step budget and not a protocol difference. Split the same runs by whether the
episode terminated on its own:

| domain | all | terminated | hit max_steps | avg messages |
|---|---|---|---|---|
| airline | 8.8 (n=80) | 10.0 (n=70) | 0.0 (n=10) | — |
| retail | 9.4 (n=160) | 10.1 (n=149) | 0.0 (n=11) | 46 |
| telecom | 8.1 (n=160) | 16.0 (n=81) | 0.0 (n=79) | 127 |

Half of telecom's episodes never finish inside 200 orchestrator steps and score zero —
`reward_basis` is None for exactly those 79, so tau2 never evaluated them. But the cause
is not the budget:

- **All 79 truncated episodes contain degenerate policy output**, averaging 67 such
  messages each, and it starts at the 24th message (median) — long before the cap. The
  model emits syntactic fragments with a literal `<tool_call>` string in the message
  body: `'Sure to the to the. <tool_call> to the to send. <tool_call> to sendon.'`
  That literal is the hermes parser's own delimiter leaking into content, i.e. the model
  is writing tool calls as prose instead of into the structured field.
- **Raising the budget does not help.** A local-user-simulator A/B (no API credit needed,
  since truncation is mechanical) moved truncation only 62% → 52% going from 200 to 400
  steps, while average length grew 146 → 244 messages; the 400-step job then died with
  `ContextWindowExceededError`. The conversations are not almost-done, they are looping.
- 80% of the *terminated* episodes show degeneration too, just less of it (9 messages).

Telecom is the only domain that needs sustained multi-turn tool interaction — 127
messages on average against retail's 46 — which is why only it is affected. Three things
it is *not*:

- **Not context overflow.** Collapse triggers at a median of ~7.4k tokens total (6.2k of
  telecom system prompt plus ~1.2k of conversation), 23% of the 32768 window.
- **Not a verbose agent.** Pre-collapse assistant messages are 268 chars at the median
  against retail's 223 and airline's 226; telecom is longer in *turns*, not in words.
- **Not fixable by decoding.** Four sampling configs over 80 episodes each, local user
  simulator (truncation is mechanical, so this needs no API credit):

  | agent sampling | truncated | degenerate |
  |---|---|---|
  | temperature 0 (paper protocol) | 62% | 95% |
  | + frequency_penalty 0.2 | 69% | 92% |
  | + repetition_penalty 1.1 | 62% | 100% |
  | temperature 0.3 | 65% | 96% |

What stands out is the ratio: when telecom collapses, ~84% of the context is the static
policy document (6212 tokens, against retail's 2710). Nothing in the harness or the
decoder moves it, so this looks like a failure mode of Qwen2.5-7B under a long static
prompt plus sustained tool calling, and the gap to the paper's 14.4 has to come from
somewhere outside what is reproducible here.

**The tau2 version matters more than anything else.** InfoPO's paper (2026-02-28) and
repo (2026-03-16) both predate tau2 v1.0.0 (2026-03-18). Between then and v1.0.1 the
user simulator was rewritten, the evaluator gained LLM-judge and hallucination
reviewers, and 3.5M lines of task data changed — telecom's score doubles between the two
versions. `scripts/sbatch_tau2_align.sh` takes `TAU2_VERSION=infopo|local`; `infopo` is a
worktree of `c5b2d22` at `${TAU2_BENCH_DIR}` with its own conda env, so the
training pipeline's v1.0.1 checkout is untouched. Point the env servers at the same tree
with `TAU2_ENV=${TAU2_ENV_PAPER}` when training for comparison.

The remaining differences, in the order they cost accuracy:

- **`tau2 run` overrides nothing.** Default agent prompt, default user-simulator prompt,
  and `EvaluationType.ALL` — which runs the NL-assertion judge on nearly every task.
  Our harness scores env-assertions only and injects a `strict` ReAct prompt.
- **tau2's agent uses native tool calling**, not a ReAct text protocol, so the *policy*
  server also needs `--enable-auto-tool-choice --tool-call-parser hermes`. Without it
  every task ends as `infrastructure_error` with zero messages and scores 0%.
- **`max_steps` is not one of them.** 50 and 200 both give 13.8 on airline at 4 trials.
  The paper body says 50, their script says 200; it does not matter.

Two upstream bugs had to be patched in both checkouts, or nothing runs at all:

- `to_litellm_messages` emits a top-level `"name"` on each tool call alongside
  `function.name`. Current vLLM rejects the duplicate as `extra_forbidden`, so every
  assistant turn after a tool call fails.
- `max_context_tokens` in `--agent-llm-args` only exists in the tau2 fork InfoPO
  vendors; here it is forwarded to litellm and rejected. `--max-model-len` on the policy
  server covers the same ground.

`tau2 run` also **prompts on stdin** to resume when the save path exists, which is an
instant `EOFError` under sbatch — the script clears prior results unless `RESUME=1`.

**Do not rebuild the user-simulator prompts from the paper's figures.** Appendix
Figures 19 and 20 print "optimized simulation guidelines" for the base and tool-enabled
user, and tau2 really does select between exactly those two files
(`data/tau2/user_simulator/simulation_guidelines{,_tools}.md`) on whether the user has
tools — so substituting them looks right. It is not: the figures are abridged summaries
and drop the mechanics the simulator runs on, including "each turn you can either send a
message or make a tool call, not both", "messages accompanying a tool call are not shown
to the agent", and the `###OUT-OF-SCOPE###` token. Swapping them in took telecom from
8.1 to 1.6. Originals are kept alongside as `*.md.tau2default`.

**The user simulator is a metered API.** Every turn of every rollout is a gpt-4o-mini
call, and a 50-turn three-domain training run burns credit fast; run 21001 exhausted the
account at step 8 of 50. When credits run out the failure is quiet from the trainer's
side — episodes come back empty and rewards go to zero rather than erroring — so check
`runlogs/<exp>/env_cluster/*.log` for `no credits remaining` before trusting a flat
reward curve.

## Measured result

### GRPO under the aligned protocol

Standard GRPO at the paper's hyperparameters (178 tasks over all three domains, batch 32,
n=5, lr 1e-6, KL off, 50 turns, binary reward, gpt-4o-mini customer). The run reached 35
of 50 steps before API credit ran out; the step-25 checkpoint was evaluated.

Standard GRPO at the paper's hyperparameters **does not lift all three domains**, under
any of four configurations tried. Reward shaping decides *which* domain benefits; one
always does, at the others' expense.

Avg@4 on the test splits, gpt-4o-mini customer (the paper's protocol):

| domain | base | binary @25 | dense @25 (local-trained) | dense @15 (4o-trained) |
|---|---|---|---|---|
| airline | 8.8 | **18.8** | 7.5 | 6.2 |
| retail | 9.4 | 10.0 | **15.6** | 11.2 |
| telecom | 8.1 | 1.9 | 4.4 | — (out of credit) |

The missing cell was filled under a paired local-customer protocol, where base scores
11.2: dense@15 gives **5.0**, i.e. −6.2 at −2.0 se. Telecom got worse in every
configuration measured — it is the one domain needing sustained multi-turn tool use.

Two explanations were proposed during this work and **both were falsified by the
experiments above**:

- *"dense looks bad on airline only because it was trained against a local customer."*
  Retraining dense with the matched gpt-4o-mini customer gave 6.2 on airline, slightly
  *below* the mismatched run's 7.5.
- *"degenerate groups are the bottleneck; restore within-group variance and it learns."*
  Dense cut degeneracy from 78.7% to 12.5% and merely moved the winning domain from
  airline to retail. Variance is necessary, not sufficient. (Under dense@15 telecom's
  collapse rate also fell, 42% → 24%, while its solve rate still halved — the policy
  loops less and simply fails earlier.)

One methodological note, learned the hard way: absolute rates are **not** transferable
across evaluation setups. Base airline scores 8.8 under gpt-4o-mini and 15.0 under the
local customer, so an unpaired comparison across two setups produced a +10.0 that a
paired control later reversed in sign. Only compare runs whose `eval_config.json`
matches, and prefer arms submitted together.

Three-domain means land at 9–11 against base 8.8 and the paper's RAGEN 16.7. Reaching
that table needs turn-level signal, which is what InfoPO's information-gain reward
supplies without touching the outcome definition — consistent with their table showing
all three domains above RAGEN rather than one.




**Training reward does not predict this.** Over 35 steps it moved 0.1113 → 0.1243
(+0.92 se), i.e. flat, while held-out airline doubled. Train reward here is binary
pass/fail on the train split under 81.8% degenerate groups, so it is dominated by task
difficulty mix and group collapse; only the aligned evaluation says anything. Two
mid-run readings of the training curve were misread as evidence of learning and of
not-learning respectively before the eval settled it.

The 81.8% degenerate-group rate matches the paper's own Table 2 for τ²-Bench (76.3%) and
is what InfoPO's information-gain reward exists to fix. Notably `pg_loss` was never
exactly zero here — with 32 tasks x 5 rollouts per batch, enough groups carry signal even
at that collapse rate, unlike the 8-task batches in the retail-only runs above where 75%
of steps had no gradient at all.


Qwen2.5-7B-Instruct, retail train (74 tasks), 45 steps / 5 epochs, batch 8 × rollout 8.
`runlogs/tau2_solverate.png`, regenerate with `scripts/tau2_plot_solverate.py`.

| | binary | dense + force-done |
|---|---|---|
| solve rate by epoch | 7.3 / 9.2 / 8.3 / 9.5 / 8.6 % | **9.5 / 10.1 / 11.6 / 15.3 / 16.4 %** |
| epoch 1 → 5 | flat | z = **+3.38** |
| reward, first half vs second | −0.25 se | **+3.51 se** |
| steps with `pg_loss == 0` | 33/44 = 75% | **0/41** |
| degenerate groups | 95.5% | **8.8%** |

The binary column is two independent runs' worth of flat (−0.25 se and +0.02 se), so
the contrast is not a seed artifact. The dense run is a single seed — repeat it before
treating 16.4% as a number rather than a direction.

## Verification

Stages 0–2 need no GPU. Stage 3 onward does.

```bash
# terminal 1 -- canned user simulator (no GPU, no API)
python scripts/tau2_fake_usersim.py --port 38002

# terminal 2 -- env server
TAU2_USER_API_BASE=http://127.0.0.1:38002/v1 tau2-env --host 127.0.0.1 --port 36205

# terminal 3
python scripts/smoke_tau2_env.py    --addr http://127.0.0.1:36205 --n-tasks 3
python scripts/smoke_tau2_leak.py   --addr http://127.0.0.1:36205 --n 5
PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/AgentGym/agentenv \
  ${TRAIN_ENV}/bin/python \
  scripts/smoke_tau2_client.py --addr http://127.0.0.1:36205
```

| Script | Checks |
|---|---|
| `smoke_tau2_env.py` | Replays each task's ground-truth tool calls; expects reward 1.0. Covers create/reset/step, tool execution, delta observations, evaluator. |
| `smoke_tau2_leak.py` | Opens episodes, abandons them mid-way, closes them; expects orchestrator threads back to zero. |
| `smoke_tau2_client.py` | `Tau2EnvClient`: prompt fetched from `/system_prompt`, ReAct parsing, code-fence/multi-line repair, tool-vs-message routing, invalid action not fatal. |
| `eval_tau2.py` | Scores a policy against a live env cluster with no training — the cheap way to A/B a reward or prompt change. Reports the *informative-group* fraction, which is what predicts whether GRPO can learn. |
| `sbatch_tau2_eval.sh` | Runs that A/B end to end (policy vLLM + user sim + one env cluster per variant). |

**When A/B-ing env-server flags, give each variant its own port block.** An earlier
version of `sbatch_tau2_eval.sh` reused one block with `pkill` in between; the previous
variant's uvicorn workers had not released the sockets, every new server died with
EADDRINUSE, and the plain `/` health check passed against the survivors — so the second
variant was silently measured with the first variant's config, twice. The script now
asserts `/config` on every port before running.

## Things that bite

**Do not read a tau2 run before it finishes.** Results accumulate in task order, and
short tasks finish first, so a partial file is biased toward easy cases in a way that
looks like a real signal. This produced three wrong conclusions during the InfoPO
alignment: telecom read 11.5% at 52/160 and 8.1% at 160/160; a frequency-penalty A/B read
0% truncation at n=7 and 60% at n=40. Always check `n` against `tasks x trials` before
comparing anything.

- **Action format.** tau2's `is_functional_tool_call` is `^\w+\s*\(.*\)$` matched against
  the *whole* string, without `re.DOTALL`. A fenced or line-wrapped tool call therefore
  degrades silently into a chat message to the customer rather than erroring.
  `_clean_action` in `agentenv/envs/tau2.py` strips fences and collapses wrapped calls.
- **Invalid actions must not raise.** `vllm_rollout.py::agent_step` catches exceptions by
  marking the trajectory `done=True, score=0`, which burns a rollout and skews the group
  baseline. `Tau2EnvClient.step` returns a corrective observation instead.
- **Abandoned episodes leak threads.** verl closes every env client at the end of each
  rollout, including trajectories that merely ran out of `MAX_ROUNDS`. Each live
  `AgentGymEnv` owns a daemon orchestrator thread parked on a `threading.Event`, so
  `_close_slot` steps `done()` first to drive the simulation to termination.
- **Observations are deltas.** `all_messages_as_observation=False` makes
  `_format_observation` return only the messages after the last assistant message.
  verl's `RolloutHandler` accumulates the conversation itself; returning full history
  would grow the context quadratically.
- **User simulator determinism.** `TAU2_USER_TEMPERATURE` defaults to 0. Raising it means
  the 8 rollouts in a GRPO group face different customer behaviour, which adds variance
  to the group baseline.
- **Context budget.** The retail policy plus rendered tool signatures is ~2.5k tokens of
  prompt, hence `MAX_PROMPT_LENGTH=8192` / `MAX_MODEL_LEN=32768` (webshop uses 2048 /
  16384).
- **Conversations that outgrow the engine window used to take the whole batch down.**
  `get_generation_prompt` re-renders the full message list and had no cap of its own;
  `truncate_output_ids` only trims the training tensors after the loop. Once a
  conversation passed `min(max_model_len, prompt_length + response_length)` vLLM
  rejected the request outside `agent_step`'s try, and every other trajectory in the
  batch died with it (run 115123, step 16, 12446 > 12288 tokens). The rollout now ends
  such a trajectory the way running out of rounds does and logs how many it evicted
  per round. With the paper's 8192 + 16384 window this fires on ~0.3% of episodes.
- **Listening ports must sit below 32768.** The kernel hands out ephemeral source
  ports from 32768-60999, so an env server bound inside that range can lose its port
  to any outbound connection (the customer API, a previous run's TIME_WAIT sockets)
  and die with `EADDRINUSE`. Defaults are now 20301 (customer) and 20401+ (env servers).
- **Checkpoints are 86GB each** (FSDP shards of a 7B with optimizer state). Set
  `CKPT_DIR` to a large disk before a long run; a full root filesystem kills the
  final save and everyone else's jobs with it.

- **Step metrics can vanish from `train.log` while training continues.** They are
  printed by a Ray task (`main_task`), and Ray's worker-to-driver log forwarding was
  observed to stop ~2 minutes into a run (progress bars from rank 0 kept arriving,
  nothing from `main_task` did). Ray still writes that task's stdout to its session
  directory; the pipeline links it as `runlogs/<exp>/main_task.out`, which is the
  file to read -- the wandb offline run has the same numbers.
