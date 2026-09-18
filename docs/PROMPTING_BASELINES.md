# Prompting / ReAct baselines

The two baselines already configured in `examples/train/` — AgentGym-RL and
ScalingInter-RL — are both RL. They answer "does this training signal beat that
training signal". They do not answer "does any of this beat prompting the base model",
which is the floor every RL curve is drawn above.

This is that floor: the environment's own ReAct prompt, driven turn by turn against an
untrained model, with no weights updated.

| | ALFWorld | WebShop |
|---|---|---|
| prompting baseline (this doc) | `scripts/eval_alfworld_openai.py` | `scripts/eval_webshop_openai.py` |
| trained checkpoint | `scripts/batch_eval_alfworld.sh` | `scripts/eval_webshop.py` |

Both baseline scripts talk to an OpenAI-compatible endpoint, so one code path covers a
hosted model (`--model gpt-5-mini`) and a local base model served by
`scripts/run_vllm_openai_server.sh` (`--model Qwen2.5-7B-Instruct --api-key EMPTY`).

## Where the prompt actually lives

Nowhere in `scripts/`. Each environment's adapter owns it, and the eval scripts read it
off the client as `env_client.conversation_start`:

```
AgentGym/agentenv/agentenv/envs/alfworld.py   AlfWorldAdapter.conversation_start_dict
AgentGym/agentenv/agentenv/envs/webshop.py    WebshopAdapter.conversation_start_dict
AgentGym/agentenv/agentenv/controller/types.py    ActionFormat: react | function_calling | code_as_action
AgentGym/agentenv/agentenv/controller/utils.py    BaseAdapter.parse_react / to_react
```

That is the point of routing the baseline through `AlfWorldEnvClient` /
`WebshopEnvClient` rather than writing a standalone prompt loop: the baseline and the
training rollout in `src/verl/workers/rollout/` read the same two opening messages from
the same dict. There is one copy of the prompt in the repo, so a prompt edit cannot
silently move the baseline relative to the trained numbers.

`--action-format` exposes the other two formats. `react` is the default, the one
training uses, and the only one behind any number here.

## Running

```bash
# 1. environment servers (ALFWorld from 36001, WebShop from 36101)
ALFWORLD_ENV=<prefix> ALFWORLD_DATA=<cache> NUM_ENVS=4 \
  bash scripts/run_alfworld_env_service.sh
WEBSHOP_ENV=<prefix> NUM_ENVS=1 bash scripts/run_webshop_env_service.sh

# 2a. baseline for a hosted model
OPENAI_API_KEY=sk-... python scripts/eval_alfworld_openai.py \
    --model gpt-5-mini \
    --env-addrs http://127.0.0.1:36001,http://127.0.0.1:36002 \
    --output-dir runs/alfworld_gpt5mini --concurrency 16

# 2b. baseline for the base model the RL runs start from
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct SERVED_MODEL_NAME=Qwen2.5-7B-Instruct \
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 PORT=8100 \
  bash scripts/run_vllm_openai_server.sh &

python scripts/eval_webshop_openai.py \
    --model Qwen2.5-7B-Instruct --api-key EMPTY \
    --base-url http://127.0.0.1:8100/v1 \
    --env-addrs http://127.0.0.1:36101 \
    --num-items 500 --output-dir runs/webshop_qwen7b_prompting
```

Per-item JSONs land in `--output-dir` and are skipped on re-run, so an interrupted
sweep resumes. `summary.json` records the full setting alongside the score.

Item-id files are generated per checkout (see the README). ALFWorld falls back to the
order in `AgentGym/agentenv-alfworld/configs/mappings_test.json`, which is what
`batch_eval_alfworld.sh` derives its split from; WebShop takes `--num-items N` to
evaluate `webshop_0..N-1`, since `webshop_i` indexes the env server's goal list
directly (`WEBSHOP_GOAL_SOURCE=human` by default, so those are the official human
goals in order).

## Settings

| | ALFWorld | WebShop |
|---|---|---|
| max rounds | 30 | 15 |
| max tokens / turn | 200 | 256 |
| temperature / top_p | 1.0 / 1.0 | 1.0 / 1.0 |
| env timeout | 2400s | 2400s |
| reported | success by task family (Pick/Look/Clean/Heat/Cool/Pick2) + All | Score (mean reward), Succ (reward == 1.0) |

These match `scripts/batch_eval_alfworld.sh` and the WebShop training rollout, so the
baseline and the trained checkpoint differ in the weights and nothing else.

## Five things that will make a number wrong

**1. ALFWorld observations carry the action list, so this is not the classic ReAct
setting.** AgentGym appends `AVAILABLE ACTIONS` to the observation. A model that could
never have guessed this environment's verb/object vocabulary is handed it every turn,
and success rates here run far above published zero-shot ReAct ALFWorld numbers for
that reason alone. The number is a valid floor *for this repo's harness*. It is not
comparable to an ALFWorld number from any paper that did not inject the action list.

**2. `--no-reinject-actions` changes what you are measuring.** By default every turn's
user message comes from `env_client.observe()` (observation + AVAILABLE ACTIONS). The
training rollout instead uses the bare `step.state`, with the action list present only
on turn 0. Evaluating a trained checkpoint with the default puts it out of
distribution; evaluating a cold API model with the flag denies it the vocabulary. Pick
per model, and read `reinject_actions` back out of `summary.json` before comparing two
runs. The two settings are not comparable to each other.

**3. The prompt is zero-shot.** `conversation_start` is an instruction plus a canned
`"OK. I'll follow your instructions"` — no in-context demonstrations, unlike the
original ReAct setups, which carry two worked examples. A weak base model failing here
is partly a format-compliance failure, and `terminated_by` in the per-item JSON
separates that from genuine task failure.

**4. Reasoning models silently return empty strings.** `max_completion_tokens` is a
hard cap that hidden reasoning tokens count against. Size it like `--max-tokens` and
the model spends the whole budget thinking, returns `""`, and the env scores an invalid
action — a configuration error that reads as a bad baseline. Hence
`--reasoning-max-tokens` (default 4096) well above `--max-tokens`, plus a low
`--reasoning-effort`.

**5. A partial run is biased, and `count` is how you catch it.** Both scripts aggregate
over completed items only. An interrupted sweep over-represents whatever finished
fastest, which on ALFWorld means the easy families. `summary.json` carries
`completed_items` against `total_items`; compare only runs where they match. Errors
that are deterministic rather than transient — a 400, or `insufficient_quota` — end the
trajectory instead of burning the retry ladder on every remaining item, so a dead key
fails the sweep quickly rather than producing a slow, plausible, wrong number.

Sampling is at temperature 1 to match the rollout, so a single pass carries real
variance. Small deltas need avg@k, not one run.
