# CoPE

Reinforcement learning for LLM agents across interactive environments, built on
[AgentGym-RL](https://github.com/PolarisDane/Agentgym-RL)'s verl fork.

The contribution is a set of auxiliary training signals that supplement the sparse
trajectory-level reward these benchmarks provide:

| module | idea |
|---|---|
| `src/verl/agent_trainer/ppo/plan_forecast.py` | forecast the plan ahead of acting and score the agent against it |
| `src/verl/trainer/ppo/info_grpo.py` | turn-level information gain from a masked-feedback counterfactual |

All of them are additive. With every flag off the training path is the stock GRPO one,
which the unit tests assert element-wise — so a run of this repo reproduces the baseline
before it reproduces the method:

```bash
pytest tests/test_additive.py
```

## Environments

| environment | train | eval | launcher |
|---|---|---|---|
| SciWorld | ✓ | ✓ | ✓ |
| TextCraft | ✓ | ✓ | |
| WebArena | ✓ | ✓ | |
| SearchQA | ✓ | ✓ | |
| BabyAI | ✓ | ✓ | ✓ |
| ALFWorld | | | ✓ |
| WebShop | | | ✓ |
| τ²-bench | ✓ | ✓ | ✓ |

Two baselines are configured alongside the method: `examples/train/AgentGym-RL/` and
`examples/train/ScalingInter-RL/`. Method variants are driven by environment variables on
the `scripts/run_*_grpo_train.sh` launchers — `PLAN_FORECAST_ENABLE`,
`INFO_INTRINSIC_WEIGHT` — rather than by forked copies of the script.

## Layout

```
src/verl/                 the training framework (fork of AgentGym-RL's verl)
  agent_trainer/ppo/      plan_forecast, sft_common, ray_trainer
  trainer/ppo/            info_grpo, intrinsic_reward, turn_structure
  workers/                actor, rollout, FSDP workers
src/envs/tau2/            the τ²-bench environment server and client
AgentGym/                 submodule: every other environment's server
examples/train/           per-environment configs, two baselines, method variants
examples/eval/            evaluation configs
scripts/                  launchers, evaluation harness, scoring, visualisation
docs/TAU2_GRPO.md         τ²-bench protocol alignment and findings
data/                     τ²-bench item-id datasets
tests/                    the additive claim, asserted element-wise
```

Training scripts read item-id files from `data/`. The τ²-bench ones are committed;
for the other environments they are generated per checkout, since they are derived
from each benchmark's own task list rather than authored here.

## Setup

```bash
git clone --recurse-submodules https://github.com/ulab-uiuc/CoPE.git
# the AgentGym submodule carries every environment server except tau2, and the
# agentenv base classes that tau2's own client subclasses -- so it is required
# even for a tau2-only run
git submodule update --init AgentGym
```

Environment servers run as separate processes from training — several need Python
versions that conflict with the training stack (WebShop is on 3.8, τ²-bench needs ≥3.12
against a 3.10 trainer), so each is its own HTTP service that training reaches over
`requests`.

For τ²-bench specifically, see `docs/TAU2_GRPO.md`: the benchmark must be pinned to
commit `c5b2d22`, and evaluation goes through tau2's own CLI. Those two choices are what
make numbers comparable to published baselines.

```bash
# τ²-bench
git clone https://github.com/sierra-research/tau2-bench.git
git -C tau2-bench checkout c5b2d22
conda create -y -p ./envs/tau2 python=3.12
./envs/tau2/bin/pip install -e tau2-bench -e src/envs/tau2 gymnasium

# the item-id files in data/ are committed, but this regenerates them
./envs/tau2/bin/python scripts/make_tau2_itemid.py \
    --domains retail airline telecom --split train --out data/
```

## Running

```bash
# a configured environment, baseline
bash examples/train/AgentGym-RL/sciworld_train.sh

# with the plan-forecast auxiliary loss
PLAN_FORECAST_ENABLE=True PLAN_FORECAST_COEF=0.01 \
  bash scripts/run_sciworld_grpo_train.sh

# tmux launchers bring up env servers + training together
bash scripts/launch_sciworld_grpo_tmux.sh
bash scripts/launch_tau2_grpo_tmux.sh

# τ²-bench with InfoPO's published hyperparameters, for a comparable run
bash scripts/launch_tau2_infopo_aligned.sh
```

τ²-bench evaluation, scoring against published numbers, and the trajectory viewer:

```bash
sbatch --export=ALL,TAU2_VERSION=infopo,MODEL_PATH=<hf-ckpt>,TAG=mine \
  scripts/sbatch_tau2_align.sh
python scripts/tau2_align_score.py --tag mine
python scripts/tau2_viz_trajectories.py "base=<tag>" "trained=<tag>" --out traj.html
```

## τ²-bench status

τ²-bench is the most recent addition and its results are negative so far, reported here
rather than omitted. Avg@4 on the test splits, gpt-4o-mini customer:

| | airline | retail | telecom |
|---|---|---|---|
| base | 8.8 | 9.4 | 8.1 |
| GRPO @ step 25 | **18.8** | 10.0 | 1.9 |
| RAGEN (published) | 15.0 | 17.5 | 17.5 |
| InfoPO (published) | 16.3 | 18.8 | 18.1 |

Plain GRPO lifts one domain at the others' expense — reward shaping only decides which
(binary → airline, dense → retail) — and telecom regresses under every configuration
tried. Telecom's failure is a decoding collapse rather than a statistical shortfall: the
policy emits bare `<tool_call>` tags with no JSON body and loops to the step cap, in 90%
of base episodes and 99% after training. Published RAGEN is itself GRPO-family and does
reach 15.0/17.5/17.5, so the gap is a missing mechanism; variance-based trajectory
filtering (`GRPO_FILTER_DEGENERATE=1`) and `info_grpo` are implemented here but not yet
run end to end.

`docs/TAU2_GRPO.md` carries the details, including two traps worth knowing before
comparing any numbers: absolute scores are not transferable across evaluation setups
(the same base model scores 8.8 on airline under gpt-4o-mini and 15.0 under a local 7B
customer), and a partially-complete tau2 run is biased toward short tasks in a way that
reads like signal.
