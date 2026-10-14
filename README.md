# CoPE

Reinforcement learning for LLM agents in interactive environments, built on
[AgentGym-RL](https://github.com/PolarisDane/Agentgym-RL)'s fork of
[verl](https://github.com/volcengine/verl).

Agentic RL benchmarks hand back one reward at the end of a trajectory. On τ²-bench, 79%
of GRPO groups have no within-group spread, so four fifths of a batch carries no
gradient at all. CoPE adds auxiliary training signals that are dense where the outcome
reward is silent — per turn, or per predicted-action — without redefining what counts as
success.

## The signals

| module | idea | enabled by |
|---|---|---|
| [`action_forecast.py`](src/verl/agent_trainer/ppo/action_forecast.py) | at each step the model predicts the next K actions it will take; the realized rollout supervises the prediction, as a separate SFT-style pass that never touches PG | `ACTION_FORECAST_ENABLE=True` |
| [`temporal_ensemble.py`](src/verl/agent_trainer/ppo/temporal_ensemble.py) | aggregate those forecasts across time into a reference distribution `q_t` and distill the policy back toward it with a KL term | `TE_ENABLE=True` |
| [`info_grpo.py`](src/verl/trainer/ppo/info_grpo.py) | turn-level information gain: how far seeing an observation moved the next-action distribution, measured against a masked-feedback counterfactual | `algorithm.adv_estimator=info_grpo` |

All of them are additive. With every flag off, the training path is element-wise
identical to stock GRPO — which the tests assert, so a run of this repo reproduces the
baseline before it reproduces the method:

```bash
pytest tests/test_additive.py
```

## Environments

| environment | train | eval | tmux launcher |
|---|---|---|---|
| SciWorld | ✓ | ✓ | ✓ |
| BabyAI | ✓ | ✓ | ✓ |
| τ²-bench | ✓ | ✓ | ✓ |
| AppWorld | ✓ | ✓ | ✓ |
| TextCraft | ✓ | ✓ | |
| WebArena | ✓ | ✓ | |
| SearchQA | ✓ | ✓ | |
| ALFWorld | | | ✓ |
| WebShop | | | ✓ |

Environment servers run as separate processes from training. Several need Python
versions that conflict with the training stack — WebShop is on 3.8, τ²-bench needs
≥3.12 against a 3.10 trainer — so each runs as its own HTTP service that training
reaches over `requests`.

## Setup

```bash
git clone --recurse-submodules https://github.com/ulab-uiuc/CoPE.git
cd CoPE
```

The `AgentGym` submodule carries every environment server and the `agentenv` base
classes, so it is required for any run. If you cloned without `--recurse-submodules`:

```bash
git submodule update --init AgentGym
```

Training runs in a Python 3.10 environment with vllm; `poetry install` sets up only the
lint and test tooling. For the training stack, follow
[AgentGym-RL's install guide](https://github.com/PolarisDane/Agentgym-RL).

<details>
<summary>τ²-bench</summary>

```bash
git clone https://github.com/sierra-research/tau2-bench.git
git -C tau2-bench checkout c5b2d22
conda create -y -p ./envs/tau2 python=3.12
./envs/tau2/bin/pip install -e tau2-bench -e AgentGym/agentenv-tau2 gymnasium
```

The pin to `c5b2d22` is not optional: τ²-bench has since become τ³-bench, and evaluation
goes through tau2's own CLI. Those two choices are what keep numbers comparable to
published baselines. See [`docs/TAU2_GRPO.md`](docs/TAU2_GRPO.md).

Item-id files for τ²-bench are committed under `data/`; to regenerate them:

```bash
./envs/tau2/bin/python scripts/make_tau2_itemid.py \
    --domains retail airline telecom --split train --out data/
```

For the other environments the item-id files are generated per checkout, since they
derive from each benchmark's own task list rather than being authored here.
</details>

## Running

The `launch_*_grpo_tmux.sh` scripts bring up the environment servers and training
together, each in its own tmux session:

```bash
bash scripts/launch_sciworld_grpo_tmux.sh
bash scripts/launch_tau2_grpo_tmux.sh
```

Method variants are environment variables on the same launcher, not forked copies of it:

```bash
ACTION_FORECAST_ENABLE=True ACTION_FORECAST_COEF=0.01 \
  bash scripts/launch_sciworld_grpo_tmux.sh

TE_ENABLE=True TE_LAMBDA=0.1 TE_ETA=0.5 \
  bash scripts/launch_sciworld_grpo_tmux.sh
```

Two baselines are configured alongside, as plain config rather than env vars:

```bash
bash examples/train/AgentGym-RL/sciworld_train.sh
bash examples/train/ScalingInter-RL/sciworld_train.sh
```

Evaluation configs live in `examples/eval/`. For τ²-bench, evaluation runs under the
published protocol and has its own scoring and trajectory viewer:

```bash
sbatch --export=ALL,TAU2_VERSION=infopo,MODEL_PATH=<hf-ckpt>,TAG=mine \
  scripts/sbatch_tau2_align.sh
python scripts/tau2_align_score.py --tag mine
python scripts/tau2_viz_trajectories.py "base=<tag>" "trained=<tag>" --out traj.html
```

## Layout

```
src/verl/                 training framework (fork of AgentGym-RL's verl)
  agent_trainer/ppo/      action_forecast, temporal_ensemble, ray_trainer
  trainer/ppo/            info_grpo, intrinsic_reward, turn_structure
  workers/                actor, rollout, FSDP workers
AgentGym/                 submodule: every environment's server and client
  agentenv-tau2/          the τ²-bench environment server
  agentenv/envs/tau2.py   the τ²-bench client
examples/train/           per-environment configs for the two baselines
examples/eval/            evaluation configs
scripts/                  launchers, evaluation harness, scoring, visualisation
data/                     τ²-bench item-id datasets
docs/TAU2_GRPO.md         τ²-bench protocol alignment and measured results
tests/                    the additive claim, asserted element-wise
```

## Status

τ²-bench is the most recent environment added and results there are **negative so far**,
reported rather than omitted. Plain GRPO lifts exactly one domain at the other two's
expense — reward shaping only decides which — and telecom regresses under every
configuration tried. `info_grpo` is implemented and unit-tested but has not been run end
to end; the shipped launchers pass `algorithm.adv_estimator=grpo`.

[`docs/TAU2_GRPO.md`](docs/TAU2_GRPO.md) has the numbers, the two falsified explanations,
and two traps worth knowing before comparing against anything: absolute scores are not
transferable across evaluation setups (the same base model scores 8.8 on airline under a
gpt-4o-mini customer and 15.0 under a local 7B one), and a partially-complete tau2 run is
biased toward short tasks in a way that reads like signal.

## Acknowledgements

Builds on [AgentGym-RL](https://github.com/PolarisDane/Agentgym-RL) and
[verl](https://github.com/volcengine/verl); `src/verl/` is a fork of the former, Apache
2.0 licensed. The τ²-bench environment wraps
[tau2-bench](https://github.com/sierra-research/tau2-bench). The information-gain
objective follows InfoPO (arXiv 2603.00656).

## License

Apache 2.0. See [LICENSE](LICENSE).
