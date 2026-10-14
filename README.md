<h1 align="center">CoPE: Auxiliary Training Signals for Agent RL</h1>

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-orange)](https://github.com/ulab-uiuc/CoPE/pulls)
[![Arxiv](https://img.shields.io/badge/arXiv-Coming%20soon-b31b1b)](https://github.com/ulab-uiuc/CoPE)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

## Introduction

**CoPE** is a reinforcement-learning stack for LLM agents in interactive environments,
built on [AgentGym-RL](https://github.com/PolarisDane/Agentgym-RL)'s fork of
[verl](https://github.com/volcengine/verl).

Agentic RL benchmarks hand back a single reward at the end of a trajectory, which leaves
most of a batch with no gradient. CoPE adds an **action forecasting loss**: at every step
the agent predicts the next K actions it will take, and the realized rollout supervises
that prediction. It runs as a separate SFT-style pass that never touches the policy
gradient, so with the flag off the training path is stock GRPO.

Five environments are supported: SciWorld, ALFWorld, WebShop, AppWorld and τ²-bench.

## Get started

### Clone with the submodule

The `AgentGym` submodule carries every environment server and the `agentenv` base
classes, so it is required for any run.

```bash
git clone --recurse-submodules https://github.com/ulab-uiuc/CoPE.git
cd CoPE
# if you already cloned without it:
git submodule update --init AgentGym
```

### Set up the environments

Training and the environment servers run as separate processes with separate
interpreters — WebShop needs Python 3.8 and τ²-bench needs ≥3.12, neither of which the
3.10 trainer can host. Each environment is its own HTTP service that training reaches
over `requests`.

```bash
# the trainer (Python 3.10 + vllm); see AgentGym-RL's install guide for the details
conda create -n agentgym-rl python=3.10

# one env per environment you intend to run, e.g. SciWorld
conda create -n agentenv-sciworld python=3.10
```

### Configure

The launchers read everything from environment variables, so a variant is one extra
variable rather than a forked script.

```bash
export CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
export TRAIN_ENV=agentgym-rl           # conda env for the trainer
export SCIWORLD_ENV=agentenv-sciworld  # conda env for the env server
export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

API keys are read from files under `.secrets/` (git-ignored), never passed on a command
line — `scontrol show job` and any `env` dump in a log would otherwise expose them:

```bash
mkdir -p .secrets && echo "sk-..." > .secrets/openai_api_key
```

### Run

The `launch_*_grpo_tmux.sh` scripts bring up the environment servers and training
together, each in its own tmux session, and wait for the servers to become healthy
before training starts.

```bash
# baseline GRPO
bash scripts/launch_sciworld_grpo_tmux.sh

# with action forecasting
ACTION_FORECAST_ENABLE=True ACTION_FORECAST_COEF=0.01 \
  bash scripts/launch_sciworld_grpo_tmux.sh

# temporal ensembling, which distils the forecasts back into the policy
TE_ENABLE=True TE_LAMBDA=0.1 TE_ETA=0.5 \
  bash scripts/launch_sciworld_grpo_tmux.sh
```

Two published baselines are configured as plain config rather than env vars:

```bash
bash examples/train/AgentGym-RL/sciworld_train.sh
bash examples/train/ScalingInter-RL/sciworld_train.sh
```

Evaluation configs live in `examples/eval/`. τ²-bench runs under the published protocol
and has its own scoring and trajectory viewer — see
[`docs/TAU2_GRPO.md`](docs/TAU2_GRPO.md), which also carries the measured results
(negative so far, reported rather than omitted) and two traps worth knowing before
comparing any numbers against a paper.

## Developing

### Install the dev tooling

```bash
poetry install --all-extras
poetry run pre-commit install
```

This covers lint and tests only; the training stack is the conda environment above.

### New branch for each feature

`git checkout -b feature/feature-name` and PR to the `main` branch.

### Before committing

```bash
pre-commit run --all-files
pytest tests/
```

New auxiliary signals must be **additive**: with the feature's flag off, the training
path has to be element-wise identical to stock GRPO. Extend `tests/test_additive.py`
rather than working around it. Comments and identifiers are English-only.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the rest.

### Check the GitHub Action result

CI runs ruff, mypy, codespell, pre-commit and pytest on Python 3.10. Make sure it passes
before asking for review.

## Acknowledgements

Builds on [AgentGym-RL](https://github.com/PolarisDane/Agentgym-RL) and
[verl](https://github.com/volcengine/verl); `src/verl/` is a fork of the former. The
τ²-bench environment wraps
[tau2-bench](https://github.com/sierra-research/tau2-bench); RAGEN and InfoPO are the
published τ²-bench baselines the numbers in `docs/TAU2_GRPO.md` are compared against.

## License

MIT, see [LICENSE](LICENSE). The vendored code under `src/verl/` carries its own
Apache 2.0 headers from upstream verl.

<p align="center">
<a href="https://star-history.com/#ulab-uiuc/CoPE">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ulab-uiuc/CoPE&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ulab-uiuc/CoPE&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ulab-uiuc/CoPE&type=Date" />
  </picture>
</a>
</p>
