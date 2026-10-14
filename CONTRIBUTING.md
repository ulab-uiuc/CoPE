# Contributing

Thanks for your interest in CoPE. This file covers the mechanics; for what the project
is and how to run it, see the [README](README.md).

## Scope

Most of `src/verl/` is a vendored fork of [AgentGym-RL](https://github.com/PolarisDane/Agentgym-RL)'s
verl, which is itself a fork of [volcengine/verl](https://github.com/volcengine/verl).
Changes to that code are kept minimal so the fork stays rebaseable. Bug fixes that belong
upstream are better sent upstream.

The code that is ours, and where contributions are most useful:

```
src/verl/agent_trainer/ppo/    action_forecast, temporal_ensemble, ray_trainer wiring
scripts/                       launchers, evaluation harness, scoring
AgentGym/agentenv-tau2/        the τ²-bench environment server (in the submodule)
```

## Setup

```bash
poetry install --all-extras
poetry run pre-commit install
```

Training and the environment servers need their own environments; `poetry install` only
covers lint and test tooling. See the README's Setup section.

## Before opening a pull request

```bash
pre-commit run --all-files
pytest tests/
```

CI runs ruff, mypy, codespell, pre-commit, and pytest on Python 3.10.

New auxiliary signals must be **additive**: with the feature's flag off, the training
path has to be element-wise identical to stock GRPO. `tests/test_additive.py` asserts
this, and a new signal should extend it rather than work around it.

Comments and identifiers are English-only.

## Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (enforced by commitizen on `commit-msg`).
- Branch names follow `type/description`, e.g. `feature/add-llm-agents`.
- Keep a pull request to one type of change (docs, feature, fix) and as small as
  it can be.

## Reporting a problem

Open an issue with the [bug report](.github/ISSUE_TEMPLATE/bug_report.yml) or
[feature request](.github/ISSUE_TEMPLATE/feature_request.yml) template. For a training
run that misbehaves, include the launcher you used, the environment variables you set,
and the relevant lines from the run log.
