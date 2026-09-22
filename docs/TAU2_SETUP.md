# τ²-bench: environment setup

Everything the two launchers (`scripts/launch_tau2_grpo_tmux.sh`, `scripts/launch_tau2_cope_tmux.sh`)
need before they can start, in the order to set it up. `docs/TAU2_GRPO.md` explains the
training protocol and the numbers; this file is only about getting a machine ready.

## 0. What runs where

A τ² training run is three kinds of process, and they cannot share a Python:

| process | Python | what it is |
|---|---|---|
| trainer (`verl.agent_trainer.main_ppo`) | 3.10, `TRAIN_ENV` | GRPO + vLLM rollout, one process per training GPU |
| env cluster (`tau2-env`, one per port) | 3.12, `TAU2_ENV` | τ²-bench environments, reached over HTTP |
| customer | hosted `gpt-4o-mini` **or** a local vLLM server | the simulated user τ² talks to |

`scripts/run_tau2_pipeline.sh` starts all three and tears them down on exit; the launchers
run it in tmux. The pipeline checks every prerequisite below and refuses to start with a
specific message when one is missing (`DRY_RUN=1` runs only the checks).

## 1. Repository and the AgentGym submodule

```bash
git clone https://github.com/ulab-uiuc/CoPE.git && cd CoPE
git submodule update --init AgentGym          # agentenv base classes + agentenv-tau2 env server
```

The τ² env server and client live in the submodule and need its `native` prompt variant
(PolarisDane/Agentgym#3). Until that PR is merged, `git submodule update` cannot fetch the
pinned commit from the upstream URL; fetch the PR ref instead:

```bash
git -C AgentGym fetch origin pull/3/head && git -C AgentGym checkout FETCH_HEAD
```

(If SSH to GitHub is blocked: `git config submodule.AgentGym.url https://github.com/PolarisDane/Agentgym.git`
before the update.)

## 2. τ²-bench at the pinned commit

```bash
git clone https://github.com/sierra-research/tau2-bench.git
git -C tau2-bench checkout c5b2d22
bash scripts/patch_tau2_bench.sh     # one upstream fix for tau2's CLI vs current vLLM; idempotent
```

The commit matters: task sets and evaluation changed later, and `c5b2d22` is what the
published numbers (and this repo's) are measured against. `TAU2_BENCH_DIR` overrides the
default `./tau2-bench`; `TAU2_DATA_DIR` defaults to `${TAU2_BENCH_DIR}/data`.

## 3. `TAU2_ENV`: the env-server environment (Python 3.12)

```bash
conda create -y -p ./envs/tau2 python=3.12
./envs/tau2/bin/pip install -e tau2-bench -e AgentGym/agentenv-tau2 gymnasium
./envs/tau2/bin/tau2-env --help            # the server entry point the pipeline looks for
```

`./envs/tau2` is the default `TAU2_ENV`. Known-good versions here: tau2 0.2.1.dev0 (the
checkout above), agentenv_tau2 0.0.1, litellm 1.101.0, openai 2.54.0, fastapi 0.141.1,
uvicorn 0.53.0. Install with `PYTHONNOUSERSITE=1` if your `~/.local` has packages that
shadow these (the pipeline runs everything with it set).

## 4. `TRAIN_ENV`: the training environment (Python 3.10)

The trainer is this repository's `src/verl` (AgentGym-RL's verl fork); the scripts put it on
`PYTHONPATH`, nothing is pip-installed from this repo. What the env must provide is the
AgentGym-RL training stack — see `docs/agentgym-rl/start/install.rst` — with one of:

- vLLM 0.6.3 (verl's vendored engine), or
- vLLM >= 0.6.6 (the SPMD engine; required on Blackwell / sm_120 GPUs).

Known-good on RTX PRO 6000 (driver 580): Python 3.10, torch 2.7.1+cu128, vllm 0.9.2,
ray 2.55.1, transformers 4.51.3, flash-attn 2.8.3. Point `TRAIN_ENV` at the env's prefix
and `CONDA_SH` at your `conda.sh` (auto-detected under the usual anaconda/miniconda paths):

```bash
export TRAIN_ENV=/path/to/conda/envs/agentgym
export CONDA_SH=/path/to/etc/profile.d/conda.sh
```

## 5. The policy model

```bash
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct      # a local snapshot directory
```

The pipeline runs with `HF_HUB_OFFLINE=1`; set `HF_HUB_OFFLINE=0` once to let
`MODEL_PATH=Qwen/Qwen2.5-7B-Instruct` download into `HF_HOME`. The rollout is TP=1, so the
model must fit one GPU next to the training shard (7B on 80–97 GB cards is the tested point;
see "Model size" in `docs/TAU2_GRPO.md`).

## 6. The customer (user simulator)

**Hosted (default, InfoPO's protocol):** `gpt-4o-mini` at temperature 0.7 through litellm.

```bash
mkdir -p .secrets && install -m 600 /dev/null .secrets/openai_api_key
printf '%s' 'sk-...' > .secrets/openai_api_key      # gitignored; KEY_FILE overrides the path
```

A step is 160 trajectories × ~18 turns ≈ 1,500 customer calls; a 50-step run cost about
$50 in gpt-4o-mini credit. The env-server logs (`runlogs/<exp>/env_cluster/env_*.log`)
show authentication/quota failures — a failing key reads as a reward collapse otherwise.

**Local (no API credit):** a Qwen customer on its own GPU, `USERSIM_MODE=local USERSIM_GPU=<id>`
with `<id>` outside `CUDA_VISIBLE_DEVICES`. Deltas are meaningful, absolute scores are not
comparable to the paper. The pipeline picks `hosted` when the key file is readable, else `local`.

## 7. Task lists

Training reads an item-id file that maps a running index to `(domain, task)`; the env server
resolves the same index with the same rule, so the file and the server's `TAU2_DOMAIN` /
`TAU2_TASK_SPLIT` must agree (the presets set both). The files in `data/` are committed;
regenerate them inside `TAU2_ENV`:

```bash
./envs/tau2/bin/python scripts/make_tau2_itemid.py --domains retail airline telecom --split train --out data/
```

The `infopo` preset uses `data/tau2_retail-airline-telecom_train.json` (178 tasks: retail 74,
airline 30, telecom 74); the `repo` preset uses `data/tau2_retail_train.json`.

## 8. GPUs, ports, disk

- `CUDA_VISIBLE_DEVICES` lists the training GPUs (default `0,1,2,3`); the InfoPO sequence
  budget (prompt 8192 + response 16384) with micro-batch 1 needs ~80 GB per card.
  `TRAIN_BATCH_SIZE × ROLLOUT_N` and `PPO_MINI_BATCH_SIZE × ROLLOUT_N` must be divisible by
  the number of training GPUs (the pipeline checks).
- Listening ports: customer 20301, env servers 20401 + one per env (`ENVS_PER_GPU` × GPUs,
  default 4 per GPU). They sit below the kernel's ephemeral range (32768–60999) on purpose.
- Checkpoints are FSDP shards + optimizer state, ~86 GB each for 7B, written every
  `SAVE_FREQ` steps (15 in the preset) and at the end. Put `CKPT_DIR` on a large disk.
  Ray's session logs go to `/tmp/ray` and its object store to `/dev/shm`.

## 9. Check, then run

```bash
DRY_RUN=1 bash scripts/launch_tau2_grpo_tmux.sh   # prerequisites + resolved config, no launch
bash scripts/launch_tau2_grpo_tmux.sh             # plain GRPO
bash scripts/launch_tau2_cope_tmux.sh             # CoPE: GRPO + action forecasting (weight 0.1)
```

Output: `runlogs/<exp>/pipeline.log` (everything), `runlogs/<exp>/train.log` (trainer),
`runlogs/<exp>/main_task.out` (a link to Ray's log of the training task — the authoritative
per-step metrics, because Ray's forwarding to the driver can stop mid-run),
`runlogs/<exp>/rollout_logs/step<N>/` (every trajectory with its reward),
`runlogs/<exp>/env_cluster/`, checkpoints under `CKPT_DIR`. `tmux attach -t tau2_grpo` /
`tau2_cope` to watch; `tmux kill-session -t <name>` stops the whole tree.

To exercise the pieces separately (a canned customer, one env server, the client) see
"Verification" in `docs/TAU2_GRPO.md`.

## 10. When it does not start

| symptom | cause / fix |
|---|---|
| `FATAL: ... tau2-env missing` | step 3 not done, or `TAU2_ENV` points elsewhere |
| `FATAL: this AgentGym checkout has no 'native' prompt variant` | submodule not at PR #3's commit (step 1) |
| `FATAL: MODEL_PATH ... not a local directory` | step 5; or set `HF_HUB_OFFLINE=0` once |
| `FATAL: USERSIM_MODE=hosted needs an API key` | step 6 |
| `ImportError: ... vLLMRollout` | vLLM version not recognised; 0.6.3 or >= 0.6.6 |
| `Expandable segments are not compatible with memory pool` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` set in the shell with vLLM >= 0.6.6; unset it |
| env server dies with `EADDRINUSE` | a port inside the ephemeral range; keep `BASE_PORT` < 32768 |
| CUDA OOM in the actor backward | `PPO_MICRO_BATCH_SIZE_PER_GPU=1` (preset default); or lower `MAX_RESPONSE_LENGTH` |
| reward collapses to ~0 mid-run | check `env_cluster/env_*.log` for `AuthenticationError` / `insufficient_quota` |
| no `step:N` lines in `train.log` while GPUs are busy | read `main_task.out` instead (Ray stopped forwarding) |
