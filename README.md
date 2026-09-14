# GRPO on τ²-bench

Reinforcement learning for conversational agents on
[τ²-bench](https://github.com/sierra-research/tau2-bench), with the evaluation protocol
aligned to published results so numbers are directly comparable.

Two things here are reusable independently of the training results:

1. **A reproducible evaluation harness.** Getting τ²-bench numbers that can sit next to a
   paper's table turns out to depend on details that are easy to get wrong — the
   benchmark version, whether you drive tau2's own CLI or your own rollout loop, and
   which reward basis is in play. `docs/TAU2_GRPO.md` documents what we found and
   `scripts/sbatch_tau2_align.sh` encodes it.
2. **Turn-level advantage estimation** (`src/tau2_grpo/`) — an information-gain reward
   computed from a masked-feedback counterfactual, following
   [InfoPO](https://arxiv.org/abs/2603.00656), plus RAGEN-style variance-based
   trajectory filtering. Both are additive: with them disabled the training path is
   byte-identical to plain GRPO.

## What we measured

Qwen2.5-7B-Instruct, Avg@4 on the test splits, tau2 at the paper-era commit `c5b2d22`,
gpt-4o-mini customer — i.e. the protocol published baselines use.

| | airline | retail | telecom |
|---|---|---|---|
| base | 8.8 | 9.4 | 8.1 |
| GRPO @ step 25 | **18.8** | 10.0 | 1.9 |
| RAGEN (published) | 15.0 | 17.5 | 17.5 |
| InfoPO (published) | 16.3 | 18.8 | 18.1 |

**Plain GRPO does not lift all three domains.** Reward shaping only decides *which* one
benefits — binary reward lifts airline (+10.0), dense partial credit lifts retail (+6.2)
— and telecom regresses under every configuration we ran. Two candidate explanations
were tested and rejected: it is not the training/evaluation customer mismatch, and it is
not solely degenerate groups (dense cut those from 78.7% to 12.5% and merely moved the
winning domain).

Telecom's failure is visible rather than statistical: the policy starts emitting bare
`<tool_call>` tags with no JSON body and loops to the step cap, in 90% of base episodes
and 99% after training. `scripts/tau2_viz_trajectories.py` renders the transcripts so
this can be read directly.

Published RAGEN — itself a GRPO-family method, with variance-based trajectory filtering
and decoupled clipping — does reach 15.0/17.5/17.5, so the gap is a missing mechanism
rather than a ceiling. Trajectory filtering is implemented here
(`GRPO_FILTER_DEGENERATE=1`) but has not been run end to end.

## Layout

```
src/tau2_grpo/
  turn_structure.py      turn boundaries derived from the rollout's observation mask
  info_grpo.py           outcome + information-gain advantage, variance-gated
  intrinsic_reward.py    masked-feedback counterfactual KL
  env/                   the τ²-bench AgentGym environment server and client
  patches/               diff against AgentGym-RL's verl (see below)
scripts/                 training, evaluation, scoring, visualisation
docs/TAU2_GRPO.md        full method, protocol alignment, and the traps
data/                    item-id datasets (178 train / 100 test across three domains)
```

## Setup

τ²-bench needs Python ≥3.12 while the training stack is on 3.10 with vllm 0.6.3, so the
environment runs as its own HTTP service and training talks to it over `requests`.

```bash
# 1. tau2-bench at the paper-era commit -- version matters, see docs/TAU2_GRPO.md
git clone https://github.com/sierra-research/tau2-bench.git
git -C tau2-bench checkout c5b2d22

conda create -y -p ./envs/tau2 python=3.12
./envs/tau2/bin/pip install -e tau2-bench -e src/tau2_grpo/env gymnasium

# 2. training stack
git clone https://github.com/PolarisDane/Agentgym-RL.git
git -C Agentgym-RL apply ../src/tau2_grpo/patches/verl-tau2-grpo.patch
cp src/tau2_grpo/{turn_structure,info_grpo,intrinsic_reward}.py \
   Agentgym-RL/verl/trainer/ppo/

# 3. an OpenAI key for the customer simulator, never in argv or the environment
mkdir -p .secrets && chmod 700 .secrets
printf '%s' "$OPENAI_API_KEY" > .secrets/openai_api_key && chmod 600 .secrets/openai_api_key
bash scripts/check_openai.sh      # five layers; all must pass
```

The verl changes ship as a patch rather than a vendored copy: they are five small edits
to a large upstream tree, and a patch keeps them reviewable and rebaseable.

## Running

```bash
# train: three domains jointly, 178 tasks, paper hyperparameters
sbatch scripts/sbatch_tau2_grpo.sh

# evaluate against published numbers (tau2's own CLI, EvaluationType.ALL)
sbatch --export=ALL,TAU2_VERSION=infopo,MODEL_PATH=<hf-ckpt>,TAG=mine \
  scripts/sbatch_tau2_align.sh
python scripts/tau2_align_score.py --tag mine

# read the transcripts
python scripts/tau2_viz_trajectories.py "base=<tag>" "trained=<tag>" --out traj.html
```

Turn-level estimation is opt-in: `algorithm.adv_estimator=info_grpo` for the
information-gain term, `GRPO_FILTER_DEGENERATE=1` for trajectory filtering.

## Things that cost us time

Recorded in full in `docs/TAU2_GRPO.md`; the ones most likely to bite a reader:

- **tau2 version dominates.** v1.0.1 rewrote the user simulator, added LLM-judge
  reviewers, and changed 3.5M lines of task data. Baseline error against published
  numbers went from ±6–9 points in inconsistent directions down to airline reproducing
  exactly, purely by checking out `c5b2d22`.
- **Absolute scores are not transferable across evaluation setups.** The same base model
  scores 8.8 on airline under gpt-4o-mini and 15.0 under a local 7B customer. An
  unpaired comparison across two setups produced a +10.0 that a paired control reversed
  in sign. Every run writes `eval_config.json`; diff it before comparing.
- **Never read a tau2 run before it finishes.** Results accumulate in task order and
  short tasks land first, so a partial file is biased toward easy cases in a way that
  looks like signal. The scorer refuses to score partial runs for this reason.
- Three upstream tau2 bugs are patched here: a duplicate top-level `name` on tool calls
  that current vLLM rejects, a `max_context_tokens` argument that only exists in some
  forks, and a resume prompt that blocks on stdin under a batch scheduler.
