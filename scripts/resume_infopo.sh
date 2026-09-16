#!/usr/bin/env bash
# Resume the InfoPO alignment work once OpenAI credits are topped up.
#
# Everything below was validated end to end before run 21001 exhausted the account at
# step 8 of 50; nothing here needs rediscovery. Check credits first -- when they are out
# the trainer does not error, it just returns empty episodes and zero reward.
#
#   bash scripts/check_openai.sh        # all five layers must pass
#   bash scripts/resume_infopo.sh       # prints the commands, runs nothing

cat <<'EOF'
=== 1. verify credits ===
bash scripts/check_openai.sh

=== 2. three-domain GRPO training (was 8/50 steps when credits ran out) ===
# Paper hyperparameters from InfoPO examples/tau2/train.sh. Env servers run on the
# 2026-02 tau2 tree so training and evaluation share one benchmark.
sbatch --nodes=1 --gres=gpu:a100:8 --time=48:00:00 \
  --export=ALL,TAU2_ENV=${TAU2_ENV_PAPER},USERSIM_MODE=hosted,\
USERSIM_LLM=openai/gpt-4o-mini-2024-07-18,TAU2_DOMAIN=retail+airline+telecom,\
TAU2_TASK_SPLIT=train,TAU2_REWARD_SHAPE=binary,TAU2_REWARD_BASIS=all,\
TAU2_PROMPT_VARIANT=base,TRAIN_BATCH_SIZE=32,ROLLOUT_N=5,PPO_MINI_BATCH_SIZE=16,\
TOTAL_EPOCHS=10,MAX_ROUNDS=50,TAU2_MAX_STEPS=200,MAX_PROMPT_LENGTH=8192,\
MAX_RESPONSE_LENGTH=16384,MAX_TOKENS_PER_TURN=1024,POLICY_LR=1e-6,USE_KL_LOSS=False,\
KL_COEF=0,ENTROPY_COEF=0.001,ROLLOUT_GPU_MEMORY_UTILIZATION=0.50,ENVS_PER_GPU=4,\
SAVE_FREQ=50,EXP_NAME=tau2_infopo_align,\
TRAIN_FILE=${ROOT}/data/tau2_retail-airline-telecom_train.json \
  scripts/sbatch_tau2_grpo.sh

=== 3. evaluate a checkpoint against the paper's table ===
# Merge FSDP shards to HF first (scripts/model_merger.py), then:
sbatch --gres=gpu:a100:2 --time=12:00:00 \
  --export=ALL,TAU2_VERSION=infopo,TAG=trained,MODEL_PATH=<hf-ckpt>,\
DOMAINS="airline retail telecom",NUM_TRIALS=4,MAX_STEPS=200,MAX_CONCURRENCY=16 \
  scripts/sbatch_tau2_align.sh

=== reference points (Avg@4, test split, paper Table 1) ===
              telecom  retail  airline
  Qwen2.5 raw    14.4    13.1      7.5   <- ours: 8.1 / 9.4 / 8.8
  RAGEN          17.5    17.5     15.0   <- what standard GRPO should be compared to
  InfoPO         18.1    18.8     16.3   <- needs their info_grpo estimator, not GRPO

=== telecom: what is already ruled out (do not redo) ===
Telecom is 6.3 points low because the policy degenerates in long tool-calling
conversations -- bare <tool_call> tags with no JSON body, looping to the step cap. It
starts at the 24th message (median) in 100% of the 79 truncated episodes. Ruled out with
a local user simulator (no credit needed): step budget, context overflow (collapse hits
at 23% of the window), agent verbosity, the paper's user-simulator prompts, parser
failure, and all of frequency_penalty / repetition_penalty / temperature. See
TAU2_GRPO.md. Remaining untested ideas need credit or a different model.

=== first thing to run when credits return ===
Restart training (step 2 above). Airline already reproduces and retail is within
sampling noise, so the protocol is aligned; telecom's gap is a model failure mode, not a
harness bug, and should not block training.

Do NOT retry substituting the paper's Figure 19/20 user-simulator prompts -- that was
tried and took telecom from 8.1 to 1.6; see TAU2_GRPO.md.
EOF
