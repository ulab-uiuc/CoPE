# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import random
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.agent_trainer.ppo import core_algos
from verl.agent_trainer.ppo.turn_spans import compute_turn_boundaries
from verl.agent_trainer.ppo.plan_forecast import (
    build_plan_forecast_batch,
    parse_k_schedule,
    active_k_range,
)
# verl.agent_trainer.ppo.sft_ablation is not present in this checkout (never
# committed), and importing it at module scope makes *every* run fail on import. The
# feature is off by default, so bind the symbol lazily and only fail if it is enabled.
try:
    from verl.agent_trainer.ppo.sft_ablation import (
        build_sft_ablation_batch,
    )
except ImportError:
    build_sft_ablation_batch = None
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.agent_dataset.rl_dataset import RLHFDataset, collate_fn
from abc import ABC, abstractmethod

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def find_latest_ckpt_path_aistudio(path, directory_format="global_step_{}"):
    if path is None:
        return None

    from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        return None

    from aistudio_checkpoint.aistudio_base_checkpointer import load_checkpoint
    with open(tracker_file, "r") as f:
        iteration, resuming_path = f.read().split("\n")
    ckpt_path = os.path.join(load_checkpoint(resuming_path=resuming_path), directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch['response_mask']

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def _filter_degenerate_groups(token_level_rewards, response_mask, index, epsilon=1e-6):
    """Zero the response mask for samples whose GRPO group has no within-group spread.

    Returns (mask, n_groups_dropped, n_groups_total).
    """
    import collections
    scores = token_level_rewards.sum(dim=-1)
    by_group = collections.defaultdict(list)
    for i in range(scores.shape[0]):
        by_group[index[i]].append(i)
    mask = response_mask.clone()
    dropped = 0
    for _gid, rows in by_group.items():
        vals = scores[rows]
        if vals.numel() > 1 and vals.std() > epsilon:
            continue
        dropped += 1
        for r in rows:
            mask[r] = 0
    return mask, dropped, len(by_group)


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        response_mask = data.batch['response_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        # Variance-based trajectory filtering (RAGEN), zeroing the mask of samples whose
        # group has no within-group spread.
        #
        # Read this before enabling it. With ppo_micro_batch_size_per_gpu=1 the policy
        # loss is a masked_mean *within each sample*, then divided by a fixed
        # gradient_accumulation. A degenerate sample already contributes 0 to the
        # numerator (its advantage is 0) while still counting in that fixed denominator,
        # so filtering does not redirect the gradient -- it scales it. On tau2, where
        # ~79% of groups are degenerate, enabling this is close to a 5x learning-rate
        # increase and little else. Prefer changing the learning rate, which is at least
        # legible.
        #
        # It also cannot simply zero the mask: a fully masked sample makes masked_mean
        # divide by zero, and with micro_bsz=1 that is most micro-batches. Run 23462
        # produced actor/grad_norm=nan from step 1 and destroyed the policy within 8
        # steps. The guard below keeps the whole batch when filtering would leave too
        # little to train on.
        if os.environ.get('GRPO_FILTER_DEGENERATE') == '1':
            filtered, n_drop, n_tot = _filter_degenerate_groups(
                token_level_rewards, response_mask, index)
            frac = n_drop / max(1, n_tot)
            # Below this many surviving groups the remaining gradient is one or two
            # trajectories wide and the update is pure variance, so keep the batch
            # intact and say so rather than taking a step off a single sample.
            if n_tot - n_drop >= max(2, int(0.1 * n_tot)):
                data.batch['response_mask'] = filtered
                data.meta_info['grpo_filter_applied'] = 1.0
            else:
                print(f"[grpo-filter] only {n_tot - n_drop}/{n_tot} groups survive; "
                      f"keeping the full batch this step")
                data.meta_info['grpo_filter_applied'] = 0.0
            data.meta_info['grpo_filtered_frac'] = frac
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'info_grpo':
        # GRPO's outcome term plus a turn-level information gain. The intrinsic tensor
        # is produced during the actor update (it needs a forward pass); when it is
        # absent this falls back to exactly plain GRPO, so the branch is safe to select
        # before the producer side is wired up.
        from verl.trainer.ppo.info_grpo import compute_info_grpo_advantage
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        intrinsic = data.batch.get('token_level_intrinsic_rewards')
        if intrinsic is None:
            intrinsic = torch.zeros_like(token_level_rewards)
        advantages, returns = compute_info_grpo_advantage(
            token_level_rewards=token_level_rewards,
            token_level_intrinsic_rewards=intrinsic,
            eos_mask=response_mask,
            index=index,
            intrinsic_weight=float(os.environ.get('INFO_INTRINSIC_WEIGHT', 0.1)),
            gate_temperature=float(os.environ.get('INFO_GATE_TEMP', 0.05)),
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


class RoundsScheduler(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_global_steps(self, global_steps: int):
        raise NotImplementedError

    @abstractmethod
    def get_rounds(self):
        raise NotImplementedError
    

class FixedRoundsScheduler(RoundsScheduler):
    def __init__(self, rounds: int):
        self.max_rounds = rounds

    def step(self):
        pass

    def set_global_steps(self, global_steps: int):
        pass

    def get_rounds(self):
        return self.max_rounds


class StepRoundsScheduler(RoundsScheduler):
    def __init__(self, steps_scaling_inter: int, rounds_ls: List[int]):
        self.rounds_ls = rounds_ls
        self.steps_scaling_inter = steps_scaling_inter
        self.max_rounds = rounds_ls[0]
        self.current_stage = 0
        self.global_steps = 1 # start from 1

    def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps
        if (self.global_steps // self.steps_scaling_inter < len(self.rounds_ls)):
            self.current_stage = self.global_steps // self.steps_scaling_inter
        else:
            self.current_stage = len(self.rounds_ls) - 1
        self.max_rounds = self.rounds_ls[self.current_stage]
    
    def step(self):
        if self.current_stage + 1 < len(self.rounds_ls) and self.global_steps % self.steps_scaling_inter == 0:
            self.current_stage += 1
            self.max_rounds = self.rounds_ls[self.current_stage]
        self.global_steps += 1

    def get_rounds(self):
        return self.max_rounds


class CoeffScheduler(ABC):
    """Anneal an auxiliary-loss coefficient over training steps.

    Drive it either by calling ``step()`` once per step or by pushing the trainer's
    own counter with ``set_global_steps`` — the latter also makes resume correct for
    free, since the counter is restored from the checkpoint.
    """

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def set_global_steps(self, global_steps: int):
        raise NotImplementedError

    @abstractmethod
    def get_coeff(self):
        raise NotImplementedError


class FixedCoeffScheduler(CoeffScheduler):
    def __init__(self, coeff: float):
        self.coeff = coeff

    def step(self):
        pass

    def set_global_steps(self, global_steps: int):
        pass

    def get_coeff(self):
        return self.coeff


class LinearCoeffScheduler(CoeffScheduler):
    def __init__(self, start_coeff: float, end_coeff: float, horizon: int):
        self.start_coeff = start_coeff
        self.end_coeff = end_coeff
        self.horizon = horizon
        self.current_coeff = start_coeff
        self.global_steps = 0

    def step(self):
        self.global_steps += 1
        self._update()

    def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps
        self._update()

    def _update(self):
        if self.horizon <= 0:
            self.current_coeff = self.end_coeff
        else:
            fraction = min(self.global_steps / self.horizon, 1.0)
            self.current_coeff = self.start_coeff + fraction * (self.end_coeff - self.start_coeff)

    def get_coeff(self):
        return self.current_coeff


class PowerCoeffScheduler(CoeffScheduler):
    def __init__(self, start_coeff: float, end_coeff: float, horizon: int, power: float = 2.0):
        self.start_coeff = start_coeff
        self.end_coeff = end_coeff
        self.horizon = horizon
        self.power = power
        self.current_coeff = start_coeff
        self.global_steps = 0

    def step(self):
        self.global_steps += 1
        self._update()

    def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps
        self._update()

    def _update(self):
        if self.horizon <= 0:
            self.current_coeff = self.end_coeff
        else:
            fraction = min(self.global_steps / self.horizon, 1.0)
            # Use power function for non-linear growth: (step/horizon)^power
            self.current_coeff = self.start_coeff + (fraction ** self.power) * (self.end_coeff - self.start_coeff)

    def get_coeff(self):
        return self.current_coeff


class CutoffCoeffScheduler(CoeffScheduler):
    def __init__(self, start_coeff: float, end_coeff: float, cutoff_step: int):
        self.start_coeff = start_coeff
        self.end_coeff = end_coeff
        self.cutoff_step = cutoff_step
        self.current_coeff = start_coeff
        self.global_steps = 0

    def step(self):
        self.global_steps += 1
        self._update()

    def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps
        self._update()

    def _update(self):
        if self.global_steps >= self.cutoff_step:
            self.current_coeff = self.end_coeff
        else:
            self.current_coeff = self.start_coeff

    def get_coeff(self):
        return self.current_coeff


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    task_scores = batch.batch["task_scores"].sum(-1)
    task_rounds = batch.batch["task_rounds"]

    response_length = batch.batch['response_mask'].sum(-1).float()
    prompt_length = batch.batch['attention_mask'].sum(-1).float() - response_length

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    response_mask = batch.batch['response_mask'].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # task score
        'critic/task_score/mean':
            torch.mean(task_scores).detach().item(),
        'critic/task_score/max':
            torch.max(task_scores).detach().item(),
        'critic/task_score/min':
            torch.min(task_scores).detach().item(),
        # task round
        'critic/task_round/mean':
            torch.mean(task_rounds).detach().item(),
        'critic/task_round/max':
            torch.max(task_rounds).detach().item(),
        'critic/task_round/min':
            torch.min(task_rounds).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    num_overall_tokens = torch.sum(batch.batch['attention_mask']).item()
    num_response_tokens = torch.sum(batch.batch['response_mask']).item()

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        else:
            raise NotImplementedError

        # Plan FORMAT reward (default OFF): per-turn shaping bonus on the advantage
        # for emitting a well-formed Thought->Plan->Action turn, to counter the
        # decay of inline-plan behaviour under RL.
        _pa = self.config.actor_rollout_ref.actor
        self.plan_format_reward_enable = bool(_pa.get('plan_format_reward_enable', False))
        self.plan_format_reward_coef = float(_pa.get('plan_format_reward_coef', 0.05))
        self.plan_format_reward_baseline = float(_pa.get('plan_format_reward_baseline', 0.5))
        self.plan_format_reward_clip = float(_pa.get('plan_format_reward_clip', 0.0))
        # penalty_only (default True): only penalize turns that DROP the plan,
        # never reward keeping it -> dormant while compliant, no length/positive bias.
        self.plan_format_reward_penalty_only = bool(_pa.get('plan_format_reward_penalty_only', True))
        # warmup: keep the format reward OFF until global_step >= warmup_steps, so
        # the policy can learn the task (and compress) first before plan pressure.
        self.plan_format_reward_warmup_steps = int(_pa.get('plan_format_reward_warmup_steps', 10))
        self.plan_format_reward_k = int(_pa.get('plan_format_reward_k',
                                                int(self.config.data.get('plan_inline_k', 3))
                                                if hasattr(self.config, 'data') else 3))

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=self.config.data,
            agentgym_config=self.config.actor_rollout_ref.agentgym,
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        assert len(self.train_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if self.config.algorithm.rounds_ctrl.type == 'fixed':
            self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
        elif self.config.algorithm.rounds_ctrl.type == 'scaling_inter_stepwise':
            self.rounds_scheduler = StepRoundsScheduler(steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
                                                   rounds_ls=self.config.algorithm.rounds_ctrl.rounds)
        else:
            raise NotImplementedError
        print(f'Total training steps: {self.total_training_steps}')

        # Plan-forecast coefficient scheduler (anneal plan_forecast_coef). Knobs live
        # on the actor config next to the other plan_forecast_* settings.
        self.plan_forecast_coeff_scheduler = self._build_plan_forecast_scheduler()

        # plan_forecast and the sft-ablation (RFT) control are MUTUALLY EXCLUSIVE:
        # they share one optimizer path and are meant to be A/B'd, never combined.
        _acfg = self.config.actor_rollout_ref.actor
        if bool(_acfg.get('plan_forecast_enable', False)) and bool(_acfg.get('sft_ablation_enable', False)):
            raise ValueError("plan_forecast_enable and sft_ablation_enable are mutually "
                             "exclusive — enable exactly one.")

        # plan_forecast horizon-growth schedule (empty = off -> fixed plan_forecast_k).
        # Parsed once here so a malformed spec fails fast at init instead of being
        # swallowed by the forecast dispatch's try/except at every step.
        self._pf_k_stages = parse_k_schedule(str(_acfg.get('plan_forecast_k_schedule', '') or ''))

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        if self.config.trainer.storage_mode == 'aistudio':
            from aistudio_checkpoint.aistudio_mnt_checkpointer import AistudioMntCheckpointer
            ckpter = AistudioMntCheckpointer()
            save_dir = ckpter.get_save_dir(step=self.global_steps)
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(save_dir,
                                                    f'global_step_{self.global_steps}')
        elif self.config.trainer.storage_mode == 'local':
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                    f'global_step_{self.global_steps}')
        else:
            raise NotImplementedError
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save,
                                              max_ckpt_to_keep=self.config.trainer.max_local_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save,
                                           max_ckpt_to_keep=self.config.trainer.max_local_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            if self.config.trainer.storage_mode == 'aistudio':
                f.write(str(self.global_steps) + "\n" + ckpter.commit(memo=self.config.trainer.experiment_name))
            elif self.config.trainer.storage_mode == 'local':
                f.write(str(self.global_steps))
            else:
                raise NotImplementedError

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            if self.config.trainer.storage_mode == 'aistudio':
                global_step_folder = find_latest_ckpt_path_aistudio(checkpoint_folder)  # None if no latest
            elif self.config.trainer.storage_mode == 'local':
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            else:
                raise NotImplementedError

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        self.rounds_scheduler.set_global_steps(self.global_steps)

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _build_plan_forecast_scheduler(self):
        """Coeff scheduler for plan_forecast_coef. anneal in {fixed,linear,power,
        cutoff}; reuses the generic *CoeffScheduler classes. start =
        plan_forecast_coef."""
        actor_cfg = self.config.actor_rollout_ref.actor
        anneal = str(actor_cfg.get('plan_forecast_coef_anneal', 'fixed')).lower()
        start = float(actor_cfg.get('plan_forecast_coef', 0.0))
        end = float(actor_cfg.get('plan_forecast_coef_end', 0.0))
        horizon = int(actor_cfg.get('plan_forecast_coef_horizon', 0))
        if anneal in ('fixed', 'none', ''):
            return FixedCoeffScheduler(coeff=start)
        if anneal == 'linear':
            return LinearCoeffScheduler(start_coeff=start, end_coeff=end, horizon=horizon)
        if anneal == 'power':
            return PowerCoeffScheduler(start_coeff=start, end_coeff=end, horizon=horizon,
                                                 power=float(actor_cfg.get('plan_forecast_coef_power', 2.0)))
        if anneal == 'cutoff':
            return CutoffCoeffScheduler(start_coeff=start, end_coeff=end,
                                                  cutoff_step=int(actor_cfg.get('plan_forecast_coef_cutoff_step', 0)))
        raise NotImplementedError(f"unknown plan_forecast_coef_anneal: {anneal}")

    def _build_plan_forecast_dataproto(self, batch: DataProto, coef: float):
        """Build a plan-forecast SFT DataProto (predict realized next-K actions).

        ``coef`` is the CURRENT (annealed) coefficient. Returns (dataproto, meta)
        or (None, meta). ``None`` when disabled, coef<=0, no ``rollout_messages``,
        or no qualifying step. Wins-gate uses per-traj reward from ``traj_return``
        (fallback: token_level_scores.sum)."""
        actor_cfg = self.config.actor_rollout_ref.actor
        if not actor_cfg.get('plan_forecast_enable', False):
            return None, {}
        if coef <= 0:
            return None, {'plan_forecast/coef': 0.0}

        messages_list = batch.non_tensor_batch.get('rollout_messages', None)
        if messages_list is None:
            return None, {'plan_forecast/skipped_no_msgs': 1.0}

        rewards = None
        if 'traj_return' in batch.batch.keys():
            rewards = batch.batch['traj_return'].tolist()
        elif 'token_level_scores' in batch.batch.keys():
            rewards = batch.batch['token_level_scores'].sum(dim=-1).tolist()

        # target: 'action' (block2-success, predict realized next-K actions) or
        # 'subgoal' (predict next-K hindsight-confirmed achieved sub-goals).
        target = str(actor_cfg.get('plan_forecast_target', 'action')).lower()
        # seq: 'separate' (block2 — synthetic prompt + bare list, standalone) or
        # 'inline_consistent' (SFT sample matches a real rollout turn: obs ->
        # Plan+Action). inline_consistent only makes sense with inline plan ON; if
        # inline is off it falls back to separate to avoid teaching an unused format.
        seq = str(actor_cfg.get('plan_forecast_seq', 'separate')).lower()
        inline_on = bool(hasattr(self.config, 'data')
                         and self.config.data.get('plan_inline_enable', False))
        if seq == 'inline_consistent' and not inline_on:
            print("[plan_forecast] seq=inline_consistent -> separate (inline plan OFF)", flush=True)
            seq = 'separate'

        # Horizon: fixed plan_forecast_k, or a per-sample draw over the active stage
        # of the horizon-growth schedule (curriculum). RNG seeded by global_step so
        # the draw is reproducible and resume-consistent.
        k_min = k_max = None
        if self._pf_k_stages:
            k_min, k_max = active_k_range(self._pf_k_stages, self.global_steps)
        rng = random.Random(self.global_steps)

        # skip_invalid: drop actions whose env result was invalid/no-effect from the
        # forecast target (per-env patterns keyed by the task name). Default off.
        skip_invalid = bool(actor_cfg.get('plan_forecast_skip_invalid', False))
        try:
            env_name = str(self.config.actor_rollout_ref.agentgym.get('task_name', 'alfworld'))
        except Exception:
            env_name = 'alfworld'

        # group knobs (both need uid aligned to rollout_messages, default off, compose):
        # group_gate = filter which groups' successes to distill (curriculum);
        # group_norm = give each kept group equal total plan-CE weight (stability).
        group_gate = str(actor_cfg.get('plan_forecast_group_gate', 'off')).lower()
        group_norm = bool(actor_cfg.get('plan_forecast_group_norm', False))
        group_ids = None
        if group_gate != 'off' or group_norm:
            _uid = batch.non_tensor_batch.get('uid', None)
            if _uid is not None:
                group_ids = list(_uid)

        assembled, meta = build_plan_forecast_batch(
            messages_list=list(messages_list),
            tokenizer=self.tokenizer,
            rewards=rewards,
            k=int(actor_cfg.get('plan_forecast_k', 3)),
            gate=str(actor_cfg.get('plan_forecast_gate', 'wins')),
            success_threshold=float(actor_cfg.get('plan_forecast_success_threshold', 0.5)),
            target=target,
            seq=seq,
            max_length=int(actor_cfg.get('plan_forecast_max_length', 4096)),
            max_samples_per_trajectory=actor_cfg.get('plan_forecast_max_samples_per_traj', None),
            k_min=k_min,
            k_max=k_max,
            rng=rng,
            skip_invalid=skip_invalid,
            env=env_name,
            group_ids=group_ids,
            group_gate=group_gate,
            group_low=float(actor_cfg.get('plan_forecast_group_low_thresh', 0.5)),
            group_high=float(actor_cfg.get('plan_forecast_group_high_thresh', 1.0)),
            group_norm=group_norm,
            group_dedup=bool(actor_cfg.get('plan_forecast_group_dedup', True)),
        )
        if assembled is None:
            return None, meta
        return DataProto.from_single_dict(assembled), meta

    def _build_sft_ablation_dataproto(self, batch: DataProto, coef: float):
        """Build the RFT-style SFT-ablation DataProto (behavior-clone this step's
        winning trajectories). Control for plan-forecast; mutually exclusive with it.
        Same win-gating / reward source as ``_build_plan_forecast_dataproto``."""
        actor_cfg = self.config.actor_rollout_ref.actor
        if not actor_cfg.get('sft_ablation_enable', False):
            return None, {}
        if build_sft_ablation_batch is None:
            raise ImportError(
                "sft_ablation_enable=True but verl.agent_trainer.ppo.sft_ablation is "
                "missing from this checkout."
            )
        if coef <= 0:
            return None, {'sft_ablation/coef': 0.0}

        messages_list = batch.non_tensor_batch.get('rollout_messages', None)
        if messages_list is None:
            return None, {'sft_ablation/skipped_no_msgs': 1.0}

        rewards = None
        if 'traj_return' in batch.batch.keys():
            rewards = batch.batch['traj_return'].tolist()
        elif 'token_level_scores' in batch.batch.keys():
            rewards = batch.batch['token_level_scores'].sum(dim=-1).tolist()

        assembled, meta = build_sft_ablation_batch(
            messages_list=list(messages_list),
            tokenizer=self.tokenizer,
            rewards=rewards,
            gate=str(actor_cfg.get('sft_ablation_gate', 'wins')),
            success_threshold=float(actor_cfg.get('sft_ablation_success_threshold', 0.5)),
            max_length=int(actor_cfg.get('sft_ablation_max_length', 4096)),
            max_samples_per_trajectory=actor_cfg.get('sft_ablation_max_samples_per_traj', None),
        )
        if assembled is None:
            return None, meta
        return DataProto.from_single_dict(assembled), meta

    def _apply_plan_format_reward(self, batch: DataProto):
        """Driver-side plan FORMAT reward: score each generated turn's text for
        Thought->Plan->Action compliance and add coef*(score-baseline) to that
        turn's advantage tokens. Pure text parsing (no worker). Fully guarded.

        Returns a metrics dict; mutates batch.batch['advantages'] in place.
        """
        from verl.agent_trainer.ppo.plan_format import (
            score_turn_format, apply_plan_format_advantage)
        from verl.agent_trainer.ppo.plan_forecast import _action_turn_indices, _to_chat_list
        if 'rollout_messages' not in batch.non_tensor_batch:
            return {'plan_format/skipped_no_msgs': 1.0}
        msgs = batch.non_tensor_batch['rollout_messages']
        advantages = batch.batch['advantages']
        response_mask = batch.batch['response_mask']
        B, T = advantages.shape
        k = self.plan_format_reward_k
        tb = compute_turn_boundaries(response_mask)             # per-traj action spans
        format_tok = torch.zeros(B, T, dtype=advantages.dtype, device=advantages.device)
        n_turns = 0
        for i in range(B):
            spans = tb[i]
            if not spans:
                continue
            convo = _to_chat_list(msgs[i])
            turns = [convo[ai].get('content', '') or '' for ai in _action_turn_indices(convo)]
            for j, (s, e) in enumerate(spans):
                if j >= len(turns):
                    break
                sc = score_turn_format(turns[j], k=k)
                format_tok[i, s:e] = sc
                n_turns += 1
        new_adv, metrics = apply_plan_format_advantage(
            advantages=advantages, response_mask=response_mask, format_tok=format_tok,
            coef=self.plan_format_reward_coef, baseline=self.plan_format_reward_baseline,
            clip=self.plan_format_reward_clip,
            penalty_only=self.plan_format_reward_penalty_only)
        batch.batch['advantages'] = new_adv
        metrics['plan_format/num_turns'] = float(n_turns)
        return metrics

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.config.trainer.storage_mode == 'aistudio':
            self._save_checkpoint()

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                # keep the dataset's step in sync for step-gated inline warmup
                # (num_workers=0 -> same object; <=1-step lag is fine for a threshold)
                try:
                    self.train_dataset.current_step = self.global_steps
                except Exception:
                    pass

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'], non_tensor_batch_keys=['item_id', 'raw_prompt'])
                gen_batch.meta_info['global_steps'] = self.global_steps
                gen_batch.meta_info['max_rounds'] = self.rounds_scheduler.get_rounds()
                # Block-1 per-turn plan reminder: when inline plan is on, re-state
                # the "Plan first" request after EVERY observation (a one-time
                # instruction decays over turns). Routed via gen meta_info. Gated by
                # plan_inline_warmup_steps: during warmup the original prompt is used
                # (no reminder), the plan prompt is introduced only at/after warmup.
                _data_cfg = getattr(self.config, 'data', None)
                _inline_warmup = int(_data_cfg.get('plan_inline_warmup_steps', 0)) if _data_cfg is not None else 0
                # per-turn reminder ARCHIVED (default OFF) -> inline uses the opening
                # standing instruction only. Set data.plan_inline_per_turn=True to revive.
                _inline_active = (_data_cfg is not None
                                  and bool(_data_cfg.get('plan_inline_enable', False))
                                  and bool(_data_cfg.get('plan_inline_per_turn', False))
                                  and self.global_steps >= _inline_warmup)
                _think_on = _data_cfg is not None and bool(_data_cfg.get('think_reminder_enable', False))
                # Mutually exclusive per-turn reminders: inline-plan takes priority.
                # plan_inline_style: 'actions' (next-K actions) | 'todo' (checkable
                # sub-goal TODO list with (done) marks; pairs with target=subgoal).
                if _inline_active:
                    _style = str(_data_cfg.get('plan_inline_style', 'actions')).lower()
                    gen_batch.meta_info['per_turn_reminder'] = 'todo' if _style == 'todo' else 'plan'
                    gen_batch.meta_info['plan_inline_k'] = int(_data_cfg.get('plan_inline_k', 3))
                    metrics['plan_inline/active'] = 1.0
                elif _think_on:
                    gen_batch.meta_info['per_turn_reminder'] = 'think'
                    metrics['think_reminder/active'] = 1.0
                elif _data_cfg is not None and bool(_data_cfg.get('plan_inline_enable', False)):
                    metrics['plan_inline/active'] = 0.0   # inline in warmup -> original prompt
                metrics.update({
                    'max_rounds': self.rounds_scheduler.get_rounds(),
                })

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == 'remax':
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = batch.batch['rewards']
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    # # 在 batch dict 进入 update 之前
                    # batch.batch['old_log_probs'] = torch.nan_to_num(
                    #     batch.batch['old_log_probs'],
                    #     nan=-10.0,
                    #     posinf=0.0,      # log prob 不应该 > 0，但兜底
                    #     neginf=-10.0,    # 这是核心防护
                    # )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # we combine with rule-based rm
                        reward_tensor = batch.batch['scores']
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # info_grpo needs a turn-level information gain, which costs an
                        # extra forward pass, so only produce it when that estimator is
                        # actually selected.
                        if self.config.algorithm.adv_estimator == 'info_grpo':
                            with _timer('intrinsic_reward', timing_raw):
                                ir = self.actor_rollout_wg.compute_intrinsic_rewards(batch)
                                batch = batch.union(ir)
                                _m = batch.batch['token_level_intrinsic_rewards']
                                metrics['info/intrinsic_mean'] = _m.mean().item()
                                metrics['info/intrinsic_nonzero_frac'] = (_m != 0).float().mean().item()

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                        # Surface the degenerate-group filter. compute_advantage stashes
                        # this in meta_info, and without lifting it into metrics there is
                        # no way to tell a filtered run from an unfiltered one -- the flag
                        # is an env var, so a silent no-op looks exactly like a real run.
                        if 'grpo_filtered_frac' in batch.meta_info:
                            metrics['grpo/filtered_frac'] = batch.meta_info['grpo_filtered_frac']

                        # Plan FORMAT reward (default OFF): per-turn shaping bonus
                        # on the advantage for well-formed Thought->Plan->Action
                        # turns, to counter inline-plan decay. Driver-side, guarded.
                        if self.plan_format_reward_enable:
                            # format reward needs a plan to score -> only fire when
                            # inline is ACTIVE (enabled and past inline warmup), and
                            # past its own warmup. Otherwise it would penalize every
                            # (plan-less) turn during the original-prompt window.
                            _inl_warm = int(self.config.data.get('plan_inline_warmup_steps', 0)) \
                                if hasattr(self.config, 'data') else 0
                            _inline_active = (hasattr(self.config, 'data')
                                              and bool(self.config.data.get('plan_inline_enable', False))
                                              and self.global_steps >= _inl_warm)
                            if self.global_steps < self.plan_format_reward_warmup_steps or not _inline_active:
                                # still in warmup -> format reward dormant
                                metrics['plan_format/warmup'] = 1.0
                            else:
                                try:
                                    with _timer('plan_format_reward', timing_raw):
                                        metrics.update(self._apply_plan_format_reward(batch))
                                    metrics['plan_format/warmup'] = 0.0
                                except Exception as e:
                                    print(f'[plan_format] skipped due to: {e}')
                                    metrics['plan_format/error'] = 1.0

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                        # Optional plan-forecast SFT update: predict the realized
                        # next-K actions (the "plan"), supervised by what actually
                        # happened. Separate forward, does NOT touch PG. Default OFF.
                        try:
                            self.plan_forecast_coeff_scheduler.set_global_steps(self.global_steps)
                            pf_coef = float(self.plan_forecast_coeff_scheduler.get_coeff())
                            metrics['plan_forecast/coef_sched'] = pf_coef
                            pf_data, pf_meta = self._build_plan_forecast_dataproto(batch, pf_coef)
                            metrics.update(pf_meta)
                            if pf_data is not None and len(pf_data) > 0:
                                pf_data.meta_info['plan_forecast_coef'] = pf_coef
                                pf_data_padded, _pf_pad = pad_dataproto_to_divisor(
                                    pf_data, self.actor_rollout_wg.world_size)
                                pf_data_padded.meta_info['plan_forecast_coef'] = pf_coef
                                with _timer('update_plan_forecast', timing_raw):
                                    pf_output = self.actor_rollout_wg.update_actor_plan_forecast(pf_data_padded)
                                metrics.update(reduce_metrics(pf_output.meta_info['metrics']))
                                metrics['plan_forecast/num_samples'] = len(pf_data)
                            else:
                                metrics['plan_forecast/num_samples'] = 0
                        except Exception as e:
                            print(f"[plan_forecast] skipped due to error: {e}", flush=True)
                            metrics['plan_forecast/error'] = 1.0

                        # Optional SFT-ablation (RFT) control: one extra SFT round on
                        # this step's WINNING trajectories, behavior-cloning the real
                        # assistant turns. Mutually exclusive with plan_forecast (asserted
                        # at init). Same optimizer path (update_actor_plan_forecast), only
                        # the target differs -> a clean A/B against plan_forecast SFT.
                        try:
                            _acfg = self.config.actor_rollout_ref.actor
                            if bool(_acfg.get('sft_ablation_enable', False)):
                                abl_coef = float(_acfg.get('sft_ablation_coef', 0.01))
                                metrics['sft_ablation/coef_sched'] = abl_coef
                                sft_data, sft_meta = self._build_sft_ablation_dataproto(batch, abl_coef)
                                metrics.update(sft_meta)
                                if sft_data is not None and len(sft_data) > 0:
                                    sft_data.meta_info['plan_forecast_coef'] = abl_coef
                                    sft_data.meta_info['sft_metric_prefix'] = 'sft_ablation'
                                    sft_padded, _sft_pad = pad_dataproto_to_divisor(
                                        sft_data, self.actor_rollout_wg.world_size)
                                    sft_padded.meta_info['plan_forecast_coef'] = abl_coef
                                    sft_padded.meta_info['sft_metric_prefix'] = 'sft_ablation'
                                    with _timer('update_sft_ablation', timing_raw):
                                        sft_output = self.actor_rollout_wg.update_actor_plan_forecast(sft_padded)
                                    metrics.update(reduce_metrics(sft_output.meta_info['metrics']))
                                    metrics['sft_ablation/num_samples'] = len(sft_data)
                                else:
                                    metrics['sft_ablation/num_samples'] = 0
                        except Exception as e:
                            print(f"[sft_ablation] skipped due to error: {e}", flush=True)
                            metrics['sft_ablation/error'] = 1.0

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                self.rounds_scheduler.step()

                if self.global_steps >= self.total_training_steps:

                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
