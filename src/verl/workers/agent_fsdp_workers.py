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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
import datetime

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        raise ValueError(
            'HSDP is not supported yet because it produces incorrect results for now. Please set fsdp_size=-1')
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        # Sub-groups created later (notably vLLM's tensor-parallel group) call
        # new_group() without a timeout and land on NCCL's compiled-in 10-minute
        # default, which the main group's 3600s below does not cover. An agent rollout
        # is 15 rounds of generation each gated on an env-server HTTP call, and with a
        # 14B policy plus a 14B user simulator one round trip can exceed 10 minutes --
        # the watchdog then tears the job down mid-rollout. Raise the default that
        # new_group() reads so every sub-group inherits the same budget.
        torch.distributed.distributed_c10d.default_pg_nccl_timeout = datetime.timedelta(seconds=3600)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                            self.ulysses_sequence_parallel_size)
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0
        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               use_liger=False,
                               role='actor'):
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload
        from torch import optim

        assert role in ['actor', 'ref']

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == 'actor' else CPUOffload(offload_params=True)
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if role == 'actor':
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        from verl.workers.rollout.agent_vllm_rollout import vLLMRollout
        from verl.workers.sharding_manager import FSDPVLLMShardingManager
        log_gpu_memory_usage('Before building vllm rollout', logger=None)
        rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                                 rollout_config=self.config.rollout,
                                                 agentgym_config=self.config.agentgym,
                                                 tokenizer=self.tokenizer,
                                                 model_hf_config=self.actor_model_config)
        log_gpu_memory_usage('After building vllm rollout', logger=None)
        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = 'dummy_hf'
        rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                           inference_engine=rollout.inference_engine,
                                                           model_config=self.actor_model_config,
                                                           full_params='hf' in self.config.rollout.load_format,
                                                           device_mesh=rollout_device_mesh)
        log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.agent_actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            # if self._is_actor:
            #     optim_config = self.config.actor.optim
            #     fsdp_config = self.config.actor.fsdp_config
            # else:
            #     optim_config = None
            #     fsdp_config = OmegaConf.create()
            optim_config = self.config.actor.optim
            fsdp_config = self.config.actor.fsdp_config
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                use_liger=self.config.model.get('use_liger', False),
                role='actor')

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer,
                                              model_config=self.actor_model_config)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False),
                                                               use_liger=self.config.model.get('use_liger', False),
                                                               role='ref')[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(model=self.actor_module_fsdp,
                                                            optimizer=self.actor.actor_optimizer,
                                                            lr_scheduler=self.actor_lr_scheduler,
                                                            tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name='update_policy', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor_world_model(self, data: DataProto):
        """Run one world-model SFT update on env-prediction data.

        The incoming ``data`` is the output of
        ``world_model_loss.build_world_model_sft_batch`` packed into a
        DataProto (expects ``input_ids``/``attention_mask``/``position_ids``/``loss_mask``).
        """
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update world-model', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name='update_world_model', logger=None) as timer:
                metrics = self.actor.update_world_model(data=data)
            delta_time = timer.last
            metrics['actor/world_model_step_time'] = delta_time

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/world_model_lr'] = lr

            log_gpu_memory_usage('After update world-model', logger=logger)

            output = DataProto(meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor_plan_forecast(self, data: DataProto):
        """One plan-forecast SFT update (predict realized next-K actions).

        Incoming ``data`` is the output of ``plan_forecast.build_plan_forecast_batch``
        (expects ``input_ids``/``attention_mask``/``position_ids``/``loss_mask``).
        Mirrors ``update_actor_world_model``; DP dispatch + caller-side
        ``pad_dataproto_to_divisor`` keep every rank's sample count equal, so no
        FSDP collective desync.
        """
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()
        log_gpu_memory_usage('Before update plan-forecast', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name='update_plan_forecast', logger=None) as timer:
                metrics = self.actor.update_plan_forecast(data=data)
            metrics['plan_forecast/step_time'] = timer.last

            self.actor_lr_scheduler.step()
            metrics['plan_forecast/lr'] = self.actor_lr_scheduler.get_last_lr()[0]

            log_gpu_memory_usage('After update plan-forecast', logger=logger)
            output = DataProto(meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_hindsight_log_probs_perstep(self, data: DataProto):
        """HCAPO-aligned PER-STEP hindsight scoring (paper §4.2). Instead of one
        front injection over the whole concatenated trajectory (cancelled by ÷π̄),
        reconstruct a SHORT per-step prompt (truncated history + current obs +
        s_final) LOCAL to each action and read its mean action-token log-prob. The
        scalar is broadcast back into the (B,T) h_log_probs layout, so downstream
        apply_hca_advantage is unchanged. Enabled by actor.hca_perstep."""
        from verl.agent_trainer.ppo.hca_perstep import iter_step_inputs
        from verl.agent_trainer.ppo.wmc_erc import _turn_spans
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        H = int(self.config.actor.get('hca_history_len', 0))   # 0 = full history (our impl), >0 = paper truncation
        temperature = self.config.rollout.temperature
        messages = data.non_tensor_batch['rollout_messages']
        response_mask = data.batch['response_mask']
        B, T = response_mask.shape
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        h_log_probs = torch.zeros(B, T, dtype=torch.float32)

        # build all per-step sequences for this shard
        seqs, amasks, locs = [], [], []          # token-id list, action-mask list, (traj, span)
        spans_per_traj = []
        for i in range(B):
            spans = _turn_spans(response_mask[i].to(torch.bool))
            spans_per_traj.append(spans)
            steps = iter_step_inputs(list(messages[i]), history_len=H)
            for k, (ctx_msgs, action_text) in enumerate(steps):
                if k >= len(spans):
                    break
                try:
                    ctx_ids = self.tokenizer.apply_chat_template(ctx_msgs, add_generation_prompt=True, tokenize=True)
                    full_ids = self.tokenizer.apply_chat_template(
                        ctx_msgs + [{"role": "assistant", "content": action_text}],
                        add_generation_prompt=False, tokenize=True)
                except Exception:
                    continue
                if len(full_ids) <= len(ctx_ids) or len(full_ids) > 2048:
                    continue
                am = [0] * len(ctx_ids) + [1] * (len(full_ids) - len(ctx_ids))
                seqs.append(full_ids); amasks.append(am); locs.append((i, spans[k]))

        # forward in chunks, scatter mean action logp back into h_log_probs[i, s:e].
        # CRITICAL: every rank MUST run the SAME number of FSDP forwards or the
        # per-layer all-gather collectives desync and deadlock (NCCL timeout) —
        # variable trajectory turn counts give variable chunk counts per rank. So
        # we all-reduce the max chunk count and pad short ranks with dummy forwards.
        CH = 16
        local_chunks = (len(seqs) + CH - 1) // CH
        nchunks_t = torch.tensor([local_chunks], device=torch.cuda.current_device())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(nchunks_t, op=torch.distributed.ReduceOp.MAX)
        nchunks = int(nchunks_t.item())
        with self.ulysses_sharding_manager:
            for c in range(nchunks):
                s0 = c * CH
                chunk = seqs[s0:s0 + CH]
                if chunk:
                    ams = amasks[s0:s0 + CH]; real = len(chunk)
                else:
                    chunk = [[pad_id]]; ams = [[0]]; real = 0    # dummy to stay in sync
                L = max(len(x) for x in chunk)
                ids = torch.tensor([[pad_id] * (L - len(x)) + x for x in chunk],
                                   dtype=torch.long, device=torch.cuda.current_device())
                att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in chunk],
                                   dtype=torch.long, device=torch.cuda.current_device())
                amt = torch.tensor([[0] * (L - len(a)) + a for a in ams],
                                   dtype=torch.long, device=torch.cuda.current_device())
                mlp = self.actor.compute_perstep_action_logp(ids, att, amt, temperature)  # (chunk,)
                for j in range(real):
                    i, span = locs[s0 + j]
                    h_log_probs[i, span[0]:span[1]] = float(mlp[j].item())

        output = DataProto.from_dict(tensors={'h_log_probs': h_log_probs})
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_pe_labels(self, data: DataProto):
        """Classify PROGRESS + EXPLORATION steps per trajectory (text classifiers) and
        return per-action-token masks (B,T): 'pe_progress_mask', 'pe_explore_mask'
        (1 on the action tokens of classified progress / exploration turns). Two
        generations per trajectory via the rollout engine. Default-off path."""
        from verl.agent_trainer.ppo.progress_credit_probe import (
            extract_goal, trajectory_steps, build_progress_messages, build_explore_messages,
            parse_progress, parse_explore)
        from verl.agent_trainer.ppo.wmc_erc import _turn_spans
        assert self._is_rollout
        data = data.to('cuda')
        env = str(self.config.actor.get('pe_clf_env', 'alfworld'))
        max_new = int(self.config.actor.get('pe_clf_max_new_tokens', 160))
        messages = data.non_tensor_batch['rollout_messages']
        response_mask = data.batch['response_mask']
        B, T = response_mask.shape
        pm = torch.zeros(B, T, dtype=torch.float32)
        em = torch.zeros(B, T, dtype=torch.float32)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        # build classification prompts (non-collective; per-traj guarded so one bad
        # trajectory never aborts this rank and desyncs it from the others).
        prompts, meta, spans_per = [], [], []        # meta: (traj_i, kind, n_steps)
        for i in range(B):
            spans = _turn_spans(response_mask[i].to(torch.bool))
            spans_per.append(spans)
            try:
                conv = list(messages[i]); goal = extract_goal(conv); steps = trajectory_steps(conv)
                for kind, build in (("P", build_progress_messages), ("E", build_explore_messages)):
                    ids = self.tokenizer.apply_chat_template(build(env, goal, steps),
                                                             add_generation_prompt=True, tokenize=True)
                    prompts.append(ids); meta.append((i, kind, len(steps)))
            except Exception as e:
                print(f'[compute_pe_labels] prompt build failed for traj {i}: {e}')
        # COLLECTIVE: EVERY rank MUST enter the rollout sharding manager + generate
        # exactly once, or the weight-sync all-gather desyncs and deadlocks (NCCL
        # timeout). Pad with a dummy prompt if this rank built none.
        try:
            with self.rollout_sharding_manager:
                texts = self.rollout.generate_text_batch(prompts or [[pad_id]], max_new_tokens=max_new)
        except Exception as e:
            print(f'[compute_pe_labels] generation failed: {e}')
            texts = []
        # scatter masks (non-collective; guarded)
        if prompts:
            for (i, kind, n_steps), txt in zip(meta, texts):
                try:
                    spans = spans_per[i]
                    idxs = parse_progress(txt, n_steps) if kind == "P" else parse_explore(txt, n_steps)
                    tgt = pm if kind == "P" else em
                    for t in idxs:
                        if t < len(spans):
                            tgt[i, spans[t][0]:spans[t][1]] = 1.0
                except Exception:
                    pass
        output = DataProto.from_dict(tensors={'pe_progress_mask': pm, 'pe_explore_mask': em})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_classification(self, data: DataProto):
        """Safe-commit text gate: plain greedy generation over pre-tokenized
        classification prompts (input_ids left-padded, attention_mask), parsed
        on-worker into a predictability bit. Returns a DataProto with tensor
        'clf_predictable' (N,) ∈ {0,1} (1 = outcome DETERMINISTIC). Default-off path.
        """
        from verl.agent_trainer.ppo.safe_commit_gate import parse_predictable
        assert self._is_rollout
        data = data.to('cuda')
        max_new = int(data.meta_info.get('clf_max_new_tokens', 24))
        input_ids = data.batch['input_ids']
        attn = data.batch['attention_mask']
        # strip left padding → list of real prompt token-id lists
        prompt_ids = [input_ids[i][attn[i].bool()].tolist() for i in range(input_ids.size(0))]
        with self.rollout_sharding_manager:
            texts = self.rollout.generate_text_batch(prompt_ids, max_new_tokens=max_new)
        pred = torch.tensor([parse_predictable(t) for t in texts], dtype=torch.long)
        output = DataProto.from_dict(tensors={'clf_predictable': pred})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_intrinsic_rewards(self, data: DataProto):
        """Turn-level information gain for adv_estimator=info_grpo.

        Costs one extra pair of forward passes per scored turn, so it is capped by
        INFO_KL_BATCH / INFO_MAX_TURNS. Returns zeros rather than raising if anything is
        missing, because info_grpo degrades to plain GRPO on a zero intrinsic tensor --
        a silently weaker signal beats a crashed run mid-training.
        """
        assert self._is_actor
        from verl.trainer.ppo.intrinsic_reward import compute_intrinsic_rewards as _cir
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        data = data.to('cuda')
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            b = data.batch
            try:
                intrinsic = _cir(
                    self.actor_module_fsdp,
                    input_ids=b['input_ids'], attention_mask=b['attention_mask'],
                    position_ids=b['position_ids'], observation_mask=b['observation_mask'],
                    response_mask=b['response_mask'],
                    kl_batch_size=int(os.environ.get('INFO_KL_BATCH', 2)),
                    max_turns_per_sample=int(os.environ.get('INFO_MAX_TURNS', 8)),
                )
            except Exception as e:
                print(f'[info_grpo] intrinsic reward failed, falling back to zeros: {e}')
                intrinsic = torch.zeros_like(b['response_mask'], dtype=torch.float32)
            output = DataProto.from_dict(tensors={'token_level_intrinsic_rewards': intrinsic})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                        offload_grad=self._is_offload_grad)
        return output.to('cpu')

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        data = data.to('cuda')
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            if isinstance(output, tuple):
                log_probs, entropys = output
                output = DataProto.from_dict(tensors={'old_log_probs': log_probs, 'entropys': entropys},
                                             meta_info={'temperature': self.config.rollout.temperature})
            else:
                output = DataProto.from_dict(tensors={'old_log_probs': output},
                                             meta_info={'temperature': self.config.rollout.temperature})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After compute_log_prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_hindsight_log_probs(self, data: DataProto):
        """Inference call: per-action-token log π(a|s, s_final) log probs under
        the outcome-conditioned context (HCAPO Generative Verification).

        Builds the realized-outcome statement prefixes here (needs the
        tokenizer) and passes their token ids to the actor via meta_info.
        Only valid when the actor was constructed with use_hindsight_hca=True.
        """
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        data = data.to('cuda')
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature

        # Build the per-trajectory FINAL-STATE hindsight hint (HCAPO §4.2: inject
        # the realized outcome s_final). s_final = the trajectory's most recent
        # env observation = the LAST contiguous run of env tokens (response_mask=0
        # & attended) in the response. This is env-agnostic: works whether the
        # trajectory ends on an action (sciworld) or an observation (webshop).
        # The hint is added to data.batch so it splits with the micro-batches; the
        # actor inserts it right BEFORE the response region (local, single forward).
        M = int(self.config.actor.get('hca_final_state_max_tokens', 96))
        responses = data.batch['responses']                       # (B, Tr)
        resp_mask = data.batch['response_mask']                   # (B, Tr) 1 on action toks
        Tr = responses.size(1)
        attn_resp = data.batch['attention_mask'][:, -Tr:]
        # Neutral, declarative hindsight knowledge — NO "judge/evaluate" instruction.
        # We only read the policy's log-prob of the (fixed) action tokens under this
        # context; an instruction to "judge" would make the model expect a judgment
        # next and distort the log-prob of the actual actions that follow.
        pre_t = self.tokenizer(
            "\n[Hindsight] Knowing in advance that this episode ultimately ends in the "
            "following final state:\n",
            add_special_tokens=False)['input_ids']
        suf_t = self.tokenizer("\n", add_special_tokens=False)['input_ids']
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        Bsz = responses.size(0)
        hints = []
        for i in range(Bsz):
            obs = attn_resp[i].bool() & (resp_mask[i] == 0)       # env-observation tokens
            idx = obs.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:                                   # last contiguous obs run
                brk = (idx[1:] - idx[:-1] > 1).nonzero(as_tuple=True)[0]
                start = int(idx[int(brk[-1].item()) + 1].item()) if brk.numel() else int(idx[0].item())
                fs_ids = responses[i, start: int(idx[-1].item()) + 1].tolist()[:M]
            else:
                fs_ids = []                                       # no obs ⇒ empty final state
            hints.append(pre_t + fs_ids + suf_t)
        K = max(len(h) for h in hints)
        hint_ids = torch.tensor([[pad_id] * (K - len(h)) + h for h in hints],
                                dtype=torch.long, device=responses.device)
        hint_mask = torch.tensor([[0] * (K - len(h)) + [1] * len(h) for h in hints],
                                 dtype=torch.long, device=responses.device)
        data.batch['hca_hint_ids'] = hint_ids
        data.batch['hca_hint_mask'] = hint_mask

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            h_log_probs = self.actor.compute_hindsight_log_probs(data=data)
            output = DataProto.from_dict(tensors={'h_log_probs': h_log_probs})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self,
                        local_path,
                        hdfs_path=None,
                        global_step=0,
                        remove_previous_ckpt=False,
                        max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt,
                                                max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        # Sub-groups created later (notably vLLM's tensor-parallel group) call
        # new_group() without a timeout and land on NCCL's compiled-in 10-minute
        # default, which the main group's 3600s below does not cover. An agent rollout
        # is 15 rounds of generation each gated on an env-server HTTP call, and with a
        # 14B policy plus a 14B user simulator one round trip can exceed 10 minutes --
        # the watchdog then tears the job down mid-rollout. Raise the default that
        # new_group() reads so every sub-group inherits the same budget.
        torch.distributed.distributed_c10d.default_pg_nccl_timeout = datetime.timedelta(seconds=3600)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)
            self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                      self.ulysses_sequence_parallel_size)
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False,
                             device_mesh=self.device_mesh,
                             cpu_offload=None)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.agent_critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(model=self.critic_module,
                                                        optimizer=self.critic_optimizer,
                                                        lr_scheduler=self.critic_lr_scheduler,
                                                        tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self,
                        local_path,
                        hdfs_path=None,
                        global_step=0,
                        remove_previous_ckpt=False,
                        max_ckpt_to_keep=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt,
                                                max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
