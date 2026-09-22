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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List
from omegaconf import DictConfig
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# vllm_version is set only by the vendored engines (<= 0.6.3). On a stock vLLM the
# third_party dispatcher takes its SPMD branch and leaves it None, which is what
# distinguishes the two engine APIs everywhere below.
_VLLM_CUSTOMIZED = vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3')


def _post_process_outputs(tokenizer, request_outputs):
    """Reproduce the vendored engine's return contract on a stock vLLM.

    verl's patched LLM (third_party/vllm/vllm_v_0_6_3/llm.py) does not return
    vLLM's List[RequestOutput]; it pads the completions into a
    (num_prompts, max_response_len) tensor and returns (token_ids, logprobs).
    The agent loop below reads exactly that -- `output[0].tolist()` -- so the SPMD
    path converts here rather than in the loop, which keeps the two engines'
    behaviour identical from the rollout's point of view.
    """
    output_token_ids = []
    logprobs = []
    for request_output in request_outputs:  # List[RequestOutput]
        for output in request_output.outputs:  # List[CompletionOutput], usually len == 1
            output_token_ids.append(torch.tensor(output.token_ids))
            logprobs_dicts = output.logprobs
            if logprobs_dicts is not None:
                logprob = []
                for logprobs_dict, tok_id in zip(logprobs_dicts, output.token_ids):
                    logprob.append(logprobs_dict[tok_id].logprob)
                logprobs.append(torch.tensor(logprob))

    pad_token_id = (tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(logprobs) > 0:
        logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
    return output_token_ids, logprobs

import os
import json
import time
import requests
import numpy as np
from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.agentgym.client import init_env_client
from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, rollout_config: DictConfig, agentgym_config: DictConfig, tokenizer, model_hf_config, model_path: str = None, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            model_path: local path to the policy weights. Only the SPMD engine needs it --
                the vendored one is handed the live FSDP module instead. Falls back to
                model_hf_config._name_or_path, which AutoConfig.from_pretrained sets.
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = rollout_config
        self.agentgym_config = agentgym_config
        assert not (not rollout_config.enforce_eager and rollout_config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if _VLLM_CUSTOMIZED:
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        # Without this the engine falls back to the model config's own limit
        # (32768 for Qwen2.5), and vLLM sizes/validates the KV cache against that
        # rather than against what a rollout actually needs. Harmless for a 7B,
        # fatal for a 14B on a 40GB card: engine init dies with "max seq len
        # (32768) is larger than the maximum number of tokens that can be stored
        # in KV cache (7472)". Same expression the rollout loop asserts against.
        engine_max_model_len = min(self.config.max_model_len,
                                   self.config.prompt_length + self.config.response_length)
        # Largest generation prompt the engine will accept, leaving room for the turn's
        # own completion. The agent loop checks trajectories against this each round --
        # see the guard in generate_sequences.
        self.max_generation_prompt_len = engine_max_model_len - self.config.max_tokens

        if _VLLM_CUSTOMIZED:
            self.inference_engine = LLM(
                actor_module,
                tokenizer=tokenizer,
                model_hf_config=model_hf_config,
                tensor_parallel_size=tensor_parallel_size,
                dtype=rollout_config.dtype,
                enforce_eager=rollout_config.enforce_eager,
                gpu_memory_utilization=rollout_config.gpu_memory_utilization,
                skip_tokenizer_init=False,
                load_format=rollout_config.load_format,
                disable_log_stats=rollout_config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=engine_max_model_len,
                enable_chunked_prefill=rollout_config.enable_chunked_prefill,
            )
            # Offload vllm model to reduce peak memory usage
            self.inference_engine.offload_model_weights()
        else:
            # Stock vLLM builds its own copy of the model from disk, so it needs a path
            # rather than the FSDP module. The weights it loads here are thrown away at
            # the first sharding-manager __enter__, which overwrites them from the
            # actor's state_dict -- hence load_format 'dummy', which skips reading the
            # checkpoint at all. verl's own 'dummy_dtensor' is a name only the vendored
            # engine knows; stock vLLM rejects it.
            if model_path is None:
                model_path = getattr(model_hf_config, '_name_or_path', None)
            assert model_path, (
                'SPMD vLLM needs a local model path; pass model_path= or use a '
                'model_hf_config carrying _name_or_path')

            load_format = rollout_config.load_format
            if load_format in ('dummy_dtensor', 'dummy_hf', 'dummy_megatron'):
                load_format = 'dummy'

            # external_launcher: every rank builds its own engine on top of the process
            # group torch.distributed already established, which is what lets each rank
            # drive its own conversations. enable_sleep_mode is what makes the sharding
            # manager's sleep()/wake_up() pair work in place of the cache-engine calls.
            self.inference_engine = LLM(
                model=model_path,
                tokenizer=model_path,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend='external_launcher',
                enable_sleep_mode=True,
                dtype=rollout_config.dtype,
                enforce_eager=rollout_config.enforce_eager,
                gpu_memory_utilization=rollout_config.gpu_memory_utilization,
                skip_tokenizer_init=False,
                load_format=load_format,
                disable_log_stats=rollout_config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=engine_max_model_len,
                enable_chunked_prefill=rollout_config.enable_chunked_prefill,
                trust_remote_code=rollout_config.get('trust_remote_code', False),
                # vLLM refuses the external launcher without an explicit seed, because
                # ranks inside a TP group must sample identically. Differentiating the
                # *data-parallel* ranks is FSDPVLLMShardingManager's job -- it reseeds
                # torch with gen_dp_rank + 1000 around each generation pass -- so one
                # constant here is what that mechanism expects, not a source of
                # duplicate rollouts within a GRPO group.
                seed=rollout_config.get('seed', 0),
            )
            # Matches the vendored engine's offload_model_weights() above: give the
            # weight memory back until the sharding manager wakes the engine up.
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=rollout_config.max_tokens,
        )

        # we may detokenize the result all together later
        if _VLLM_CUSTOMIZED:
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)
        kwargs["n"] = 1  # because we have repeated task n times

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer


    def _init_cache_engine(self):
        """Rebuild the KV cache before a generation pass.

        Only the vendored engine needs this. Under SPMD the sharding manager owns the
        engine's memory across the whole rollout -- it calls wake_up() on __enter__ and
        sleep(level=1) on __exit__ -- so doing it again per pass would either free the
        weights the sharding manager just loaded or double-allocate the cache.
        """
        if _VLLM_CUSTOMIZED and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

    def _free_cache_engine(self):
        """Counterpart to _init_cache_engine; a no-op under SPMD for the same reason."""
        if _VLLM_CUSTOMIZED and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

    def _engine_generate(self, prompt_token_ids, sampling_params, use_tqdm=False):
        """Generate from pre-tokenized prompts, returning (token_ids, logprobs).

        Both engines are driven through here so the agent loop sees one contract: a
        padded (num_prompts, max_response_len) tensor as element 0. The vendored engine
        pads internally; stock vLLM returns List[RequestOutput] and is padded by
        _post_process_outputs.
        """
        if _VLLM_CUSTOMIZED:
            return self.inference_engine.generate(prompts=None,
                                                  prompt_token_ids=prompt_token_ids,
                                                  sampling_params=sampling_params,
                                                  use_tqdm=use_tqdm)
        request_outputs = self.inference_engine.generate(
            prompts=[{'prompt_token_ids': list(ids)} for ids in prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=use_tqdm)
        return _post_process_outputs(self.tokenizer, request_outputs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for _ in range(n):
                # only keep not pad part
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][i])
                attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                item_id_str = prompts.non_tensor_batch["item_id"][i]
                try:
                    task_name = item_id_str.split("_")[0]
                    item_id = int(item_id_str.split("_")[-1])
                except (ValueError, IndexError):
                    task_name = self.agentgym_config.task_name
                    item_id = i # Fallback to batch index if we can't parse it (might still be wrong but better than crash)

                handler = RolloutHandler(
                    messages=[
                        Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt
                    ],
                    task_name=task_name,
                    item_id=item_id,
                    score=0,
                    done=False,
                    input_ids=list(input_ids),
                    prompt_ids=list(input_ids),
                    response_ids=[],
                    attention_mask=list(attention_mask),
                    prompt_attention_mask=list(attention_mask),
                    response_attention_mask=[],
                    position_ids=list(position_ids),
                    prompt_position_ids=list(position_ids),
                    response_position_ids=[],
                    loss_mask=[0] * len(input_ids),
                    prompt_loss_mask=[0] * len(input_ids),
                    response_loss_mask=[],
                    observation_mask=[0] * len(input_ids),
                    prompt_observation_mask=[0] * len(input_ids),
                    response_observation_mask=[],
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                )
                assert len(handler.input_ids) == len(handler.attention_mask) == len(handler.position_ids) == len(handler.loss_mask) == len(handler.observation_mask), f"RolloutHandler has mismatched length: input_ids={len(handler.input_ids)}, attention_mask={len(handler.attention_mask)}, position_ids={len(handler.position_ids)}, loss_mask={len(handler.loss_mask)}, observation_mask={len(handler.observation_mask)}"
                handler_list.append(handler)
        return handler_list

    def _shape_task_reward(self, task_score: float, task_done: bool, task_name: str = "") -> float:
        if str(task_name).lower() == "sciworld":
            return 1.0 if bool(task_done) and float(task_score) >= 100.0 else 0.0
        elif str(task_name).lower() == "webshop":
            return 1.0 if bool(task_done) and float(task_score) >= 1.0 else 0.0
        elif str(task_name).lower() == "tau2":
            # Pass tau2's score through unchanged. It is already 1.0 exactly when every
            # applicable check passes, and under TAU2_REWARD_SHAPE=dense the values in
            # between are the partial-credit signal that keeps GRPO groups from being
            # uniformly zero -- binarising here would throw that away.
            return float(task_score) if bool(task_done) else 0.0
        else:
            return task_score

    @torch.no_grad()
    def generate_text_batch(self, prompt_token_ids, max_new_tokens: int = 24):
        """Plain (single-turn) greedy text generation for the safe-commit text gate.

        Unlike generate_sequences (a full multi-turn agent rollout), this just runs
        the underlying LLM on a list of pre-tokenized prompts and returns the decoded
        completions. Used to classify per-turn outcome predictability. Greedy/short.
        """
        self._init_cache_engine()
        try:
            sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                                max_tokens=int(max_new_tokens))
            output = self._engine_generate(list(prompt_token_ids), sp, use_tqdm=False)
            # verl's vLLM wrapper returns the response-token tensor as output[0]
            # (same as generate_sequences: response_ids = output[0].tolist()), a
            # (num_prompts, resp_len) tensor — NOT a list of vLLM RequestOutput.
            resp = output[0] if isinstance(output, (tuple, list)) else output
            resp_ids = resp.tolist() if hasattr(resp, "tolist") else list(resp)
            texts = [self.tokenizer.decode(r, skip_special_tokens=True) for r in resp_ids]
        finally:
            self._free_cache_engine()
        return texts

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        self._init_cache_engine()

        global_steps = prompts.meta_info.get('global_steps', None)
        max_rounds = prompts.meta_info.get('max_rounds', 10)
        cur_device = prompts.batch["input_ids"].device

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # repeat for self.config.n times to rollout
        batch_size = prompts.batch['input_ids'].size(0)
        batch_size *= self.config.n
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)
        env_clients = [init_env_client(self.agentgym_config) for _ in range(batch_size)]
        time.sleep(self.config.send_interval) # take a break before sendng request
        all_done_flag = False
        # Native tool-calling protocol (tau2 / InfoPO alignment): the env server labels
        # each observation with the role it came from -- "tool: <result>" for a tool
        # call, "user: <reply>" for the customer. Tool results go back as tool-role
        # messages and customer replies as bare user turns, exactly as tau2's own
        # agent sees them; the ReAct harness instead feeds every observation, label
        # included, as a user turn.
        native_tools = bool(self.agentgym_config.get("native_tools", False))
        def _split_role(state: str):
            if native_tools:
                if state.startswith("tool: "):
                    return "tool", state[len("tool: "):]
                if state.startswith("user: "):
                    return "user", state[len("user: "):]
            return "user", state
        for idx, rollout_handler in enumerate(rollout_handler_ls):
            try:
                env_clients[idx].reset(rollout_handler.item_id)
                task = env_clients[idx].observe()
                _, task = _split_role(task)
                rollout_handler.add_user_message(self.tokenizer, task)
            except TimeoutError:
                print(f"Reset Timeout: Webarena Env Timeout. item id = {rollout_handler.item_id}")
                rollout_handler.done = True
                rollout_handler.score = 0

        rounds = 0
        task_rounds = [0] * batch_size
        rollout_bar = tqdm(total = max_rounds, desc="Running rounds", disable=torch.distributed.get_rank() != 0)
        def agent_step(i, idx):
            content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
            rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
            task_rounds[idx] += 1
            try:
                step_output = env_clients[idx].step(content)
                state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                    step_output.state,
                    step_output.reward,
                    step_output.done,
                )
                role, body = _split_role(state)
                if role == "tool":
                    rollout_handler_ls[idx].add_tool_message(self.tokenizer, body)
                else:
                    rollout_handler_ls[idx].add_user_message(self.tokenizer, body)
                return step_output.done
            except Exception as e:
                rollout_handler_ls[idx].score = 0
                rollout_handler_ls[idx].done = True
                print(f"Rollou step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                return True
        while rounds < max_rounds and not all_done_flag:
            # get generation prompt
            generation_prompt_idxs = []
            not_done_idxs = []
            over_budget = 0
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                if rollout_handler.done:
                    continue
                prompt_ids = rollout_handler.get_generation_prompt(self.tokenizer)
                # get_generation_prompt re-renders the whole message list and has no cap
                # of its own; truncate_output_ids only trims the *training* tensors, and
                # only after this loop. So a conversation that outgrows the engine's
                # context window reaches vLLM unchecked, and vLLM rejects the request
                # with "decoder prompt (length N) is longer than the maximum model
                # length" -- an exception raised outside agent_step's try, which takes
                # down every other trajectory in the batch with it.
                #
                # End the trajectory instead, exactly as running out of rounds does: it
                # keeps whatever score the environment last gave it.
                if len(prompt_ids) > self.max_generation_prompt_len:
                    rollout_handler.done = True
                    over_budget += 1
                    continue
                generation_prompt_idxs.append(prompt_ids)
                not_done_idxs.append(idx)
            if over_budget:
                print(f"[rollout] round {rounds + 1}: ended {over_budget} trajectory(ies) "
                      f"that outgrew the {self.max_generation_prompt_len}-token prompt budget")
            if not not_done_idxs:
                break

            rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active agents per gpu: {len(not_done_idxs)}")
            # users can customize different sampling_params at different run
            with self.update_sampling_params(**kwargs):
                output = self._engine_generate(generation_prompt_idxs,
                                               self.sampling_params,
                                               use_tqdm=False)
            response_ids = output[0].tolist()
            all_done_flag = True
            time.sleep(self.config.send_interval) # take a break before sendng request
            if len(not_done_idxs) > 0:
                with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                    step_dones = list(executor.map(
                        lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                    ))
                    all_done_flag = all(step_dones)
            rounds += 1
            rollout_bar.update(1)
        
        # process ids
        rollout_bar.close()
        response_ids, response_attention_mask, response_position_ids, response_loss_mask, response_observation_mask = [], [], [], [], []
        response_turn_ids = []   # Temporal Ensembling: action-turn index per token (-1 = none)
        scores, messages = [], []
        
        for rollout_handler in rollout_handler_ls:
            # check length
            rollout_handler.truncate_output_ids()
            assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask) == len(rollout_handler.observation_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=}, 
            {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}, {len(rollout_handler.observation_mask)=}"""
            assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
            response_observation_mask.append(torch.tensor(rollout_handler.response_observation_mask, dtype=torch.int, device=cur_device))
            response_turn_ids.append(torch.tensor(
                rollout_handler.response_turn_ids or [-1] * len(rollout_handler.response_loss_mask),
                dtype=torch.long, device=cur_device))
            scores.append(self._shape_task_reward(
                task_score=rollout_handler.score,
                task_done=rollout_handler.done,
                task_name=rollout_handler.task_name,
            ))
            messages.append(rollout_handler.messages)
        
        # pad to length
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_observation_mask = pad_sequence(response_observation_mask, batch_first=True, padding_value=0)
        if response_observation_mask.shape[1] < self.config.response_length:
            response_observation_mask = pad_sequence_to_length(response_observation_mask, self.config.response_length, 0)
        # Pad turn ids with -1, never 0: 0 is a valid turn index, and padding with it
        # would fold padding positions into the first action turn.
        response_turn_ids = pad_sequence(response_turn_ids, batch_first=True, padding_value=-1)
        if response_turn_ids.shape[1] < self.config.response_length:
            response_turn_ids = pad_sequence_to_length(response_turn_ids, self.config.response_length, -1)
        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        prompt_length = input_ids.size(-1)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
        response_mask = response_loss_mask
        observation_mask = response_attention_mask * (1 - response_mask)

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32) # (bs, response_length)
        valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
        for i in range(len(scores)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if global_steps:
            try:
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        records = {
                            "item_id": rollout_handler_ls[idx].item_id,
                            "conversations": [msg.to_dict() for msg in msgs],
                            "reward": scores[idx]
                        }
                        json_msg.append(records)
                    json.dump(json_msg, f, ensure_ascii=True, indent=4)
            except Exception as e:
                print(e)

        # close clients
        for client in env_clients:
            try:
                client.close()
            except Exception as e:
                print(f"Error during closing env: {e}")

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                'observation_mask': observation_mask,
                'scores': reward_tensor,
                'task_rounds': torch.tensor(task_rounds, dtype=torch.float32).to(input_ids.device),
                'task_scores': reward_tensor
            },
            batch_size=batch_size)
        # Temporal Ensembling needs per-token turn ids. Added only when te_enable is set,
        # so with TE off the batch has exactly the keys it had before.
        if bool(self.config.get('te_enable', False)):
            batch['turn_ids'] = response_turn_ids[:, :response_length]

        # One-time structural check for turn-level advantage estimators: the turn count
        # derived from observation_mask must match the rollout's own round count. The
        # mask is per-token and the round counter is per-episode, so a mismatch means
        # the two disagree about what a turn is -- which would silently misalign any
        # per-turn credit. Off by default; TAU2_CHECK_TURNS=1 to run it.
        if os.environ.get("TAU2_CHECK_TURNS") == "1":
            from verl.trainer.ppo.turn_structure import turn_boundaries_from_observation_mask
            tb = turn_boundaries_from_observation_mask(observation_mask, response_mask)
            derived = tb.sum(dim=-1)
            rounds = torch.tensor(task_rounds, device=derived.device, dtype=derived.dtype)
            bad = (derived - rounds).abs() > 1  # opening turn has no prior observation
            if bad.any():
                i = int(bad.nonzero()[0][0])
                print(f"[turn-check] MISMATCH sample {i}: derived={int(derived[i])} "
                      f"rounds={int(rounds[i])}")
            else:
                print(f"[turn-check] OK: {int(derived.min())}..{int(derived.max())} turns "
                      f"per sample, matches task_rounds")

        # Expose per-sample chat history so downstream passes (e.g. the
        # world-model SFT update) can re-assemble fresh chat-template data
        # instead of reusing the rollout sequence in-place.
        rollout_messages_np = np.empty(batch_size, dtype=object)
        for i, msgs in enumerate(messages):
            rollout_messages_np[i] = [m.to_dict() for m in msgs]
        non_tensor_batch = {'rollout_messages': rollout_messages_np}

        # free vllm cache engine
        self._free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
