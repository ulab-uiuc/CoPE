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
Single Process Actor
"""

import itertools
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.agent_trainer.ppo.sft_common import (
    compute_sft_loss_from_logits,
    compute_observation_mask,
    compute_traj_lm_loss,
)
from verl.workers.agent_actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']

# Fixed width (tokens per action slot) of the per-token Temporal Ensembling scores.
TE_SLOT_MAX_TOK = 128


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        model_config=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature, return_hidden_states: bool = False,
                             te_cov_pos=None):
        """
        Returns:
            entropy: (bs, response_len)
            log_probs: (bs, response_len)
            full_hidden (optional, when return_hidden_states=True):
                (bs, seqlen, hidden) — last-layer hidden states over the FULL
                input sequence (prompt + response), needed for state-value
                queries at positions in the prompt region (turn 0).
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           output_hidden_states=return_hidden_states,
                                           use_cache=False)  # prevent model thinks we are generating
                if te_cov_pos is not None:
                    # Logits are packed here, so response positions do not map to rows the
                    # way they do on the dense path. Fail loudly rather than read wrong rows.
                    raise NotImplementedError(
                        'te_mix=fullvocab does not support use_remove_padding / ulysses sp yet')
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # also pull the last-layer hidden state if requested
                last_hidden_rmpad = None
                if return_hidden_states:
                    last_hidden_rmpad = output.hidden_states[-1].squeeze(0)  # (total_nnz_sp, hidden)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    if last_hidden_rmpad is not None:
                        last_hidden_rmpad = gather_outpus_and_unpad(last_hidden_rmpad,
                                                                    gather_dim=0,
                                                                    unpad_dim=0,
                                                                    padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

                if return_hidden_states:
                    full_hidden = pad_input(hidden_states=last_hidden_rmpad,
                                            indices=indices,
                                            batch=batch_size,
                                            seqlen=seqlen)  # (bsz, seqlen, hidden)
                    return entropy, log_probs, full_hidden

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           output_hidden_states=return_hidden_states,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                # Full-vocab TE-KL needs the complete logits rows at the covered positions.
                # Keep only those rows ([P, V], P ~ 30) instead of carrying the whole
                # [bsz, response_len, V] block, which is GBs at 4096 x 152k.
                if te_cov_pos is not None:
                    te_valid_cov = te_cov_pos >= 0
                    if te_valid_cov.any():
                        te_bi, te_ci = te_valid_cov.nonzero(as_tuple=True)
                        te_pos = te_cov_pos[te_bi, te_ci].long().clamp(0, logits.shape[1] - 1)
                        self._te_logits_rows = logits[te_bi, te_pos]
                        self._te_logits_bi = te_bi
                        self._te_logits_ci = te_ci
                    else:
                        self._te_logits_rows = None

                if return_hidden_states:
                    full_hidden = output.hidden_states[-1]  # (bsz, seqlen, hidden)
                    return entropy, log_probs, full_hidden

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # Skip the update when grad_norm is non-finite (NaN/inf). Stepping the optimizer
        # with a non-finite gradient permanently corrupts the weights -- a single bad
        # step yields irreversible policy collapse (outputs degenerate, reward -> 0).
        # Zero the poisoned grads so they cannot linger into the next accumulation.
        if not torch.isfinite(grad_norm):
            print(f"[optimizer_step] non-finite grad_norm ({grad_norm}); skipping optimizer.step()", flush=True)
            self.actor_optimizer.zero_grad()
            return grad_norm
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: log_prob tensor and,
            optionally, entropy tensor over response tokens.
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        return_entropy = data.meta_info.get('return_entropy', False)

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            if return_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if return_entropy else None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if return_entropy:
                entropys = entropys[revert_indices]

        if return_entropy:
            return log_probs, entropys
        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        # Temporal Ensembling settings must be read HERE. Below, `data.select(...).batch`
        # drops meta_info and the loop rebinds `data` to that bare TensorDict, so reading
        # meta_info inside the loop silently yields 0 and the TE term never reaches the
        # loss (a lambda=0 dry-run cannot reveal this).
        te_lambda = float(data.meta_info.get('te_lambda', 0.0))
        te_gbar = float(data.meta_info.get('te_gbar', 0.0))
        te_center = bool(self.config.get('te_center', True))
        te_mix = str(self.config.get('te_mix', 'token')).lower()

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'responses', 'response_mask']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        # TE tensors are dropped by the select unless listed. fullvocab computes its KL
        # even at lambda=0 (reported, not applied) so a dry-run exercises the same path.
        # With TE off none of these keys exist and select_keys is unchanged.
        if te_lambda > 0.0 or te_mix == 'fullvocab':
            for _k in ('te_log_q', 'te_valid', 'turn_ids',
                       'te_cov_pos', 'te_cov_ids', 'te_cov_prs'):
                if _k in data.batch.keys():
                    select_keys.append(_k)
        # traj_lm needs the env-observation tokens, so keep observation_mask when the
        # rollout carried one.
        if 'observation_mask' in data.batch.keys():
            select_keys.append('observation_mask')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                response_mask = data['response_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                _cov = data['te_cov_pos'] if (te_mix == 'fullvocab'
                                              and 'te_cov_pos' in data.keys()) else None
                self._te_logits_rows = None
                entropy, log_prob = self._forward_micro_batch(
                    micro_batch=data, temperature=temperature, te_cov_pos=_cov)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                             log_prob=log_prob,
                                                                             advantages=advantages,
                                                                             eos_mask=response_mask,
                                                                             cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                # ---- Temporal Ensembling: lambda_TE * KL(pi_theta || q) ----
                # Metrics go through append_to_dict so they are averaged across
                # micro-batches; a plain metrics[...] = keeps only the last one.
                _te_metrics = {}
                _fv_rows = getattr(self, '_te_logits_rows', None)
                if te_mix == 'fullvocab' and _fv_rows is not None:
                    # Full-vocabulary per-token KL: the only form with a fixed point that
                    # cannot be evaded. The log-ratio at realized tokens has no
                    # normalization (policy moved mass to non-action text: response
                    # length 931 -> 3264); removing its common mode left an unbounded
                    # linear term (entropy 0.37 -> 6.72).
                    from verl.agent_trainer.ppo.temporal_ensemble import fullvocab_te_kl_rows
                    _bi, _ci = self._te_logits_bi, self._te_logits_ci
                    _kl_te, _npos = fullvocab_te_kl_rows(
                        _fv_rows, data['te_cov_ids'][_bi, _ci], data['te_cov_prs'][_bi, _ci],
                        float(self.config.get('te_eta', 0.5)))
                    if te_lambda > 0.0:   # lambda=0: report only, same code path
                        policy_loss = policy_loss + te_lambda * _kl_te
                    _te_metrics = {
                        'te/kl': _kl_te.detach().item(),
                        'te/lambda': te_lambda,
                        'te/fv_positions_mb': float(_npos),
                    }
                    self._te_logits_rows = None
                elif te_lambda > 0.0 and 'te_log_q' in data.keys() and 'turn_ids' in data.keys():
                    # Legacy te_mix=token/seq. Both were shown to degenerate; kept only to
                    # reproduce those runs.
                    _tid = data['turn_ids']
                    _logq = data['te_log_q']
                    _valid = data['te_valid'].to(_logq.dtype)
                    if _logq.shape == log_prob.shape:
                        _logp_ref = log_prob
                    else:
                        from verl.agent_trainer.ppo.temporal_ensemble import segment_sum_by_turn
                        _logp_ref = segment_sum_by_turn(log_prob, _tid, _logq.shape[1])
                    _den = _valid.sum().clamp(min=1.0)
                    _kld_te = core_algos.kl_penalty(
                        logprob=_logp_ref, ref_logprob=_logq,
                        kl_penalty=self.config.get('te_kl_type', 'low_var_kl'))
                    _kl_raw = ((_kld_te * _valid).sum() / _den).detach()
                    if te_center:
                        with torch.no_grad():
                            _g = 1.0 - torch.exp(_logq - _logp_ref.detach())
                            _w = (_g - te_gbar) * _valid
                        _kl_te = (_w * _logp_ref).sum() / _den
                        _w_absmean = float(_w.abs().sum() / _den)
                    else:
                        _kl_te = (_kld_te * _valid).sum() / _den
                        _w_absmean = float('nan')
                    policy_loss = policy_loss + te_lambda * _kl_te
                    _te_metrics = {
                        'te/kl': _kl_raw.item(),
                        'te/loss_term': _kl_te.detach().item(),
                        'te/w_absmean': _w_absmean,
                        'te/gbar': te_gbar,
                        'te/lambda': te_lambda,
                        'te/turns_used': float(_valid.sum()),
                    }

                # Full-trajectory LM SFT: next-token CE over the WHOLE response region
                # (obs tokens AND the agent's own tokens), no masking. Default off.
                traj_lm_coef = float(self.config.get('traj_lm_coef', 0.0))
                if traj_lm_coef > 0:
                    _obs_mask = compute_observation_mask(
                        attention_mask=data['attention_mask'],
                        response_mask=response_mask,
                        observation_mask=data.get('observation_mask', None))
                    # gate='wins': only clone trajectories with positive GRPO advantage
                    # (better than group mean) — avoids BC'ing losing behavior, which
                    # otherwise anchors the policy to base and slows early learning.
                    _row_mask = None
                    if str(self.config.get('traj_lm_gate', 'all')).lower() == 'wins':
                        _adv_pt = (advantages * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1.0)
                        _row_mask = (_adv_pt > 0).to(log_prob.dtype)
                        metrics['actor/traj_lm_frac_wins'] = _row_mask.mean().detach().item()
                    traj_lm_loss = compute_traj_lm_loss(log_prob, response_mask, _obs_mask, row_mask=_row_mask)
                    policy_loss = policy_loss + traj_lm_coef * traj_lm_loss
                    metrics['actor/traj_lm_loss'] = traj_lm_loss.detach().item()
                    metrics['actor/traj_lm_coef'] = traj_lm_coef

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                data.update(_te_metrics)   # empty dict when TE is off
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    def compute_te_log_prob(self, data: DataProto):
        """Temporal Ensembling: score every forecast slot under the frozen policy.

        Inference only (no graph, no backward). The batch carries input_ids /
        attention_mask / position_ids; non_tensor_batch['slot_spans'] holds each
        sample's [(start, end), ...] already shifted for left padding. Called by the
        trainer only when te_enable is set.

        Returns four tensors, all indexed by sample:
          [B, K]                            sequence log-prob per slot
          [B, K, TE_SLOT_MAX_TOK]           per-token log-probs (NaN padded)
          [B, K_slot, TOPM_MAXTOK, TOPM]    top-M token ids per action token (-1 padded)
          [B, K_slot, TOPM_MAXTOK, TOPM]    their probabilities (0 padded)
        Everything goes back as tensors in the batch: this runs under
        Dispatch.DP_COMPUTE_PROTO, which concatenates batch tensors across workers but
        keeps meta_info from the first shard only (returning Python lists in meta_info
        silently lost 3/4 of the scores on 4 GPUs). Widths are fixed for the same
        reason -- per-worker maxima differ and would not concatenate.
        """
        from verl.agent_trainer.ppo.temporal_ensemble import (
            slot_logprobs_from_logits, slot_token_logprobs_from_logits,
            slot_topk_from_logits, pad_slot_dim, TE_TOPM, TE_TOPM_MAXTOK)
        self.actor_module.eval()
        mbs = int(self.config.get('te_micro_batch_size_per_gpu', 1))
        ids = data.batch['input_ids']
        attn = data.batch['attention_mask']
        pos = data.batch['position_ids']
        spans_all = data.non_tensor_batch['slot_spans']
        out, out_tok = [], []
        _fv = str(self.config.get('te_mix', 'token')).lower() == 'fullvocab'
        topm_ids, topm_prs = [], []
        for b0 in range(0, ids.size(0), mbs):
            b1 = min(b0 + mbs, ids.size(0))
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.actor_module(input_ids=ids[b0:b1],
                                           attention_mask=attn[b0:b1],
                                           position_ids=pos[b0:b1],
                                           use_cache=False).logits
            spans = list(spans_all[b0:b1])
            out += slot_logprobs_from_logits(logits, ids[b0:b1], spans)
            out_tok += slot_token_logprobs_from_logits(logits, ids[b0:b1], spans)
            if _fv:
                _i, _p = slot_topk_from_logits(logits, ids[b0:b1], spans)
                topm_ids.append(_i)
                topm_prs.append(_p)
            del logits
        torch.cuda.empty_cache()

        K = int(self.config.get('action_forecast_k', 3))   # at most K slots per sample
        t = torch.full((len(out), max(K, 1)), float('nan'), dtype=torch.float32)
        for i, r in enumerate(out):
            r = r[:K]
            if r:
                t[i, :len(r)] = torch.tensor(r, dtype=torch.float32)
        # Slots longer than TE_SLOT_MAX_TOK are truncated; the length check on the
        # assemble side then drops them rather than misaligning (actions average 3.3
        # tokens, so the bound is loose).
        TMAX = TE_SLOT_MAX_TOK
        tt = torch.full((len(out_tok), max(K, 1), TMAX), float('nan'), dtype=torch.float32)
        for i, r in enumerate(out_tok):
            for j, toks in enumerate(r[:K]):
                if toks:
                    m = min(len(toks), TMAX)
                    tt[i, j, :m] = torch.tensor(toks[:m], dtype=torch.float32)
        if _fv and topm_ids:
            K_slot = max(x.shape[1] for x in topm_ids)
            ti_ = torch.cat([pad_slot_dim(x, K_slot) for x in topm_ids], 0)
            tp_ = torch.cat([pad_slot_dim(x, K_slot) for x in topm_prs], 0)
        else:
            ti_ = torch.full((len(out), 1, TE_TOPM_MAXTOK, TE_TOPM), -1, dtype=torch.int32)
            tp_ = torch.zeros((len(out), 1, TE_TOPM_MAXTOK, TE_TOPM), dtype=torch.float32)
        return t, tt, ti_, tp_

    def update_action_forecast(self, data: DataProto):
        """SFT update on a action-forecast batch (predict the realized next-K actions).

        Takes a chat-template-assembled SFT batch
        (``input_ids``/``attention_mask``/``position_ids``/``loss_mask``, loss only
        on the realized next-K action tokens) and CE-from-logits over it.
        Scaled by ``action_forecast_coef``. Does NOT touch PG.
        """
        self.actor_module.train()

        coef = data.meta_info.get('action_forecast_coef',
                                  float(self.config.get('action_forecast_coef', 0.0)))
        # This is a separate Adam step, so ``coef`` only rescales the gradient and Adam's
        # normalization largely undoes it. ``action_forecast_lr_scale`` scales the learning
        # rate of this step instead (1.0 = the policy lr), which does change its size.
        af_lr_scale = float(self.config.get('action_forecast_lr_scale', 1.0))
        # metric namespace: 'action_forecast' (default) or 'sft_ablation' when the
        # RFT-style control reuses this same optimizer path (mutually exclusive).
        mp = data.meta_info.get('sft_metric_prefix', 'action_forecast')
        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'loss_mask']
        if 'loss_weight' in data.batch.keys():   # per-sample group-norm weight
            select_keys.append('loss_weight')
        if 'token_weight' in data.batch.keys():  # per-token call/message balancing weight
            select_keys.append('token_weight')
        batch = data.select(batch_keys=select_keys).batch

        mini_batch_size = self.config.get('sft_mini_batch_size',
                                          self.config.ppo_mini_batch_size)
        micro_batch_size = self.config.get('sft_micro_batch_size_per_gpu',
                                           self.config.ppo_micro_batch_size_per_gpu)

        metrics: dict = {}
        dataloader = batch.split(mini_batch_size) if mini_batch_size else [batch]

        for mini_batch in dataloader:
            micro_batches = mini_batch.split(micro_batch_size) if micro_batch_size else [mini_batch]
            gradient_accumulation = max(1, len(micro_batches))

            self.actor_optimizer.zero_grad()
            for micro in micro_batches:
                micro = micro.cuda()
                loss_mask = micro['loss_mask']
                if loss_mask.sum().item() == 0:
                    continue

                # Per-sample group-norm weight (mean 1; identity when absent). It goes
                # into the token-level aggregation of the loss rather than scaling the
                # aggregated scalar -- see compute_sft_loss_from_logits for why the
                # latter only works at one sample per micro-batch.
                lw = micro['loss_weight'] if 'loss_weight' in micro.keys() else None
                tw = micro['token_weight'] if 'token_weight' in micro.keys() else None

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = self.actor_module(
                        input_ids=micro['input_ids'],
                        attention_mask=micro['attention_mask'],
                        position_ids=micro['position_ids'],
                        use_cache=False,
                    )
                    af_loss = compute_sft_loss_from_logits(
                        logits=output.logits,
                        labels=micro['input_ids'],
                        loss_mask=loss_mask,
                        sample_weight=lw,
                        token_weight=tw,
                    )

                loss = coef * af_loss / gradient_accumulation
                loss.backward()

                append_to_dict(metrics, {
                    f'{mp}/sft_loss': af_loss.detach().item(),
                    f'{mp}/coef': coef,
                    f'{mp}/loss_weight_mean': (float(lw.float().mean().item()) if lw is not None else 1.0),
                    # Spread within this micro-batch. Always 0 at one sample per
                    # micro-batch; the batch-level spread is reported as
                    # action_forecast/batch_loss_weight_std by build_action_forecast_batch.
                    f'{mp}/loss_weight_std': (float(lw.float().std().item()) if (lw is not None and lw.numel() > 1) else 0.0),
                    f'{mp}/valid_tokens': loss_mask.sum().detach().item(),
                })

            if af_lr_scale != 1.0:
                _saved_lrs = [g['lr'] for g in self.actor_optimizer.param_groups]
                for g in self.actor_optimizer.param_groups:
                    g['lr'] = g['lr'] * af_lr_scale
            grad_norm = self._optimizer_step()
            if af_lr_scale != 1.0:
                for g, _lr in zip(self.actor_optimizer.param_groups, _saved_lrs):
                    g['lr'] = _lr
            append_to_dict(metrics, {f'{mp}/grad_norm': grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
