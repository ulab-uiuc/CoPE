from dataclasses import dataclass
from typing import List, Literal
from transformers import PreTrainedTokenizer
import torch


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

# Hybrid-reasoning models (e.g. Qwen3) open every assistant turn with
# <think>...</think> by default. An agent turn only gets a few hundred tokens, so
# thinking eats the budget and breaks Thought/Action parsing. The official switch is
# enable_thinking=False, which does not strip tags: it pre-fills an empty
# "<think>\n\n</think>\n\n" at the end of the generation prompt.
#
# add_assistant_message decides which loss-mask branch to take by matching the exact
# token suffix of input_ids, so the assistant prefix must match what
# apply_chat_template produced token for token. Rather than hard-code a model list,
# probe the tokenizer: if passing enable_thinking=False changes the rendered prompt,
# it is a thinking model. Older tokenizers (Qwen2.5) ignore the unknown kwarg and
# render identically, so they keep the original path.
_THINKING_CACHE: dict = {}
_EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


def _supports_thinking(tokenizer) -> bool:
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    if key in _THINKING_CACHE:
        return _THINKING_CACHE[key]
    probe = [{"role": "user", "content": "x"}]
    try:
        default = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
        nothink = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
        res = default != nothink
    except Exception:
        res = False
    _THINKING_CACHE[key] = res
    return res


def _thinking_kwargs(tokenizer) -> dict:
    return {"enable_thinking": False} if _supports_thinking(tokenizer) else {}


def _action_token_mask(tokenizer, content: str, response_ids, task_name: str):
    """Bool list, same length as ``response_ids``, marking the bare-action tokens.

    Temporal Ensembling scores a distribution over the action string, so the policy
    side must cover exactly those tokens. Including the Thought text made it ~9x
    longer than the forecast side, and the sequence log-probs then differed by
    10-28 nats from length alone.

    Uses return_offsets_mapping to map characters to tokens (the extracted action was
    a literal substring of the content in 2307/2307 measured turns). Any failure
    falls back to marking every token, i.e. the pre-TE behaviour, rather than
    silently dropping the turn.
    """
    n = len(response_ids)
    try:
        from verl.agent_trainer.ppo.plan_forecast import extract_action
        act = extract_action(content, env=(task_name or "").lower())
        if not act:
            return [True] * n
        cs = content.rindex(act)
        ce = cs + len(act)
        enc = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if len(offs) != n:   # tokenized differently from encode(content); give up
            return [True] * n
        mask = [(a < ce and b > cs) for (a, b) in offs]
        return mask if any(mask) else [True] * n
    except Exception:
        return [True] * n


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    def to_dict(self):
        return {'role': self.role, 'content': self.content}
    def __repr__(self):
        return str(self.to_dict())
    def __str__(self):
        return self.__repr_


class RolloutHandler:
    def __init__(
        self,
        messages: List[Message],
        task_name: str,
        item_id: int,
        score: float,
        done: bool,
        input_ids: List[int],
        prompt_ids: List[int],
        response_ids: List[int],
        attention_mask: List[int],
        prompt_attention_mask: List[int],
        response_attention_mask: List[int],
        position_ids: List[int],
        prompt_position_ids: List[int],
        response_position_ids: List[int],
        loss_mask: List[int],
        prompt_loss_mask: List[int],
        response_loss_mask: List[int],
        observation_mask: List[int],
        prompt_observation_mask: List[int],
        response_observation_mask: List[int],
        max_response_len: int = 8192,
        max_model_len: int = 32768,
    ):
        self.messages = messages
        self.task_name = task_name
        self.item_id = item_id
        self.score = score
        self.done = done
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.attention_mask = attention_mask
        self.prompt_attention_mask = prompt_attention_mask
        self.response_attention_mask = response_attention_mask
        self.position_ids = position_ids
        self.prompt_position_ids = prompt_position_ids
        self.response_position_ids = response_position_ids
        self.loss_mask = loss_mask
        self.prompt_loss_mask = prompt_loss_mask
        self.response_loss_mask = response_loss_mask
        self.observation_mask = observation_mask
        self.prompt_observation_mask = prompt_observation_mask
        self.response_observation_mask = response_observation_mask
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len  
        self.format_config: dict = {
            "qwen": {
                "assistat_prefix_msg": "\n<|im_start|>assistant\n",
                "assistat_suffix_msg": "<|im_end|>",
                "user_prefix_msg": "\n<|im_start|>user\n",
                "user_suffix_msg": "<|im_end|>",
            }
        }
        # Temporal Ensembling: which action turn each token belongs to (-1 = not part
        # of a bare action). Purely additive -- input_ids and every mask above are
        # untouched, and the constructor signature is unchanged. Maintained even when TE
        # is off (one list append per message); vllm_rollout only puts it in the batch
        # when te_enable is set.
        self.turn_ids = [-1] * len(self.input_ids)
        self.prompt_turn_ids = [-1] * len(self.prompt_ids)
        self.response_turn_ids = []
        self._assistant_turn = -1   # becomes 0 on the first add_assistant_message

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        conversations = [
            msg.to_dict() for msg in self.messages
        ]
        return tokenizer.apply_chat_template(conversations, add_generation_prompt=True,
                                             tokenize=True, **_thinking_kwargs(tokenizer))
    
    
    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='assistant', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["assistat_prefix_msg"]
        if _supports_thinking(tokenizer):
            # must match the generation prompt rendered with enable_thinking=False
            prefix_msg = prefix_msg + _EMPTY_THINK_BLOCK
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["assistat_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        response = tokenizer.encode(content, add_special_tokens=False)
        self._assistant_turn += 1
        _act_mask = _action_token_mask(tokenizer, content, response, self.task_name)
        _resp_turn = [self._assistant_turn if m else -1 for m in _act_mask]
        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = response
            _loss_mask = [1] * len(response)
            _turn = _resp_turn
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + response
            _loss_mask = [0] * len(prefix_token_ids) + [1] * len(response)
            _turn = [-1] * len(prefix_token_ids) + _resp_turn
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )
        append_token_ids += suffix_token_ids
        _loss_mask += [1] * len(suffix_token_ids)
        _turn += [-1] * len(suffix_token_ids)   # template tokens are not action text
        assert len(_turn) == len(append_token_ids), (len(_turn), len(append_token_ids))
        self.turn_ids += _turn
        _observation_mask = [0] * len(append_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.observation_mask += _observation_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask) == len(self.observation_mask), f"""Rollout Handler has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}, {len(self.observation_mask)=}"""
        
    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='user', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["user_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["user_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)

        _content_obs_mask = [1] * len(content_token_ids)

        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = content_token_ids
            _loss_mask = [0] * len(content_token_ids)
            _observation_mask = list(_content_obs_mask)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            _loss_mask = [0] * len(prefix_token_ids) + [0] * len(content_token_ids)
            _observation_mask = [0] * len(prefix_token_ids) + list(_content_obs_mask)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )

        append_token_ids += suffix_token_ids
        _loss_mask += [0] * len(suffix_token_ids)
        _observation_mask += [0] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        self.turn_ids += [-1] * len(append_token_ids)   # observations are no action turn
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.observation_mask += _observation_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask) == len(self.observation_mask), f"""Rollout Handler has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}, {len(self.observation_mask)=}"""
        
    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.observation_mask = self.observation_mask[: self.max_model_len]
        self.turn_ids = self.turn_ids[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
        self.response_observation_mask = self.observation_mask[len(self.prompt_observation_mask) :][: self.max_response_len]
        self.response_turn_ids = self.turn_ids[len(self.prompt_turn_ids) :][: self.max_response_len]
