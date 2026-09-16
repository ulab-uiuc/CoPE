"""Temporal Ensembling (TE)：把 plan_forecast 的时间视角预测聚合成策略的参考分布 q_t。

论文对应：
    q_t^F(a)   = (1/K_t) Σ_{k=1..K_t} π^k(a | h_{t-k})            时间混合
    q_t(a)     = (1-η) π^0_θ̄(a | h_t) + η q_t^F(a; θ̄)            冻结参考
    L_policy   = L_GRPO + λ_TE · E_{h_t~d_θ̄} KL(π_θ^0(·|h_t) ‖ q_t)

与 plan_forecast SFT 的关系：SFT 训练成员（af-loss），TE 把成员蒸馏回策略。
两者**必须共用同一套 forecast block 构造**，所以 π^k 是 block 的链式条件概率
p(a_t | h_s, a_s..a_{t-1})，而不是边缘 p(a_t | h_s)。这与论文"把 af-loss 按目标
时刻重新分组"是一致的（af-loss 里出现的就是链式分解项）。

⚠️ 成员筛选按**真实轮号**，不能按 slot 下标：
   skip_invalid=True 时，若发出轮 s 自己的动作无效，会被自己的过滤器滤掉，
   此时 action_turns[0] > s，slot 1 已经是 k>=1 的合法成员。
   规则：target t 的成员 = {(s,j) : action_turns[j] == t 且 s < t}
   （见 tests/test_te_action_turns.py，以及 /data1/logs/temporal_ensemble_design.md §0）

全模块只在 te_enable=True 时被调用；关闭时 ray_trainer 根本不 import 这里的东西。
"""
import math
from typing import Dict, List, Optional, Tuple

import torch

from verl.agent_trainer.ppo.plan_forecast import (
    build_plan_targets, _to_chat_list, DEFAULT_PLAN_PROMPT,
)


# --------------------------------------------------------------------------
# 1) 构造打分样本
# --------------------------------------------------------------------------
def build_te_scoring_samples(messages, tokenizer, k: int = 3,
                             skip_invalid: bool = False, env: str = "sciworld",
                             max_length: int = 4096) -> List[Dict]:
    """一条轨迹 → 每个动作轮一条打分样本（带 slot→轮号 与 slot→token span）。

    编码路径与 plan_forecast 的 build_plan_forecast_samples **逐字一致**：
        prefix = convo[:obs_s+1] + {'role':'user', 'content': PROMPT.format(k=realized)}
        target = {'role':'assistant', 'content': "\\n".join(items)}
    注意 target 是裸的换行连接。
    这里用 tokenizer(full_text, return_offsets_mapping=True)，与 encode_sft_sample
    的 tokenizer(full_text) 得到同一串 token id，额外拿到字符偏移用于定位 slot。

    返回的每条样本：
        input_ids / attention_mask / loss_mask : torch.LongTensor，与 SFT 同构
        slot_turns : List[int]        每个 slot 的真实动作轮号
        slot_spans : List[(s,e)]      每个 slot 的动作正文在 input_ids 里的 [s,e)
        src_turn   : int              这条 forecast 的发出轮 s
    """
    convo = _to_chat_list(messages)
    out: List[Dict] = []
    for tgt in build_plan_targets(messages, k=k, skip_invalid=skip_invalid, env=env):
        items = tgt.get('actions') or []
        turns = tgt.get('action_turns') or []
        src = tgt.get('src_turn')
        if not items or src is None or len(turns) != len(items):
            continue
        # 至少要有一个 k>=1 的成员（按轮号判断，不能按 slot 下标）
        if not any(t > src for t in turns):
            continue

        prefix = list(convo[:tgt['prefix_end'] + 1])
        prefix.append({'role': 'user',
                       'content': DEFAULT_PLAN_PROMPT.format(k=len(items))})
        content = "\n".join(items)
        target_msgs = [{'role': 'assistant', 'content': content}]
        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                prefix + target_msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue

        enc = tokenizer(full_text, add_special_tokens=False,
                        return_offsets_mapping=True)
        ids = enc['input_ids']
        offs = enc['offset_mapping']
        if not ids:
            continue

        # content 在 full_text 中的位置（取最后一次出现：它就在结尾附近）
        try:
            cpos = full_text.rindex(content)
        except ValueError:
            continue
        # 每个 item 在 content 内的字符区间（items 用 "\n" 连接）
        spans_tok: List[Tuple[int, int]] = []
        off_in_content = 0
        ok = True
        for it in items:
            cs, ce = cpos + off_in_content, cpos + off_in_content + len(it)
            off_in_content += len(it) + 1        # +1 是分隔的 "\n"
            idx = [j for j, (a, b) in enumerate(offs) if a < ce and b > cs]
            if not idx:
                ok = False
                break
            spans_tok.append((min(idx), max(idx) + 1))
        if not ok:
            continue

        # prefix_len：与 encode_sft_sample 完全同一套判定
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)['input_ids']
        if full_text.startswith(prefix_text):
            prefix_len = len(prefix_ids)
        else:
            common = 0
            for i in range(min(len(prefix_ids), len(ids))):
                if prefix_ids[i] != ids[i]:
                    break
                common = i + 1
            prefix_len = common

        input_ids = torch.tensor(ids, dtype=torch.long)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[prefix_len:] = 1

        # 左截断（与 encode_sft_sample 同样从左边丢），span 同步平移
        if input_ids.size(0) > max_length:
            drop = input_ids.size(0) - max_length
            input_ids = input_ids[drop:]
            loss_mask = loss_mask[drop:]
            spans_tok = [(s - drop, e - drop) for (s, e) in spans_tok]
            if any(s < 0 for (s, _) in spans_tok):
                continue                        # 有 slot 被截掉了，整条丢弃

        out.append(dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            loss_mask=loss_mask,
            slot_turns=list(turns),
            slot_spans=spans_tok,
            src_turn=int(src),
        ))
    return out


# --------------------------------------------------------------------------
# 2) 从 logits 取每个 slot 的序列级 log-prob
# --------------------------------------------------------------------------
@torch.no_grad()
def slot_logprobs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                              slot_spans: List[List[Tuple[int, int]]],
                              ) -> List[List[float]]:
    """每条样本的每个 slot：该 slot 动作正文 token 的 log-prob **之和**（序列级）。

    shift 约定：位置 i 的 logits 预测 token i+1，所以 token 位置 p 的 log-prob
    取 gathered[p-1]（gathered 已经是 shift 过的 [B, T-1]）。
    """
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tok = input_ids[:, 1:]
    gathered = torch.gather(logp, 2, tok.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
    res: List[List[float]] = []
    for b, spans in enumerate(slot_spans):
        row = []
        for (s, e) in spans:
            lo, hi = max(s - 1, 0), max(e - 1, 0)
            row.append(float(gathered[b, lo:hi].sum()) if hi > lo else float('nan'))
        res.append(row)
    return res


def slot_token_logprobs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                                    slot_spans: List[List[Tuple[int, int]]],
                                    ) -> List[List[List[float]]]:
    """同 slot_logprobs_from_logits，但返回**逐 token** log-prob（不求和）。

    逐 token 混合需要它：成员和 rollout 打的是同一串动作 token（已验证 98.0%
    逐 token 完全一致、0% 长度不等，另 2.0% 仅末尾换行 token 不同且都是
    extract_action 未命中的退化样本），位置可 1:1 对齐。

    返回 res[b][slot] = [该 slot 每个 token 的 log-prob]（空 span 给 []）。
    """
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tok = input_ids[:, 1:]
    gathered = torch.gather(logp, 2, tok.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
    res: List[List[List[float]]] = []
    for b, spans in enumerate(slot_spans):
        row = []
        for (s, e) in spans:
            lo, hi = max(s - 1, 0), max(e - 1, 0)
            row.append([float(x) for x in gathered[b, lo:hi]] if hi > lo else [])
        res.append(row)
    return res


# --------------------------------------------------------------------------
# 3) 合成 q_t
# --------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 全词表逐 token KL（te_mix='fullvocab'）所需的 top-M 提取与混合
#
# 为什么需要它（2026-09-15/16 两次失败的共同根因）：
#   只在"实际发生的 token"上算 log q - log π，得到的是**无归一化**的目标：
#     - 原始 KL 形式：有不动点，但只覆盖动作 token，模型把概率质量挪到非动作
#       文本上即可规避 -> response_length 931->3264，成功率归零。
#     - 扣共模后：逃逸路径堵死了，但损失退化成线性项 w·log π（w 为常数），
#       w>0 时最小值在 log π->-inf、w<0 时在 π->1，**两个方向都无界、没有不动点**
#       -> 熵 0.37->6.72，策略输出变成词沙拉。
#   真 KL 需要在整个词表上求和：它在 π_θ=q 处取 0 且梯度为 0（有不动点），
#   而把质量挪给任何别的 token 都会让 KL 变大（无逃逸路径）。两个毛病同时解决。
#
# 实现要点：
#   q(v) = [(1-η)·p0(v) + η·qF(v)] / Z，其中
#     p0 = 冻结策略的下一 token 分布 —— **不需要传输**，在 dp_actor 里用当前
#          logits 的 detach 版取得（第一个内层 epoch θ=θ̄ 时严格相等）；
#     qF = 成员预测分布的平均，只以 top-M 稀疏形式运送；
#     Z  = (1-η) + η·(qF 的 top-M 质量)，修正 top-M 截断带来的次归一化。
#   于是 q(v) >= (1-η)p0(v)/Z 处处成立，对数比有界(<= -log(1-η))，
#   无需任何平滑，也不会除零。
# ---------------------------------------------------------------------------

TE_TOPM = 64          # 每个动作 token 位置保留的词表 top-M
TE_TOPM_MAXTOK = 16   # 每个动作最多保留多少 token（实测中位 2，16 覆盖 98.51%）


def slot_topk_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                          slot_spans: List[List[Tuple[int, int]]],
                          topm: int = TE_TOPM, maxtok: int = TE_TOPM_MAXTOK):
    """每条样本的每个 slot，取其动作 token 位置上的下一 token 分布 top-M。

    shift 约定与 slot_logprobs_from_logits 一致：token 位置 p 的分布来自 logits[p-1]。
    返回 (ids, probs)，形状均为 [B, n_slot_max, maxtok, topm]，
    ids 用 -1 填充、probs 用 0 填充（padding 位置对混合无贡献）。
    """
    B = logits.shape[0]
    n_slot = max((len(sp) for sp in slot_spans), default=0)
    n_slot = max(n_slot, 1)
    ids = torch.full((B, n_slot, maxtok, topm), -1, dtype=torch.int32)
    prs = torch.zeros((B, n_slot, maxtok, topm), dtype=torch.float32)
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)      # [B, T-1, V]
    for b, spans in enumerate(slot_spans):
        for j, (st, en) in enumerate(spans[:n_slot]):
            lo, hi = max(st - 1, 0), max(en - 1, 0)
            hi = min(hi, lo + maxtok)
            if hi <= lo:
                continue
            v, i = torch.topk(lp[b, lo:hi], k=min(topm, lp.shape[-1]), dim=-1)
            n = hi - lo
            m = v.shape[-1]
            prs[b, j, :n, :m] = v.exp().cpu()
            ids[b, j, :n, :m] = i.to(torch.int32).cpu()
    return ids, prs


def assemble_q(member_lp: Dict[int, List[float]], pi0_lp: Dict[int, float],
               eta: float) -> Dict[int, Dict[str, float]]:
    """把成员 log-prob 合成 log q_t。

    member_lp[t] : [log π^k(a_t|h_{t-k}) for k>=1]（已按 t>s 筛过）
    pi0_lp[t]    : log π^0_θ̄(a_t|h_t)，来自 old_log_probs 按 turn_ids 分段求和
    返回 {t: {'log_q','log_qF','K_t'}}；无成员（含 t=0）时 log_q = pi0，
    此时 TE-KL 退化为信任域项 KL(π_θ‖π_θ̄)，与论文 q_0 = π^0_θ̄ 一致。
    """
    out: Dict[int, Dict[str, float]] = {}
    for t, p0 in pi0_lp.items():
        ms = [x for x in member_lp.get(t, []) if x == x]          # 滤 NaN
        if not ms or eta <= 0.0:
            out[t] = dict(log_q=float(p0), log_qF=float('nan'), K_t=0)
            continue
        mt = torch.tensor(ms, dtype=torch.float64)
        log_qF = float(torch.logsumexp(mt, 0) - math.log(len(ms)))
        a = torch.tensor(math.log1p(-eta) + float(p0), dtype=torch.float64)
        b = torch.tensor(math.log(eta) + log_qF, dtype=torch.float64)
        out[t] = dict(log_q=float(torch.logaddexp(a, b)), log_qF=log_qF, K_t=len(ms))
    return out


def segment_sum_by_turn(values: torch.Tensor, turn_ids: torch.Tensor,
                        n_turn: int) -> torch.Tensor:
    """按轮号对 per-token 值分段求和 → [B, n_turn]。turn_ids 中 -1 的位置被忽略。

    用 one_hot+einsum 而非 for 循环，既快又**保住梯度**（策略侧 log π_θ 需要）。
    """
    valid = (turn_ids >= 0)
    oh = torch.nn.functional.one_hot(turn_ids.clamp(min=0), n_turn).to(values.dtype)
    oh = oh * valid.unsqueeze(-1).to(oh.dtype)
    return torch.einsum('bt,btn->bn', values, oh)


# --------------------------------------------------------------------------
# 4) batch 组装（trainer 侧调用）
# --------------------------------------------------------------------------
def build_te_batch(messages_list, tokenizer, k: int, skip_invalid: bool, env: str,
                   max_length: int = 4096, pad_token_id: int = 0,
                   traj_subsample: float = 1.0, rng=None):
    """把所有轨迹的打分样本 padding 成一个 DataProto，并返回索引。

    索引 index[i] = (traj_idx, slot_turns, src_turn)，用于把第 i 条样本的
    slot log-prob 归到 (traj_idx, target_turn) 上。

    traj_subsample < 1 时随机抽一部分轨迹打分（KL 是期望，子采样无偏，方差变大），
    这是文档 §6 给的首选降本手段。
    """
    import random as _random
    from verl import DataProto
    from tensordict import TensorDict

    if rng is None:
        rng = _random
    samples, index = [], []
    n_traj_scored = 0
    for ti, msgs in enumerate(messages_list):
        if traj_subsample < 1.0 and rng.random() > traj_subsample:
            continue
        got = build_te_scoring_samples(msgs, tokenizer, k=k, skip_invalid=skip_invalid,
                                       env=env, max_length=max_length)
        if got:
            n_traj_scored += 1
        for smp in got:
            samples.append(smp)
            index.append((ti, smp['slot_turns'], smp['src_turn']))
    if not samples:
        return None, [], {'te/n_samples': 0.0, 'te/n_traj_scored': 0.0}

    L = max(int(s['input_ids'].size(0)) for s in samples)
    B = len(samples)
    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    shifted_spans = []
    for b, s in enumerate(samples):
        n = int(s['input_ids'].size(0))
        off = L - n                                    # 左填充，与 span 同步平移
        input_ids[b, off:] = s['input_ids']
        attn[b, off:] = 1
        pos[b, off:] = torch.arange(n, dtype=torch.long)
        shifted_spans.append([(a + off, e + off) for (a, e) in s['slot_spans']])

    import numpy as np
    batch = DataProto(
        batch=TensorDict({'input_ids': input_ids, 'attention_mask': attn,
                          'position_ids': pos}, batch_size=B),
        non_tensor_batch={'slot_spans': np.array(shifted_spans, dtype=object)},
    )
    meta = {'te/n_samples': float(B), 'te/n_traj_scored': float(n_traj_scored),
            'te/max_len': float(L)}
    return batch, index, meta


TE_COV_MAX = 128      # 每条轨迹最多保留多少个被覆盖的动作 token


def pad_slot_dim(x: torch.Tensor, k: int) -> torch.Tensor:
    """把 [B, n_slot, ...] 的 slot 维补齐到 k（int32 用 -1、浮点用 0 填充）。

    各 micro-batch 的 slot 数可能不同，跨 worker 拼接前必须统一宽度。
    放在 TE 模块里而不是 dp_actor：后者的函数清单受惰性审计约束。
    """
    if x.shape[1] >= k:
        return x
    sh = list(x.shape)
    sh[1] = k - x.shape[1]
    fill = -1 if x.dtype == torch.int32 else 0
    return torch.cat([x, torch.full(sh, fill, dtype=x.dtype)], 1)


def assemble_te_topm(index, topm_ids, topm_prs, turn_ids: torch.Tensor,
                     n_members_needed=None, cov_max: int = TE_COV_MAX):
    """把成员的 top-M 分布归并成与 rollout 对齐的紧凑张量。

    输出（紧凑表示，避免 [bsz, resp_len, M] 那种 128x4096x64 的巨张量）：
        te_cov_pos [bsz, C]      每个被覆盖 token 在 response 中的位置，-1 = padding
        te_cov_ids [bsz, C, M]   该位置 qF 的 top-M token id，-1 = padding
        te_cov_prs [bsz, C, M]   对应概率（成员平均后），0 = padding
    qF = 成员分布的**算术平均**。成员之间 top-M 的 id 会重叠，必须按 id 合并求和，
    不能简单拼接 —— 所以用稠密 scatter_add 分块做（每块 64 个位置，64x152k float
    约 39MB），再取 top-M。这是正确性优先于技巧的选择。
    """
    bsz = turn_ids.shape[0]
    M = topm_prs.shape[-1] if topm_prs is not None and topm_prs.numel() else TE_TOPM
    # (traj, turn) -> [(sample_idx, slot_idx), ...]
    members = {}
    for si, (ti, slot_turns, src) in enumerate(index):
        for j, t in enumerate(slot_turns):
            if t <= src:                      # k=0 按论文排除
                continue
            members.setdefault((int(ti), int(t)), []).append((si, j))

    rows_traj, rows_pos, rows_src = [], [], []
    for (ti, t), ms in sorted(members.items()):
        if ti >= bsz:
            continue
        pos = (turn_ids[ti] == t).nonzero(as_tuple=True)[0]
        n = int(pos.numel())
        if n == 0:
            continue
        # 成员的 token 数必须与 rollout 一致，否则丢弃（不截断错位，同 token 版）
        good = []
        for (si, j) in ms:
            valid_tok = int((topm_ids[si, j, :, 0] >= 0).sum())
            if valid_tok == n:
                good.append((si, j))
        if not good or n > topm_ids.shape[2]:
            continue
        for u in range(min(n, cov_max)):
            rows_traj.append(ti)
            rows_pos.append(int(pos[u]))
            rows_src.append([(si, j, u) for (si, j) in good])

    P = len(rows_traj)
    te_cov_pos = torch.full((bsz, cov_max), -1, dtype=torch.int32)
    te_cov_ids = torch.full((bsz, cov_max, M), -1, dtype=torch.int32)
    te_cov_prs = torch.zeros((bsz, cov_max, M), dtype=torch.float32)
    meta = {'te/fv_positions': float(P)}
    if P == 0:
        return te_cov_pos, te_cov_ids, te_cov_prs, meta

    V = int(topm_ids.max().item()) + 1 if topm_ids.numel() else 1
    V = max(V, 1)
    slot_in_traj = {}
    CH = 64
    for c0 in range(0, P, CH):
        c1 = min(c0 + CH, P)
        dense = torch.zeros((c1 - c0, V), dtype=torch.float32)
        for r in range(c0, c1):
            srcs = rows_src[r]
            for (si, j, u) in srcs:
                ii = topm_ids[si, j, u].long()
                pp = topm_prs[si, j, u].float()
                keep = ii >= 0
                dense[r - c0].scatter_add_(0, ii[keep].clamp(max=V - 1),
                                           pp[keep] / len(srcs))
        v, i = torch.topk(dense, k=min(M, V), dim=-1)
        for r in range(c0, c1):
            ti = rows_traj[r]
            k = slot_in_traj.get(ti, 0)
            if k >= cov_max:
                continue
            slot_in_traj[ti] = k + 1
            te_cov_pos[ti, k] = rows_pos[r]
            m = v.shape[-1]
            te_cov_ids[ti, k, :m] = i[r - c0].to(torch.int32)
            te_cov_prs[ti, k, :m] = v[r - c0]
    meta['te/fv_mass_mean'] = float(te_cov_prs.sum(-1)[te_cov_pos >= 0].mean()) \
        if (te_cov_pos >= 0).any() else 0.0
    return te_cov_pos, te_cov_ids, te_cov_prs, meta


def fullvocab_te_kl(logits: torch.Tensor, cov_pos: torch.Tensor,
                    cov_ids: torch.Tensor, cov_prs: torch.Tensor,
                    eta: float):
    """全词表逐 token KL(π_θ ‖ q)，q = [(1-η)p0 + η qF] / Z。

    logits  : [B, T, V]（response 段，与 cov_pos 同一坐标系），带梯度
    cov_pos : [B, C]    被覆盖 token 在 response 中的位置，-1 = padding
    cov_ids : [B, C, M] qF 的 top-M token id，-1 = padding
    cov_prs : [B, C, M] 对应概率

    p0 取 softmax(logits).detach() —— 第一个内层 epoch θ=θ̄ 时严格等于冻结策略。
    由此 q(v) >= (1-η)p0(v)/Z 处处成立，对数比有界(<= -log(1-η))，无需平滑。

    恒等式（推导见设计文档）：
        KL(π‖q) = KL(π‖p0) - log((1-η)/Z)
                  - Σ_{v∈topM} π(v)·log1p( η·qF(v) / ((1-η)·p0(v)) )
    其中 KL(π‖p0) 在 p0=π.detach() 处取值与梯度均为 0，保留它是为了公式完整
    （也便于测试里用非自指的 p0 验证分解式）。

    梯度 = -Σ_topM ∇π(v)·log1p(...)，因 Σ_v ∇π(v)=0 而自动中心化：
    质量只在 token 之间转移，总量守恒 —— 这正是"扣共模"版本缺失的归一化约束。
    不动点在 π=qF：此时括号内为常数，梯度恰好归零。

    返回 (kl_mean, n_pos)。n_pos=0 时 kl_mean 为 0 张量（惰性）。
    """
    B, T, V = logits.shape
    valid = cov_pos >= 0
    n_pos = int(valid.sum())
    if n_pos == 0 or eta <= 0.0:
        return logits.sum() * 0.0, 0
    bi, ci = valid.nonzero(as_tuple=True)
    pos = cov_pos[bi, ci].long().clamp(0, T - 1)
    return fullvocab_te_kl_rows(logits[bi, pos], cov_ids[bi, ci], cov_prs[bi, ci], eta)


def fullvocab_te_kl_rows(z: torch.Tensor, ids: torch.Tensor, prs: torch.Tensor,
                         eta: float):
    """fullvocab_te_kl 的核心：直接吃已经取好的 logits 行。

    z   : [P, V] 被覆盖位置的 logits（带梯度）
    ids : [P, M] qF 的 top-M token id，-1 = padding
    prs : [P, M] 对应概率
    dp_actor 里用这个入口 —— 只把那几十行 [P,V] 带出前向，避免整块
    [bsz, resp_len, V]（resp_len=4096、V=152k 下是 GB 级）。
    """
    if z.numel() == 0 or eta <= 0.0:
        return z.sum() * 0.0, 0
    n_pos = z.shape[0]
    logp = torch.log_softmax(z.float(), dim=-1)
    p = logp.exp()
    logp0 = logp.detach()
    ids = ids.long()                                     # [P, M]
    prs = prs.float()                                    # [P, M]
    keep = ids >= 0
    ids_c = ids.clamp(min=0)
    qf = prs * keep                                      # padding 置 0
    mass = qf.sum(-1)                                    # top-M 覆盖的质量
    Z = (1.0 - eta) + eta * mass                         # [P] 次归一化修正
    p0_sel = logp0.gather(1, ids_c).exp()                # [P, M]
    ratio = (eta * qf) / ((1.0 - eta) * p0_sel.clamp_min(1e-20))
    corr = torch.log1p(ratio) * keep                     # [P, M]
    p_sel = p.gather(1, ids_c)
    kl_p_p0 = (p * (logp - logp0)).sum(-1)               # 值与梯度均为 0（自指时）
    kl = kl_p_p0 - torch.log((1.0 - eta) / Z) - (p_sel * corr).sum(-1)
    return kl.mean(), n_pos


def assemble_te_tensors_token(index, slot_tok_lp, old_log_probs: torch.Tensor,
                              turn_ids: torch.Tensor, eta: float,
                              traj_return=None):
    """**逐 token** 版：返回与 old_log_probs 同形的 [bsz, T] te_log_q。

    为什么必须逐 token（2026-09-15 dry-run 实测）：
      序列级混合下 gain = log qF - log π⁰ ≈ -7.6（= 每 token -2.3 × 动作 3.3 token），
      qF 在 q=(1-η)π⁰+η·qF 里的权重只有 3e-4，q ≈ (1-η)π⁰，于是
      log q - log π⁰ ≡ log(1-η) 对所有 target 都是同一个常数，k3 梯度
      1 - e^{kl} 也是常数 —— 退化成"对所有动作等量压低"，不含方向信息。
      换成逐 token 后每 token gain ≈ -0.6，η=0.5 时 qF 权重就有 0.33-0.41，
      且 win/fail 的 q 真正分开（Δ ≈ 0.14 nats/token）。

    对齐前提已离线验证：成员 slot span 的 token 串与 rollout 侧 turn_ids 标出的
    动作 token 串，1677 对里 98.0% 逐 token 完全相同、0% 长度不等；余下 2.0%
    长度相同仅末尾换行 token 不同。长度不等的成员在这里直接丢弃（计入
    te/token_len_mismatch），绝不做截断对齐 —— 错位混合比不混合更糟。
    """
    bsz, T = old_log_probs.shape
    te_log_q = old_log_probs.float().clone()          # 默认 = π⁰ → KL 恒为 0
    te_valid = torch.zeros_like(te_log_q, dtype=torch.bool)

    members = {}          # (traj, t) -> [[tok logp...], ...]
    n_k0 = 0
    for (ti, slot_turns, src), row in zip(index, slot_tok_lp):
        for j, t in enumerate(slot_turns):
            if j >= len(row):
                continue
            toks = row[j]
            if not toks or any(v != v for v in toks):     # 空或含 NaN
                continue
            if t <= src:                                  # k=0，按论文排除
                n_k0 += 1
                continue
            members.setdefault((ti, int(t)), []).append(list(toks))

    Ks, gains, n_mismatch = [], [], 0
    log1m_eta = math.log1p(-eta) if eta < 1.0 else float('-inf')
    log_eta = math.log(eta) if eta > 0.0 else float('-inf')
    for (ti, t), ms in members.items():
        if ti >= bsz:
            continue
        pos = (turn_ids[ti] == t).nonzero(as_tuple=True)[0]
        n = int(pos.numel())
        if n == 0:
            continue
        good = [m for m in ms if len(m) == n]
        n_mismatch += len(ms) - len(good)
        if not good or eta <= 0.0:
            continue
        p0_vec = old_log_probs[ti, pos].float()
        mt = torch.tensor(good, dtype=torch.float64)                  # [K, n]
        log_qF = torch.logsumexp(mt, 0) - math.log(mt.shape[0])       # [n]
        a = log1m_eta + p0_vec.to(torch.float64)
        b = log_eta + log_qF
        te_log_q[ti, pos] = torch.logaddexp(a, b).to(te_log_q.dtype)
        te_valid[ti, pos] = True
        Ks.append(mt.shape[0])
        gains.append((ti, float((log_qF - p0_vec.to(torch.float64)).mean())))

    # 共模基线 g_bar：k3 估计量对 log π 的梯度是 g = 1 - exp(log q - log π⁰)。
    # 因为 gain 恒为负、q < π⁰，g 恒为正 —— 即"对所有动作 token 一律往下压"。
    # 2026-09-15 实测：共模 ≈0.20/token，而 win/fail 的差只有 0.13，共模比信号还大；
    # 且 TE 只覆盖裸动作 token，模型可以把概率质量挪到非动作文本上来规避，
    # 于是 response_length 从 931 涨到 3264、成功率归零、te/kl 自我满足地跌到 0。
    # 扣掉 g_bar 后该项零均值：只有"比平均更难预测"的 token 被压，写废话不再有收益。
    # ⚠️ 必须在**整个 batch** 上算。dp_actor 的 micro_batch_size_per_gpu=1，
    # 在 micro-batch 内中心化 = 按轨迹中心化，会把 win/fail 差异整个抹平。
    _g_bar = float('nan')
    if int(te_valid.sum()) > 0:
        with torch.no_grad():
            _kl_raw = (te_log_q - old_log_probs.float())[te_valid]
            _g_bar = float((1.0 - torch.exp(_kl_raw)).mean())
    meta = {
        'te/members_mean': float(sum(Ks) / len(Ks)) if Ks else 0.0,
        'te/targets_with_members': float(len(Ks)),
        'te/k0_slots_excluded': float(n_k0),
        'te/token_len_mismatch': float(n_mismatch),
        'te/tokens_covered': float(int(te_valid.sum())),
        'te/g_bar': _g_bar,
        'te/gain_tok_all': float(sum(g for _, g in gains) / len(gains)) if gains else float('nan'),
    }
    if traj_return is not None and gains:
        gw = [g for ti, g in gains if float(traj_return[ti]) > 0.5]
        gf = [g for ti, g in gains if float(traj_return[ti]) <= 0.5]
        meta['te/gain_tok_win'] = float(sum(gw) / len(gw)) if gw else float('nan')
        meta['te/gain_tok_fail'] = float(sum(gf) / len(gf)) if gf else float('nan')
        meta['te/n_win_targets'] = float(len(gw))
        meta['te/n_fail_targets'] = float(len(gf))
    return te_log_q, te_valid, meta


def assemble_te_tensors(index, slot_lp, old_log_probs: torch.Tensor,
                        turn_ids: torch.Tensor, eta: float,
                        traj_return=None):
    """把 slot log-prob 归并成 [bsz, n_turn] 的 te_log_q / te_valid。

    index[i]   = (traj_idx, slot_turns, src_turn)
    slot_lp[i] = [该样本每个 slot 的序列级 log-prob]
    old_log_probs / turn_ids : [bsz, T]，rollout 的冻结 log-prob 与轮号

    成员筛选：只保留 slot_turns[j] > src_turn 的（论文的 k>=1）。
    ⚠️ 不能按 slot 下标 j>=1 筛 —— skip_invalid=True 时 slot1 可能已是 k>=1。
    """
    bsz = old_log_probs.shape[0]
    n_turn = int(turn_ids.max().item()) + 1 if (turn_ids >= 0).any() else 1
    n_turn = max(n_turn, 1)
    # π^0 分量：按轮对 old_log_probs 分段求和（不额外做前向）
    pi0 = segment_sum_by_turn(old_log_probs.float(), turn_ids, n_turn)   # [bsz,n_turn]
    has_turn = segment_sum_by_turn(torch.ones_like(old_log_probs.float()),
                                   turn_ids, n_turn) > 0                 # [bsz,n_turn]

    members = {}          # (traj, t) -> [logp,...]
    n_k0 = 0
    for (ti, slot_turns, src), row in zip(index, slot_lp):
        for j, t in enumerate(slot_turns):
            if j >= len(row):
                continue
            v = row[j]
            if v != v:                       # NaN
                continue
            if t <= src:                     # k=0，按论文排除（用 rollout actor 代替）
                n_k0 += 1
                continue
            members.setdefault((ti, int(t)), []).append(float(v))

    te_log_q = pi0.clone()                   # 默认 = π^0（无成员/t=0 时的退化分支）
    te_valid = has_turn.clone()
    # gains 存 (traj_idx, gain) —— **不能**只存 gain 再靠 zip 去配 members.items()：
    # gains 仅在通过过滤且 log_qF 非 NaN 时才追加，而 members.items() 是全部条目，
    # 按位置 zip 会把 gain 配到错误的轨迹上，gain_win/gain_fail 随之失去意义。
    # (2026-09-15 修，dry-run 前两步的 gain_win 数值即受此影响，已作废)
    # gains 存 (traj_idx, gain, n_tok)。n_tok = 该轮裸动作的 token 数，用来算
    # **每 token** 的 gain。序列级 gain 会随动作长度线性放大（-1.6 nats/tok × 5 tok
    # = -8），单看它无法判断 forecast 到底差多少；per-token 才是可比的量，也直接
    # 决定 qF 在混合 q=(1-η)π⁰+η·qF 里占多大权重。
    Ks, gains = [], []
    for (ti, t), ms in members.items():
        if t >= n_turn or ti >= bsz or not has_turn[ti, t]:
            continue
        r = assemble_q({t: ms}, {t: float(pi0[ti, t])}, eta)[t]
        te_log_q[ti, t] = r['log_q']
        Ks.append(r['K_t'])
        if r['log_qF'] == r['log_qF']:
            n_tok = int((turn_ids[ti] == t).sum())
            gains.append((ti, r['log_qF'] - float(pi0[ti, t]), max(n_tok, 1)))

    meta = {
        'te/members_mean': float(sum(Ks) / len(Ks)) if Ks else 0.0,
        'te/targets_with_members': float(len(Ks)),
        'te/targets_total': float(has_turn.sum()),
        'te/k0_slots_excluded': float(n_k0),
        'te/gain_all': float(sum(g for _, g, _ in gains) / len(gains)) if gains else float('nan'),
        'te/span_len_mean': float(sum(n for _, _, n in gains) / len(gains)) if gains else 0.0,
        'te/gain_tok_all': float(sum(g / n for _, g, n in gains) / len(gains)) if gains else float('nan'),
    }
    # 分成功/失败统计 gain —— 这是判断方法是否成立的关键指标
    if traj_return is not None and gains:
        gw = [(g, n) for ti, g, n in gains if float(traj_return[ti]) > 0.5]
        gf = [(g, n) for ti, g, n in gains if float(traj_return[ti]) <= 0.5]
        meta['te/gain_win'] = float(sum(g for g, _ in gw) / len(gw)) if gw else float('nan')
        meta['te/gain_fail'] = float(sum(g for g, _ in gf) / len(gf)) if gf else float('nan')
        meta['te/gain_tok_win'] = float(sum(g / n for g, n in gw) / len(gw)) if gw else float('nan')
        meta['te/gain_tok_fail'] = float(sum(g / n for g, n in gf) / len(gf)) if gf else float('nan')
        meta['te/n_win_targets'] = float(len(gw))    # 样本量，太小则 gain_win 不可信
        meta['te/n_fail_targets'] = float(len(gf))
    return te_log_q, te_valid, meta
