#!/usr/bin/env python3
"""
Offline Entropy Analysis on MiniMax-M2.1 BabyAI Trajectories
============================================================
用小模型（Qwen3-1.7B，本地 Transformers）对已采集的 MiniMax-M2.1 BabyAI 轨迹做 forward，
统计小模型在每个 turn 位置的 token 熵。

这才是确定 prefix 切分点的正确方式：
  - 小模型熵高  => 小模型对这段轨迹"看不懂/预测不准"，适合放在 prefix 里由大模型提供
  - 小模型熵低  => 小模型已能自主跟上，适合作为 rollout 起始位置

统计三种粒度（均从小模型视角计算）：
  A. 每个 turn 首 token 的熵   —— 最轻量的信号
  B. Action 首 token 的熵      —— 决策时刻的不确定性（[[ 之后）
  C. 每个 turn 所有 token 的平均熵  —— 最稳定的信号

Teacher-Forcing 实现说明：
  - 本地加载 Qwen3-1.7B 模型与 tokenizer，直接做 teacher-forcing forward
  - prefix 只包含 target turn 之前的历史，不包含 target 本身
  - 统一使用 Qwen3-1.7B 的 apply_chat_template 生成 prefix/full prompt
  - 用 token 级边界切分 target span：target_ids = full_ids[len(prefix_ids):]

使用方式：
  conda activate verl
  python entropy_offline_minimax_traj.py \
      --traj_dir /Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356 \
      --model_path /Data/public/Qwen3-1.7B \
      --tokenizer_name /Data/public/Qwen3-1.7B \
      --output_dir /Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax \
      --max_samples 100 \
      --top_k 100
"""

import os
import sys
import json
import math
import re
import logging
import argparse
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

# ---------- 加载 Qwen3-1.7B tokenizer ----------
# 使用与 vLLM forward 模型 Qwen3-1.7B 对应的 tokenizer
# 确保精确计算 token 边界
TOKENIZER = None
MODEL = None
MODEL_DEVICE = None

def get_tokenizer(tokenizer_name: str = None):
    """延迟加载 tokenizer

    Args:
        tokenizer_name: tokenizer 模型名，默认使用 Qwen3-1.7B
    """
    global TOKENIZER
    if TOKENIZER is None:
        if tokenizer_name is None:
            tokenizer_name = "Qwen/Qwen3-1.7B"
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return TOKENIZER


def get_model(model_path: str, cuda_device: int = 0):
    """延迟加载 causal LM（teacher-forcing forward 用）。"""
    global MODEL, MODEL_DEVICE
    if MODEL is not None:
        return MODEL

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        MODEL_DEVICE = torch.device(f"cuda:{cuda_device}")
    else:
        dtype = torch.float32
        MODEL_DEVICE = torch.device("cpu")

    # 优先使用 SDPA (Scaled Dot Product Attention)，在 L20 上可获得 3x 加速
    # 如果 SDPA 不可用，自动回退到 eager
    attn_impl = "sdpa"
    
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    # 不使用 device_map="auto"：避免对 accelerate 的依赖；单卡直接 to(device)
    MODEL.to(MODEL_DEVICE)
    MODEL.eval()
    return MODEL


def _tokenize_with_bos(tokenizer, prompt: str) -> List[int]:
    """tokenize(prompt, add_special_tokens=False) + 可选 BOS，使首 token 也有条件上下文。"""
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    # Qwen tokenizer 可能没有显式 bos；用 eos 作为稳定起始上下文（只用于对齐 logprob）
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is not None:
        ids = [bos] + ids
    return ids


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """返回 needle 在 haystack 中首次出现的起始 index；找不到返回 None。"""
    if not needle or len(needle) > len(haystack):
        return None
    first = needle[0]
    max_i = len(haystack) - len(needle)
    for i in range(max_i + 1):
        if haystack[i] != first:
            continue
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _find_prefix_boundary(prefix_ids: List[int], full_ids: List[int], max_trim: int = 8) -> Optional[int]:
    """
    在 full_ids 中找 prefix 边界。

    注意：BPE 可能在拼接边界产生跨界 merge，导致 tokenize(prefix) 的尾部 token 与 tokenize(full) 不一致。
    这里允许裁掉 prefix_ids 末尾最多 max_trim 个 token，以找到稳定的“前缀匹配”边界。
    返回 boundary（full_ids 中 target 追加段起点 index）。
    """
    if len(full_ids) == 0:
        return None
    if len(prefix_ids) == 0:
        return 0

    # 优先尝试完全匹配
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return len(prefix_ids)

    # 允许裁掉末尾 token 以抵消跨界 merge
    trim_max = min(max_trim, len(prefix_ids))
    for trim in range(1, trim_max + 1):
        cand = prefix_ids[:-trim]
        if len(cand) == 0:
            return 0
        if full_ids[: len(cand)] == cand:
            return len(cand)
    return None

# ---------- logging ----------
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("OfflineEntropyAnalysis")
DEBUG_MODE = False  # 全局调试开关

def debug_log(msg: str):
    """仅在调试模式下打印日志"""
    if DEBUG_MODE:
        logger.info(f"[DEBUG] {msg}")


# =============================================================================
# 数据加载
# =============================================================================

def load_trajectories(traj_dir: str, max_samples: int = -1) -> List[Dict]:
    """
    加载 MiniMax 轨迹数据（jsonl 格式）。
    每条数据包含 conversations 列表（role/content 交替），item_id, reward, success 等。
    """
    # 优先尝试几种常见的文件名
    candidates = list(Path(traj_dir).glob("*_trajectories.jsonl"))
    if not candidates:
        candidates = list(Path(traj_dir).glob("*.jsonl"))

    if not candidates:
        raise FileNotFoundError(f"No jsonl file found in {traj_dir}")

    # 优先匹配带前缀的文件名（如 babyai_trajectories.jsonl, textcraft_trajectories.jsonl）
    traj_path = None
    for c in candidates:
        if "trajectories" in c.name:
            traj_path = c
            break
    if traj_path is None:
        traj_path = candidates[0]

    trajectories = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    if max_samples > 0:
        trajectories = trajectories[:max_samples]

    logger.info(f"Loaded {len(trajectories)} trajectories from {traj_path}")
    return trajectories


def parse_user_turns(conversations: List[Dict]) -> List[str]:
    """
    从对话列表中提取所有 user 轮次的内容。
    返回: List[str]，每个元素是一个 user turn 的完整文本
    """
    return [m["content"] for m in conversations if m.get("role") == "user"]


def parse_assistant_turns(conversations: List[Dict]) -> List[str]:
    """
    从对话列表中提取所有 assistant 轮次的内容。
    返回: List[str]，每个元素是一个 turn 的完整文本
    """
    return [m["content"] for m in conversations if m.get("role") == "assistant"]


def build_prefix_for_turn(conversations: List[Dict], turn_idx: int, role: str = "assistant") -> List[Dict]:
    """
    构建到第 turn_idx 个指定角色轮次之前的历史消息上下文。
    用于让小模型预测该 turn 的 token 分布。

    conversations 格式: [user, assistant, user, assistant, ...]
    turn_idx: 第几个指定角色的 turn（0-indexed）
    role: "assistant" 或 "user"

    重要：prefix 只包含 target turn 之前的历史，不包含 target 本身。
    这样才能正确实现 teacher-forcing：给模型 prefix，让模型预测 target 的每个 token。
    """
    prefix_messages = []
    role_count = 0
    for msg in conversations:
        msg_role = msg.get("role")
        if msg_role == role:
            if role_count == turn_idx:
                # 到达目标 turn，停止，不包含 target 本身
                break
            role_count += 1
        prefix_messages.append(msg)
    return prefix_messages


# =============================================================================
# 熵计算工具
# =============================================================================

def compute_entropy_from_logprobs(log_probs_tensor: torch.Tensor) -> float:
    """
    直接从原始 log_probs tensor 计算熵（不需要 top-k decode）。
    这是最高效的方式，因为：
    1. 不需要 topk 操作
    2. 不需要 tokenizer decode
    3. 熵 = -sum(p * log(p)) = -sum(exp(lp) * lp)，直接用 log_probs 计算
    """
    # log_probs_tensor: [V] (vocab logprobs for one position)
    probs = torch.exp(log_probs_tensor)
    # 过滤极小概率
    mask = probs > 1e-12
    if mask.sum() == 0:
        return 0.0
    filtered_probs = probs[mask]
    filtered_log_probs = log_probs_tensor[mask]
    entropy = -torch.sum(filtered_probs * filtered_log_probs)
    return float(entropy.item())


def compute_entropy(lp_dict: Dict[str, float]) -> float:
    """从 {token: logprob} 字典计算香农熵（top-k 近似）。"""
    if not lp_dict:
        return 0.0
    log_probs = list(lp_dict.values())
    probs = [math.exp(lp) for lp in log_probs]
    return -sum(p * lp for p, lp in zip(probs, log_probs) if p > 1e-12)


def find_action_token_idx(token_list: List[Dict]) -> Optional[int]:
    """定位 action 开始后的第一个 token 位置。

    支持两种格式：
    - Qwen/QiFormer: [[action]]
    - MiniMax: Thought: ... Action: ...
    """
    text = "".join(t["decoded"] for t in token_list)

    # 优先匹配 MiniMax 格式: Action:
    m = re.search(r"Action:\s*\n?", text)
    if m:
        target = m.end()
        char_count = 0
        for i, t in enumerate(token_list):
            char_count += len(t["decoded"])
            if char_count >= target:
                return min(i + 1, len(token_list) - 1)
        return None

    # 备选: Qwen 格式: [[action]]
    m = re.search(r"\[\[", text)
    if not m:
        return None
    target = m.end()
    char_count = 0
    for i, t in enumerate(token_list):
        char_count += len(t["decoded"])
        if char_count >= target:
            return min(i + 1, len(token_list) - 1)
    return None


# =============================================================================
# vLLM logprobs 请求 (Teacher-Forcing 实现)
# =============================================================================

def apply_qwen_chat_template(messages: List[Dict], tokenizer) -> str:
    """
    使用 Qwen3 的 apply_chat_template 生成 prompt。
    与 vLLM 服务端的 chat template 完全一致。
    """
    # 将 {role: xxx, content: yyy} 格式转换为 [(role, content), ...]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return formatted

def get_logprobs_for_turn(
    prefix_messages: List[Dict],
    target_content: str,
    target_role: str,
    model_path: str,
    tokenizer_name: str,
    top_k: int,
    debug: bool = False,
) -> Tuple[List[Dict], Optional[str]]:
    """
    本地 teacher-forcing forward 获取 target_content 的 token 级别 logprobs/top-k logprobs。

    实现要点（offset_mapping 方案）：
    1) 只对 full_prompt 做一次 tokenization，返回 offset_mapping
    2) 用字符位置确定 target_content 在 full_prompt 中的区间
    3) 用 offset_mapping 找出与该字符区间有交集的 token（精确 token 边界）
    4) 因果对齐：logits[t] 预测的是 token_id[t+1]；因此 token_id[j] 的 logprob 来自 logits[j-1]

    熵统计只使用 target 部分的 token，不混入 prefix 或额外生成的 token。
    """
    tokenizer = get_tokenizer(tokenizer_name)
    model = get_model(model_path)

    # ========== 构造 prompt（统一使用 apply_chat_template）==========
    prefix_prompt = apply_qwen_chat_template(prefix_messages, tokenizer) if prefix_messages else ""
    target_message = {"role": target_role, "content": target_content}
    full_prompt = apply_qwen_chat_template(prefix_messages + [target_message], tokenizer)

    # ========== 用 offset_mapping 找 target_content 的 token span ==========
    # 关键修复：不再用字符位置切分 appended_text，而是直接在 full_prompt 中搜索 target_content
    # 这样可以避免 prefix_prompt 和 full_prompt 开头不一致导致的问题
    #
    # 1) 对 full_prompt 做单次 tokenization（带 offset_mapping）
    #    注意：apply_chat_template() 已经生成了 chat-formatted 文本（含 <|im_start|> 等），
    #    因此这里用 add_special_tokens=False，不再额外注入 special tokens，避免 offset 错位
    enc = tokenizer(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]  # list of (start_char, end_char)

    # offset_mapping 可能返回 tensor 或其他格式，确保转为 list
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets is None or len(offsets) != len(full_ids):
        logger.warning("[Skip] tokenizer did not return valid offset_mapping")
        return [], None

    # 2) 直接在 full_prompt 中搜索 target_content，而不是在 appended_text 中搜索
    #    使用分层匹配：原始精确匹配 -> 可逆规范化匹配 -> 双锚点匹配
    #    搜索范围是整个 full_prompt（不限制只在后半部分）
    search_text = full_prompt

    # ========== 可逆规范化函数 ==========
    def reversible_normalize(s: str):
        """
        可逆规范化：处理格式差异，但保留每个规范化位置到原始位置的映射
        处理：
        - \r\n -> \n
        - \r -> \n
        - 连续空白(space/tab) -> 单个空格
        - 行尾空白(trailing whitespace) -> 去除
        
        返回：
        - normalized_s: 规范化后的字符串
        - norm_spans: list of (orig_start, orig_end)，每个规范化字符对应的原始开区间
        """
        normalized_chars = []
        norm_spans = []  # normalized_pos -> (orig_start, orig_end)
        
        # 使用 buffer 机制来正确处理行尾空白
        # 思路：先把字符缓冲起来，遇到 \n 时先删除 buffer 末尾的空白，再写入 \n
        buffer = []  # 缓冲的字符
        buffer_spans = []  # 每个缓冲字符对应的 (orig_start, orig_end)
        
        i = 0
        while i < len(s):
            c = s[i]
            
            # 处理 \r\n
            if c == '\r' and i + 1 < len(s) and s[i + 1] == '\n':
                # 先把 buffer 中非 trailing whitespace 的部分写入
                # 找到 buffer 中最后一个非空白的位置
                while buffer and (buffer[-1] == ' ' or buffer[-1] == '\t'):
                    buffer.pop()
                    buffer_spans.pop()
                # 写入 \n
                normalized_chars.extend(buffer)
                norm_spans.extend(buffer_spans)
                normalized_chars.append('\n')
                norm_spans.append((i, i + 2))  # \r\n -> \n，原始区间是 [i, i+2)
                buffer.clear()
                buffer_spans.clear()
                i += 2  # 跳过 \r\n
                continue
            
            # 处理单独的 \r
            if c == '\r':
                # 同上，先清 trailing whitespace
                while buffer and (buffer[-1] == ' ' or buffer[-1] == '\t'):
                    buffer.pop()
                    buffer_spans.pop()
                normalized_chars.extend(buffer)
                norm_spans.extend(buffer_spans)
                normalized_chars.append('\n')
                norm_spans.append((i, i + 1))  # \r -> \n
                buffer.clear()
                buffer_spans.clear()
                i += 1
                continue
            
            # 遇到换行符 \n
            if c == '\n':
                # 先删除 buffer 末尾的 trailing whitespace
                while buffer and (buffer[-1] == ' ' or buffer[-1] == '\t'):
                    buffer.pop()
                    buffer_spans.pop()
                # 把 buffer 写入结果
                normalized_chars.extend(buffer)
                norm_spans.extend(buffer_spans)
                # 写入 \n
                normalized_chars.append('\n')
                norm_spans.append((i, i + 1))
                buffer.clear()
                buffer_spans.clear()
                i += 1
                continue
            
            # 处理连续空白 (space/tab) - 行内空白压缩
            if c == ' ' or c == '\t':
                start = i
                # 跳过后续空白
                i += 1
                while i < len(s) and (s[i] == ' ' or s[i] == '\t'):
                    i += 1
                # 压缩成单个空格，加入 buffer
                buffer.append(' ')
                buffer_spans.append((start, i))  # 覆盖整个空白区间
                continue
            
            # 普通字符 - 加入 buffer
            buffer.append(c)
            buffer_spans.append((i, i + 1))
            i += 1
        
        # 处理末尾剩余的 buffer（文件最后一行可能没有换行符）
        # 先删除末尾的 trailing whitespace
        while buffer and (buffer[-1] == ' ' or buffer[-1] == '\t'):
            buffer.pop()
            buffer_spans.pop()
        normalized_chars.extend(buffer)
        norm_spans.extend(buffer_spans)
        
        return ''.join(normalized_chars), norm_spans

    # ========== 分层匹配 ==========
    content_start_char = None  # 在 full_prompt 中的字符起始位置
    matched_content_len = len(target_content)
    match_method = None  # "exact", "normalized", "anchor", or None
    
    # Debug 信息
    debug_info = {
        "method1_exact": False,
        "method2_normalized": False,
        "method3_anchor1": None,
        "method3_anchor2": None,
    }

    # 方法1：原始精确匹配（直接在 full_prompt 中搜索）
    pos = search_text.find(target_content)
    if pos != -1:
        content_start_char = pos
        matched_content_len = len(target_content)
        match_method = "exact"
        debug_info["method1_exact"] = True
        debug_log(f"Method1 exact match: pos={pos}, len={matched_content_len}")

    # 方法2：可逆规范化后匹配
    if content_start_char is None:
        normalized_search, norm_spans = reversible_normalize(search_text)
        normalized_target, _ = reversible_normalize(target_content)
        pos = normalized_search.find(normalized_target)

        debug_info["method2_normalized"] = (pos != -1)

        if pos != -1:
            # 用 norm_spans 映射回原始位置
            # 起点：首个规范化字符对应区间的起点
            # 终点：最后一个规范化字符对应区间的终点
            if pos < len(norm_spans):
                content_start_char = norm_spans[pos][0]
                norm_end_pos = pos + len(normalized_target)
                if norm_end_pos > 0 and norm_end_pos <= len(norm_spans):
                    orig_end_pos = norm_spans[norm_end_pos - 1][1]
                else:
                    orig_end_pos = len(search_text)
                matched_content_len = orig_end_pos - content_start_char
                match_method = "normalized"
                debug_log(f"Method2 normalized match: norm_pos={pos}, orig_pos={content_start_char}, len={matched_content_len}")

    # 方法3：双锚点匹配（仅当前面方法失败时）
    if content_start_char is None and len(target_content) >= 20:
        # 构造规范化文本
        normalized_search, norm_spans = reversible_normalize(search_text)

        # 选择两个锚点：前段(10-30字符) 和 后段(末尾10-30字符)
        anchor_len = min(20, len(target_content) // 3)
        anchor1 = target_content[:anchor_len]
        anchor2 = target_content[-anchor_len:]

        # 规范化锚点
        norm_anchor1, _ = reversible_normalize(anchor1)
        norm_anchor2, _ = reversible_normalize(anchor2)

        pos1 = normalized_search.find(norm_anchor1)
        pos2 = normalized_search.rfind(norm_anchor2)  # 从后向前找

        debug_info["method3_anchor1"] = (norm_anchor1, pos1)
        debug_info["method3_anchor2"] = (norm_anchor2, pos2)

        if pos1 != -1 and pos2 != -1:
            # 检查锚点间距离是否与 target 长度一致（允许 ±20% 误差）
            expected_dist = len(target_content)
            actual_dist = pos2 - pos1 + len(norm_anchor2)
            if 0.8 * expected_dist <= actual_dist <= 1.2 * expected_dist:
                # 双锚点约束满足，用 norm_spans 映射回原始位置
                if pos1 < len(norm_spans):
                    content_start_char = norm_spans[pos1][0]
                    norm_end_pos = pos2 + len(norm_anchor2)
                    if norm_end_pos > 0 and norm_end_pos <= len(norm_spans):
                        orig_end_pos = norm_spans[norm_end_pos - 1][1]
                    else:
                        orig_end_pos = len(search_text)
                    matched_content_len = orig_end_pos - content_start_char
                    match_method = "anchor"
                    debug_log(f"Method3 dual-anchor match: anchor1_pos={pos1}, anchor2_pos={pos2}, orig_start={content_start_char}, len={matched_content_len}")

    if content_start_char is None:
        # 所有方法都失败，明确 skip
        if debug or DEBUG_MODE:
            debug_log(
                f"[Skip] target_content not found. "
                f"full_prompt_len={len(full_prompt)}, "
                f"target_content_len={len(target_content)}, "
                f"search_text_len={len(search_text)}, "
                f"target_content[:300]={repr(target_content[:300])}, "
                f"search_text[:500]={repr(search_text[:500])}, "
                f"debug_info={debug_info}"
            )
        logger.warning(
            f"[Skip] target_content not found. target_len={len(target_content)}, full_prompt_len={len(full_prompt)}, match_method={match_method}"
        )
        return [], None

    # 计算结束字符位置
    content_end_char = content_start_char + matched_content_len

    # 3) 用 offset_mapping 找出与 [content_start_char, content_end_char) 有交集的 token
    #    过滤掉 offset 为 (0, 0) 的特殊 token（如 <|im_end|> 等）
    target_token_indices: List[int] = []
    skipped_zero_offset_tokens = 0
    for idx, (tok_start, tok_end) in enumerate(offsets):
        # 跳过无实际字符覆盖的特殊 token
        if tok_start == 0 and tok_end == 0:
            skipped_zero_offset_tokens += 1
            continue
        # 判断：token 区间与 content 区间有交集
        if tok_end > content_start_char and tok_start < content_end_char:
            target_token_indices.append(idx)

    if len(target_token_indices) == 0:
        logger.warning("[Skip] no tokens found for target_content via offset_mapping")
        return [], None

    if debug or DEBUG_MODE:
        debug_log(
            f"content_start_char={content_start_char}, content_end_char={content_end_char}, "
            f"target_token_count={len(target_token_indices)}, skipped_zero_offset={skipped_zero_offset_tokens}"
        )
        first_tok_idx = target_token_indices[0] if target_token_indices else None
        if first_tok_idx is not None:
            first_decoded = tokenizer.decode([full_ids[first_tok_idx]], clean_up_tokenization_spaces=False)
            debug_log(f"target_token_list[0] decoded={repr(first_decoded)}")

    # ========== teacher-forcing forward ==========
    input_ids = torch.tensor([full_ids], device=MODEL_DEVICE)
    # inference_mode 比 no_grad 更激进，可略微提升性能
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits  # [1, T, V]
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]

    # 对齐说明：logits[t] 预测 token_id[t+1]，所以 token_id[j] 的 logprob 来自 log_probs[j-1]
    # target_token_indices 已经由 offset_mapping 确定，不需要再做前缀差

    # ========== 构造 target_token_list ==========
    target_token_list: List[Dict[str, Any]] = []
    for j in target_token_indices:
        t = j - 1
        if t < 0:
            # 第一个 token 之前没有 logit 对应，跳过
            continue
        gold_id = full_ids[j]
        gold_lp = float(log_probs[0, t, gold_id].item())

        # top-k
        k = min(top_k, log_probs.shape[-1])
        topv, topi = torch.topk(log_probs[0, t, :], k=k, dim=-1)
        # 用 logsumexp 聚合同字符串的 token 概率，避免 key collision
        # topv 是 log probabilities，直接用 logsumexp 合并
        top_logprobs: Dict[str, float] = {}
        seen_tokens: Dict[str, List[float]] = {}  # 调试用：记录哪些 token 被聚合
        for tok_id, lp in zip(topi.tolist(), topv.tolist()):
            tok_str = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            if tok_str not in seen_tokens:
                seen_tokens[tok_str] = []
            seen_tokens[tok_str].append(float(lp))

        # 对每个 unique string，用 logsumexp 合并其多个 token id 的概率
        for tok_str, lps in seen_tokens.items():
            if len(lps) == 1:
                top_logprobs[tok_str] = lps[0]
            else:
                # logsumexp: log(exp(lp1) + exp(lp2) + ...)
                max_lp = max(lps)
                summed = sum(math.exp(lp - max_lp) for lp in lps)
                top_logprobs[tok_str] = max_lp + math.log(summed)

        if debug or DEBUG_MODE:
            # 统计 collision 数量
            collisions = sum(1 for lps in seen_tokens.values() if len(lps) > 1)
            if collisions > 0:
                debug_log(f"[Collision] t={t}, collisions={collisions}, details={seen_tokens}")

        decoded = tokenizer.decode([gold_id], clean_up_tokenization_spaces=False)
        target_token_list.append(
            {"decoded": decoded, "logprob": gold_lp, "top_logprobs": top_logprobs}
        )

    # ========== 最小自检 ==========
    if len(target_token_list) == 0:
        logger.warning("[Skip] No tokens extracted for target")
        return [], None

    if (debug or DEBUG_MODE) and target_token_list:
        debug_log(f"extracted_target_tokens={len(target_token_list)}; first_decoded={repr(target_token_list[0]['decoded'])}")

    return target_token_list, None


# =============================================================================
# 单条轨迹的熵分析
# =============================================================================

def analyze_trajectory(
    traj: Dict,
    model_path: str,
    tokenizer_name: str,
    top_k: int,
    max_turns: int,
    save_per_token_entropy: bool = False,
) -> Dict:
    """
    对一条 MiniMax 轨迹，逐个 user/assistant turn 请求小模型 logprobs，统计熵。
    分别计算并区分 user 和 assistant 的熵。
    """
    conversations = traj.get("conversations", [])
    item_id = traj.get("item_id", "unknown")
    success = traj.get("success", 0)
    reward = traj.get("reward", 0)

    assistant_turns = parse_assistant_turns(conversations)
    user_turns = parse_user_turns(conversations)

    if not assistant_turns and not user_turns:
        return {"item_id": item_id, "success": success, "reward": reward,
                "entropy_A_assistant": [], "entropy_B_assistant": [], "entropy_C_assistant": [],
                "entropy_A_user": [], "entropy_C_user": [],
                "turn_lengths_assistant": [], "turn_lengths_user": []}

    # Assistant 熵存储
    entropy_A_assistant = []   # 首 token 熵
    entropy_B_assistant = []   # Action 首 token 熵
    entropy_C_assistant = []   # Turn 平均熵
    turn_lengths_assistant = []
    cumsum_lengths_assistant = []
    entropy_per_token_assistant = []

    # User 熵存储
    entropy_A_user = []   # 首 token 熵
    entropy_C_user = []   # Turn 平均熵
    turn_lengths_user = []
    cumsum_lengths_user = []
    entropy_per_token_user = []

    # 处理 assistant turns
    n_assistant_turns = len(assistant_turns) if max_turns == -1 else min(len(assistant_turns), max_turns)
    cumsum_assistant = 0

    for turn_idx in range(n_assistant_turns):
        # 构建该 turn 之前的上下文（不包含 target turn 本身）
        prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="assistant")

        # target 是 assistant turn 的内容
        target_content = assistant_turns[turn_idx]

        # 请求小模型 logprobs（teacher-forcing）
        # 不再使用 KV cache 复用，因为 prompt_cache_id 不是稳定的 OpenAI 兼容字段
        target_token_list, _ = get_logprobs_for_turn(
            prefix_messages=prefix_msgs,
            target_content=target_content,
            target_role="assistant",
            model_path=model_path,
            tokenizer_name=tokenizer_name,
            top_k=top_k,
        )
        
        n_tok = len(target_token_list)
        turn_lengths_assistant.append(n_tok)
        cumsum_assistant += n_tok
        cumsum_lengths_assistant.append(cumsum_assistant)

        if n_tok == 0:
            entropy_A_assistant.append(0.0)
            entropy_B_assistant.append(None)
            entropy_C_assistant.append(0.0)
            if save_per_token_entropy:
                entropy_per_token_assistant.append([])
            continue

        # A: 首 token 熵
        ent_A = compute_entropy(target_token_list[0]["top_logprobs"])
        entropy_A_assistant.append(ent_A)

        # C: turn 平均熵
        all_ents = [compute_entropy(t["top_logprobs"]) for t in target_token_list]
        entropy_C_assistant.append(sum(all_ents) / n_tok)

        # 可选：保存 per-token 熵
        if save_per_token_entropy:
            entropy_per_token_assistant.append(all_ents)

        # B: [[ 之后的 action token 熵
        act_idx = find_action_token_idx(target_token_list)
        if act_idx is not None and act_idx < n_tok:
            entropy_B_assistant.append(compute_entropy(target_token_list[act_idx]["top_logprobs"]))
        else:
            entropy_B_assistant.append(None)

    # 处理 user turns
    # 注意：user turn 的语义与 assistant 不同
    # - Assistant: 给定 prefix，预测 assistant 的回复（正常的 teacher-forcing）
    # - User: 给定 prefix，预测 user 的发言（这是一种"逆 teacher-forcing"，语义上不太自然）
    # 但为了保持一致性，我们仍然按相同方式处理
    n_user_turns = len(user_turns) if max_turns == -1 else min(len(user_turns), max_turns)
    cumsum_user = 0

    for turn_idx in range(n_user_turns):
        # 构建该 turn 之前的上下文（不包含 target turn 本身）
        prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="user")

        # target 是 user turn 的内容
        target_content = user_turns[turn_idx]

        # 请求小模型 logprobs（teacher-forcing）
        target_token_list, _ = get_logprobs_for_turn(
            prefix_messages=prefix_msgs,
            target_content=target_content,
            target_role="user",
            model_path=model_path,
            tokenizer_name=tokenizer_name,
            top_k=top_k,
        )
        
        n_tok = len(target_token_list)
        turn_lengths_user.append(n_tok)
        cumsum_user += n_tok
        cumsum_lengths_user.append(cumsum_user)

        if n_tok == 0:
            entropy_A_user.append(0.0)
            entropy_C_user.append(0.0)
            if save_per_token_entropy:
                entropy_per_token_user.append([])
            continue

        # A: 首 token 熵
        ent_A = compute_entropy(target_token_list[0]["top_logprobs"])
        entropy_A_user.append(ent_A)

        # C: turn 平均熵
        all_ents = [compute_entropy(t["top_logprobs"]) for t in target_token_list]
        entropy_C_user.append(sum(all_ents) / n_tok)

        # 可选：保存 per-token 熵
        if save_per_token_entropy:
            entropy_per_token_user.append(all_ents)

    # 计算相对位置
    n_turns = max(n_assistant_turns, n_user_turns)
    if n_turns > 1:
        relative_positions = [turn_idx / (n_turns - 1) for turn_idx in range(n_turns)]
    else:
        relative_positions = [0.0]

    return {
        "item_id": item_id,
        "success": success,
        "reward": reward,
        "num_assistant_turns": n_assistant_turns,
        "num_user_turns": n_user_turns,
        "total_tokens": cumsum_assistant + cumsum_user,
        "turn_lengths_assistant": turn_lengths_assistant,
        "turn_lengths_user": turn_lengths_user,
        "cumsum_lengths_assistant": cumsum_lengths_assistant,
        "cumsum_lengths_user": cumsum_lengths_user,
        "relative_positions": relative_positions,
        # Assistant 熵
        "entropy_A_assistant": entropy_A_assistant,
        "entropy_B_assistant": entropy_B_assistant,
        "entropy_C_assistant": entropy_C_assistant,
        # User 熵
        "entropy_A_user": entropy_A_user,
        "entropy_C_user": entropy_C_user,
        # Per-token 熵（可选）
        "entropy_per_token_assistant": entropy_per_token_assistant if save_per_token_entropy else None,
        "entropy_per_token_user": entropy_per_token_user if save_per_token_entropy else None,
    }


# =============================================================================
# 聚合统计
# =============================================================================

def aggregate_entropy(all_results: List[Dict]) -> Dict:
    # Assistant 熵聚合
    agg = {
        "all": {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
        "success": {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
    }
    # User 熵聚合
    agg_user = {
        "all": {"A": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
        "success": {"A": defaultdict(list), "C": defaultdict(list), "length": defaultdict(list)},
    }

    # H(q): 按相对位置 q = t/(T-1) 聚合 (0%, 10%, 20%, ..., 100%)
    # 使用更细的粒度：20个桶 (0-5%, 5-10%, ..., 95-100%)
    agg_by_q = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }
    agg_by_q_user = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }

    # H(q | length_bin): 按轨迹总长度分桶 + 相对位置
    agg_by_q_and_bin = {
        "all": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "success": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    agg_by_q_and_bin_user = {
        "all": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "success": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }

    # H(turn | length_bin): 按轨迹总长度分桶
    # 分桶: 0-2k, 2k-4k, 4k-6k, 6k-8k, 8k-10k, 10k+
    agg_by_bin = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }
    agg_by_bin_user = {
        "all": defaultdict(lambda: defaultdict(list)),
        "success": defaultdict(lambda: defaultdict(list)),
    }

    for res in all_results:
        if "error" in res:
            continue
        is_success = bool(res.get("success", 0))

        # ========== Assistant 熵聚合 ==========
        # 原始按 turn 索引聚合
        for t_idx, val in enumerate(res.get("entropy_A_assistant", [])):
            agg["all"]["A"][t_idx].append(val)
            if is_success:
                agg["success"]["A"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("entropy_B_assistant", [])):
            if val is not None:
                agg["all"]["B"][t_idx].append(val)
                if is_success:
                    agg["success"]["B"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("entropy_C_assistant", [])):
            agg["all"]["C"][t_idx].append(val)
            if is_success:
                agg["success"]["C"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("turn_lengths_assistant", [])):
            agg["all"]["length"][t_idx].append(val)
            if is_success:
                agg["success"]["length"][t_idx].append(val)

        # ========== User 熵聚合 ==========
        for t_idx, val in enumerate(res.get("entropy_A_user", [])):
            agg_user["all"]["A"][t_idx].append(val)
            if is_success:
                agg_user["success"]["A"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("entropy_C_user", [])):
            agg_user["all"]["C"][t_idx].append(val)
            if is_success:
                agg_user["success"]["C"][t_idx].append(val)
        for t_idx, val in enumerate(res.get("turn_lengths_user", [])):
            agg_user["all"]["length"][t_idx].append(val)
            if is_success:
                agg_user["success"]["length"][t_idx].append(val)

        # ========== H(q): 按相对位置聚合 (Assistant) ==========
        n_turns = res.get("num_assistant_turns", 0)
        relative_positions = res.get("relative_positions", [])
        entropy_C_assistant = res.get("entropy_C_assistant", [])

        for turn_idx, q in enumerate(relative_positions):
            if turn_idx >= len(entropy_C_assistant):
                continue
            val = entropy_C_assistant[turn_idx]
            q_bin = int(q * 20)
            q_bin = min(q_bin, 19)
            agg_by_q["all"][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q["success"][q_bin][turn_idx].append(val)

            # H(q | length_bin)
            total_tokens = res.get("total_tokens", 0)
            length_bin = get_length_bin(total_tokens)
            agg_by_q_and_bin["all"][length_bin][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q_and_bin["success"][length_bin][q_bin][turn_idx].append(val)

        # H(turn | length_bin) - Assistant
        for turn_idx, val in enumerate(entropy_C_assistant):
            length_bin = get_length_bin(total_tokens)
            agg_by_bin["all"][length_bin][turn_idx].append(val)
            if is_success:
                agg_by_bin["success"][length_bin][turn_idx].append(val)

        # ========== H(q): 按相对位置聚合 (User) ==========
        entropy_C_user = res.get("entropy_C_user", [])
        for turn_idx, q in enumerate(relative_positions):
            if turn_idx >= len(entropy_C_user):
                continue
            val = entropy_C_user[turn_idx]
            q_bin = int(q * 20)
            q_bin = min(q_bin, 19)
            agg_by_q_user["all"][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q_user["success"][q_bin][turn_idx].append(val)

            # H(q | length_bin) - User
            length_bin = get_length_bin(total_tokens)
            agg_by_q_and_bin_user["all"][length_bin][q_bin][turn_idx].append(val)
            if is_success:
                agg_by_q_and_bin_user["success"][length_bin][q_bin][turn_idx].append(val)

        # H(turn | length_bin) - User
        for turn_idx, val in enumerate(entropy_C_user):
            length_bin = get_length_bin(total_tokens)
            agg_by_bin_user["all"][length_bin][turn_idx].append(val)
            if is_success:
                agg_by_bin_user["success"][length_bin][turn_idx].append(val)

    def get_length_bin(total_tokens: int) -> int:
        """根据 token 总数返回长度分桶索引"""
        if total_tokens < 2000:
            return 0
        elif total_tokens < 4000:
            return 1
        elif total_tokens < 6000:
            return 2
        elif total_tokens < 8000:
            return 3
        elif total_tokens < 10000:
            return 4
        else:
            return 5

    def compute_percentiles(vals: List[float]) -> Dict[str, float]:
        """计算百分位数: 25%, 50%, 75%, 90%, 95%, 99%"""
        if not vals:
            return {}
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        result = {}
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int(n * p / 100)
            if idx >= n:
                idx = n - 1
            result[f"p{p}"] = sorted_vals[idx]
        return result


def t_test_two_groups(vals1: List[float], vals2: List[float]) -> Dict[str, float]:
    """对两组样本进行独立样本 t 检验，返回 t 统计量和 p 值。"""
    if len(vals1) < 2 or len(vals2) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}

    import numpy as np
    t_stat, p_value = np.ttest_ind(vals1, vals2)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }

    def summarize(d, include_percentiles=False):
        out = {}
        for t_idx, vals in sorted(d.items()):
            n = len(vals)
            if n == 0:
                continue
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            result = {"mean": mean, "std": std, "count": n}
            if include_percentiles:
                result.update(compute_percentiles(vals))
            out[str(t_idx)] = result
        return out

    # 按相对位置 q 聚合: 每个 q_bin 下收集所有 turn 的熵，求平均
    def summarize_by_q(agg_by_q_dict):
        out = {}
        for q_bin in range(20):  # 0-19 (0-5%, 5-10%, ..., 95-100%)
            all_ents = []
            for turn_idx, ents in agg_by_q_dict[q_bin].items():
                all_ents.extend(ents)
            if all_ents:
                mean = sum(all_ents) / len(all_ents)
                std = math.sqrt(sum((e - mean) ** 2 for e in all_ents) / len(all_ents)) if len(all_ents) > 1 else 0.0
                # 将 q_bin 转换为百分比标签
                q_label = f"q{q_bin * 5:02d}-{(q_bin + 1) * 5:02d}"
                out[q_label] = {"mean": mean, "std": std, "count": len(all_ents)}
        return out

    # 按轨迹长度分桶聚合
    def summarize_by_bin(agg_by_bin_dict):
        bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
        out = {}
        for bin_idx in range(6):
            all_ents = []
            for turn_idx, ents in agg_by_bin_dict[bin_idx].items():
                all_ents.extend(ents)
            if all_ents:
                mean = sum(all_ents) / len(all_ents)
                std = math.sqrt(sum((e - mean) ** 2 for e in all_ents) / len(all_ents)) if len(all_ents) > 1 else 0.0
                out[bin_labels[bin_idx]] = {"mean": mean, "std": std, "count": len(all_ents)}
        return out

    # 按相对位置 + 轨迹长度分桶聚合: H(q | length_bin)
    def summarize_by_q_and_bin(agg_by_q_and_bin_dict):
        bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
        out = {}
        for bin_idx in range(6):
            bin_label = bin_labels[bin_idx]
            q_dict = agg_by_q_and_bin_dict.get(bin_idx, {})
            # 对每个 q_bin 求平均
            q_means = []
            for q_bin in range(20):
                all_ents = []
                for turn_idx, ents in q_dict.get(q_bin, {}).items():
                    all_ents.extend(ents)
                if all_ents:
                    q_means.append(sum(all_ents) / len(all_ents))
            if q_means:
                overall_mean = sum(q_means) / len(q_means)
                out[bin_label] = {
                    "q_means": q_means,
                    "overall_mean": overall_mean,
                    "count": sum(len(ents) for q_ents in q_dict.values() for ents in q_ents.values())
                }
        return out

    return {
        "all_episodes": {
            "A_first_token":  summarize(agg["all"]["A"]),
            "B_action_token": summarize(agg["all"]["B"]),
            "C_turn_mean":    summarize(agg["all"]["C"]),
            "turn_lengths":   summarize(agg["all"]["length"], include_percentiles=True),
            "entropy_by_q":   summarize_by_q(agg_by_q["all"]),
            "entropy_by_q_and_length_bin": summarize_by_q_and_bin(agg_by_q_and_bin["all"]),
            "entropy_by_length_bin": summarize_by_bin(agg_by_bin["all"]),
            # User 熵
            "A_first_token_user":  summarize(agg_user["all"]["A"]),
            "C_turn_mean_user":    summarize(agg_user["all"]["C"]),
            "turn_lengths_user":   summarize(agg_user["all"]["length"], include_percentiles=True),
            "entropy_by_q_user":   summarize_by_q(agg_by_q_user["all"]),
            "entropy_by_q_and_length_bin_user": summarize_by_q_and_bin(agg_by_q_and_bin_user["all"]),
            "entropy_by_length_bin_user": summarize_by_bin(agg_by_bin_user["all"]),
        },
        "success_only": {
            "A_first_token":  summarize(agg["success"]["A"]),
            "B_action_token": summarize(agg["success"]["B"]),
            "C_turn_mean":    summarize(agg["success"]["C"]),
            "turn_lengths":   summarize(agg["success"]["length"], include_percentiles=True),
            "entropy_by_q":   summarize_by_q(agg_by_q["success"]),
            "entropy_by_q_and_length_bin": summarize_by_q_and_bin(agg_by_q_and_bin["success"]),
            "entropy_by_length_bin": summarize_by_bin(agg_by_bin["success"]),
            # User 熵
            "A_first_token_user":  summarize(agg_user["success"]["A"]),
            "C_turn_mean_user":    summarize(agg_user["success"]["C"]),
            "turn_lengths_user":   summarize(agg_user["success"]["length"], include_percentiles=True),
            "entropy_by_q_user":   summarize_by_q(agg_by_q_user["success"]),
            "entropy_by_q_and_length_bin_user": summarize_by_q_and_bin(agg_by_q_and_bin_user["success"]),
            "entropy_by_length_bin_user": summarize_by_bin(agg_by_bin_user["success"]),
        },
        "total_episodes":   len(all_results),
        "success_episodes": sum(1 for r in all_results if r.get("success", 0)),
    }


# =============================================================================
# 绘图
# =============================================================================

def plot_entropy(summary: Dict, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))

    # ========== (0,0)：按绝对 turn 的熵曲线 ==========
    labels = {
        "A_first_token":  "A: First-token Entropy (Think start)",
        "B_action_token": "B: Action first-token Entropy (after Action: or [[)",
        "C_turn_mean":    "C: Mean Token Entropy per Turn",
    }
    colors = {"A_first_token": "steelblue", "B_action_token": "coral", "C_turn_mean": "seagreen"}

    ax = axes[0, 0]
    subset = summary["all_episodes"]
    for metric_key, metric_label in labels.items():
        data = subset.get(metric_key, {})
        if not data:
            continue
        steps = sorted(data.keys(), key=lambda x: int(x))
        means = [data[s]["mean"] for s in steps]
        stds  = [data[s]["std"]  for s in steps]
        xs    = [int(s) + 1 for s in steps]

        ax.plot(xs, means, marker="o", markersize=4,
                label=metric_label, color=colors[metric_key])
        ax.fill_between(xs,
                        [m - sd for m, sd in zip(means, stds)],
                        [m + sd for m, sd in zip(means, stds)],
                        alpha=0.15, color=colors[metric_key])

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("H(turn) - Entropy by Absolute Turn Index", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (0,1)：Success vs All ==========
    ax = axes[0, 1]
    for metric_key, metric_label in labels.items():
        all_data = summary["all_episodes"].get(metric_key, {})
        succ_data = summary["success_only"].get(metric_key, {})
        if not all_data:
            continue
        steps = sorted(all_data.keys(), key=lambda x: int(x))
        all_means = [all_data[s]["mean"] for s in steps]
        succ_means = [succ_data[s]["mean"] for s in steps]
        xs = [int(s) + 1 for s in steps]

        ax.plot(xs, all_means, marker="o", markersize=4, linestyle="-",
                label=f"{metric_label} (all)", color=colors[metric_key], alpha=0.7)
        ax.plot(xs, succ_means, marker="s", markersize=4, linestyle="--",
                label=f"{metric_label} (success)", color=colors[metric_key], alpha=0.4)

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("All vs Success Episodes", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ========== (0,2)：H(q) - 按相对位置 q = t/(T-1) ==========
    ax = axes[0, 2]
    entropy_by_q = summary["all_episodes"].get("entropy_by_q", {})
    if entropy_by_q:
        # 解析 q 标签 (格式: q00-05, q05-10, ..., q95-100)
        q_labels = sorted(entropy_by_q.keys(), key=lambda x: (int(x[1:3]), int(x[4:6])))
        xs = [int(x[1:3]) + 2.5 for x in q_labels]  # 中点: 2.5, 7.5, 12.5, ..., 97.5
        means = [entropy_by_q[q]["mean"] for q in q_labels]
        stds = [entropy_by_q[q]["std"] for q in q_labels]

        ax.bar(xs, means, width=4, color="purple", alpha=0.6, label="H(q)")
        ax.plot(xs, means, marker="o", markersize=6, color="purple")
        ax.fill_between(xs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="purple")
        ax.axhline(y=sum(means)/len(means), color="gray", linestyle=":", alpha=0.5, label=f"avg={sum(means)/len(means):.3f}")

    ax.set_xlabel("Relative Position q = t/(T-1) (%)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("H(q) - Entropy by Relative Position (0%=start, 100%=end)", fontsize=11)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (1,0)：Turn Length 百分位数曲线 ==========
    ax = axes[1, 0]
    length_data = summary["all_episodes"].get("turn_lengths", {})
    if length_data:
        steps = sorted(length_data.keys(), key=lambda x: int(x))
        xs = [int(s) + 1 for s in steps]

        for p, ls, color in [("p50", "-", "blue"), ("p75", "--", "green"),
                              ("p90", ":", "orange"), ("p95", "-.", "red")]:
            vals = [length_data[s].get(p, 0) for s in steps]
            ax.plot(xs, vals, marker="o", markersize=3, linestyle=ls,
                    label=f"Length {p}", color=color)

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Length", fontsize=11)
    ax.set_title("Turn Length Percentiles", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (1,1)：H(q | length_bin) - 按轨迹总长度分桶后的相对位置熵 ==========
    ax = axes[1, 1]
    entropy_by_q_and_bin = summary["all_episodes"].get("entropy_by_q_and_length_bin", {})
    bin_labels = ["0-2k", "2k-4k", "4k-6k", "6k-8k", "8k-10k", "10k+"]
    bin_colors = plt.cm.viridis([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if entropy_by_q_and_bin:
        xs = list(range(0, 100, 5))  # 0, 5, 10, ..., 95
        for bin_idx, bin_label in enumerate(bin_labels):
            if bin_label not in entropy_by_q_and_bin:
                continue
            bin_data = entropy_by_q_and_bin[bin_label]
            q_means = bin_data.get("q_means", [])
            if q_means:
                ax.plot(xs[:len(q_means)], q_means, marker="o", markersize=3, linestyle="-",
                        label=f"{bin_label} (n={bin_data.get('count', '?')})",
                        color=bin_colors[bin_idx], alpha=0.8)

    ax.set_xlabel("Relative Position q = t/(T-1) (%)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("H(q | length_bin) - Entropy by Relative Position for Each Length Bin", fontsize=11)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ========== (1,2)：各分桶的统计信息（柱状图）==========
    ax = axes[1, 2]
    entropy_by_bin = summary["all_episodes"].get("entropy_by_length_bin", {})
    if entropy_by_bin:
        bins = list(entropy_by_bin.keys())
        means = [entropy_by_bin[b]["mean"] for b in bins]
        counts = [entropy_by_bin[b]["count"] for b in bins]

        # 柱状图显示均值
        x_pos = range(len(bins))
        bars = ax.bar(x_pos, means, color=bin_colors[:len(bins)], alpha=0.7, edgecolor="black")

        # 在柱子上标注 count
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"n={count}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Trajectory Length Bin", fontsize=11)
    ax.set_ylabel("Mean Entropy", fontsize=11)
    ax.set_title("Mean Entropy by Trajectory Length Bin", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bins, rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # ========== (2,0)：按绝对 turn 的熵曲线（仅 success）==========
    ax = axes[2, 0]
    for metric_key, metric_label in [("C_turn_mean", "C: Mean Token Entropy")]:
        # All episodes
        all_data = summary["all_episodes"].get(metric_key, {})
        if not all_data:
            continue
        steps = sorted(all_data.keys(), key=lambda x: int(x))
        all_means = [all_data[s]["mean"] for s in steps]
        all_stds = [all_data[s]["std"] for s in steps]
        xs = [int(s) + 1 for s in steps]

        ax.plot(xs, all_means, marker="o", markersize=4,
                label="All episodes", color="blue", alpha=0.7)
        ax.fill_between(xs,
                        [m - s for m, s in zip(all_means, all_stds)],
                        [m + s for m, s in zip(all_means, all_stds)],
                        alpha=0.15, color="blue")

        # Success episodes
        succ_data = summary["success_only"].get(metric_key, {})
        if succ_data:
            succ_means = [succ_data[s]["mean"] for s in steps]
            succ_stds = [succ_data[s]["std"] for s in steps]
            ax.plot(xs, succ_means, marker="s", markersize=4,
                    label="Success only", color="green", alpha=0.7)
            ax.fill_between(xs,
                            [m - s for m, s in zip(succ_means, succ_stds)],
                            [m + s for m, s in zip(succ_means, succ_stds)],
                            alpha=0.15, color="green")

    ax.set_xlabel("Turn Step (1-indexed)", fontsize=11)
    ax.set_ylabel("Token Entropy", fontsize=11)
    ax.set_title("C: Mean Turn Entropy (All vs Success)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== (2,1)：熵的分布热力图 (turn vs entropy bin) ==========
    ax = axes[2, 1]
    all_turn_data = summary["all_episodes"].get("C_turn_mean", {})
    if all_turn_data:
        # 收集所有 turn 的熵值
        import numpy as np
        turn_entropies = []
        for turn_idx in range(15):
            if str(turn_idx) in all_turn_data:
                # 用均值代替实际分布做简单热力图
                turn_entropies.append(all_turn_data[str(turn_idx)]["mean"])
            else:
                turn_entropies.append(0)

        # 绘制条形图展示 turn 间熵变化
        xs = list(range(1, len(turn_entropies) + 1))
        colors = plt.cm.RdYlGn_r([v / max(turn_entropies) for v in turn_entropies])
        ax.bar(xs, turn_entropies, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(y=np.mean(turn_entropies), color="red", linestyle="--", label=f"mean={np.mean(turn_entropies):.3f}")

    ax.set_xlabel("Turn Step", fontsize=11)
    ax.set_ylabel("Mean Entropy", fontsize=11)
    ax.set_title("Entropy by Turn (Color: High=Red, Low=Green)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ========== (2,2)：相对位置 q 的详细分布 ==========
    ax = axes[2, 2]
    if entropy_by_q:
        # 显示每个 q 桶的样本数量
        q_labels = sorted(entropy_by_q.keys(), key=lambda x: (int(x[1:3]), int(x[4:6])))
        counts = [entropy_by_q[q]["count"] for q in q_labels]
        means = [entropy_by_q[q]["mean"] for q in q_labels]

        x_pos = range(len(q_labels))
        ax.bar(x_pos, counts, color="steelblue", alpha=0.7, label="Sample count")

        # 在柱子上标注均值
        for i, (c, m) in enumerate(zip(counts, means)):
            ax.text(i, c + max(counts) * 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xlabel("Relative Position q (5% bins)", fontsize=11)
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.set_title("Sample Distribution across Relative Positions", fontsize=11)
        ax.set_xticks(x_pos[::4])  # 每4个显示一个
        ax.set_xticklabels([q_labels[i] for i in range(0, len(q_labels), 4)], rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved: {output_path}")


# =============================================================================
# 文件写入
# =============================================================================

def safe_write(path: str, record: Dict):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# =============================================================================
# 主流程
# =============================================================================

def run_analysis(args):
    global DEBUG_MODE
    DEBUG_MODE = getattr(args, 'debug', False)

    # 加载数据
    trajectories = load_trajectories(args.traj_dir, args.max_samples)
    # 预加载模型/Tokenizer，避免中途多次加载
    _ = get_tokenizer(args.tokenizer_name)
    _ = get_model(args.model_path, args.cuda_device)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"offline_results_{ts}.jsonl")
    summary_file = os.path.join(args.output_dir, f"offline_summary_{ts}.json")
    plot_file    = os.path.join(args.output_dir, f"offline_plot_{ts}.png")
    open(results_file, "w").close()

    all_results = []

    # 本地单卡推理：顺序处理，保证稳定与正确性
    pbar = tqdm(total=len(trajectories), desc="Offline entropy analysis", unit="traj")
    for traj in trajectories:
        res = analyze_trajectory(
            traj=traj,
            model_path=args.model_path,
            tokenizer_name=args.tokenizer_name,
            top_k=args.top_k,
            max_turns=args.max_turns,
            save_per_token_entropy=args.save_per_token_entropy,
        )
        safe_write(results_file, res)
        all_results.append(res)
        pbar.update(1)
    pbar.close()

    # ========== 聚合分析和可视化（暂时禁用）==========
    # logger.info("Aggregating...")
    # summary = aggregate_entropy(all_results)
    #
    # has_error = False
    # if summary is None:
    #     summary = {"all_episodes": {}, "success_only": {}, "total_episodes": len(all_results), "success_episodes": 0}
    #     has_error = True
    # if not all_results:
    #     summary = {"all_episodes": {}, "success_only": {}, "total_episodes": 0, "success_episodes": 0}
    #     has_error = True
    #
    # with open(summary_file, "w") as f:
    #     json.dump(summary, f, indent=2)
    #
    # if has_error:
    #     return
    #
    # print("\n" + "=" * 65)
    # print("Results:")
    # print(f"Total: {summary['total_episodes']}, Success: {summary['success_episodes']}")
    # print("=" * 65)
    #
    # plot_entropy(summary, plot_file)

    logger.info(f"Done! Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline entropy analysis: small model on MiniMax-M2.1 BabyAI trajectories"
    )
    parser.add_argument("--traj_dir", type=str,
                        default="/Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356",
                        help="Directory containing trajectories jsonl (e.g., babyai_trajectories.jsonl)")
    parser.add_argument("--output_dir", type=str,
                        default="/Data/wyh/datasets/Verl-Data/outputs/entropy_offline_minimax",
                        help="Output directory")
    parser.add_argument("--model_path", type=str,
                        default="/Data/public/Qwen3-1.7B",
                        help="Local model path or HF repo id for Qwen3-1.7B")
    parser.add_argument("--tokenizer_name", type=str, default="/Data/public/Qwen3-1.7B",
                        help="Tokenizer name/path (default: /Data/public/Qwen3-1.7B)")
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA device index to load model onto (default: 0)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max trajectories to analyze, -1 = all")
    parser.add_argument("--max_turns", type=int, default=-1,
                        help="Max assistant turns to analyze per trajectory (-1 = all)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="top-k logprobs per token position")
    parser.add_argument("--save_per_token_entropy", action="store_true",
                        help="Save per-token entropy for detailed analysis (larger output file)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging for first few samples")
    args = parser.parse_args()

    logger.info("Config:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    run_analysis(args)


if __name__ == "__main__":
    main()
