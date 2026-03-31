#!/usr/bin/env python3
"""性能分析脚本：测量各部分的耗时"""
import os
import sys
import time
import json
import math
import torch

# Set CUDA device before importing transformers
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM

TOKENIZER = None
MODEL = None
MODEL_DEVICE = None

def get_tokenizer(tokenizer_name="/Data/public/Qwen3-1.7B"):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return TOKENIZER

def get_model(model_path="/Data/public/Qwen3-1.7B", cuda_device=0):
    global MODEL, MODEL_DEVICE
    if MODEL is not None:
        return MODEL
    
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        MODEL_DEVICE = torch.device(f"cuda:{cuda_device}")
    else:
        dtype = torch.float32
        MODEL_DEVICE = torch.device("cpu")
    
    print(f"Loading model to {MODEL_DEVICE} with dtype {dtype}...")
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    MODEL.to(MODEL_DEVICE)
    MODEL.eval()
    print("Model loaded!")
    return MODEL

def apply_qwen_chat_template(messages, tokenizer):
    """Apply chat template"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text

def parse_assistant_turns(conversations):
    """提取所有 assistant 的 content"""
    return [c["content"] for c in conversations if c.get("role") == "assistant"]

def parse_user_turns(conversations):
    """提取所有 user 的 content"""
    return [c["content"] for c in conversations if c.get("role") == "user"]

def build_prefix_for_turn(conversations, turn_idx, role="assistant"):
    """构建该 turn 之前的上下文（不包含 target turn 本身）"""
    prefix_messages = []
    role_count = 0
    for msg in conversations:
        msg_role = msg.get("role", "")
        if msg_role == role:
            if role_count == turn_idx:
                break
            role_count += 1
        prefix_messages.append(msg)
    return prefix_messages

# ====== 性能测量 ======

def benchmark_single_turn(traj, turn_idx=0, role="assistant"):
    """测量处理单个 turn 的各部分耗时"""
    tokenizer = get_tokenizer()
    model = get_model()
    
    conversations = traj.get("conversations", [])
    
    if role == "assistant":
        turns = parse_assistant_turns(conversations)
        target_content = turns[turn_idx] if turn_idx < len(turns) else None
    else:
        turns = parse_user_turns(conversations)
        target_content = turns[turn_idx] if turn_idx < len(turns) else None
    
    if target_content is None:
        return None
    
    # 1. Build prefix
    t0 = time.time()
    prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role=role)
    t_prefix = time.time() - t0
    
    # 2. Apply chat template
    t0 = time.time()
    prefix_prompt = apply_qwen_chat_template(prefix_msgs, tokenizer) if prefix_msgs else ""
    target_message = {"role": role, "content": target_content}
    full_prompt = apply_qwen_chat_template(prefix_msgs + [target_message], tokenizer)
    t_template = time.time() - t0
    
    # 3. Tokenize
    t0 = time.time()
    enc = tokenizer(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    t_tokenize = time.time() - t0
    
    # 4. Find target token span
    t0 = time.time()
    search_text = full_prompt
    pos = search_text.find(target_content)
    if pos == -1:
        print(f"Warning: target content not found!")
        return None
    content_start_char = pos
    content_end_char = content_start_char + len(target_content)
    
    target_token_indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == 0 and tok_end == 0:
            continue
        if tok_end > content_start_char and tok_start < content_end_char:
            target_token_indices.append(idx)
    t_find_span = time.time() - t0
    
    # 5. Model forward
    t0 = time.time()
    input_ids = torch.tensor([full_ids], device=MODEL_DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    t_forward = time.time() - t0
    
    # 6. Extract logprobs and top-k (per token)
    t0 = time.time()
    top_k = 100
    for j in target_token_indices:
        t = j - 1
        if t < 0:
            continue
        gold_id = full_ids[j]
        gold_lp = float(log_probs[0, t, gold_id].item())
        
        # top-k
        k = min(top_k, log_probs.shape[-1])
        topv, topi = torch.topk(log_probs[0, t, :], k=k, dim=-1)
        
        # Decode each token
        seen_tokens = {}
        for tok_id, lp in zip(topi.tolist(), topv.tolist()):
            tok_str = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            if tok_str not in seen_tokens:
                seen_tokens[tok_str] = []
            seen_tokens[tok_str].append(float(lp))
        
        # Merge duplicate strings
        for tok_str, lps in seen_tokens.items():
            if len(lps) > 1:
                max_lp = max(lps)
                summed = sum(math.exp(lp - max_lp) for lp in lps)
                merged_lp = max_lp + math.log(summed)
    t_logprob = time.time() - t0
    
    # Total
    total = t_prefix + t_template + t_tokenize + t_find_span + t_forward + t_logprob
    
    print(f"\n=== Single Turn Performance ===")
    print(f"Prefix build:     {t_prefix*1000:8.2f} ms ({t_prefix/total*100:5.1f}%)")
    print(f"Chat template:    {t_template*1000:8.2f} ms ({t_template/total*100:5.1f}%)")
    print(f"Tokenize:         {t_tokenize*1000:8.2f} ms ({t_tokenize/total*100:5.1f}%)")
    print(f"Find span:        {t_find_span*1000:8.2f} ms ({t_find_span/total*100:5.1f}%)")
    print(f"Model forward:    {t_forward*1000:8.2f} ms ({t_forward/total*100:5.1f}%)")
    print(f"Logprob/top-k:    {t_logprob*1000:8.2f} ms ({t_logprob/total*100:5.1f}%)")
    print(f"-------------------")
    print(f"Total:            {total*1000:8.2f} ms")
    print(f"Input tokens:     {len(full_ids)}")
    print(f"Target tokens:    {len(target_token_indices)}")
    
    return {
        "t_prefix": t_prefix,
        "t_template": t_template,
        "t_tokenize": t_tokenize,
        "t_find_span": t_find_span,
        "t_forward": t_forward,
        "t_logprob": t_logprob,
        "total": total,
    }

def benchmark_full_trajectory(traj):
    """测量处理整条轨迹的耗时"""
    tokenizer = get_tokenizer()
    model = get_model()
    
    conversations = traj.get("conversations", [])
    assistant_turns = parse_assistant_turns(conversations)
    user_turns = parse_user_turns(conversations)
    
    total_turns = len(assistant_turns) + len(user_turns)
    
    t0 = time.time()
    processed = 0
    
    # Process assistant turns
    for turn_idx in range(len(assistant_turns)):
        prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="assistant")
        target_content = assistant_turns[turn_idx]
        
        prefix_prompt = apply_qwen_chat_template(prefix_msgs, tokenizer) if prefix_msgs else ""
        target_message = {"role": "assistant", "content": target_content}
        full_prompt = apply_qwen_chat_template(prefix_msgs + [target_message], tokenizer)
        
        enc = tokenizer(full_prompt, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        
        search_text = full_prompt
        pos = search_text.find(target_content)
        if pos == -1:
            continue
        content_start_char = pos
        content_end_char = content_start_char + len(target_content)
        
        target_token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_end > content_start_char and tok_start < content_end_char:
                target_token_indices.append(idx)
        
        input_ids = torch.tensor([full_ids], device=MODEL_DEVICE)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        
        top_k = 100
        for j in target_token_indices:
            t = j - 1
            if t < 0:
                continue
            gold_id = full_ids[j]
            gold_lp = float(log_probs[0, t, gold_id].item())
            k = min(top_k, log_probs.shape[-1])
            topv, topi = torch.topk(log_probs[0, t, :], k=k, dim=-1)
            for tok_id in topi.tolist():
                tok_str = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
        
        processed += 1
    
    # Process user turns
    for turn_idx in range(len(user_turns)):
        prefix_msgs = build_prefix_for_turn(conversations, turn_idx, role="user")
        target_content = user_turns[turn_idx]
        
        prefix_prompt = apply_qwen_chat_template(prefix_msgs, tokenizer) if prefix_msgs else ""
        target_message = {"role": "user", "content": target_content}
        full_prompt = apply_qwen_chat_template(prefix_msgs + [target_message], tokenizer)
        
        enc = tokenizer(full_prompt, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        
        search_text = full_prompt
        pos = search_text.find(target_content)
        if pos == -1:
            continue
        content_start_char = pos
        content_end_char = content_start_char + len(target_content)
        
        target_token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_end > content_start_char and tok_start < content_end_char:
                target_token_indices.append(idx)
        
        input_ids = torch.tensor([full_ids], device=MODEL_DEVICE)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        
        top_k = 100
        for j in target_token_indices:
            t = j - 1
            if t < 0:
                continue
            gold_id = full_ids[j]
            gold_lp = float(log_probs[0, t, gold_id].item())
            k = min(top_k, log_probs.shape[-1])
            topv, topi = torch.topk(log_probs[0, t, :], k=k, dim=-1)
            for tok_id in topi.tolist():
                tok_str = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
        
        processed += 1
    
    t_total = time.time() - t0
    
    print(f"\n=== Full Trajectory Performance ===")
    print(f"Total turns:      {total_turns}")
    print(f"Processed:        {processed}")
    print(f"Total time:       {t_total*1000:8.2f} ms")
    print(f"Time per turn:    {t_total/processed*1000:8.2f} ms")
    
    return t_total

if __name__ == "__main__":
    # Load a sample trajectory
    traj_path = '/Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356/babyai_trajectories.jsonl'
    with open(traj_path) as f:
        traj = json.loads(f.readline())
    
    print(f"Trajectory: {traj['item_id']}")
    print(f"Conversations: {len(traj['conversations'])} messages")
    
    # Benchmark single turn
    result = benchmark_single_turn(traj, turn_idx=0, role="assistant")
    
    # Benchmark full trajectory
    t_traj = benchmark_full_trajectory(traj)
    
    print(f"\n=== Estimated throughput ===")
    # Assume average 5 turns per trajectory, 1000 trajectories
    est_time_per_traj = t_traj
    print(f"Estimated time for 100 trajectories: {est_time_per_traj*100/60:.1f} min")
    print(f"Estimated time for 1000 trajectories: {est_time_per_traj*1000/60:.1f} min")
