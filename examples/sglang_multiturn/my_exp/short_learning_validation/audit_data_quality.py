#!/usr/bin/env python3
"""
Data quality audit script for textcraft validated parquet.

Audits:
1. Placeholder pollution ( <your next action>, [my next action] )
2. prefix_actions vs prompt assistant actions mismatch
3. Multi-action concatenation in assistant responses
4. task_id prefix_actions inconsistency (same task_id, different prefix_actions)
5. Goal / task_id collisions (different task_id, same goal)
6. Assistant prefix old logprobs / prefix_token_count consistency
7. Missing or empty fields

Outputs:
- corruption_report.json: detailed report
- cleaned parquet at new path (does NOT overwrite original)
"""

import pandas as pd
import numpy as np
import re
import json
import os
from collections import defaultdict
from datetime import datetime

INPUT_PATH = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet"
OUTPUT_DIR = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/cleaned"
REPORT_PATH = os.path.join(OUTPUT_DIR, "corruption_report.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Loading parquet from: {INPUT_PATH}")
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(df)} samples, columns: {list(df.columns)}")
print()

report = {
    "timestamp": timestamp,
    "input_path": INPUT_PATH,
    "total_samples": len(df),
    "issues": {},
    "bad_sample_ids": defaultdict(list),
    "summary": {},
}


# ─────────────────────────────────────────────
# Issue 1: Placeholder pollution
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 1: Placeholder pollution")
print("=" * 60)

placeholder_patterns = [
    r'<your next action>',
    r'\[my next action\]',
    r'<your next action',
    r'\[my next action',
]

placeholder_samples = []
for i in range(len(df)):
    row = df.iloc[i]
    prompt = row.get('prompt', [])
    # Convert numpy array to accessible list
    if isinstance(prompt, np.ndarray):
        prompt_list = [prompt[j] for j in range(len(prompt))]
    elif isinstance(prompt, list):
        prompt_list = list(prompt)
    else:
        prompt_list = []
    if not prompt_list:
        continue
    for msg in prompt_list:
        if not isinstance(msg, dict):
            continue
        # msg can be a numpy void (dict-like scalar) — extract values safely
        if isinstance(msg, np.void):
            content = str(msg['content']) if 'content' in msg.dtype.names else ''
            role = str(msg['role']) if 'role' in msg.dtype.names else ''
        else:
            content = str(msg.get('content', ''))
            role = str(msg.get('role', ''))
        if role == 'assistant':
            for pattern in placeholder_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    placeholder_samples.append(i)
                    break

report["issues"]["placeholder_pollution"] = {
    "count": len(placeholder_samples),
    "sample_ids": placeholder_samples[:50],
    "description": "Assistant messages contain placeholder text instead of proper Action: [[ xxx ]] blocks",
}
print(f"  Placeholder pollution: {len(placeholder_samples)} samples")
for idx in placeholder_samples[:5]:
    print(f"    Sample {idx}")
print()


# ─────────────────────────────────────────────
# Issue 2: prefix_actions vs prompt assistant actions mismatch
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 2: prefix_actions vs prompt assistant mismatch")
print("=" * 60)

def extract_prompt_actions(prompt):
    """Extract actions from assistant messages in prompt.

    prompt can be a numpy object array (shape N,) containing dict items,
    or a plain Python list of dicts.
    """
    actions = []

    # Convert numpy array to accessible list
    if isinstance(prompt, np.ndarray):
        # numpy object array — iterate element by element
        prompt_list = [prompt[i] for i in range(len(prompt))]
    elif isinstance(prompt, list):
        prompt_list = prompt
    else:
        return actions

    for item in prompt_list:
        if isinstance(item, np.void):
            role = str(item['role']) if 'role' in item.dtype.names else ''
            content = str(item['content']) if 'content' in item.dtype.names else ''
        elif isinstance(item, dict):
            role = str(item.get('role', ''))
            content = str(item.get('content', ''))
        else:
            continue
        if role == 'assistant':
            matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', content, re.DOTALL)
            for m in matches:
                action = ' '.join(m.split()).strip()
                if action:
                    actions.append(action)
    return actions

prefix_mismatch_samples = []
prefix_mismatch_reasons = defaultdict(list)

for i in range(len(df)):
    row = df.iloc[i]
    extra = row.get('extra_info', {})
    interaction_kwargs = extra.get('interaction_kwargs', {})
    prefix_actions_meta = interaction_kwargs.get('prefix_actions', [])
    prompt_actions = extract_prompt_actions(row.get('prompt', []))

    # Normalize
    meta_norm = [' '.join(a.split()).strip() for a in prefix_actions_meta if ' '.join(a.split()).strip()]
    prompt_norm = [' '.join(a.split()).strip() for a in prompt_actions if ' '.join(a.split()).strip()]

    if meta_norm != prompt_norm:
        prefix_mismatch_samples.append(i)
        reason = ""
        if len(meta_norm) != len(prompt_norm):
            reason = f"len_diff: meta={len(meta_norm)} vs prompt={len(prompt_norm)}"
        elif meta_norm and prompt_norm and meta_norm[-1] != prompt_norm[-1]:
            reason = f"last_diff: meta={meta_norm[-1]!r} vs prompt={prompt_norm[-1]!r}"
        else:
            reason = f"order_diff"
        prefix_mismatch_reasons[reason].append(i)

report["issues"]["prefix_actions_mismatch"] = {
    "count": len(prefix_mismatch_samples),
    "sample_ids": prefix_mismatch_samples[:50],
    "reasons": {k: len(v) for k, v in prefix_mismatch_reasons.items()},
    "description": "extra_info.prefix_actions does not match the assistant actions in the prompt",
}
print(f"  Prefix actions mismatch: {len(prefix_mismatch_samples)} samples")
for reason, indices in list(prefix_mismatch_reasons.items())[:5]:
    print(f"    {reason}: {len(indices)} samples")
print()


# ─────────────────────────────────────────────
# Issue 3: Multi-action concatenation
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 3: Multi-action concatenation")
print("=" * 60)

multi_action_samples = []
for i in range(len(df)):
    row = df.iloc[i]
    prompt = row.get('prompt', [])
    # Convert numpy array to accessible list
    if isinstance(prompt, np.ndarray):
        prompt_list = [prompt[j] for j in range(len(prompt))]
    elif isinstance(prompt, list):
        prompt_list = list(prompt)
    else:
        prompt_list = []
    for msg in prompt_list:
        if isinstance(msg, np.void):
            role = str(msg['role']) if 'role' in msg.dtype.names else ''
            content = str(msg['content']) if 'content' in msg.dtype.names else ''
        elif isinstance(msg, dict):
            role = str(msg.get('role', ''))
            content = str(msg.get('content', ''))
        else:
            continue
        if role == 'assistant':
            matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', content, re.DOTALL)
            for m in matches:
                action_text = m.strip()
                get_count = len(re.findall(r'\bget\b', action_text, re.IGNORECASE))
                craft_count = len(re.findall(r'\bcraft\b', action_text, re.IGNORECASE))
                if get_count > 1 or craft_count > 1:
                    multi_action_samples.append(i)
                    break

report["issues"]["multi_action_concatenation"] = {
    "count": len(set(multi_action_samples)),
    "sample_ids": list(set(multi_action_samples))[:50],
    "description": "A single Action: [[ xxx ]] block contains multiple concatenated actions",
}
print(f"  Multi-action concatenation: {len(set(multi_action_samples))} samples")
print()


# ─────────────────────────────────────────────
# Issue 4: task_id prefix_actions inconsistency
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 4: task_id prefix_actions inconsistency")
print("=" * 60)

task_prefix_map = defaultdict(list)
for i in range(len(df)):
    row = df.iloc[i]
    extra = row.get('extra_info', {})
    interaction_kwargs = extra.get('interaction_kwargs', {})
    task_id = interaction_kwargs.get('task_id', None)
    prefix_actions = interaction_kwargs.get('prefix_actions', [])
    if task_id is not None:
        key = tuple([' '.join(a.split()).strip() for a in prefix_actions])
        task_prefix_map[task_id].append(key)

task_prefix_inconsistent = []
for task_id, prefix_lists in task_prefix_map.items():
    if len(set(prefix_lists)) > 1:
        task_prefix_inconsistent.append(task_id)

report["issues"]["task_prefix_inconsistency"] = {
    "count": len(task_prefix_inconsistent),
    "task_ids": task_prefix_inconsistent[:50],
    "description": "Same task_id has different prefix_actions across samples",
}
print(f"  task_id prefix inconsistency: {len(task_prefix_inconsistent)} task_ids")
print()


# ─────────────────────────────────────────────
# Issue 5: Goal / task_id collisions
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 5: Goal / task_id collisions")
print("=" * 60)

task_goal_map = defaultdict(list)
for i in range(len(df)):
    row = df.iloc[i]
    extra = row.get('extra_info', {})
    interaction_kwargs = extra.get('interaction_kwargs', {})
    task_id = interaction_kwargs.get('task_id', None)
    prompt = row.get('prompt', [])
    # For goal extraction, do string-level search on raw prompt repr to avoid numpy iteration issues
    prompt_raw = str(prompt)
    if 'Your goal:' not in prompt_raw or task_id is None:
        continue
    goal = None
    # Iterate numpy array element by element
    if isinstance(prompt, np.ndarray):
        for pi in range(len(prompt)):
            item = prompt[pi]
            if isinstance(item, np.void):
                role = str(item['role']) if 'role' in item.dtype.names else ''
                content = str(item['content']) if 'content' in item.dtype.names else ''
            elif isinstance(item, dict):
                role = str(item.get('role', ''))
                content = str(item.get('content', ''))
            else:
                continue
            if role == 'user' and 'Your goal:' in content:
                match = re.search(r'Your goal:\s*(.+)', content)
                if match:
                    goal = match.group(1).strip()
                    break
    if goal:
        task_goal_map[goal].append(task_id)

goal_collision = {g: tids for g, tids in task_goal_map.items() if len(set(tids)) > 1}

report["issues"]["goal_taskid_collision"] = {
    "count": len(goal_collision),
    "examples": {g: tids[:5] for g, tids in list(goal_collision.items())[:10]},
    "description": "Same goal maps to different task_ids",
}
print(f"  Goal/task_id collisions: {len(goal_collision)} goals")
print()


# ─────────────────────────────────────────────
# Issue 6: prefix_token_count consistency
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 6: prefix_token_count / olp consistency")
print("=" * 60)

olp_mismatch = []
mask_mismatch = []
for i in range(len(df)):
    row = df.iloc[i]
    ptc = row.get('prefix_token_count', None)
    olp = row.get('assistant_prefix_old_log_probs', None)
    mask = row.get('prefix_mask', None)
    if ptc is None or olp is None or mask is None:
        continue
    olp_len = len(olp) if hasattr(olp, '__len__') else 1
    mask_sum = int(mask.sum()) if hasattr(mask, 'sum') else sum(mask)
    if olp_len != ptc:
        olp_mismatch.append(i)
    if mask_sum != ptc:
        mask_mismatch.append(i)

report["issues"]["prefix_token_consistency"] = {
    "olp_len_mismatch_count": len(olp_mismatch),
    "mask_sum_mismatch_count": len(mask_mismatch),
    "description": "prefix_token_count should equal len(olp) and mask.sum() for all samples",
}
print(f"  olp_len != prefix_token_count: {len(olp_mismatch)} samples")
print(f"  mask.sum() != prefix_token_count: {len(mask_mismatch)} samples")
print()


# ─────────────────────────────────────────────
# Issue 7: Missing or empty required fields
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 7: Missing or empty required fields")
print("=" * 60)

required_fields = ['prompt', 'extra_info', 'assistant_prefix_old_log_probs', 'prefix_token_count', 'prefix_mask']
missing_field_samples = defaultdict(list)
for i in range(len(df)):
    row = df.iloc[i]
    for field in required_fields:
        val = row.get(field, None)
        if val is None:
            missing_field_samples[field].append(i)
        elif isinstance(val, (list, np.ndarray)) and len(val) == 0:
            missing_field_samples[field].append(i)

report["issues"]["missing_fields"] = {
    field: len(indices) for field, indices in missing_field_samples.items()
}
for field, indices in missing_field_samples.items():
    if indices:
        print(f"  Missing/empty '{field}': {len(indices)} samples")
print()


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)

# Collect all bad sample IDs
all_bad = set()
all_bad.update(placeholder_samples)
all_bad.update(prefix_mismatch_samples)
all_bad.update(multi_action_samples)

# task_prefix_inconsistent: map task_ids back to sample indices
for tid in task_prefix_inconsistent:
    for i in range(len(df)):
        row = df.iloc[i]
        extra = row.get('extra_info', {})
        interaction_kwargs = extra.get('interaction_kwargs', {})
        if interaction_kwargs.get('task_id') == tid:
            all_bad.add(i)

report["bad_sample_ids"] = sorted(all_bad)
report["summary"] = {
    "total_samples": len(df),
    "total_bad_samples": len(all_bad),
    "good_samples": len(df) - len(all_bad),
    "placeholder_pollution": len(placeholder_samples),
    "prefix_actions_mismatch": len(prefix_mismatch_samples),
    "multi_action_concatenation": len(set(multi_action_samples)),
    "task_prefix_inconsistency": len(task_prefix_inconsistent),
    "goal_taskid_collision": len(goal_collision),
    "olp_mismatch": len(olp_mismatch),
    "mask_mismatch": len(mask_mismatch),
}

for key, val in report["summary"].items():
    print(f"  {key}: {val}")
print()


# ─────────────────────────────────────────────
# Generate cleaned parquet
# ─────────────────────────────────────────────
print("=" * 60)
print("Generating cleaned parquet")
print("=" * 60)

bad_ids = set(report["bad_sample_ids"])
good_mask = ~df.index.isin(bad_ids)
df_clean = df[good_mask].reset_index(drop=True)

cleaned_path = os.path.join(OUTPUT_DIR, f"textcraft_validated_cleaned_{timestamp}.parquet")
df_clean.to_parquet(cleaned_path, index=False)

print(f"  Original samples: {len(df)}")
print(f"  Bad samples removed: {len(bad_ids)}")
print(f"  Cleaned samples: {len(df_clean)}")
print(f"  Cleaned parquet saved to: {cleaned_path}")
print()


# ─────────────────────────────────────────────
# Save report
# ─────────────────────────────────────────────
with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"Report saved to: {REPORT_PATH}")
