#!/usr/bin/env python3
"""
Data quality audit script v2 for textcraft validated parquet.

NEW RULES (v2):
- prefix_actions_mismatch: 只比较 prompt 历史中"最后 len(prefix_actions) 个 action"，
  而不是比较整个 prompt 的所有 assistant actions。
  理由：prompt 可能被历史污染，只要当前 segment 的最后 N 个 action 对齐就认为该 segment 干净。
  口径：只标记为 suspicious flag，不做 hard delete。

- task_prefix_inconsistency: 不再把 task-level inconsistency 直接映射成 sample-level 全删。
  新口径：对每个 task_id，取 prefix_actions 的众数版本，只删除"非众数"的罕见异常样本。
  众数版本保留。

- multi_action_concatenation: 降级为 suspicious flag，不做 hard delete。
  理由：多 action 拼接可能是数据构建过程的历史遗留，不一定是当前样本的错。

保留 hard delete 的规则：
- placeholder_pollution（明确的占位符）
- prefix_token_count / olp 不一致
- missing_fields
"""

import pandas as pd
import numpy as np
import re
import json
import os
from collections import defaultdict, Counter
from datetime import datetime

INPUT_PATH = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet"
OUTPUT_DIR = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/cleaned_v2"
REPORT_PATH = os.path.join(OUTPUT_DIR, "corruption_report_v2.json")

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
    "warnings": {},        # suspicious but not hard-deleted
    "bad_sample_ids": [],  # actually removed
    "good_sample_ids": [], # kept
}


# ─────────────────────────────────────────────
# Helper: extract all actions from prompt
# ─────────────────────────────────────────────
def get_prompt_role_content(prompt):
    """Iterate prompt (numpy object array or list) and yield (role, content) pairs."""
    if isinstance(prompt, np.ndarray):
        items = [prompt[j] for j in range(len(prompt))]
    elif isinstance(prompt, list):
        items = list(prompt)
    else:
        return
    for item in items:
        if isinstance(item, np.void):
            role = str(item['role']) if 'role' in item.dtype.names else ''
            content = str(item['content']) if 'content' in item.dtype.names else ''
        elif isinstance(item, dict):
            role = str(item.get('role', ''))
            content = str(item.get('content', ''))
        else:
            continue
        yield role, content


def extract_all_actions(prompt):
    """Extract all [[ action ]] from assistant messages."""
    actions = []
    for role, content in get_prompt_role_content(prompt):
        if role == 'assistant':
            matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', content, re.DOTALL)
            for m in matches:
                action = ' '.join(m.split()).strip()
                if action:
                    actions.append(action)
    return actions


def extract_last_n_actions(prompt, n):
    """Extract the LAST n [[ action ]] from assistant messages."""
    all_actions = extract_all_actions(prompt)
    return all_actions[-n:] if len(all_actions) >= n else all_actions


def extract_goal(prompt):
    """Extract the goal from the first user message containing 'Your goal:'."""
    for role, content in get_prompt_role_content(prompt):
        if role == 'user' and 'Your goal:' in content:
            match = re.search(r'Your goal:\s*(.+)', content)
            if match:
                return match.group(1).strip()
    return None


# ─────────────────────────────────────────────
# Issue 1: Placeholder pollution — HARD DELETE
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 1: Placeholder pollution — HARD DELETE")
print("=" * 60)

placeholder_patterns = [
    r'<your next action>', r'\[my next action\]',
    r'<your next action', r'\[my next action',
]

placeholder_samples = []
for i in range(len(df)):
    prompt = df.iloc[i].get('prompt', [])
    for role, content in get_prompt_role_content(prompt):
        if role == 'assistant':
            for pattern in placeholder_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    placeholder_samples.append(i)
                    break

report["issues"]["placeholder_pollution"] = {
    "count": len(placeholder_samples),
    "sample_ids": placeholder_samples[:20],
    "severity": "HARD_DELETE",
    "description": "Assistant messages contain placeholder text",
}
print(f"  Placeholder pollution: {len(placeholder_samples)} samples")
for idx in placeholder_samples[:3]:
    print(f"    Sample {idx}")
print()


# ─────────────────────────────────────────────
# Issue 2: prefix_actions_mismatch — SUSPICIOUS FLAG (v2)
# 只比较最后 len(prefix_actions) 个 action
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 2: prefix_actions_mismatch — SUSPICIOUS FLAG (v2)")
print("=" * 60)

# Build per-sample analysis
prefix_mismatch_info = {}  # sample_id -> dict with details

for i in range(len(df)):
    row = df.iloc[i]
    extra = row.get('extra_info', {})
    interaction_kwargs = extra.get('interaction_kwargs', {})
    prefix_actions_meta = interaction_kwargs.get('prefix_actions', [])
    prompt_all_actions = extract_all_actions(row.get('prompt', []))

    # Normalize meta prefix_actions
    meta_norm = tuple([' '.join(a.split()).strip() for a in prefix_actions_meta if ' '.join(a.split()).strip()])
    n = len(meta_norm)

    # Compare ONLY the last n prompt actions with meta prefix_actions
    last_n_prompt_actions = tuple(prompt_all_actions[-n:]) if n > 0 else ()

    mismatch = (meta_norm != last_n_prompt_actions)

    # Detailed reason
    reason = "ok"
    if mismatch:
        if len(prompt_all_actions) < n:
            reason = f"prompt_short: prompt has {len(prompt_all_actions)} actions but meta needs {n}"
        elif meta_norm != last_n_prompt_actions:
            reason = f"segment_mismatch: meta_last={meta_norm[-1]!r} vs prompt_last={last_n_prompt_actions[-1]!r}"

    prefix_mismatch_info[i] = {
        "meta_len": n,
        "prompt_all_len": len(prompt_all_actions),
        "meta_last": meta_norm[-1] if meta_norm else None,
        "prompt_last": last_n_prompt_actions[-1] if last_n_prompt_actions else None,
        "mismatch": mismatch,
        "reason": reason,
        "meta_full": list(meta_norm),
        "prompt_segment": list(last_n_prompt_actions),
    }

# Count mismatches
mismatch_samples = [i for i, v in prefix_mismatch_info.items() if v["mismatch"]]
report["warnings"]["prefix_actions_mismatch"] = {
    "count": len(mismatch_samples),
    "sample_ids": mismatch_samples[:20],
    "severity": "SUSPICIOUS_ONLY",
    "description": "Last N prompt actions != prefix_actions (prompt may have historical pollution — NOT hard deleted)",
}
print(f"  Segment mismatch (suspicious): {len(mismatch_samples)} samples")
# Show breakdown
reason_counts = Counter(v["reason"] for v in prefix_mismatch_info.values() if v["mismatch"])
for reason, cnt in reason_counts.most_common(5):
    print(f"    {reason}: {cnt}")
print()


# ─────────────────────────────────────────────
# Issue 3: Multi-action concatenation — SUSPICIOUS FLAG (v2)
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 3: Multi-action concatenation — SUSPICIOUS FLAG (v2)")
print("=" * 60)

multi_action_samples = []
for i in range(len(df)):
    prompt = df.iloc[i].get('prompt', [])
    for role, content in get_prompt_role_content(prompt):
        if role == 'assistant':
            matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', content, re.DOTALL)
            for m in matches:
                action_text = m.strip()
                get_count = len(re.findall(r'\bget\b', action_text, re.IGNORECASE))
                craft_count = len(re.findall(r'\bcraft\b', action_text, re.IGNORECASE))
                if get_count > 1 or craft_count > 1:
                    multi_action_samples.append(i)
                    break

report["warnings"]["multi_action_concatenation"] = {
    "count": len(multi_action_samples),
    "sample_ids": list(set(multi_action_samples))[:20],
    "severity": "SUSPICIOUS_ONLY",
    "description": "Multi-action concatenation in prompt (NOT hard deleted)",
}
print(f"  Multi-action concatenation (suspicious): {len(set(multi_action_samples))} samples")
print()


# ─────────────────────────────────────────────
# Issue 4: task_id prefix_actions inconsistency — SOFT DELETE (v2)
# 只删除非众数版本，保留众数版本
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 4: task_id prefix_actions inconsistency — SOFT DELETE (v2)")
print("=" * 60)

# Step 1: for each task_id, count all prefix_actions versions
task_prefix_versions = defaultdict(Counter)  # task_id -> Counter of (prefix_actions_tuple -> count)
for i in range(len(df)):
    row = df.iloc[i]
    extra = row.get('extra_info', {})
    interaction_kwargs = extra.get('interaction_kwargs', {})
    task_id = interaction_kwargs.get('task_id', None)
    prefix_actions = interaction_kwargs.get('prefix_actions', [])
    if task_id is not None:
        key = tuple([' '.join(a.split()).strip() for a in prefix_actions])
        task_prefix_versions[task_id][key] += 1

# Step 2: for each task_id, find the mode version
task_mode_version = {}  # task_id -> most_common prefix_actions tuple
for task_id, counter in task_prefix_versions.items():
    task_mode_version[task_id] = counter.most_common(1)[0][0]

# Step 3: mark samples as suspicious if their prefix_actions != mode version
task_inconsistency_bad = []  # samples to delete (non-mode versions)
task_inconsistency_warn = []  # samples that are mode versions but task has other versions
for i in range(len(df)):
    row = df.iloc[i]
    extra = row.get('extra_info', {})
    interaction_kwargs = extra.get('interaction_kwargs', {})
    task_id = interaction_kwargs.get('task_id', None)
    prefix_actions = interaction_kwargs.get('prefix_actions', [])
    if task_id is None:
        continue
    key = tuple([' '.join(a.split()).strip() for a in prefix_actions])
    mode = task_mode_version[task_id]
    if key != mode:
        task_inconsistency_bad.append(i)
    elif len(task_prefix_versions[task_id]) > 1:
        # This is a mode sample but the task has other versions — keep but flag
        task_inconsistency_warn.append(i)

# Also: for task_ids with ONLY one version, no issue
clean_task_ids = [tid for tid, counter in task_prefix_versions.items() if len(counter) == 1]
multi_version_task_ids = [tid for tid, counter in task_prefix_versions.items() if len(counter) > 1]

report["issues"]["task_prefix_inconsistency"] = {
    "total_task_ids": len(task_prefix_versions),
    "clean_task_ids_count": len(clean_task_ids),
    "multi_version_task_ids_count": len(multi_version_task_ids),
    "mode_samples_kept": len(task_inconsistency_warn),
    "non_mode_samples_deleted": len(task_inconsistency_bad),
    "non_mode_sample_ids": task_inconsistency_bad[:20],
    "severity": "SOFT_DELETE",
    "description": "Delete non-mode prefix_actions versions; keep mode versions (even if task has multiple versions)",
}
print(f"  Total task_ids: {len(task_prefix_versions)}")
print(f"  task_ids with only 1 version (clean): {len(clean_task_ids)}")
print(f"  task_ids with multiple versions: {len(multi_version_task_ids)}")
print(f"  Mode samples (kept, flagged as warning): {len(task_inconsistency_warn)}")
print(f"  Non-mode samples (deleted): {len(task_inconsistency_bad)}")
print()


# ─────────────────────────────────────────────
# Issue 5: Goal / task_id collisions — WARNING ONLY
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 5: Goal / task_id collisions — WARNING")
print("=" * 60)

task_goal_map = defaultdict(list)
for i in range(len(df)):
    prompt = df.iloc[i].get('prompt', [])
    extra = df.iloc[i].get('extra_info', {})
    interaction_kwargs = extra.get('interaction_kwargs', {})
    task_id = interaction_kwargs.get('task_id', None)
    goal = extract_goal(prompt)
    if goal and task_id is not None:
        task_goal_map[goal].append(task_id)

goal_collision = {g: tids for g, tids in task_goal_map.items() if len(set(tids)) > 1}

report["warnings"]["goal_taskid_collision"] = {
    "count": len(goal_collision),
    "examples": {g: sorted(set(tids))[:5] for g, tids in list(goal_collision.items())[:10]},
    "severity": "WARNING_ONLY",
    "description": "Same goal maps to different task_ids — informational only",
}
print(f"  Goal/task_id collisions: {len(goal_collision)} goals")
print()


# ─────────────────────────────────────────────
# Issue 6: prefix_token_count / olp consistency — HARD DELETE
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 6: prefix_token_count / olp consistency — HARD DELETE")
print("=" * 60)

olp_mismatch = set()
mask_mismatch = set()
for i in range(len(df)):
    row = df.iloc[i]
    ptc = row.get('prefix_token_count', None)
    olp = row.get('assistant_prefix_old_log_probs', None)
    mask = row.get('prefix_mask', None)
    if ptc is None or olp is None or mask is None:
        olp_mismatch.add(i)
        continue
    olp_len = len(olp) if hasattr(olp, '__len__') else 1
    mask_sum = int(mask.sum()) if hasattr(mask, 'sum') else sum(mask)
    if olp_len != ptc:
        olp_mismatch.add(i)
    if mask_sum != ptc:
        mask_mismatch.add(i)

report["issues"]["prefix_token_consistency"] = {
    "olp_len_mismatch_count": len(olp_mismatch),
    "mask_sum_mismatch_count": len(mask_mismatch),
    "severity": "HARD_DELETE",
    "description": "prefix_token_count should equal len(olp) and mask.sum() for all samples",
}
print(f"  olp_len != prefix_token_count: {len(olp_mismatch)} samples (deleted)")
print(f"  mask.sum() != prefix_token_count: {len(mask_mismatch)} samples (deleted)")
print()


# ─────────────────────────────────────────────
# Issue 7: Missing or empty required fields — HARD DELETE
# ─────────────────────────────────────────────
print("=" * 60)
print("Issue 7: Missing or empty required fields — HARD DELETE")
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
        print(f"  Missing/empty '{field}': {len(indices)} samples (deleted)")
print()


# ─────────────────────────────────────────────
# Build final good/bad lists
# ─────────────────────────────────────────────
print("=" * 60)
print("BUILDING FINAL GOOD/BAD LISTS")
print("=" * 60)

# Hard delete: placeholder + olp/mask mismatch + missing fields
hard_delete = set()
hard_delete.update(placeholder_samples)
hard_delete.update(olp_mismatch)
hard_delete.update(mask_mismatch)
for indices in missing_field_samples.values():
    hard_delete.update(indices)

# Soft delete: non-mode task_prefix_inconsistency
soft_delete = set(task_inconsistency_bad)

# Suspicious flags (NOT deleted, just logged)
all_bad_ids = sorted(hard_delete | soft_delete)
all_good_ids = sorted(set(range(len(df))) - hard_delete - soft_delete)

report["bad_sample_ids"] = all_bad_ids
report["good_sample_ids"] = all_good_ids

print(f"  Hard delete (placeholder + tokenization + missing): {len(hard_delete)}")
print(f"  Soft delete (non-mode prefix_actions): {len(soft_delete)}")
print(f"  Total bad samples: {len(all_bad_ids)}")
print(f"  Total good samples: {len(all_good_ids)}")
print()


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
report["summary"] = {
    "total_samples": len(df),
    "total_bad_samples": len(all_bad_ids),
    "good_samples": len(all_good_ids),
    "hard_delete_reasons": {
        "placeholder_pollution": len(placeholder_samples),
        "olp_mismatch": len(olp_mismatch),
        "mask_mismatch": len(mask_mismatch),
        "missing_fields": sum(len(v) for v in missing_field_samples.values()),
    },
    "soft_delete_reasons": {
        "task_prefix_inconsistency_non_mode": len(task_inconsistency_bad),
    },
    "suspicious_warnings_only": {
        "prefix_actions_mismatch": len(mismatch_samples),
        "multi_action_concatenation": len(set(multi_action_samples)),
        "goal_taskid_collision": len(goal_collision),
    },
}

print("=" * 60)
print("SUMMARY")
print("=" * 60)
for key, val in report["summary"].items():
    print(f"  {key}: {val}")
print()


# ─────────────────────────────────────────────
# Generate cleaned v2 parquet
# ─────────────────────────────────────────────
print("=" * 60)
print("Generating cleaned v2 parquet")
print("=" * 60)

good_mask = df.index.isin(all_good_ids)
df_clean = df[good_mask].reset_index(drop=True)

cleaned_path = os.path.join(OUTPUT_DIR, f"textcraft_validated_cleaned_v2_{timestamp}.parquet")
df_clean.to_parquet(cleaned_path, index=False)

print(f"  Original samples: {len(df)}")
print(f"  Removed: {len(all_bad_ids)}")
print(f"  Cleaned v2 samples: {len(df_clean)}")
print(f"  Cleaned v2 parquet: {cleaned_path}")
print()


# ─────────────────────────────────────────────
# Save report
# ─────────────────────────────────────────────
with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"Report saved: {REPORT_PATH}")
print()


# ─────────────────────────────────────────────
# Show sample details for first 5 suspicious mismatch samples
# ─────────────────────────────────────────────
print("=" * 60)
print("SAMPLE DETAILS: First 5 segment mismatch (suspicious)")
print("=" * 60)
for i in mismatch_samples[:5]:
    info = prefix_mismatch_info[i]
    goal = extract_goal(df.iloc[i].get('prompt', []))
    print(f"  Sample {i}:")
    print(f"    goal: {goal}")
    print(f"    reason: {info['reason']}")
    print(f"    meta_prefix_actions ({info['meta_len']}): {info['meta_full']}")
    print(f"    prompt_last ({info['meta_len']}): {info['prompt_segment']}")
    print(f"    prompt_total_actions: {info['prompt_all_len']}")
    print()
