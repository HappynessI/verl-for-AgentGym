#!/usr/bin/env python3
"""
Light validation: 模拟 TextCraftInteraction.start_interaction + prefix_replay
对 parquet 里 task_id=81 / goal=emerald block 的样本做完整的环境交互验证

验证目标：
1. goal 能否正确绑定（env 的 goal == parquet 里的 goal）
2. prefix_actions replay 后状态正确
3. craft action 返回 reward=1, done=True
"""
import json, re, sys, asyncio, numpy as np

# ── Parquet ──────────────────────────────────────────────────────────────────
PARQUET = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/cleaned_v2/textcraft_validated_cleaned_v2_20260326_000658.parquet"

# ── TextCraft server ───────────────────────────────────────────────────────────
import requests

BASE = "http://127.0.0.1:36001"


def extract_goal_from_prompt(prompt_messages):
    """从 prompt 数组里提取第一个非 system 的 Goal: 字段。"""
    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()
    if not isinstance(prompt_messages, list):
        prompt_messages = list(prompt_messages)
    for msg in prompt_messages:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '')
            m = re.search(r'Goal:\s*craft\s+(.+?)\.?$', content, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
    return None


def normalize_goal(g):
    """标准化比较：去掉 namespace、前后空格、_"""
    if g is None:
        return None
    return g.lower().replace("minecraft:", "").replace("_", " ").replace("'", "").strip()


def extract_action(text):
    """从 assistant content 里提取 action（复现 TextCraftInteraction.extract_action）。"""
    text = text.strip()
    text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|im_end\|>', '', text)

    # Method 1: [[ ... ]]
    matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
    if matches:
        action = ' '.join(matches[-1].split()).strip()
        if action:
            return action, 'double_bracket'

    # Method 2: Action:\nxxx
    m = re.search(r'Action:\s*\n\s*(.+?)(?:\n|$)', text, re.DOTALL)
    if m:
        action = ' '.join(m.group(1).split()).strip()
        if action:
            return action, 'action_colon'

    # Method 3: Action: xxx
    m = re.search(r'Action:\s*(.+?)(?:\n|$)', text, re.DOTALL)
    if m:
        action = ' '.join(m.group(1).split()).strip()
        if action:
            return action, 'action_inline'

    return None, 'no_match'


async def simulate_start_interaction(instance_id, interaction_kwargs, prompt=None):
    """
    模拟 TextCraftInteraction.start_interaction 的完整流程：
    1. 从 interaction_kwargs 提取 goal
    2. POST /create {goal: ...}
    3. 校验 goal 匹配
    4. replay prefix_actions
    """
    # Extract goal (from interaction_kwargs, with fallback from prompt)
    expected_goal = interaction_kwargs.get('goal')
    if expected_goal is None and prompt is not None:
        prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else (prompt if isinstance(prompt, list) else [])
        expected_goal = extract_goal_from_prompt(prompt_list)
        if expected_goal:
            print(f"    [start_interaction] goal not in kwargs — extracted from prompt: {expected_goal!r}")

    prefix_actions = interaction_kwargs.get('prefix_actions', [])
    if isinstance(prefix_actions, np.ndarray):
        prefix_actions = prefix_actions.tolist()

    # Build create body
    create_body = {}
    if expected_goal is not None:
        create_body['goal'] = expected_goal

    # POST /create
    resp = requests.post(f"{BASE}/create", json=create_body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    env_id = data.get('id', data.get('env_id', '?'))
    init_obs = data.get('observation', '')

    # ── FAIL-FAST: goal 校验 ──────────────────────────────────────────────
    actual_goal_in_obs = None
    m = re.search(r'Goal:\s*craft\s+(.+?)\.?$', init_obs, re.IGNORECASE | re.MULTILINE)
    if m:
        actual_goal_in_obs = m.group(1).strip()

    match_ok = (
        expected_goal is not None
        and actual_goal_in_obs is not None
        and normalize_goal(actual_goal_in_obs) == normalize_goal(expected_goal)
    )

    if not match_ok:
        print(f"    [start_interaction] *** FAIL-FAST: goal mismatch!")
        print(f"       expected_goal={expected_goal!r}, actual={actual_goal_in_obs!r}")
        raise ValueError(f"Goal mismatch: expected={expected_goal}, actual={actual_goal_in_obs}")

    print(f"    [start_interaction] goal validation passed: {expected_goal!r}")

    # ── Replay prefix actions ────────────────────────────────────────────────
    session = {'env_id': env_id, 'done': False, 'step_count': 0}
    for action in prefix_actions:
        if session['done']:
            print(f"    [replay] SKIP {action!r} (env done)")
            break
        r = requests.post(f"{BASE}/step", json={"id": env_id, "action": action}, timeout=30).json()
        obs = r.get('observation', '')
        done = r.get('done', False)
        session['done'] = done
        session['step_count'] += 1
        print(f"    [replay] {action!r} → obs={obs[:80]!r}, reward={r.get('reward')}, done={done}")

    final_obs = init_obs
    return {
        'env_id': env_id,
        'expected_goal': expected_goal,
        'actual_goal': actual_goal_in_obs,
        'goal_match': match_ok,
        'prefix_actions_replayed': len(prefix_actions),
        'session': session,
    }


async def main():
    print("=" * 70)
    print("LIGHT VALIDATION: TextCraftInteraction full chain with goal fix")
    print("=" * 70)

    # Load parquet
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    print(f"\nLoaded {len(df)} rows from parquet")

    # Find emerald block samples
    emerald_rows = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        prompt = row.get('prompt')
        if prompt is None:
            continue
        prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else (prompt if isinstance(prompt, list) else list(prompt))
        for msg in prompt_list:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                if 'emerald' in content.lower() and 'block' in content.lower():
                    m = re.search(r'Goal:\s*craft\s+(.+?)\.?$', content, re.IGNORECASE)
                    if m:
                        emerald_rows.append((idx, row))
                        break

    print(f"Found {len(emerald_rows)} emerald block samples")
    print()

    results = []
    for row_idx, row in emerald_rows:  # validate all
        print(f"\n{'='*70}")
        print(f"SAMPLE row_idx={row_idx}")
        print(f"{'='*70}")

        # Show row info
        extra = row.get('extra_info', {})
        ik = extra.get('interaction_kwargs', {})
        prompt = row.get('prompt')
        prefix_actions = ik.get('prefix_actions', [])
        if isinstance(prefix_actions, np.ndarray):
            prefix_actions = prefix_actions.tolist()
        goal_from_kwargs = ik.get('goal', 'MISSING')
        goal_from_prompt = extract_goal_from_prompt(prompt)

        print(f"  task_id: {ik.get('task_id')}")
        print(f"  goal (from interaction_kwargs): {goal_from_kwargs!r}")
        print(f"  goal (from prompt): {goal_from_prompt!r}")
        print(f"  prefix_actions: {prefix_actions}")
        print(f"  data_source: {row.get('data_source')}")
        print()

        # Run full interaction simulation
        try:
            sim_result = await simulate_start_interaction(
                instance_id=f"val_{row_idx}",
                interaction_kwargs=ik,
                prompt=prompt,
            )

            env_id = sim_result['env_id']
            session = sim_result['session']

            print(f"\n  [STEP] Model generation — from prompt continuation...")

            # The model would generate: "craft 1 emerald block using 9 emerald"
            # But we simulate by directly calling the correct action
            craft_action = "craft 1 emerald block using 9 emerald"
            r = requests.post(f"{BASE}/step", json={"id": env_id, "action": craft_action}, timeout=30).json()
            print(f"    [rollout] action={craft_action!r}")
            print(f"    [rollout] obs={r.get('observation')!r}")
            print(f"    [rollout] reward={r.get('reward')}, done={r.get('done')}")

            results.append({
                'row_idx': row_idx,
                'goal_match': sim_result['goal_match'],
                'expected_goal': sim_result['expected_goal'],
                'actual_goal': sim_result['actual_goal'],
                'craft_reward': r.get('reward', 0.0),
                'craft_done': r.get('done', False),
            })

            # Close
            requests.post(f"{BASE}/close", json={"id": env_id}, timeout=10)

        except Exception as e:
            print(f"  *** ERROR: {e}")
            results.append({
                'row_idx': row_idx,
                'goal_match': False,
                'expected_goal': goal_from_kwargs,
                'actual_goal': 'N/A',
                'craft_reward': 0.0,
                'craft_done': False,
                'error': str(e),
            })

        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'row_idx':<10} {'goal_match':<12} {'expected_goal':<25} {'actual_goal':<25} {'craft_reward':<15} {'craft_done'}")
    print("-" * 110)
    for r in results:
        print(f"{r['row_idx']:<10} {str(r['goal_match']):<12} {str(r['expected_goal']):<25} {str(r['actual_goal']):<25} {r['craft_reward']:<15} {r['craft_done']}")

    print()
    goal_matches = all(r['goal_match'] for r in results)
    craft_rewards = all(r.get('craft_reward', 0) > 0 for r in results)
    craft_dones = all(r.get('craft_done', False) for r in results)

    if goal_matches and craft_rewards and craft_dones:
        print("✅ ALL CHECKS PASSED:")
        print("   - Goal binding: ✅ All samples matched expected goal")
        print("   - Craft reward: ✅ All samples got reward=1 on craft")
        print("   - Done flag:    ✅ All samples set done=True")
        print("   → The fix is working correctly")
    else:
        print("❌ SOME CHECKS FAILED:")
        print(f"   - Goal matches: {goal_matches}")
        print(f"   - Craft rewards: {[r.get('craft_reward') for r in results]}")
        print(f"   - Done flags: {[r.get('craft_done') for r in results]}")


if __name__ == '__main__':
    asyncio.run(main())
