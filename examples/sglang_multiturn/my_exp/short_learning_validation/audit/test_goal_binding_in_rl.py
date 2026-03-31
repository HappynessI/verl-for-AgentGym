#!/usr/bin/env python3
"""
Direct test of TextCraftInteraction.start_interaction with goal fix.
This simulates exactly what tool_agent_loop.py does:
  interaction.start_interaction(request_id, **interaction_kwargs_with_prompt)
"""
import asyncio, json, sys, re
import numpy as np

sys.path.insert(0, '/Data/wyh/verl')
from verl.interactions.textcraft_interaction import TextCraftInteraction

PARQUET = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/cleaned_v2/textcraft_validated_cleaned_v2_20260326_000658.parquet"


def extract_goal_from_prompt(prompt_arr):
    if hasattr(prompt_arr, 'tolist'):
        prompt_arr = prompt_arr.tolist()
    if not isinstance(prompt_arr, list):
        prompt_arr = list(prompt_arr)
    for msg in prompt_arr:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '')
            m = re.search(r'Goal:\s*craft\s+(.+?)\.?$', content, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
    return None


async def main():
    print("=" * 70)
    print("GOAL-BINDING FIX: TextCraftInteraction integration test")
    print("=" * 70)

    # Load parquet
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    print(f"\nLoaded {len(df)} rows")

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

    print(f"Emerald block samples: {[r[0] for r in emerald_rows]}")
    print()

    # Instantiate TextCraftInteraction
    interaction = TextCraftInteraction({
        'env_server_base': 'http://127.0.0.1:36001',
        'timeout': 600,
        'max_retries': 3,
    })

    results = []
    for row_idx, row in emerald_rows:
        print(f"\n{'='*70}")
        print(f"TEST row_idx={row_idx}")
        print(f"{'='*70}")

        extra = row.get('extra_info', {})
        ik = extra.get('interaction_kwargs', {})
        prompt = row.get('prompt')
        prefix_actions = list(ik.get('prefix_actions', []))
        goal_from_kwargs = ik.get('goal', 'MISSING')
        goal_from_prompt = extract_goal_from_prompt(prompt)

        print(f"  interaction_kwargs.goal: {goal_from_kwargs!r}")
        print(f"  prompt-extracted goal:   {goal_from_prompt!r}")
        print(f"  prefix_actions: {prefix_actions}")

        instance_id = f"rl_test_{row_idx}"

        # Simulate tool_agent_loop.py: merge interaction_kwargs with raw_prompt
        interaction_kwargs_with_prompt = {
            **ik,
            'prompt': prompt,  # ← THIS is the critical injection from tool_agent_loop patch
        }

        try:
            # This is exactly what tool_agent_loop.run() does after our fix:
            # interaction_kwargs_with_prompt = {**interaction_kwargs, "prompt": kwargs.get("raw_prompt")}
            # await interaction.start_interaction(request_id, **interaction_kwargs_with_prompt)
            await interaction.start_interaction(instance_id, **interaction_kwargs_with_prompt)
            print(f"  start_interaction: OK")

            session = interaction.instance_sessions.get(instance_id)
            if session:
                env_id = session['env_id']
                import requests
                base = 'http://127.0.0.1:36001'

                # Replay prefix actions
                for action in prefix_actions:
                    r = requests.post(f"{base}/step", json={"id": env_id, "action": action}, timeout=30).json()
                    print(f"    [replay] {action!r} → reward={r.get('reward')}, done={r.get('done')}")

                # Rollout: craft action
                craft_action = "craft 1 emerald block using 9 emerald"
                r = requests.post(f"{base}/step", json={"id": env_id, "action": craft_action}, timeout=30).json()
                print(f"\n  [rollout] action={craft_action!r}")
                print(f"            reward={r.get('reward')}, done={r.get('done')}")
                print(f"            obs={r.get('observation')!r}")

                craft_reward = r.get('reward', 0.0)
                craft_done = r.get('done', False)

                results.append({
                    'row_idx': row_idx,
                    'goal_from_kwargs': goal_from_kwargs,
                    'goal_from_prompt': goal_from_prompt,
                    'craft_reward': craft_reward,
                    'craft_done': craft_done,
                })

                requests.post(f"{base}/close", json={"id": env_id}, timeout=10)
            else:
                print(f"  session not found!")
                results.append({'row_idx': row_idx, 'error': 'no session', 'craft_reward': 0.0, 'craft_done': False})

        except Exception as e:
            print(f"  *** ERROR: {e}")
            results.append({'row_idx': row_idx, 'error': str(e), 'craft_reward': 0.0, 'craft_done': False})

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  row {r['row_idx']}: kwargs.goal={r.get('goal_from_kwargs')!r}, "
              f"prompt.goal={r.get('goal_from_prompt')!r}, "
              f"reward={r.get('craft_reward')}, done={r.get('craft_done')}")

    print()
    all_pass = all(r.get('craft_reward', 0) > 0 and r.get('craft_done', False) for r in results)
    all_fail = all(r.get('craft_reward', 0) == 0 and not r.get('craft_done', True) for r in results)
    if all_pass:
        print("✅ ALL PASS: goal binding fix works in verl RL pipeline")
    elif all_fail:
        print("❌ ALL FAIL: rewards still 0")
    else:
        print(f"⚠ MIXED: {[r.get('craft_reward') for r in results]}")


if __name__ == '__main__':
    asyncio.run(main())
