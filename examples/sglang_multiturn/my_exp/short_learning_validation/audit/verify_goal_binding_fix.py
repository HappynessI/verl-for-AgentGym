#!/usr/bin/env python3
"""
最小验证脚本：测试 goal 绑定修复是否生效
直接调用 TextCraft server 的 /create 接口，传入 goal 参数
"""
import requests, json, time

BASE = "http://127.0.0.1:36001"
TASK_ID = 81  # 这只是 parquet 里的索引，不是 server 的任务 ID
GOAL = "emerald block"  # 从 parquet 的 Goal: 字段解析出来的

def run_episode(episode_num, goal_override=None):
    goal = goal_override or GOAL
    print(f"\n{'='*60}")
    print(f"EPISODE {episode_num} — goal={goal!r}")
    print(f"{'='*60}")

    # ── Create with goal specified ─────────────────────────────────────────
    create_resp = requests.post(
        f"{BASE}/create",
        json={"goal": goal},  # ← 关键：显式传入 goal
        timeout=30
    )
    try:
        create_data = create_resp.json()
    except Exception as e:
        print(f"  [CREATE] FAILED: {e}, raw={create_resp.text[:300]}")
        return None

    if create_resp.status_code != 200:
        print(f"  [CREATE] HTTP {create_resp.status_code}: {create_resp.text[:300]}")
        return None

    env_id = create_data.get("id", "?")
    init_obs = create_data.get("observation", "")
    init_reward = create_data.get("reward", 0.0)
    init_done = create_data.get("done", False)

    # ── Extract actual goal from observation ──────────────────────────────
    import re
    m = re.search(r'Goal:\s*craft\s+(.+?)\.?$', init_obs, re.IGNORECASE | re.MULTILINE)
    actual_goal = m.group(1).strip() if m else "N/A"

    print(f"  [CREATE] env_id={env_id}")
    print(f"  [CREATE] observation (first 400 chars):")
    for line in init_obs[:400].split('\n'):
        print(f"    {line}")
    print(f"  [CREATE] actual_goal_in_env={actual_goal!r}")
    print(f"  [CREATE] goal matches: {goal.lower().replace('_',' ').strip() == actual_goal.lower().replace('_',' ').strip()}")

    if goal.lower().replace('_',' ').strip() != actual_goal.lower().replace('_',' ').strip():
        print(f"  *** MISMATCH: expected={goal!r}, actual={actual_goal!r} ***")

    # ── Step 1: inventory ─────────────────────────────────────────────────
    s1 = requests.post(f"{BASE}/step", json={"id": env_id, "action": "inventory"}, timeout=30).json()
    print(f"\n  [STEP 1] action='inventory'")
    print(f"           reward={s1.get('reward')}, done={s1.get('done')}, obs={s1.get('observation')!r}")

    # ── Step 2: get 9 emerald ──────────────────────────────────────────────
    s2 = requests.post(f"{BASE}/step", json={"id": env_id, "action": "get 9 emerald"}, timeout=30).json()
    print(f"  [STEP 2] action='get 9 emerald'")
    print(f"           reward={s2.get('reward')}, done={s2.get('done')}, obs={s2.get('observation')!r}")

    # ── Step 3: craft ─────────────────────────────────────────────────────
    craft_action = "craft 1 emerald block using 9 emerald"
    s3 = requests.post(f"{BASE}/step", json={"id": env_id, "action": craft_action}, timeout=30).json()
    print(f"  [STEP 3] action={craft_action!r}")
    print(f"           reward={s3.get('reward')}, done={s3.get('done')}, obs={s3.get('observation')!r}")

    # ── Summary ─────────────────────────────────────────────────────────────
    r3 = s3.get("reward", 0.0)
    d3 = s3.get("done", False)
    print(f"\n  [RESULT] Episode {episode_num}: craft_reward={r3}, craft_done={d3}")
    print(f"           actual_goal={actual_goal!r}, goal_matches={goal.lower().replace('_',' ').strip() == actual_goal.lower().replace('_',' ').strip()}")

    # Close
    try:
        requests.post(f"{BASE}/close", json={"id": env_id}, timeout=10)
    except:
        pass

    return {"episode": episode_num, "goal": goal, "actual_goal": actual_goal,
            "craft_reward": r3, "craft_done": d3}


print(f"TextCraft goal-binding fix verification")
print(f"Server: {BASE}")
print(f"Test goal: {GOAL!r}")
print()

results = []
for ep in [1, 2, 3]:
    r = run_episode(ep, goal_override=GOAL)
    if r:
        results.append(r)
    time.sleep(0.3)

# ── Comparison ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("COMPARISON: 3 episodes with goal-binding fix")
print("=" * 70)
print(f"{'Ep':<5} {'expected_goal':<25} {'actual_goal_in_env':<25} {'craft_reward':<15} {'craft_done'}")
print("-" * 90)
for r in results:
    match = r["goal"].lower().replace("_"," ").strip() == r["actual_goal"].lower().replace("_"," ").strip()
    print(f"{r['episode']:<5} {r['goal']:<25} {r['actual_goal']:<25} {r['craft_reward']:<15} {r['craft_done']}")

print()
craft_rewards = [r["craft_reward"] for r in results]
craft_dones = [r["craft_done"] for r in results]

if all(r > 0 for r in craft_rewards):
    print("✅ PASS: All 3 episodes got reward=1, done=True on craft action")
    print("   → Goal binding fix is WORKING correctly")
elif all(r == 0 for r in craft_rewards):
    print("❌ FAIL: All 3 episodes still get reward=0, done=False")
    print("   → Either server is still running old code, or env.step() has a bug")
else:
    print(f"⚠ MIXED: rewards={craft_rewards}, dones={craft_dones}")
