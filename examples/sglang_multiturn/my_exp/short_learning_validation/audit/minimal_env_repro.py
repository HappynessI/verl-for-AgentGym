#!/usr/bin/env python3
"""
Minimal environment reproduction for task_id=81 (craft emerald block).
Runs 3 independent episodes, prints every step's response fields.
"""
import requests, json, time

BASE = "http://127.0.0.1:36001"
TASK_ID = 81
GOAL_KEYWORD = "emerald block"

def run_episode(episode_num):
    print(f"\n{'='*60}")
    print(f"EPISODE {episode_num}")
    print(f"{'='*60}")

    # ── Step 0: Create environment ─────────────────────────────────────────
    create_resp = requests.post(f"{BASE}/create", json={"task_id": TASK_ID}, timeout=30)
    try:
        create_data = create_resp.json()
    except Exception as e:
        print(f"  [CREATE] FAILED to parse JSON: {e}")
        print(f"  Raw: {create_resp.text[:500]}")
        return None

    if create_resp.status_code != 200:
        print(f"  [CREATE] HTTP {create_resp.status_code}")
        print(f"  Raw: {create_resp.text[:500]}")
        return None

    env_id = create_data.get("id", "unknown")
    init_obs = create_data.get("observation", "")
    init_reward = create_data.get("reward", 0.0)
    init_done = create_data.get("done", False)

    print(f"  [CREATE] env_id={env_id}, reward={init_reward}, done={init_done}")
    print(f"  [CREATE] observation (first 600 chars):")
    for line in init_obs[:600].split('\n'):
        print(f"    {line}")
    if len(init_obs) > 600:
        print(f"    ... ({len(init_obs)-600} more chars)")

    # Extract goal text
    goal_match = init_obs.split("Goal:")[-1].split("\n")[0].strip() if "Goal:" in init_obs else "N/A"
    craft_match = init_obs.split("Crafting commands:")[-1].split("\n")[0].strip() if "Crafting commands:" in init_obs else "N/A"
    print(f"  [CREATE] crafting_cmd: {craft_match}")
    print(f"  [CREATE] goal: {goal_match}")

    # Also check the full observation for any task spec
    print(f"\n  [CREATE] full observation ({len(init_obs)} chars):")
    print(f"    {init_obs[:300]!r}")
    print()

    # ── Step 1: inventory ─────────────────────────────────────────────────
    step1_resp = requests.post(f"{BASE}/step", json={"id": env_id, "action": "inventory"}, timeout=30)
    step1_data = step1_resp.json()
    obs1 = step1_data.get("observation", "")
    r1 = step1_data.get("reward", 0.0)
    d1 = step1_data.get("done", False)
    print(f"  [STEP 1] action='inventory'")
    print(f"           reward={r1}, done={d1}")
    print(f"           obs: {obs1!r}")

    # ── Step 2: get 9 emerald ───────────────────────────────────────────────
    step2_resp = requests.post(f"{BASE}/step", json={"id": env_id, "action": "get 9 emerald"}, timeout=30)
    step2_data = step2_resp.json()
    obs2 = step2_data.get("observation", "")
    r2 = step2_data.get("reward", 0.0)
    d2 = step2_data.get("done", False)
    print(f"  [STEP 2] action='get 9 emerald'")
    print(f"           reward={r2}, done={d2}")
    print(f"           obs: {obs2!r}")

    # ── Step 3: craft 1 emerald block using 9 emerald ────────────────────────
    craft_action = "craft 1 emerald block using 9 emerald"
    step3_resp = requests.post(f"{BASE}/step", json={"id": env_id, "action": craft_action}, timeout=30)
    step3_data = step3_resp.json()
    obs3 = step3_data.get("observation", "")
    r3 = step3_data.get("reward", 0.0)
    d3 = step3_data.get("done", False)
    print(f"  [STEP 3] action={craft_action!r}")
    print(f"           reward={r3}, done={d3}")
    print(f"           obs: {obs3!r}")

    # ── Step 4: Try Task Completed! ────────────────────────────────────────
    step4_resp = requests.post(f"{BASE}/step", json={"id": env_id, "action": "Task Completed!"}, timeout=30)
    step4_data = step4_resp.json()
    obs4 = step4_data.get("observation", "")
    r4 = step4_data.get("reward", 0.0)
    d4 = step4_data.get("done", False)
    print(f"  [STEP 4] action='Task Completed!'")
    print(f"           reward={r4}, done={d4}")
    print(f"           obs: {obs4!r}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  [SUMMARY] Episode {episode_num}:")
    print(f"    task_id={TASK_ID}, goal={goal_match}")
    print(f"    craft action: reward={r3}, done={d3}")
    print(f"    craft obs: {obs3!r}")
    print(f"    final: reward={r4}, done={d4}")

    # Close
    try:
        requests.post(f"{BASE}/close", json={"id": env_id}, timeout=10)
    except Exception as e:
        print(f"  [CLOSE] failed: {e}")

    return {
        "episode": episode_num,
        "task_id": TASK_ID,
        "goal": goal_match,
        "craft_cmd": craft_match,
        "craft_reward": r3,
        "craft_done": d3,
        "craft_obs": obs3,
        "final_reward": r4,
        "final_done": d4,
    }


# ─── Run 3 independent episodes ───────────────────────────────────────────────
print(f"TextCraft minimal reproduction for task_id={TASK_ID}")
print(f"Base URL: {BASE}")

results = []
for ep in [1, 2, 3]:
    result = run_episode(ep)
    if result:
        results.append(result)
    time.sleep(0.5)

# ─── Comparison table ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("COMPARISON TABLE: 3 independent episodes")
print("=" * 60)
print(f"{'Episode':<10} {'task_id':<10} {'Goal':<40} {'craft_reward':<15} {'craft_done':<12} {'final_reward':<15} {'final_done'}")
print("-" * 120)
for r in results:
    print(f"{r['episode']:<10} {r['task_id']:<10} {str(r['goal'])[:40]:<40} {r['craft_reward']:<15} {r['craft_done']:<12} {r['final_reward']:<15} {r['final_done']}")

# ─── Historical MiniMax sample goal ──────────────────────────────────────────
print()
print("=" * 60)
print("HISTORICAL MINIMAX SAMPLES (from trajectories.jsonl)")
print("=" * 60)
TRAJ = "/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl"
with open(TRAJ) as f:
    for i, line in enumerate(f):
        if i < 200:
            continue
        if i > 203:
            break
        obj = json.loads(line)
        conversations = obj.get("conversations", [])
        # First user message has the task
        for msg in conversations:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "craft" in content.lower() and "emerald" in content.lower():
                    print(f"  line {i}:")
                    print(f"    item_id: {obj.get('item_id','?')}")
                    print(f"    success: {obj.get('success','?')}")
                    print(f"    reward: {obj.get('reward','?')}")
                    print(f"    goal content: {content!r}")
                break

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
craft_rewards = [r["craft_reward"] for r in results]
craft_dones   = [r["craft_done"]   for r in results]
all_zero_none = all(r == 0.0 and not d for r, d in zip(craft_rewards, craft_dones))
any_positive  = any(r > 0 for r in craft_rewards)
all_positive  = all(r > 0 for r in craft_rewards)

if all_zero_none:
    print("RESULT: All 3 episodes → reward=0, done=False for the craft action")
    print("→ SITUATION A: Environment's completion checker is broken for task_id=81")
elif all_positive:
    print("RESULT: All 3 episodes → reward>0 for the craft action")
    print("→ SITUATION C: Environment works correctly, problem is in RL run task binding")
else:
    print("RESULT: Mixed results across episodes")
    print("→ SITUATION B: Environment is unstable/state-dependent")

print()
print(f"Craft rewards: {craft_rewards}")
print(f"Craft dones:   {craft_dones}")
