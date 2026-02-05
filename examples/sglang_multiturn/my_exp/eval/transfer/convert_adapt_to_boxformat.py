#!/usr/bin/env python3
"""
将adapt格式的textcraft轨迹转换为box format格式。

旧格式（adapt）：
    > think: XXX
    OK.
    > inventory
    > get XXX
    > craft XXX

新格式（box format）：
    Think: XXX
    Action: [[ inventory ]]
    Action: [[ get XXX ]]
    Action: [[ craft XXX ]]
    Action: [[ Task Completed! ]]
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any


# 新的系统提示词
NEW_SYSTEM_PROMPT = '''You are a Minecraft Assistant. Your goal is to craft items by managing resources and recipes.

**CORE PROTOCOL (Strictly Follow):**
1. **THINK FIRST**: Before any action, analyze the current state using the "Reasoning Logic" below.
2. **ONE ACTION**: Output exactly ONE action per turn.
3. **BOX FORMAT**: Wrap your command in `[[ ]]`. Example: `Action: [[ inventory ]]`
4. **NO HALLUCINATION**: Do NOT simulate the Environment's response. Stop immediately after outputting the Action.

**REASONING LOGIC (The Algorithm):**
When trying to acquire an item [Target]:
1. **Check Inventory**: Do you already have [Target]?
   - If YES -> Task Complete / Proceed to next step.
   - If NO -> Go to step 2.
2. **Check Recipe**: Is there a crafting recipe for [Target]?
   - If YES -> Check if you have the [Ingredients].
     - If you have Ingredients -> `craft [Target] ...`
     - If you miss Ingredients -> **NEW SUB-GOAL**: Get the missing [Ingredient] (Repeat Logic from Step 1).
   - If NO (Base Material) -> `get [Target]` directly from environment.
3. **Handle Errors**:
   - If `get` fails -> The item might require crafting from a sub-ingredient (e.g., gold ingot needs gold nuggets). Check recipes again.
   - If `craft` fails -> Check if you have the exact items.

**CORE COMMAND SET (API):**
* `craft [target] using [ingredients]` (e.g., "craft 1 stick using 2 bamboo")
* `get [item]` (e.g., "get 3 log")
* `inventory` (Check what you have)

**INTERACTION EXAMPLES:**

[Environment]
Goal: craft powered rail
Recipes:
- craft 6 powered rail using 1 redstone, 1 stick, 6 gold ingot
- craft 1 gold ingot using 9 gold nugget
- craft 1 stick using 2 bamboo

[You]
Think: Recipe for powered rail needs redstone, stick, gold ingot. I need to check what I have.
Action: [[ inventory ]]

[Environment]
Inventory: [redstone] (1)

[You]
Think: I have redstone. I miss stick and gold ingot. Gold ingot has a recipe (needs gold nugget). I should get gold nugget first.
Action: [[ get 54 gold nugget ]]

[Environment]
Got 54 gold nugget.

[You]
Think: Now I have nuggets. I can craft gold ingots.
Action: [[ craft 6 gold ingot using 54 gold nugget ]]

[Environment]
Crafted 6 gold ingot.

[You]
Think: Now I need a stick. Recipe says stick needs bamboo. I need to get bamboo.
Action: [[ get 2 bamboo ]]

[Environment]
Got 2 bamboo.

[You]
Think: I have bamboo. I can craft the stick.
Action: [[ craft 1 stick using 2 bamboo ]]

[Environment]
Crafted 1 stick.

[You]
Think: I have redstone, gold ingots, and stick. I can finally craft the powered rail.
Action: [[ craft 6 powered rail using 1 redstone, 1 stick, 6 gold ingot ]]

[Environment]
Crafted 6 powered rail.

[You]
Think: Goal achieved.
Action: [[ Task Completed! ]]
'''


def convert_assistant_content(content: str) -> str:
    """
    转换assistant消息中的动作格式。
    
    处理规则：
    1. `> think: XXX` 后面跟 `OK.` -> `Think: XXX`
    2. `> inventory` -> `Action: [[ inventory ]]`
    3. `> get XXX` -> `Action: [[ get XXX ]]`
    4. `> craft XXX` -> `Action: [[ craft XXX ]]`
    5. 如果think包含 `Task Completed!`，添加 `Action: [[ Task Completed! ]]`
    """
    if not content or content.strip() == "OK.":
        return content
    
    lines = content.split('\n')
    result_lines = []
    i = 0
    has_task_completed = False
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 跳过空行
        if not stripped:
            result_lines.append(line)
            i += 1
            continue
        
        # 跳过 OK. 行（这是环境对think的响应）
        if stripped == "OK.":
            i += 1
            continue
        
        # 处理 > think: XXX
        if stripped.startswith('> think:'):
            think_content = stripped[8:].strip()  # 去掉 "> think:"
            result_lines.append(f"Think: {think_content}")
            
            # 检查是否包含 Task Completed
            if 'Task Completed' in think_content or 'task completed' in think_content.lower():
                has_task_completed = True
            
            i += 1
            continue
        
        # 处理 > inventory
        if stripped == '> inventory' or stripped == '> inventory:':
            result_lines.append("Action: [[ inventory ]]")
            i += 1
            continue
        
        # 处理 > get XXX
        if stripped.startswith('> get '):
            item = stripped[6:].strip()  # 去掉 "> get "
            # 移除末尾的句号
            if item.endswith('.'):
                item = item[:-1]
            result_lines.append(f"Action: [[ get {item} ]]")
            i += 1
            continue
        
        # 处理 > craft XXX
        if stripped.startswith('> craft '):
            craft_content = stripped[8:].strip()  # 去掉 "> craft "
            result_lines.append(f"Action: [[ craft {craft_content} ]]")
            i += 1
            continue
        
        # 其他行保持不变（环境响应等）
        result_lines.append(line)
        i += 1
    
    # 如果有 Task Completed 但没有对应的 Action，添加一个
    result_text = '\n'.join(result_lines)
    if has_task_completed and 'Action: [[ Task Completed!' not in result_text:
        result_lines.append("Action: [[ Task Completed! ]]")
    
    return '\n'.join(result_lines)


def is_system_prompt(content: str) -> bool:
    """检查是否为系统提示词（包含示例的长文本）"""
    indicators = [
        "You are given few useful crafting recipes",
        "Here is a demo of",
        "For any other natural language or thoughts",
        "> think: I should check if I can fetch"
    ]
    return any(ind in content for ind in indicators)


def convert_trajectory(trajectory: Dict[str, Any], replace_system_prompt: bool = True) -> Dict[str, Any]:
    """
    转换单条轨迹。
    
    Args:
        trajectory: 原始轨迹数据
        replace_system_prompt: 是否替换系统提示词
    
    Returns:
        转换后的轨迹
    """
    converted = trajectory.copy()
    conversations = []
    
    for turn in trajectory.get('conversations', []):
        new_turn = turn.copy()
        role = turn.get('role', '')
        content = turn.get('content', '')
        
        if role == 'user':
            # 检查是否是系统提示词
            if replace_system_prompt and is_system_prompt(content):
                new_turn['content'] = NEW_SYSTEM_PROMPT
            # 其他user消息保持不变（任务指令、环境响应）
        
        elif role == 'assistant':
            # 转换assistant的动作格式
            new_turn['content'] = convert_assistant_content(content)
        
        conversations.append(new_turn)
    
    converted['conversations'] = conversations
    return converted


def filter_successful_trajectories(trajectories: List[Dict], keep_failed: bool = False) -> List[Dict]:
    """
    过滤轨迹，可选只保留成功的轨迹。
    
    使用数据中的 success 字段判断（1=成功，0=失败）
    """
    if keep_failed:
        return trajectories
    
    filtered = []
    for traj in trajectories:
        # 使用数据中的 success 字段
        if traj.get('success', 0) == 1:
            filtered.append(traj)
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Convert adapt format trajectories to box format')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input jsonl file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output jsonl file path')
    parser.add_argument('--no-replace-prompt', action='store_true',
                        help='Do not replace system prompt with new format')
    parser.add_argument('--keep-failed', action='store_true',
                        help='Keep failed trajectories (default: only keep successful)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # 读取轨迹
    print(f"Reading trajectories from: {input_path}")
    trajectories = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # 过滤轨迹
    if not args.keep_failed:
        trajectories = filter_successful_trajectories(trajectories, keep_failed=False)
        print(f"After filtering successful: {len(trajectories)} trajectories")
    
    # 转换轨迹
    converted = []
    for i, traj in enumerate(trajectories):
        try:
            conv_traj = convert_trajectory(traj, replace_system_prompt=not args.no_replace_prompt)
            converted.append(conv_traj)
            
            if args.verbose and i < 3:
                print(f"\n=== Sample {i} ===")
                for turn in conv_traj['conversations'][:4]:
                    print(f"[{turn['role']}]: {turn['content'][:200]}...")
                    
        except Exception as e:
            print(f"Error converting trajectory {i}: {e}")
            continue
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for traj in converted:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')
    
    print(f"\nConverted {len(converted)} trajectories")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()

