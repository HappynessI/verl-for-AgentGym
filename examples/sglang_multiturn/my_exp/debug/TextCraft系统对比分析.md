# TextCraft评估系统对比分析

## 测试结果对比

| 系统 | 模型 | 成功率 | 样本数 | Prompt类型 | Action格式 |
|------|------|--------|---------|-----------|-----------|
| **AgentGym** | Qwen2.5-7B | **41%** | 100 | ReAct (Thought+Action) | react |
| **AgentGym** | Qwen2.5-3B | 14% | 100 | ReAct (Thought+Action) | react |
| **AgentGym** | Qwen2.5-1.5B | 10% | 100 | ReAct (Thought+Action) | react |
| **Verl (ADaPT)** | Qwen3-1.7B | **24.2%** | 99 | ADaPT few-shot | think: + action |
| **Verl (原始)** | Qwen3-1.7B | 54.5% | - | 简单instruct | Action: + action |

## 1. 核心区别

### 1.1 Prompt格式

#### **AgentGym - ReAct格式**
```
You are given few useful crafting recipes to craft items in Minecraft. 
Crafting commands are of the format "craft [target object] using [input ingredients]".
Every round I will give you an observation, you have to respond an action based on the state and instruction. 
You can "get" an object (ingredients) from the inventory or the environment, 
look-up the game inventory by "inventory", 
or "craft" (target) using any of the crafting commands.

Your output must strictly follow this format:
"Thought:
your thoughts.

Action:
your next action"

Reminder: 
1. Always specify the quantity when using "get" and "craft" commands.
   - Example of get: get 1 lapis lazuli
   - Example1 of craft: craft 1 blue dye using 1 lapis lazuli
   - Example2 of craft: craft 1 golden carrot using 8 gold nugget, 1 carrot
2. When using "get" command, do not specify whether the item comes from the inventory or the environment.
3. You can use ONLY crafting commands provided, do not use your own crafting commands. 
   However, if the crafting command uses a generic ingredient like "planks", 
   you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.
```

**特点：**
- ✅ 明确的 `Thought: ... Action: ...` 格式
- ✅ 详细的规则说明和示例
- ✅ 强制要求结构化输出
- ✅ Chat template封装 (chatml格式)

#### **Verl - ADaPT few-shot格式**
```
You are given few useful crafting recipes to craft items in Minecraft. 
Crafting commands are of the format "craft [target object] using [input ingredients]". 
You can either "fetch" an object (ingredients) from the inventory or the environment 
or "craft" (target) using any of the crafting commands.

Your output must strictly follow this format:"Thought:
your thoughts.

Action:
your next action"

For any other natural language or thoughts, use prefix 'think: '.

Here is a demo of how to fetch and craft objects.

Goal: craft dark oak sign

> think: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.

> inventory: 
Inventory: [stick] (1) [dark oak planks] (8)

> get dark oak sign
Could not find dark oak sign

> think: I cannot get dark oak sign directly, I need to craft it...
... (完整few-shot示例)
```

**特点：**
- ❌ Few-shot示例但没有 `Thought:` + `Action:` 结构
- ❌ 使用 `think:` 前缀而非结构化格式
- ❌ 没有chat template，直接字符串拼接
- ❌ 没有"OK."的标记

### 1.2 Action提取逻辑

#### **AgentGym - 严格的ReAct解析**
```python
# 从 agentenv/controller 代码推测
# action_format="react" 时提取逻辑：
def extract_action(text):
    # 查找 "Action:" 后面的内容
    action_match = re.search(r'Action:\s*\n(.+)', text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        # 只取第一行
        action = action.split('\n')[0]
        return action
    return None
```

**特点：**
- ✅ 明确查找 `Action:` 标记
- ✅ 只提取action部分，过滤thought
- ✅ 处理多行输出，取第一行

#### **Verl - 简单正则匹配**
```python
# verl/interactions/textcraft_interaction.py
def extract_action(self, text: str) -> Optional[str]:
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    
    # 尝试多种模式
    patterns = [
        r'Action:\s*(.+)',           # Action: ...
        r'craft\s+[\d\w\s,]+',       # craft ...
        r'get\s+[\d\w\s]+',          # get ...
        r'inventory',                 # inventory
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0) if 'Action:' not in pattern else match.group(1)
    return None
```

**特点：**
- ⚠️ 多种fallback模式
- ⚠️ 依赖模型自发输出"Action:"
- ❌ ADaPT格式下无法正确提取（只输出think:）

## 2. 参数设置对比

| 参数 | AgentGym | Verl (ADaPT尝试) | 影响 |
|------|----------|------------------|------|
| **temperature** | 未明确 (推测0.7-1.0) | 0.0 (greedy) | AgentGym更多样性 |
| **max_new_tokens** | 8192 (max_length) | 150 | AgentGym允许完整thought |
| **top_p** | 未明确 | 1.0 | - |
| **do_sample** | 推测True | False | AgentGym有随机性 |
| **stop_tokens** | 无 | ['\n'] | Verl强制单行导致输出截断 |
| **chat_template** | chatml | 无 (直接拼接) | **关键区别** |

## 3. Chat Template的影响

### AgentGym使用的chatml格式:
```
<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
```

**优点：**
- 模型训练时见过这种格式
- 自然地产生结构化输出
- 支持多轮对话状态管理

### Verl的直接拼接:
```python
prompt = ADAPT_FEW_SHOT_PROMPT
for msg in messages:
    if msg["role"] == "user":
        prompt += f"\n{msg['content']}\n>"
    elif msg["role"] == "assistant":
        prompt += f" {msg['content']}\n"
```

**问题：**
- ❌ 模型不熟悉这种格式
- ❌ `>` 前缀容易让模型混乱
- ❌ 没有明确的角色标记

## 4. 失败原因分析

### Verl ADaPT格式失败案例 (24.2%成功率)

**案例1 - Session 0 (observer)**
```
问题：模型有所有材料但craft命令总是失败
用户：Could not find enough items to craft minecraft:observer
助手：think: I have all the required ingredients. Why is the crafting command not working?

循环40轮，一直重复相同的思考，没有成功craft
```

**根本原因：**
1. **Action提取失败**：模型输出的是 `think: ...` 格式，但action extractor找不到可执行的action
2. **环境反馈混乱**：由于没有提取到action，环境返回了错误信息，导致模型进入死循环
3. **Prompt格式不匹配**：ADaPT的few-shot示例用 `think:` 和直接action，但verl的extractor期待 `Action:` 标记

**案例2 - Session 1 (crimson planks)**
```
助手：think: I need to fetch crimson planks. I can't fetch them directly...
用户：Could not find enough items to craft minecraft:crimson_planks

模型一直输出think，没有实际执行craft命令
```

**根本原因：**
- 同上，action extraction完全失败
- 模型进入"思考"模式，但没有输出可执行action

### AgentGym成功案例 (41%成功率)

```
用户：Crafting commands:...
Goal: craft slime block.

助手：Thought: To craft a slime block, I need to collect 9 slime balls first.

Action:
get 9 slime ball

用户：Got 9 slime ball

助手：Thought: Now that I have 9 slime balls, I can craft the slime block...

Action:
craft 1 slime block using 9 slime ball

用户：Crafted 1 minecraft:slime_block
```

**成功关键：**
1. ✅ 明确的 `Thought:` / `Action:` 分隔
2. ✅ Action extractor能正确提取
3. ✅ Chat template保证格式一致性
4. ✅ 环境正确执行并反馈

## 5. 核心问题总结

### Verl当前问题：

1. **Prompt-Extractor不匹配**
   - ADaPT prompt教模型用 `think:` 前缀
   - 但verl extractor期待 `Action:` 标记
   - 导致95%的输出无法提取action

2. **Chat Template缺失**
   - 直接字符串拼接不符合模型预训练格式
   - 模型难以理解对话结构
   - `>` 前缀造成混乱

3. **Stop Token过于激进**
   - `stop=['\n']` 导致输出只有一行
   - 无法完成 `Thought: ...\n\nAction: ...` 的结构
   - 只输出 `think: ...` 就停止了

4. **温度和采样设置**
   - `temperature=0.0, do_sample=False` 导致输出过于确定性
   - 遇到困境时无法探索其他策略
   - AgentGym的适度随机性帮助跳出死循环

## 6. 修复建议

### 方案A：完全采用AgentGym格式（推荐）

```python
# 1. 使用ReAct风格的system prompt
system_prompt = """You are given few useful crafting recipes to craft items in Minecraft. 
Crafting commands are of the format "craft [target object] using [input ingredients]".
Every round I will give you an observation, you have to respond an action based on the state and instruction. 
You can "get" an object (ingredients) from the inventory or the environment, 
look-up the game inventory by "inventory", 
or "craft" (target) using any of the crafting commands.

Your output must strictly follow this format:
"Thought:
your thoughts.

Action:
your next action"

Reminder: 
1. Always specify the quantity when using "get" and "craft" commands.
2. When using "get" command, do not specify whether the item comes from the inventory or the environment.
3. You can use ONLY crafting commands provided."""

# 2. 使用chat template
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": system_prompt},
        *messages
    ],
    tokenize=False,
    add_generation_prompt=True
)

# 3. 调整生成参数
outputs = model.generate(
    **inputs,
    max_new_tokens=512,      # 允许完整thought+action
    temperature=0.7,          # 适度随机性
    top_p=0.9,
    do_sample=True,
    # 不使用stop_tokens
)

# 4. 修改action extractor
def extract_action(self, text: str) -> Optional[str]:
    # 查找Action:标记
    action_match = re.search(r'Action:\s*\n?(.+?)(?:\n|$)', text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        # 只取第一行作为action
        action = action.split('\n')[0].strip()
        return action
    return None
```

### 方案B：修复ADaPT格式（不推荐）

如果坚持使用ADaPT格式，需要：

```python
# 1. 修改action extractor以处理think:格式
def extract_action(self, text: str) -> Optional[str]:
    # 移除think:前缀的行
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('think:') or line.startswith('>'):
            continue
        # 检查是否是有效action
        if any(keyword in line.lower() for keyword in ['craft', 'get', 'inventory']):
            return line
    return None

# 2. 移除stop=['\n']限制
outputs = model.generate(
    **inputs,
    max_new_tokens=300,  # 允许多行
    temperature=0.3,     # 适度随机
    do_sample=True,
    # 不使用stop_tokens
)

# 3. 添加chat template包装
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": ADAPT_FEW_SHOT_PROMPT}, *messages],
    tokenize=False
)
```

## 7. 推荐方案

**强烈推荐方案A：采用AgentGym的ReAct格式**

理由：
1. ✅ 已验证有效 (41% vs 24.2%)
2. ✅ 格式清晰，易于解析
3. ✅ 符合模型训练惯例
4. ✅ 工程实现简单
5. ✅ 可扩展性强

ADaPT格式的问题：
- ❌ 设计用于单轮生成，不适合多轮交互
- ❌ `think:` 前缀不是标准格式
- ❌ Few-shot示例冗长，占用token
- ❌ 需要额外的解析逻辑

## 8. 实验验证建议

1. 先用AgentGym格式测试Qwen3-1.7B (预期30-40%)
2. 对比温度参数影响 (0.0 vs 0.3 vs 0.7)
3. 对比max_new_tokens (150 vs 300 vs 512)
4. 确认chat_template的必要性
5. 如果成功，再迁移到GRPO训练

---

**结论：** Verl的ADaPT实现有三个关键缺陷：
1. Prompt格式与action extractor不匹配
2. 缺少chat template导致模型理解困难  
3. Stop token设置过于激进

采用AgentGym的ReAct格式可以解决所有这些问题。

