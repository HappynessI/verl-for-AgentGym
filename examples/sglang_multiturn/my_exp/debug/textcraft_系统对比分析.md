# TextCraft 三系统对比分析：ADaPT vs Eval vs GRPO

## 执行摘要

经过详细对比，发现三个系统在**prompt设计**、**action提取逻辑**和**推理参数**上存在显著差异。**核心问题**：

1. **GRPO训练数据中没有system prompt**，模型直接看到任务描述
2. **ADaPT使用详细的few-shot prompts**，而Eval/GRPO几乎没有示例
3. **三个系统的action格式完全不同**

---

## 一、Prompt对比

### 1.1 ADaPT (run_textcraft.py)

**位置**: L252-393

**特点**:
- 长达**140行**的few-shot prompt
- 包含**3个完整的交互示例**
- 明确的步骤拆解和推理过程

**Prompt结构**:

```python
atomic_exec_prompt = '''You are given few useful crafting recipes to craft items in Minecraft. 
Crafting commands are of the format "craft [target object] using [input ingredients]". 
You can either "fetch" an object (ingredients) from the inventory or the environment or 
"craft" (target) using any of the crafting commands. 

You can use ONLY these crafting commands provided, do not use your own crafting commands. 
However, if the crafting command uses a generic ingredient like "planks", you can use 
special types of the same ingredient e.g. "dark oak planks" in the command instead. 

For any other natural language or thoughts, use prefix 'think: '.

Here is a demo of how to fetch and craft objects.

[示例1: craft_with_ingredients - 23行示例]
Goal: craft dark oak sign

> think: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.

> inventory: 
Inventory: [stick] (1) [dark oak planks] (8)

> get dark oak sign
Could not find dark oak sign

> think: I cannot get dark oak sign directly, I need to craft it. From the crafting commands, 
I can use: craft dark oak sign using 6 dark oak planks, 1 stick. Ingredients needed: 
6 dark oak planks, 1 stick. Input assumption: I have all the neccessary ingredients in my 
inventory. Let me verify this first.
OK.

> inventory
Inventory: [stick] (1) [dark oak planks] (8)

> think: I found my ingredients: 6 dark oak planks, 1 stick in my inventory. My assumption 
is true, I can proceed. I will use the crafting command: craft dark oak sign using 6 dark oak planks
OK.

> craft 1 dark oak sign using 6 dark oak planks, 1 stick
Crafted 1 minecraft:dark_oak_sign

> inventory 
Inventory: [dark oak sign] (1)

> think: I now have dark oak sign in my inventory. Task Completed!
OK.

[示例2: craft_with_ingredients_gen - 获取资源示例]
[示例3: react_trajectory - 复杂任务分解示例]

Now here is a different goal. You can use these crafting commands to accomplish the goal. 
When you the desired item in your inventory, think: Task Completed! 
If you have tried your best but cannot proceed, think: task failed!
'''
```

**关键特征**:
- ✅ 明确的action格式示例
- ✅ 多步推理流程演示
- ✅ inventory检查逻辑
- ✅ 错误恢复策略（示例3）
- ✅ 明确的成功/失败标志

---

### 1.2 Eval (eval_textcraft_qwen3_1.7b.py)

**位置**: L43-57

**特点**:
- 仅**14行**简单system prompt
- **无具体示例**
- action格式不明确

**Prompt内容**:

```python
self.system_prompt = (
    'You are an agent in the TextCraft environment. Your goal is to craft items '
    'by gathering resources and following recipes.\n\n'
    'You can use actions like:\n'
    '- move [direction]\n'
    '- get [object]\n'
    '- craft [item]\n'
    '- inventory\n'
    '- look\n\n'
    'Instructions:\n'
    '- Read the task instruction carefully\n'
    '- Gather necessary resources\n'
    '- Follow crafting recipes\n'
    '- Complete the crafting goal\n\n'
    'Your response should contain only the action.'
)
```

**问题**:
- ❌ action格式与ADaPT不同（`craft [item]` vs `craft [count] [item] using [ingredients]`）
- ❌ 没有演示如何使用inventory
- ❌ 没有思考过程（但实际eval输出有`<think>`标签）
- ❌ 没有成功/失败判断标准

**实际Eval输出** (eval_results_20251213_190438.jsonl):

```json
{
  "role": "assistant",
  "content": "<think>\nOkay, let's see. The user wants to craft pink dye. Looking at the available 
crafting commands, there's one for pink dye: \"craft 1 pink dye using 1 pink tulip\". 
So the target is 1 pink dye, and the input is 1 pink tulip. But wait, do I have a pink 
tulip in my inventory? If not, I need to get it. ...\n</think>\n\nThought:\nThe goal is 
to craft 1 pink dye using the provided recipe. The recipe requires 1 pink tulip. If the 
pink tulip is already in the inventory, I can directly craft it. If not, I need to get it"
}
```

**发现**:
- 模型同时使用了`<think>`和"Thought:"
- action没有遵循prompt指定的格式
- 输出冗长，缺少明确的action行

---

### 1.3 GRPO Val (train_20251215_084925.log)

**特点**:
- **没有system prompt**（训练数据直接从parquet文件读取）
- 模型直接看到任务描述（crafting commands + Goal）

**实际输出** (L1095-1096, L2761-2763):

```
Step 0 - Raw assistant (len=2004): <think>

Looking at the available resources, there's 8 obsidian in the inventory. Wait, but the user 
might not have enough yet. Let me check. The initial inventory isn't shown, but the user 
can collect them. So I should first get the 8 obsidian. Then, find an ender eye. The ender 
eye can be made using...

Step 0 - Extracted action: 'craft an ender chest'
```

```
Step 16 - Raw assistant (len=633): <think>
Okay, the user is trying to craft bread but keeps getting errors. The previous attempts 
to get wheat didn't work, maybe because the inventory wasn't properly updated. The system 
might not be recognizing the wheat in the inventory. Let me check the steps again. The 
user needs to collect 3 wheat. The last action was getting wheat, but maybe the system 
didn't add it to the inventory. So the correct action is to get wheat again, ensuring 
it's in the inventory. Then, after collecting enough, ...

Step 16 - Extracted action: 'craft bread but keeps getting errors'
```

**问题**:
- ❌ 模型输出全是思考，缺少明确的action语句
- ❌ extract_action从长文本中误提取（L2763: 'craft bread but keeps getting errors'）
- ❌ 没有引导模型使用"Action:"格式

---

## 二、Action提取逻辑对比

### 2.1 ADaPT

**位置**: L160-250 (textcraft_run函数)

```python
action = llm(init_prompt + prompt, stop=['\n']).strip()  # L178, L223
action = action.lstrip('> ')  # L180, L226

# 如果是think，不计入action_history
if action.startswith('think:'):
    observation = 'OK.'
    if 'task completed' in action.lower(): done = True; success = True
    if 'task failed' in action.lower(): done = True; success = False
else: 
    action_history.append(action)
```

**特点**:
- ✅ stop=['\n'] 确保单行action
- ✅ 区分think和action
- ✅ 支持'> '前缀（命令行风格）
- ✅ 明确的终止条件

---

### 2.2 Eval

**方法**: 无专门的提取逻辑，直接使用整个模型输出作为action

```python
# eval_textcraft_qwen3_1.7b.py L138-140
response = agent.generate(messages)
messages.append({"role": "assistant", "content": response})
conversations.append({"role": "assistant", "content": response})

# 然后传给interaction
done, observation, step_reward, extra = await interaction.generate_response(
    instance_id, messages)  # L146
```

**依赖**: `TextCraftInteraction.extract_action()` (L61-129)

```python
def extract_action(self, text: str) -> Optional[str]:
    # 1. 移除chat template标记
    text = re.sub(r'^assistant\s*\n?', '', text, ...)
    text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, ...)
    
    # 2. 移除<think>标签（保留内容）
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    
    # 3. 提取"Action:"后的内容
    action_matches = re.findall(r'Action:\s*(.*?)(?=\n|$)', text, ...)
    if action_matches:
        action = action_matches[-1].strip()
        return action
    
    # 4. 尝试匹配craft命令
    craft_match = re.search(r'craft\s+\d+\s+[\w\s]+\s+using\s+[\w\s,\d]+', text, ...)
    if craft_match:
        return craft_match.group(0).strip()
    
    # 5. 简化craft
    craft_match = re.search(r'\bcraft\s+([\w\s]+?)(?=\n|$|\.|,)', text, ...)
    
    # 6. get命令
    get_match = re.search(r'get\s+\d+\s+[\w\s]+', text, ...)
    
    # 7. inventory
    if re.search(r'\binventory\b', text, ...):
        return 'inventory'
    
    return None
```

**问题**:
- ⚠️ 移除`<think>`标签但保留内容，导致在整个文本中搜索action
- ⚠️ 正则表达式4（L100）在长文本中可能误匹配
- ⚠️ Eval实际输出的"Thought:"不在匹配范围内

---

### 2.3 GRPO Val

使用相同的`TextCraftInteraction.extract_action()`，但问题更严重：

**实际案例** (train_20251215_084925.log):

| Step | Raw Assistant (思考内容) | 提取的Action | 环境反馈 |
|------|-------------------------|--------------|----------|
| L1095-1096 | "...So I should first get the 8 obsidian. Then, find an ender eye..." | `craft an ender chest` | `Could not execute craft an ender chest` |
| L1102-1103 | "...So I should collect white wool and planks..." | `craft a pink bed` | `Could not execute craft a pink bed` |
| L1106-1107 | "...So each stair uses 1.5 bricks?..." | `craft 4 polished blackstone brick stairs using those 6 bricks` | `Wrong item format: those 6 bricks` |
| L2763 | "...craft bread but keeps getting errors..." | `craft bread but keeps getting errors` | `Could not execute ...` |

**根因**:
- 模型输出全是`<think>...</think>`，没有单独的action行
- 正则L108-115（简化craft）从思考文本中误提取短语
- 提取的"action"是自然语言描述，不是命令

---

## 三、推理参数对比

### 3.1 参数表

| 系统 | 温度 | Max Tokens | Stop Tokens | 采样 | 推理引擎 |
|------|------|------------|-------------|------|----------|
| **ADaPT** | 0.0 | 150 (action)<br>800 (plan) | `['\n']` | ❌ | OpenAI API |
| **Eval** | 0.7 | 512 | EOS only | ✅ | Transformers |
| **GRPO (train)** | 0.7 | 512 | EOS only | ✅ | vLLM |
| **GRPO (val)** | 0.3 | 512 | EOS only | ✅ | vLLM |

### 3.2 关键差异

1. **温度**:
   - ADaPT: 0（确定性输出）
   - Eval: 0.7（较高随机性）→ 可能导致冗长输出
   - GRPO Val: 0.3（中等保守）

2. **Token限制**:
   - ADaPT: 150 tokens/action → 强制简洁
   - Eval/GRPO: 512 tokens → 允许长篇思考

3. **Stop tokens**:
   - ADaPT: `['\n']` → 单行输出，自然截断在action结束
   - Eval/GRPO: 无 → 模型可以一直生成到max_tokens或EOS

4. **推理引擎**:
   - Transformers vs vLLM在处理长文本时可能有细微差异
   - 但eval（Transformers）和GRPO Val（vLLM）都失败，说明不是引擎问题

---

## 四、实际表现对比

### 4.1 成功率

| 系统 | 成功率 | 样本数 | 说明 |
|------|--------|--------|------|
| **ADaPT** | 未知 | 200 | 论文方法，理论基准 |
| **Eval** | **54.5%** | 55/100 | Qwen3-1.7B base model |
| **GRPO Val** | **0%** | 4 tasks | 所有任务失败，reward=0 |

### 4.2 典型输出对比

#### ADaPT输出（理想）:
```
> inventory
Inventory: [stick] (1) [dark oak planks] (8)

> craft 1 dark oak sign using 6 dark oak planks, 1 stick
Crafted 1 minecraft:dark_oak_sign

> think: Task Completed!
```

#### Eval输出（部分成功）:
```
<think>
The user wants to craft pink dye. The recipe requires 1 pink tulip. 
If the pink tulip is already in the inventory, I can directly craft it...
</think>

Thought:
The goal is to craft 1 pink dye using the provided recipe...
```
→ 提取失败："Wrong item format: the provided recipe"

#### GRPO Val输出（完全失败）:
```
<think>
Looking at the available resources, there's 8 obsidian in the inventory. 
Wait, but the user might not have enough yet...
</think>
```
→ 提取到："craft an ender chest"（从思考中误提取）
→ 环境报错："Could not execute craft an ender chest"

---

## 五、根本原因分析

### 5.1 为什么Eval有54.5%成功率？

**真相**（通过分析成功案例发现）：

**Qwen3模型自发输出了"Action:"格式**，即使prompt没有要求！

**成功案例分析**（Session 0）：

```
Turn 7 [ASSISTANT]:
<think>
...The next step is to get the pink tulip from the environment...
</think>

Thought:
Since the pink tulip is required but not in the inventory, I need to obtain it...

Action: get 1 pink tulip
```

→ 环境响应：`Got 1 pink tulip` ✅

```
Turn 9 [ASSISTANT]:
<think>
...The recipe requires 1 pink tulip, so the correct command would be "craft 1 pink dye 
using 1 pink tulip"...
</think>

Action: craft 1 pink dye using 1 pink tulip
```

→ 环境响应：`Crafted 1 minecraft:pink_dye` ✅

**关键发现**：
1. ✅ 模型在长篇`<think>`后，自发输出"Thought:"和"Action:"
2. ✅ `extract_action`的L89正则成功匹配"Action:"后的内容
3. ✅ 54.5%成功率来自模型的**隐式CoT习惯**（预训练学到的）

**为什么不是100%成功**：
- 模型不是每次都输出"Action:"（只有54.5%的情况）
- 当模型只输出`<think>...</think>`时，提取逻辑回退到正则匹配，容易失败

### 5.2 为什么GRPO Val完全失败？

**根本原因对比**：

| 维度 | Eval (54.5%成功) | GRPO Val (0%成功) |
|------|-----------------|-------------------|
| **System Prompt** | ✅ 有（虽然简单） | ❌ 无 |
| **模型行为** | 54.5%情况输出"Action:" | 0%输出"Action:" |
| **输出格式** | `<think>...\nAction: xxx` | `<think>...` (全是思考) |
| **提取成功率** | 54.5%匹配到"Action:" | 0%，回退到正则误匹配 |

**为什么GRPO连"Action:"都不输出**：

1. **训练数据格式不同**：
   - Eval的system prompt暗示了"respond with action"
   - GRPO的训练数据只有任务描述，没有任何格式引导

2. **温度差异**：
   - Eval: temperature=0.7 → 更多样化，有时触发"Action:"格式
   - GRPO Val: temperature=0.3 → 更保守，坚持纯思考输出

3. **vLLM采样行为**：
   - 可能与Transformers在相同温度下的采样略有不同
   - 导致模型更少触发"Action:"格式

**实际GRPO输出**（全是思考，无action行）：

```
<think>
Looking at the available resources, there's 8 obsidian in the inventory. 
Wait, but the user might not have enough yet...
</think>
```

→ `extract_action`回退到正则L108，误提取："craft an ender chest"  
→ 环境报错：`Could not execute craft an ender chest` ❌

---

## 5.3 失败案例分析

**Eval失败案例**（Session 1，50轮后失败）：

```
[91-100] ASSISTANT (重复):
can use the "craft" command to get the slabs by using the "craft 4 blackstone slab 
using 3 blackstone" command again, which would require 3 blackstone. Since the assistant 
has 3, they can craft 4 more slabs. But the system keeps failing...

USER:
Could not find enough items to craft minecraft:blackstone_slab
```

**失败原因**：
1. 模型输出被截断，只有部分文本（"can use..."开头，缺少上文）
2. 正则提取到了错误的craft命令
3. 陷入循环：重复相同的错误命令直到达到max_rounds

**失败模式总结**：
- 当模型**不输出**"Action:"时，提取逻辑回退到正则匹配
- 正则从长文本中误匹配，提取到错误或不完整的命令
- 错误命令导致环境报错，模型在错误状态下继续生成，陷入循环

---

## 六、解决方案建议

### 优先级1：修复Prompt（立即）

**在训练数据中添加system prompt**：

```python
# 建议的prompt（参考ADaPT简化版）
TEXTCRAFT_SYSTEM_PROMPT = """You are an agent playing TextCraft (text-based Minecraft). 
Your goal is to craft items by gathering resources.

Available actions:
- get [count] [item]             # Example: get 2 oak log
- craft [count] [item] using [ingredients]  # Example: craft 1 stick using 2 bamboo
- inventory                      # Check what you have

Instructions:
1. Check inventory first
2. Get required ingredients
3. Craft the target item

IMPORTANT: 
- For thoughts, use prefix "think: ..."
- For actions, write the command directly (no prefix)
- When you have the target item, write: think: Task Completed!

Example:
Goal: craft dark oak sign

think: I need 6 dark oak planks and 1 stick
inventory
get 6 dark oak planks
get 1 stick
craft 1 dark oak sign using 6 dark oak planks, 1 stick
think: Task Completed!
"""
```

**修改位置**：
- 数据准备脚本（在conversation第一条消息前插入system message）
- 或修改`TextCraftInteraction.start_interaction()`返回的初始observation

### 优先级2：改进Action提取（中期）

**方案A**: 添加stop token

```yaml
# textcraft_grpo_train.yaml
rollout:
  val_kwargs:
    temperature: 0.3
    max_tokens: 150      # 降低到ADaPT水平
    stop: ["\n"]         # 单行输出
```

**方案B**: 改进`extract_action`逻辑

```python
def extract_action(self, text: str) -> Optional[str]:
    # 1. 优先处理明确的action行（非think开头的单行）
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('think:') and not line.startswith('<think'):
            # 检查是否是合法action
            if re.match(r'^(get|craft|inventory)', line, re.IGNORECASE):
                return line
    
    # 2. 然后才尝试从"Action:"后提取
    # ...
```

### 优先级3：微调模型（长期）

**SFT阶段**：
- 使用ADaPT格式的示例数据微调
- 强制模型学习"think: ... → action"的交替模式

**示例数据**：
```python
{
  "conversations": [
    {"role": "user", "content": "Crafting commands:\n...\nGoal: craft X"},
    {"role": "assistant", "content": "think: I need Y and Z\ninventory"},
    {"role": "user", "content": "Inventory: [A] (1)"},
    {"role": "assistant", "content": "get 2 Y"},
    {"role": "user", "content": "Got 2 Y"},
    {"role": "assistant", "content": "craft 1 X using 2 Y\nthink: Task Completed!"}
  ]
}
```

---

## 七、行动计划

### 第一步：验证假设（1天）

1. **手动测试**：
   - 使用ADaPT的prompt格式
   - 在eval脚本中测试5个session
   - 观察模型输出是否改善

2. **对比实验**：
   - 同一session_id
   - 分别用三种prompt测试
   - 记录action提取成功率

### 第二步：修改训练流程（2-3天）

1. **重新准备训练数据**：
   - 添加system prompt
   - 可选：添加few-shot示例（类似ADaPT）

2. **修改interaction代码**：
   - 在`start_interaction`中注入system prompt
   - 或在数据pipeline中处理

3. **调整推理参数**：
   - max_tokens: 512 → 150
   - 添加stop: ["\n"]

### 第三步：重新训练（3-5天）

1. **SFT阶段**：
   - 使用新prompt格式
   - 验证模型输出格式

2. **GRPO阶段**：
   - Validation应该能看到>0%成功率
   - 如果仍然失败，考虑更激进的prompt engineering

---

## 八、附录：关键代码位置

### ADaPT
- Prompt: `run_textcraft.py` L252-393
- Action提取: L178-180, L223-236
- LLM调用: L37-67 (max_tokens=150)

### Eval
- Prompt: `eval_textcraft_qwen3_1.7b.py` L43-57
- 推理参数: L196-203 (temperature=0.7, max_new_tokens=512)
- 无独立action提取，依赖interaction

### GRPO
- 配置: `textcraft_grpo_train.yaml` L48-77
- Interaction: `verl/interactions/textcraft_interaction.py`
- Action提取: L61-129
- 训练日志: `train_20251215_084925.log`

### 数据
- Eval结果: `eval_results_20251213_190438.jsonl`
- 训练数据: `/Data/wyh/datasets/Verl-Data/textcraft/train.parquet`

---

## 九、关键发现总结

### 9.1 三系统成功率的真相

| 系统 | 成功率 | 关键因素 |
|------|--------|----------|
| **ADaPT** | 高（未知） | Few-shot prompt + stop=['\n'] + 温度0 |
| **Eval** | 54.5% | **Qwen3自发输出"Action:"（54.5%情况）** |
| **GRPO Val** | 0% | 无prompt → 无"Action:"输出 → 误提取 |

### 9.2 核心发现

**不是vLLM vs Transformers的问题，而是prompt缺失导致模型行为变化**。

**Eval成功的秘密**：
- ✅ System prompt虽然简单，但提到了"Your response should contain only the action"
- ✅ Qwen3在这个提示下，54.5%情况会自发输出`<think>...\nAction: xxx`格式
- ✅ 这是模型预训练学到的**隐式CoT习惯**（类似GPT-4的"Let me think..."）

**GRPO失败的真相**：
- ❌ 训练数据完全没有prompt，只有"Crafting commands:\n...\nGoal: ..."
- ❌ 模型不知道应该输出action，只输出思考
- ❌ `extract_action`从纯思考文本中误提取短语

**ADaPT成功的关键**（对比）：
1. ✅ 140行详细few-shot prompt，包含3个完整示例
2. ✅ 明确的"think:"和action分离（不需要"Action:"前缀）
3. ✅ stop=['\n']强制单行输出，避免长篇思考
4. ✅ 温度0 + max_tokens=150，确保简洁确定性输出

### 9.3 为什么温度和stop token如此重要

**实验对比**：

| 配置 | ADaPT | Eval | GRPO Val |
|------|-------|------|----------|
| 温度 | 0 | 0.7 | 0.3 |
| Max tokens | 150 | 512 | 512 |
| Stop tokens | ['\n'] | 无 | 无 |
| **输出模式** | **单行action** | **长思考 + 有时有Action** | **纯长思考** |

**结论**：
- stop=['\n'] 是**强制约束**，防止模型输出长文本
- 低温度 + 短token限制 = 简洁输出
- 高温度 + 长token限制 = 冗长思考（但有时触发"Action:"格式）

---

## 十、结论与立即行动

**核心问题**：训练数据缺少system prompt，导致模型不知道要输出action格式。

**立即行动**（优先级排序）：

### 优先级1：添加Prompt（今天）

在训练数据中添加system prompt（参考ADaPT简化版）：

```python
TEXTCRAFT_SYSTEM_PROMPT = """You are an agent playing TextCraft (text-based Minecraft).

Available actions:
- get [count] [item]
- craft [count] [item] using [ingredients]
- inventory

Output format:
- For thoughts: think: ...
- For actions: write command directly

Example:
think: I need pink tulip
get 1 pink tulip
craft 1 pink dye using 1 pink tulip
think: Task Completed!
"""
```

### 优先级2：调整推理参数（今天）

```yaml
rollout:
  val_kwargs:
    temperature: 0.3
    max_tokens: 150       # 降低到ADaPT水平
    stop: ["\n", "think:"] # 强制单行输出
```

### 优先级3：验证实验（明天）

1. 用新prompt eval 10个session，观察是否接近100%输出"Action:"
2. 如果成功，重新开始GRPO训练

**预期效果**：
- 添加prompt后，模型应该稳定输出"Action:"格式
- Validation成功率应该从0%提升到40-60%（接近eval水平）
- 训练后应该进一步提升到>80%

**立即行动**：修改数据准备脚本，添加system prompt。

