# Webshop Integration for verl

这个文档介绍如何在新版verl (0.7.x)上集成Webshop环境进行online RL训练。

## 架构概览

Webshop集成使用verl的`BaseInteraction`机制，通过HTTP API与Webshop环境服务器通信：

```
verl训练框架
    ↓
AgentLoop (tool_agent_loop.py)
    ↓
WebshopInteraction (通过HTTP调用)
    ↓
Webshop环境服务器 (AgentGym)
```

## 文件结构

```
verl/
├── verl/interactions/
│   └── webshop_interaction.py          # Webshop交互实现
├── examples/
│   ├── sglang_multiturn/config/
│   │   └── webshop_interaction.yaml    # Interaction配置
│   ├── data_preprocess/
│   │   └── webshop_multiturn_w_interaction.py  # 数据预处理
│   └── test_webshop_interaction.py     # 测试脚本
```

## 快速开始

### 1. 启动Webshop环境服务器

首先需要启动Webshop后端服务：

**方法1：使用启动脚本（推荐）**
```bash
cd /Data/wyh/verl
bash examples/start_webshop_server.sh
```

**方法2：手动启动**
```bash
# 进入AgentGym-RL的webshop目录
cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop

# 启动服务器（监听36001端口）
python -m agentenv_webshop.launch --host 0.0.0.0 --port 36001
```

**验证服务器是否正常运行：**
```bash
curl http://127.0.0.1:36001/
# 应该返回: "ok"
```

⚠️ **注意**：如果返回 "This is environment BabyAI." 或其他信息，说明端口被其他环境服务器占用，需要先停止它。

保持这个终端运行，服务器会持续提供环境服务。

### 2. 测试集成

运行测试脚本验证集成是否成功：

```bash
cd /Data/wyh/verl

# 运行测试
python examples/test_webshop_interaction.py
```

测试会验证：
- WebshopInteraction类的初始化
- 与环境服务器的连接
- 动作提取功能
- 完整的交互流程（创建环境 → 执行动作 → 获取奖励 → 清理）

### 3. 准备训练数据

处理Webshop数据集：

```bash
cd /Data/wyh/verl

# 准备训练数据
python examples/data_preprocess/webshop_multiturn_w_interaction.py \
    --input_file /Data/wyh/datasets/AgentGym-RL-Data/train/webshop_train.json \
    --local_save_dir ~/data/webshop \
    --num_samples 100  # 可选：用于快速测试
```

这会生成一个包含`interaction_kwargs`的parquet文件。

### 4. 配置训练

创建训练配置文件 `examples/sglang_multiturn/config/webshop_grpo.yaml`：

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: ~/data/webshop/train.parquet
  val_files: ~/data/webshop/test.parquet  # 可选
  max_prompt_length: 1024
  max_response_length: 3000
  train_batch_size: 32
  return_raw_chat: True

actor_rollout_ref:
  model:
    path: /Data/public/Qwen2.5-3B-Instruct
  
  rollout:
    name: sglang
    temperature: 0.7
    top_p: 0.9
    prompt_length: 1024
    response_length: 3000
    
    multi_turn:
      enable: True
      max_user_turns: 15  # Webshop最大交互轮数
      max_assistant_turns: 15
      interaction_config_path: examples/sglang_multiturn/config/webshop_interaction.yaml
    
    agent:
      default_agent_loop: tool_agent  # 使用tool_agent_loop
      num_workers: 4

algorithm:
  adv_estimator: grpo

trainer:
  total_epochs: 5
  save_freq: 1
  project_name: webshop_rl
  experiment_name: qwen2.5-3b-webshop
```

### 5. 启动训练

```bash
cd /Data/wyh/verl

# 确保Webshop服务器正在运行
# 然后启动训练
python -m verl.trainer.main_ppo \
    --config-path examples/sglang_multiturn/config \
    --config-name webshop_grpo
```

## 核心组件说明

### WebshopInteraction类

实现了`BaseInteraction`接口的四个关键方法：

1. **`start_interaction(instance_id, session_id)`**
   - 创建环境实例
   - Reset到指定任务
   - 返回初始观察

2. **`generate_response(instance_id, messages)`**
   - 从messages中提取agent的动作
   - 执行动作并获取环境反馈
   - 返回 (done, observation, reward, info)

3. **`calculate_score(instance_id)`**
   - 返回累积奖励

4. **`finalize_interaction(instance_id)`**
   - 关闭环境释放资源

### 动作格式

Agent的动作应该是以下格式之一：
- `search[keywords]` - 搜索商品
- `click[item]` - 点击按钮或商品

示例：
```
search[red shirt size large]
click[B0B7QY8VXW]
click[size: large]
click[Buy Now]
```

### 数据格式

训练数据必须包含`interaction_kwargs`：

```python
{
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Instruction: Find..."}
    ],
    "extra_info": {
        "interaction_kwargs": {
            "name": "webshop",      # 必须与yaml中的name匹配
            "session_id": 123,      # Webshop任务ID
        }
    }
}
```

## 常见问题

### 1. 连接错误：Cannot connect to Webshop server

**原因**：Webshop服务器未启动或端口不正确

**解决**：
```bash
# 检查服务器是否运行
curl http://127.0.0.1:36001/

# 如果没有响应，启动服务器
cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop
python -m agentenv_webshop.launch --port 36001
```

### 2. 动作提取失败：No action found in messages

**原因**：Agent生成的响应不包含有效的动作格式

**解决**：
- 确保system prompt明确说明动作格式
- 在prompt中提供示例
- 检查模型输出是否包含 `search[...]` 或 `click[...]`

### 3. 训练中断：Environment timeout

**原因**：环境服务器响应超时

**解决**：
- 增加`timeout`配置（默认600秒）
- 检查服务器负载
- 减少并行worker数量

## 性能优化建议

1. **并行度调整**
   - `agent.num_workers`: 控制并行处理的样本数
   - 建议：CPU核心数的1-2倍

2. **批次大小**
   - `train_batch_size`: 训练批次大小
   - 建议：根据GPU内存调整，通常32-128

3. **交互轮数**
   - `max_user_turns`: 最大交互轮数
   - Webshop建议：10-15轮

4. **服务器资源**
   - Webshop服务器可以启动多个实例在不同端口
   - 配置多个interaction，使用不同的`env_server_base`

## 扩展到其他环境

基于相同的方式，可以集成其他AgentGym环境：

1. 创建对应的Interaction类（继承`BaseInteraction`）
2. 实现四个核心方法
3. 创建配置文件
4. 准备数据（添加`interaction_kwargs`）

支持的环境包括：
- TextCraft
- BabyAI
- SciWorld
- WebArena
- 等等

## 参考资料

- [verl文档](https://verl.readthedocs.io/)
- [AgentGym-RL](https://github.com/WooooDyy/AgentGym)
- [Webshop环境](https://github.com/princeton-nlp/WebShop)

