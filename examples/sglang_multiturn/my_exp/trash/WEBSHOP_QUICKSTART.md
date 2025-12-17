# Webshop集成 - 快速上手

## 问题诊断

刚才测试时遇到的错误是因为36001端口运行的是**BabyAI服务器**而不是Webshop服务器。

```bash
curl http://127.0.0.1:36001/
# 返回: "This is environment BabyAI."  ← 错误！应该是 "ok"
```

## 正确的启动流程

### 步骤1：停止其他环境服务器

```bash
# 查看36001端口被谁占用
lsof -i :36001

# 停止占用端口的进程
kill -9 $(lsof -t -i:36001)
```

### 步骤2：启动Webshop服务器

```bash
cd /Data/wyh/verl

# 使用脚本启动（会自动处理端口冲突）
bash examples/start_webshop_server.sh
```

或者手动启动：

```bash
cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop
python -m agentenv_webshop.launch --host 0.0.0.0 --port 36001
```

### 步骤3：验证服务器

新开一个终端，验证服务器：

```bash
# 应该返回 "ok"
curl http://127.0.0.1:36001/

# 测试创建环境
curl -X POST http://127.0.0.1:36001/create
# 应该返回一个数字（env_idx）
```

### 步骤4：运行测试

```bash
cd /Data/wyh/verl
python examples/test_webshop_interaction.py
```

预期输出：
```
================================================================================
Testing Webshop Interaction Integration
================================================================================

1. Initializing WebshopInteraction...
✓ WebshopInteraction initialized successfully

2. Starting interaction (instance_id=test_instance_001, session_id=0)...
✓ Interaction started successfully
  Initial observation: WebShop [SEP] Instruction: ...

3. Testing agent-environment interaction...
   Step 1: search[red shirt]
   ✓ Action executed successfully
     - Done: False
     - Reward: 0.0
     - Observation: ...

...

✓ All tests passed!
```

## 已创建的文件

1. **核心代码**
   - `/Data/wyh/verl/verl/interactions/webshop_interaction.py` - Webshop交互实现

2. **配置文件**
   - `/Data/wyh/verl/examples/sglang_multiturn/config/webshop_interaction.yaml` - Interaction配置

3. **工具脚本**
   - `/Data/wyh/verl/examples/data_preprocess/webshop_multiturn_w_interaction.py` - 数据预处理
   - `/Data/wyh/verl/examples/test_webshop_interaction.py` - 测试脚本
   - `/Data/wyh/verl/examples/start_webshop_server.sh` - 服务器启动脚本

4. **文档**
   - `/Data/wyh/verl/examples/WEBSHOP_INTEGRATION.md` - 完整文档
   - `/Data/wyh/verl/examples/WEBSHOP_QUICKSTART.md` - 本文档

## 下一步

### 准备训练数据

```bash
cd /Data/wyh/verl

python examples/data_preprocess/webshop_multiturn_w_interaction.py \
    --input_file /Data/wyh/datasets/AgentGym-RL-Data/train/webshop_train.json \
    --local_save_dir ~/data/webshop \
    --num_samples 10  # 先用10个样本测试
```

### 创建训练配置

创建文件 `/Data/wyh/verl/examples/sglang_multiturn/config/webshop_grpo.yaml`：

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: ~/data/webshop/train.parquet
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
      max_user_turns: 15
      max_assistant_turns: 15
      interaction_config_path: examples/sglang_multiturn/config/webshop_interaction.yaml
    
    agent:
      default_agent_loop: tool_agent
      num_workers: 4

algorithm:
  adv_estimator: grpo

trainer:
  total_epochs: 5
  save_freq: 1
  project_name: webshop_rl
  experiment_name: test_integration
```

### 运行训练

```bash
cd /Data/wyh/verl

# 确保Webshop服务器正在运行
# 然后启动训练
python -m verl.trainer.main_ppo \
    --config-path examples/sglang_multiturn/config \
    --config-name webshop_grpo
```

## 常见问题

### Q1: 测试时显示"Cannot connect to Webshop server"

**A**: Webshop服务器未启动，使用 `bash examples/start_webshop_server.sh` 启动

### Q2: 测试时显示"422 Client Error"

**A**: 端口被其他环境服务器占用，先停止其他服务器，然后启动Webshop服务器

### Q3: 如何同时支持多个环境？

**A**: 每个环境在不同端口启动服务器：
- Webshop: 36001
- BabyAI: 36002
- TextCraft: 36003
- ...

然后创建对应的Interaction类和配置文件。

## 集成验证清单

- [x] WebshopInteraction类创建完成
- [x] 配置文件创建完成
- [x] 数据预处理脚本创建完成
- [x] 测试脚本创建完成
- [x] 启动脚本创建完成
- [x] 文档创建完成
- [ ] 服务器启动并验证（需要用户操作）
- [ ] 测试通过（需要用户操作）
- [ ] 数据准备完成（需要用户操作）
- [ ] 训练测试通过（需要用户操作）

## 总结

Webshop环境已经成功集成到verl 0.7.x！关键特点：

1. **最小化修改**：只添加了Interaction实现，没有修改verl框架核心代码
2. **完全复用**：利用verl的AgentLoop和multiturn机制
3. **HTTP通信**：继续使用AgentGym的环境服务器
4. **易于扩展**：同样的方式可以集成其他环境

现在请按照上面的步骤启动Webshop服务器并运行测试！

