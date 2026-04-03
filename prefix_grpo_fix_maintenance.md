# Prefix-GRPO 修复维护说明

最后更新：2026-04-02

本文件记录从 `/Data/wyh/h200_grpo_from_oss` 同步到本仓库的 Prefix-GRPO 关键修复与当前默认口径。

## 当前默认口径

- 主实验脚本：
  `/Data/wyh/verl/examples/sglang_multiturn/my_exp/rl/run_textcraft_grpo_validated.sh`
- 主训练配置：
  `/Data/wyh/verl/examples/sglang_multiturn/config/textcraft_grpo_train.yaml`
- 默认 prefix 配置：
  `optimize_prefix_tokens=true`
  `prefix_loss_mode=split`
  `prefix_advantage_mode=constant`
  `prefix_advantage_constant=1.0`
  `use_kl_loss=false`

## 已同步的关键点

- 主实验脚本已暴露 prefix 相关控制项：
  `prefix_loss_mode`
  `prefix_advantage_mode`
  `prefix_advantage_constant`
  `prefix_clip_ratio`
  `prefix_clip_ratio_low`
  `prefix_clip_ratio_high`
  `prefix_clip_ratio_c`
- 脚本默认值已切到：
  `split + constant`
- 脚本启动时会通过 Hydra override 将上述参数传入 actor / algorithm
- 主配置中的路径型默认值已改为 `null`，由脚本在运行时传入，避免仓库内保留强绑定的本机绝对路径

## Python 侧状态

以下核心实现与 H200 验证版本一致，本轮无需额外同步：

- `/Data/wyh/verl/verl/trainer/ppo/ray_trainer.py`
- `/Data/wyh/verl/verl/workers/actor/dp_actor.py`
- `/Data/wyh/verl/verl/interactions/textcraft_interaction.py`
- `/Data/wyh/verl/envs/AgentGym/agentenv-textcraft/agentenv_textcraft/env_wrapper.py`

## 备注

- 当前实现里：
  `split` 模式支持 prefix 独立 clip；
  `joint` 模式将 prefix token 与 continuation token 放进统一 PPO 目标，但会共用 clip。
- 如果后续需要“joint + prefix 独立 clip”同时成立，需要单独补一个 hybrid loss 实现，而不是只改配置。
