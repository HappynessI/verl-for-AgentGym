#!/usr/bin/env python3
"""
Debug entry point for TextCraft GRPO training.

目标：尽量忠实复现原始 GRPO base 实验，只做最小化 debug 和防 OOM 改动。

修改原则：
- 明确设置 GRPO 算法
- 保持原实验采样参数 (temperature=0.8, top_p=0.95)
- 只做必要的 OOM 防护，不过度改变语义
- 删除可疑的 reload 逻辑
"""

import os
import sys

# ===== 1. ADD PATHS =====
sys.path.insert(0, "/Data/wyh/verl/examples/sglang_multiturn/my_exp")
sys.path.insert(0, "/Data/wyh/verl")

import logging
from datetime import datetime
from debug.add_debug_logging import apply_debug_patches

# ===== 2. CONFIGURE LOGGING =====
log_dir = "/Data/wyh/verl/examples/sglang_multiturn/my_exp/debug/logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f"{log_dir}/rollout_dumps", exist_ok=True)
log_file = os.path.join(log_dir, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
print(f"[DEBUG] Log file: {log_file}")

# ===== 3. APPLY DEBUG PATCHES (只打一次) =====
print("[DEBUG] Applying debug patches...")
apply_debug_patches(
    dump_dir=f"{log_dir}/rollout_dumps",
    samples_per_step=2
)

# ===== 4. SET ENVIRONMENT =====
# 使用 0,1,2,4 四卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

# ===== 5. HYDRA CONFIG =====
os.chdir("/Data/wyh/verl")

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

config_dir = "/Data/wyh/verl/verl/trainer/config"
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(
        config_name="ppo_trainer",
        overrides=[
            # ===== A. 明确 GRPO 语义 =====
            "algorithm.adv_estimator=grpo",
            
            # ===== B. 数据配置 (原实验值，仅微调防 OOM) =====
            "data.train_files=/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet",
            "data.val_files=/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet",
            "data.train_batch_size=4",  # 原实验值: 16 → 大幅减至最小
            "data.val_batch_size=2",     # 原实验值: 4 → 最小
            "data.max_prompt_length=1024",  # 原实验值: 2048 → 减小
            "data.max_response_length=2048",  # 原实验值: 16384 → 减小
            "+data.apply_chat_template_kwargs.enable_thinking=true",
            
            # ===== C. Ray/系统配置 =====
            "ray_kwargs.ray_init.num_cpus=16",
            "transfer_queue.enable=false",
            "global_profiler.tool=null",
            
            # ===== D. 模型配置 (原实验值) =====
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.actor.strategy=fsdp",
            "actor_rollout_ref.actor.use_dynamic_bsz=false",
            "actor_rollout_ref.model.path=/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo/merged_step_100",
            "actor_rollout_ref.model.enable_gradient_checkpointing=true",
            # ===== D1. Offload 配置 (防 OOM) =====
            "actor_rollout_ref.model.enable_activation_offload=true",  # 激活值 offload
            "actor_rollout_ref.actor.fsdp_config.param_offload=true",  # Actor 参数 offload
            "actor_rollout_ref.ref.fsdp_config.param_offload=true",  # Ref 参数 offload
            "actor_rollout_ref.actor.optim.lr=5e-6",  # 原实验值
            
            # ===== E. PPO 训练参数 (保守调整防 OOM) =====
            "actor_rollout_ref.actor.ppo_mini_batch_size=4",  # 原实验值: 16 → 最小
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",  # 原实验值: 8 → 最小
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048",  # 原实验值: 24576 → 大幅减小
            
            # ===== F. Rollout 配置 (原实验采样参数保持不变!) =====
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.n=2",           # 原实验值: 8 → 最小 (GRPO 需要 >1)
            "actor_rollout_ref.rollout.temperature=0.8",   # 原实验值: 0.8 (保持不变!)
            "actor_rollout_ref.rollout.top_p=0.95",        # 原实验值: 0.95 (保持不变!)
            
            # ===== G. Token 长度 (保守调整防 OOM) =====
            "actor_rollout_ref.rollout.prompt_length=1024",  # 原实验值: 16384 → 大幅减小
            "actor_rollout_ref.rollout.response_length=1024",  # 原实验值: 16384 → 大幅减小
            "actor_rollout_ref.rollout.max_model_len=2048",   # 原实验值: 20480 → 大幅减小
            "actor_rollout_ref.rollout.max_num_batched_tokens=1024",  # 原实验值: 16384 → 大幅减小
            "actor_rollout_ref.rollout.max_num_seqs=8",  # 原实验值: 64 → 最小
            
            # ===== H. vLLM 资源 (保守调整防 OOM) =====
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",  # 原实验值: 0.7 → 降低
            "actor_rollout_ref.rollout.enforce_eager=true",   # 原实验值
            "actor_rollout_ref.rollout.free_cache_engine=true",
            "actor_rollout_ref.rollout.calculate_log_probs=true",
            "+actor_rollout_ref.rollout.debug_print=true",  # debug 输出
            
            # 修复 vLLM generation config 被 HF 覆盖
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config=vllm',
            
            # ===== I. Validation 配置 (原实验值) =====
            "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
            "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",  # 最小
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            
            # ===== J. Trainer 配置 =====
            "trainer.n_gpus_per_node=4",  # 使用 4 卡
            "trainer.nnodes=1",
            "trainer.total_epochs=1",  # debug 只跑 1 个 epoch
            "trainer.save_freq=100",
            "trainer.test_freq=100",
            "trainer.val_before_train=false",  # 关闭训练前 validation
            "trainer.default_local_dir=/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo",
            "trainer.project_name=textcraft_grpo",
            "trainer.experiment_name=textcraft_grpo_debug",
            "trainer.resume_mode=disable",
        ]
    )

# ===== 6. VERIFY AND RUN =====
from verl.trainer.main_ppo import run_ppo

print("\n" + "=" * 60)
print("[DEBUG] Verifying GRPO config:")
print(f"  algorithm.adv_estimator = {cfg.algorithm.adv_estimator}")
print(f"  actor_rollout_ref.rollout.n = {cfg.actor_rollout_ref.rollout.n}")
print(f"  actor_rollout_ref.rollout.temperature = {cfg.actor_rollout_ref.rollout.temperature}")
print(f"  actor_rollout_ref.rollout.top_p = {cfg.actor_rollout_ref.rollout.top_p}")
print(f"  data.train_batch_size = {cfg.data.train_batch_size}")
print(f"  data.max_response_length = {cfg.data.max_response_length}")
print(f"  trainer.val_before_train = {cfg.trainer.val_before_train}")
print("=" * 60 + "\n")

print("[DEBUG] Starting GRPO training (minimal debug modifications)...")
run_ppo(cfg)
