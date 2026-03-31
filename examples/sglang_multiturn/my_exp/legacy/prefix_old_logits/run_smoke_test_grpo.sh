#!/bin/bash
# Smoke Test 脚本：验证 prefix old_logprobs GRPO 训练链路
# 修复版 - 针对 OOM、长度预算、TEST_FREQ 等问题

set -e

# ==================== 配置 ====================
# 使用已适配的训练数据
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/textcraft_validated_prefix_with_old_logprobs_step200_training.parquet"}

MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}

# GPU 配置 - 使用 GPU 2（完全空闲，45GB 可用显存）
GPU_IDS=${GPU_IDS:-"2"}
NUM_GPUS=${NUM_GPUS:-1}

# ========== 核心修复1: OOM 预防 ==========
# 极端保守配置：单卡 + 极小内存占用
NUM_EPOCHS=1
TRAIN_BATCH_SIZE=1
MICRO_BATCH_SIZE=1
ROLLOUT_N=1
SAVE_FREQ=999

# ========== 核心修复2: TEST_FREQ=999 ==========
TEST_FREQ=999
VAL_BEFORE_TRAIN=false

# ========== 核心修复3: 长度预算收紧 ==========
# 实际数据 prompt 长度范围: 833-1943 (median=1089, p90=1384)
# max_prompt_length 控制数据过滤门槛 (需 >= 实际最大 prompt)
MAX_PROMPT_LENGTH=2048       # 必须 >= 实际 prompt 最大值 (1943)，否则过滤后无样本
PROMPT_LENGTH_FOR_ROLLOUT=2048  # rollour prompt_length 需 >= 数据 prompt 最大值
RESPONSE_LENGTH=128          # 单轮生成 128 tokens
MAX_MODEL_LEN=4096          # 从 3072 恢复到 4096，vLLM 需要足够 max_model_len

# ========== 核心修复4: vLLM 内存占用 ==========
# GPU 2 完全空闲（45GB），使用保守的 gpu_memory_utilization=0.30
# 0.30 在空闲 GPU 上工作良好，且有足够空间容纳训练峰值
# - vLLM profiling 峰值：model 3.4GB + grad 0.6GB = 4GB（轻松满足）
# - vLLM 稳定后分配：model 3.4GB + KV cache 9.45GB ≈ 13GB
# - 训练 optimizer.step 峰值：PyTorch 17.2GB + vLLM 13GB ≈ 30GB < 45GB
# - 即使 GPU 被其他进程占用部分显存，也有充足余量
GPU_MEMORY_UTIL=0.30
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=64
PPO_MAX_TOKEN_LEN=2048

# Prefix optimization 配置
OPTIMIZE_PREFIX_TOKENS=true
PREFIX_LOSS_WEIGHT=1.0

# 其他配置
USE_KL_LOSS=false
LEARNING_RATE=1e-6

# 输出目录
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_prefix_smoke_test"

# ==================== 打印配置 ====================
echo "================================================================================"
echo "  Smoke Test: Prefix Old Logprobs GRPO Training (OOM-Fixed v2)"
echo "================================================================================"
echo ""
echo "【长度预算修复】"
echo "  实际数据 prompt 长度: 833-1943 (median=1089, p90=1384)"
echo "  MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH (数据过滤，需 >= 实际最大值)"
echo "  PROMPT_LENGTH_FOR_ROLLOUT=$PROMPT_LENGTH_FOR_ROLLOUT"
echo "  RESPONSE_LENGTH=$RESPONSE_LENGTH"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo ""
echo "【OOM 预防】"
echo "  GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
echo "  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "  MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo ""
echo "【TEST_FREQ】"
echo "  TEST_FREQ=$TEST_FREQ"
echo "  VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN"
echo "================================================================================"
echo ""

# ==================== 检查数据 ====================
echo "检查数据文件..."
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi
echo "✓ 数据文件存在"

python3 -c "
import pandas as pd
df = pd.read_parquet('$DATA_PATH')
print(f'样本数: {len(df)}')
print(f'列: {df.columns.tolist()}')
if 'assistant_prefix_old_log_probs' in df.columns:
    print('✓ 找到 assistant_prefix_old_log_probs')
else:
    print('✗ 缺少 assistant_prefix_old_log_probs')
    exit(1)
"
echo ""

# ==================== 清理残留 Ray 进程 ====================
echo "清理残留 Ray 进程..."
ray stop --force 2>/dev/null || true
sleep 3
echo ""

# ==================== 切换环境 ====================
cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl
echo "Python 版本: $(python --version)"
echo ""

# ==================== 验证长度预算 ====================
echo "验证长度预算..."
REQUIRED_MAX_MODEL_LEN=$((PROMPT_LENGTH_FOR_ROLLOUT + RESPONSE_LENGTH))
if [ "$MAX_MODEL_LEN" -lt "$REQUIRED_MAX_MODEL_LEN" ]; then
    echo "错误: MAX_MODEL_LEN=$MAX_MODEL_LEN < PROMPT + RESPONSE=$REQUIRED_MAX_MODEL_LEN"
    exit 1
fi
if [ "$MAX_PROMPT_LENGTH" -lt 1943 ]; then
    echo "警告: MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH < 实际最大值 1943，部分样本将被过滤"
fi
echo "✓ 长度预算检查通过"
echo ""

# ==================== 启动训练 ====================
# vLLM 使用自己的 CUDA memory pool，与 PYTORCH_CUDA_ALLOC_CONF=expandable_segments 不兼容
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 不要设置此变量！
export VERL_LOGGING_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_LOGGING_LEVEL=WARNING
export PYTHONWARNINGS=ignore
# 注意：不要设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vLLM 使用自己的 CUDA memory pool，与 expandable_segments 不兼容

echo "开始 Smoke Test..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

python3 -m verl.trainer.main_ppo \
    --config-path='/Data/wyh/verl/examples/sglang_multiturn/config' \
    --config-name='textcraft_grpo_train' \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=1 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.prompt_length=$PROMPT_LENGTH_FOR_ROLLOUT \
    actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.3 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=textcraft_grpo_prefix_smoke_test \
    trainer.experiment_name=smoke_test_${TIMESTAMP} \
    trainer.resume_mode=disable \
    '+algorithm.grpo_single_sample_adv=true' \
    algorithm.optimize_prefix_tokens=true \
    algorithm.prefix_loss_weight=1.0 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Smoke Test 完成！"
else
    echo "✗ Smoke Test 失败 (exit code: $EXIT_CODE)"
fi
echo "================================================================================"
echo ""
echo "日志文件: $LOG_FILE"

# ==================== 验证关键日志 ====================
echo ""
echo "================================================================================"
echo "验证关键日志"
echo "================================================================================"

echo ""
echo "1. 数据过滤后样本数:"
grep -E "filter dataset len" "$LOG_FILE" | tail -3 || echo "未找到"

echo ""
echo "2. 检查 prefix old_logprobs 被读到:"
grep -E "PREFIX_OPT|PREFIX_DEBUG|assistant_prefix_old_log_probs|use_cached_prefix" "$LOG_FILE" | head -10 || echo "未找到"

echo ""
echo "3. 检查 loss 和 grad_norm:"
grep -E "actor/continuation_loss|actor/prefix_loss|actor/pg_loss|actor/grad_norm|actor/ppo_kl" "$LOG_FILE" | head -20 || echo "未找到"

echo ""
echo "4. 检查 OOM / max_tokens 错误:"
grep -i "OOM\|OutOfMemory\|max_tokens must\|CUDA.*out of memory" "$LOG_FILE" | head -10 || echo "未找到 OOM 错误"

echo ""
echo "5. 检查训练 step 推进:"
grep -E "Training Progress|global_step" "$LOG_FILE" | tail -5 || echo "未找到 step 日志"

exit $EXIT_CODE
