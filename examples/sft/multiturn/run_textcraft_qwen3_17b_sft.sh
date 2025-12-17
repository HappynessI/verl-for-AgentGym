#!/bin/bash
set -x

# 显式设置默认参数
nproc_per_node=2
save_path=/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/new_ckp

# 如果命令行提供了参数，则覆盖默认值
if [ "$#" -ge 2 ]; then
    nproc_per_node=$1
    save_path=$2
    shift 2
fi

echo "训练配置:"
echo "  GPU数量: $nproc_per_node"
echo "  保存路径: $save_path"
echo ""

# ============================================================================
# 日志配置
# ============================================================================
LOG_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "日志文件: $LOG_FILE"
echo ""

# ============================================================================
# GPU 配置
# ============================================================================
# 显式设置使用的GPU（如果外部未设置CUDA_VISIBLE_DEVICES）
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1  # 默认使用GPU 0和1
    echo "CUDA_VISIBLE_DEVICES not set, using default: 0,1"
else
    echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# Fix: Add CUDA libraries to LD_LIBRARY_PATH
# 修复 libcusparseLt.so.0 找不到的问题
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "开始训练，日志输出到: $LOG_FILE"
echo "可以使用以下命令查看实时日志:"
echo "  tail -f $LOG_FILE"
echo ""

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/Data/wyh/datasets/Verl-Data/train/textcraft/train_gemini_adapt.parquet \
    data.val_files=/Data/wyh/datasets/Verl-Data/train/textcraft/train_gemini_adapt.parquet \
    data.train_batch_size=32 \
    data.micro_batch_size=2 \
    data.max_length=8192 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    model.partial_pretrain=/Data/public/Qwen3-1.7B \
    model.enable_gradient_checkpointing=true \
    optim.lr=1e-5 \
    optim.betas=[0.9,0.95] \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir=$save_path \
    trainer.project_name=textcraft-sft \
    trainer.experiment_name=textcraft-sft-qwen3-1.7b-gemini-adapt \
    trainer.logger=[console,wandb] \
    trainer.seed=42 \
    trainer.total_epochs=20 \
    trainer.save_freq=80 \
    trainer.test_freq=80 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true $@ 2>&1 | tee -a "$LOG_FILE"

# ============================================================================
# 训练完成后自动评估
# ============================================================================

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "训练完成，退出码: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
echo "完整日志已保存到: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "查找最新checkpoint..." | tee -a "$LOG_FILE"
    CKPT_DIR="${save_path}/textcraft-sft/textcraft-sft-qwen3-1.7b-gemini-adapt/checkpoints"
    
    if [ -d "$CKPT_DIR" ]; then
        # 找到最新的checkpoint（按global_step数字排序）
        LATEST_CKPT=$(ls -d ${CKPT_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
        
        if [ -n "$LATEST_CKPT" ]; then
            echo "最新checkpoint: $LATEST_CKPT" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            echo "================================================================================" | tee -a "$LOG_FILE"
            echo "开始自动评估..." | tee -a "$LOG_FILE"
            echo "================================================================================" | tee -a "$LOG_FILE"
            
            # 设置eval脚本参数
            export MODEL_PATH="$LATEST_CKPT"
            export MAX_SAMPLES=100  # 全量测试
            export CUDA_VISIBLE_DEVICES=2  # 使用GPU 2进行eval
            
            # 运行eval
            cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval
            bash run_textcraft_eval.sh 2>&1 | tee -a "$LOG_FILE"
            
            EVAL_EXIT_CODE=${PIPESTATUS[0]}
            echo "" | tee -a "$LOG_FILE"
            echo "================================================================================" | tee -a "$LOG_FILE"
            if [ $EVAL_EXIT_CODE -eq 0 ]; then
                echo "评估完成！" | tee -a "$LOG_FILE"
                echo "结果保存在: /Data/wyh/datasets/Verl-Data/outputs/textcraft_eval/" | tee -a "$LOG_FILE"
            else
                echo "评估失败，退出码: $EVAL_EXIT_CODE" | tee -a "$LOG_FILE"
            fi
            echo "================================================================================" | tee -a "$LOG_FILE"
        else
            echo "未找到checkpoint文件" | tee -a "$LOG_FILE"
        fi
    else
        echo "Checkpoint目录不存在: $CKPT_DIR" | tee -a "$LOG_FILE"
    fi
else
    echo "训练失败，跳过评估" | tee -a "$LOG_FILE"
fi
    
# ============================================================================
# TextCraft SFT 训练配置详解
# ============================================================================
#
# 【数据配置】
#   data.train_files - 训练数据路径 (350条Gemini采样ADaPT格式轨迹)
#   data.val_files - 验证数据路径 (当前复用训练集)
#   data.train_batch_size=32 - 全局batch size (所有GPU合计)
#   data.micro_batch_size=2 - 每张GPU的micro batch size
#                             实际梯度累积步数 = 256 / (2 * 2 GPUs) = 64
#   data.max_length=4096 - 最大序列长度 (token数)
#   data.multiturn.enable=true - 启用多轮对话格式
#   data.multiturn.messages_key=messages - 多轮对话数据的字段名
#
# 【模型配置】
#   model.partial_pretrain - 基础模型路径: Qwen3-1.7B
#   model.enable_gradient_checkpointing=true - 启用梯度检查点（节省显存）
#   ulysses_sequence_parallel_size=2 - 序列并行度（必须=GPU数）
#   use_remove_padding=true - 移除padding提高训练效率
#
# 【优化器配置】（这些是训练超参数，不是推理参数！）
#   optim.lr=1e-5 - 学习率 (1×10⁻⁵)
#   optim.betas=[0.9,0.95] - Adam优化器的beta参数
#   optim.weight_decay=0.01 - 权重衰减系数（L2正则化）
#   optim.lr_warmup_steps_ratio=0.1 - 学习率预热步数比例（前10%步数线性增加）
#   optim.clip_grad=1.0 - 梯度裁剪阈值（防止梯度爆炸）
#   optim.lr_scheduler=cosine - 学习率调度器（余弦退火）
#
# 【训练流程配置】
#   trainer.seed=42 - 随机种子（保证可复现）
#   trainer.total_epochs=10 - 训练轮数
#   trainer.save_freq=40 - 每40步保存checkpoint
#   trainer.test_freq=40 - 每40步在验证集测试
#
# 【日志配置】
#   trainer.logger=[console,wandb] - 同时输出到控制台和wandb
#   trainer.project_name=textcraft-sft - wandb项目名
#   trainer.experiment_name=textcraft-sft-qwen3-1.7b-gpt4o - wandb实验名
#
# 【推理参数在哪里？】
#   temperature、top_p等是推理时的参数，不在训练脚本中！
#   这些参数在评估脚本中设置：
#     /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/run_textcraft_eval.sh
#
# ============================================================================
# 训练命令示例
# ============================================================================
#
# 1. 基础训练（使用默认GPU 0,1，已启用wandb）:
#    bash run_textcraft_qwen3_17b_sft.sh 2 \
#      /Data/wyh/datasets/Verl-Data/outputs/textcraft_sft
#
# 2. 使用不同的GPU（例如GPU 2,3）:
#    CUDA_VISIBLE_DEVICES=2,3 bash run_textcraft_qwen3_17b_sft.sh 2 \
#      /Data/wyh/datasets/Verl-Data/outputs/textcraft_sft
#
# 3. 修改实验名称:
#    bash run_textcraft_qwen3_17b_sft.sh 2 \
#      /Data/wyh/datasets/Verl-Data/outputs/textcraft_sft \
#      trainer.experiment_name=textcraft-sft-exp002
#
# 4. 修改保存和测试频率（每20步一次）:
#    bash run_textcraft_qwen3_17b_sft.sh 2 \
#      /Data/wyh/datasets/Verl-Data/outputs/textcraft_sft \
#      trainer.save_freq=20 \
#      trainer.test_freq=20
#
# ============================================================================
# 评估说明
# ============================================================================
#
# 训练过程中会在 save_freq 指定的步数保存checkpoint，保存路径：
#   $save_path/textcraft-sft/textcraft-sft-qwen3-1.7b-gemini-adapt/checkpoints/
#
# 可以使用以下脚本进行评估：
#   /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/run_textcraft_eval.sh
#
# 评估示例:
#   1. 修改eval脚本中的MODEL_PATH指向checkpoint路径
#   2. 运行: bash run_textcraft_eval.sh
#
# ============================================================================


