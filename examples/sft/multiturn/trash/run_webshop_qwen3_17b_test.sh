#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_webshop_qwen3_06b_test.sh <nproc_per_node> <save_path> [other_configs...]"
    echo ""
    echo "Examples:"
    echo "  # 使用默认GPU (全部可见GPU)"
    echo "  bash $0 2 /output"
    echo ""
    echo "  # 指定使用GPU 0,1"
    echo "  CUDA_VISIBLE_DEVICES=0,1 bash $0 2 /output"
    echo ""
    echo "  # 启用 wandb 日志"
    echo "  bash $0 2 /output trainer.logger=[console,wandb]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# ============================================================================
# GPU 配置说明:
# - 通过 CUDA_VISIBLE_DEVICES 环境变量指定使用哪些GPU
# - nproc_per_node 参数指定使用的GPU数量（必须 <= 可见GPU数）
# 
# 示例:
#   CUDA_VISIBLE_DEVICES=0,1 使用GPU 0和1
#   CUDA_VISIBLE_DEVICES=2,3,4,5 使用GPU 2,3,4,5
# ============================================================================

# Fix: Add CUDA libraries to LD_LIBRARY_PATH
# 修复 libcusparseLt.so.0 找不到的问题
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/Data/wyh/datasets/Parquet-Data/webshop/train.parquet \
    data.val_files=/Data/wyh/datasets/Parquet-Data/webshop/train.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=2 \
    data.max_length=4096 \
    model.partial_pretrain=/Data/public/Qwen3-1.7B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=webshop-sft-test \
    trainer.experiment_name=webshop-sft-qwen3-1.7b-sp2 \
    trainer.logger=console \
    trainer.total_epochs=8 $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
    
# ============================================================================
# 未显式设置的参数将使用 sft_trainer.yaml 中的默认值
# ============================================================================
#
# 训练参数:
#   - optim.lr=1e-5 (学习率)
#   - optim.weight_decay=0.01 (权重衰减)
#   - data.train_batch_size=256 (全局batch size)
#   - model.enable_gradient_checkpointing=True (梯度检查点)
#
# 日志配置:
#   - trainer.logger=console (默认只输出到控制台)
#   - 可选值: console, wandb, tensorboard
#
# GPU配置:
#   - trainer.n_gpus_per_node=8 (默认值，会被 nproc_per_node 覆盖)
#
# ============================================================================
# 常用配置示例
# ============================================================================
#
# 1. 指定GPU位置 (使用GPU 2和3):
#    CUDA_VISIBLE_DEVICES=2,3 bash run_webshop_qwen3_17b_test.sh 2 /output
#
# 2. 启用 wandb 日志:
#    bash run_webshop_qwen3_17b_test.sh 2 /output \
#      trainer.logger=[console,wandb] \
#      trainer.project_name=my-project \
#      trainer.experiment_name=exp-001
#
# 3. 修改学习率和batch size:
#    bash run_webshop_qwen3_17b_test.sh 2 /output \
#      optim.lr=5e-5 \
#      data.train_batch_size=128
#
# 4. 完整训练 (更多steps + wandb):
#    CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_webshop_qwen3_17b_test.sh 4 /output \
#      trainer.total_training_steps=1000 \
#      trainer.logger=[console,wandb] \
#      trainer.save_freq=100
#
# ============================================================================

