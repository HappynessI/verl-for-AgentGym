#!/bin/bash
set -x

# ============================================================================
# 可配置参数（可通过环境变量覆盖）
# ============================================================================
nproc_per_node=${NPROC_PER_NODE:-1}
save_path=${SAVE_PATH:-}
model_path=${MODEL_PATH:-/Data/public/Qwen3-1.7B}
gpu_ids=${CUDA_VISIBLE_DEVICES:-0}

# ============================================================================
# 数据集配置
# ============================================================================
# 支持的数据集选项: success_only, success_220, mixed, mixed_242
DATASET_NAME=${DATASET_NAME:-mixed_242}

# 数据集路径映射
declare -A DATASET_PATHS
DATASET_PATHS[success_only]="/Data/wyh/datasets/SFT-Data/textcraft_success_only.parquet"
DATASET_PATHS[success_220]="/Data/wyh/datasets/SFT-Data/textcraft_success_220_only.parquet"
DATASET_PATHS[mixed]="/Data/wyh/datasets/SFT-Data/textcraft_mixed.parquet"
DATASET_PATHS[mixed_242]="/Data/wyh/datasets/SFT-Data/textcraft_mixed_242.parquet"

# 验证数据集选择
if [[ -z "${DATASET_PATHS[$DATASET_NAME]}" ]]; then
    echo "错误: 无效的 DATASET_NAME='$DATASET_NAME'"
    echo "支持的选项: ${!DATASET_PATHS[@]}"
    echo ""
    echo "可用数据集:"
    for key in "${!DATASET_PATHS[@]}"; do
        echo "  - $key: ${DATASET_PATHS[$key]}"
    done
    exit 1
fi

data_path=${DATASET_PATHS[$DATASET_NAME]}

# ============================================================================
# 实验配置
# ============================================================================
# 实验名称：包含数据集信息
experiment_name="textcraft-sft-$(basename $model_path)-${DATASET_NAME}"

# 保存路径：自动区分不同实验
if [ -z "$save_path" ]; then
    save_path="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/$(basename $model_path)-sft_${DATASET_NAME}"
fi

# ============================================================================
# 日志配置
# ============================================================================
LOG_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${DATASET_NAME}_${TIMESTAMP}.log"
echo "日志文件: $LOG_FILE"
echo ""

# ============================================================================
# GPU 配置
# ============================================================================
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_ids
    echo "CUDA_VISIBLE_DEVICES not set, using default: $gpu_ids"
else
    echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix: Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "训练配置:"
echo "  GPU数量: $nproc_per_node"
echo "  保存路径: $save_path"
echo "  模型路径: $model_path"
echo "  数据文件: $data_path"
echo "  数据集: $DATASET_NAME"
echo "  实验名称: $experiment_name"
echo ""
echo "可以使用以下命令查看实时日志:"
echo "  tail -f $LOG_FILE"
echo ""

# ============================================================================
# 运行训练
# ============================================================================
run_training() {
    local dataset_name=$1
    local save_path=$2
    local model_path=$3
    
    local data_path=${DATASET_PATHS[$dataset_name]}
    local experiment_name="textcraft-sft-$(basename $model_path)-${dataset_name}"
    
    torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        --master_port=29501 \
         -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$data_path \
        data.val_files=$data_path \
        data.train_batch_size=16 \
        data.micro_batch_size=2 \
        data.micro_batch_size_per_gpu=2 \
        +data.gradient_accumulation_steps=8 \
        data.max_length=4096 \
        data.truncation=left \
        +data.pad_mode=right \
        data.multiturn.enable=true \
        data.multiturn.messages_key=conversations \
        model.partial_pretrain=$model_path \
        model.enable_gradient_checkpointing=true \
        model.fsdp_config.cpu_offload=False \
        model.fsdp_config.offload_params=False \
        model.fsdp_config.model_dtype=bf16 \
        optim.lr=1e-5 \
        optim.betas='[0.9,0.95]' \
        optim.weight_decay=0.01 \
        optim.lr_warmup_steps_ratio=0.1 \
        optim.clip_grad=1.0 \
        optim.lr_scheduler=cosine \
        trainer.default_local_dir=$save_path \
        trainer.project_name=textcraft-sft \
        trainer.experiment_name=$experiment_name \
        trainer.logger='["console","wandb"]' \
        trainer.seed=42 \
        trainer.total_epochs=15 \
        trainer.save_freq=200 \
        trainer.test_freq=100 \
        ulysses_sequence_parallel_size=1 \
        use_remove_padding=true 2>&1 | tee -a "$LOG_FILE"
    
    return ${PIPESTATUS[0]}
}

# ============================================================================
# 训练
# ============================================================================
echo "开始训练: $DATASET_NAME"
echo "========================================"
run_training "$DATASET_NAME" "$save_path" "$model_path"

# ============================================================================
# 训练完成后自动评估
# ============================================================================

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "训练完成，退出码: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "查找最新checkpoint..." | tee -a "$LOG_FILE"
    CKPT_DIR="${save_path}"
    
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
            bash run_textcraft_eval_vllm_server.sh 2>&1 | tee -a "$LOG_FILE"
            
            EVAL_EXIT_CODE=${PIPESTATUS[0]}
            echo "" | tee -a "$LOG_FILE"
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
