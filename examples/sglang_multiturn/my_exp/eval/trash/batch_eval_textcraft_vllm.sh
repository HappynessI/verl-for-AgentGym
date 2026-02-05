#!/bin/bash
# TextCraft批量评估脚本 - 测试不同的采样参数
# 
# 使用方法:
#   bash batch_eval_textcraft_vllm.sh
#
# 功能:
#   - 测试多组temperature和top_p参数组合
#   - 每组参数重复采样4次
#   - 自动汇总所有结果

set -e

# ============================================================================
# 配置区域
# ============================================================================

# 基础配置
MODEL_PATH="${MODEL_PATH:-/Data/public/Qwen3-1.7B}"
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet"
BASE_OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval_batch"
TEXTCRAFT_SERVER="http://127.0.0.1:36003"

# GPU配置
export CUDA_VISIBLE_DEVICES=2
GPU_MEMORY_UTILIZATION=0.9

# 评估配置
MAX_SAMPLES=-1  # -1表示全部样本
NUM_SAMPLES_PER_TASK=4  # 每个任务采样4次
MAX_ROUNDS=25
MAX_LENGTH=8192
MAX_NEW_TOKENS=512
SEED=42

# 定义要测试的参数组合（温度和top_p）
# 格式: "temperature,top_p,描述"
PARAM_COMBINATIONS=(
    "0.0,1.0,greedy"           # 贪心解码
    "0.3,0.95,low_temp"        # 低温度
    "0.5,0.95,medium_temp"     # 中等温度
    "0.7,0.95,high_temp"       # 高温度
    "0.7,0.9,high_temp_low_p"  # 高温度+低top_p
)

# ============================================================================
# 脚本开始
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="$BASE_OUTPUT_DIR/batch_$TIMESTAMP"
mkdir -p "$BATCH_DIR"

SUMMARY_FILE="$BATCH_DIR/batch_summary.txt"
RESULTS_CSV="$BATCH_DIR/batch_results.csv"

echo "================================================================================" | tee "$SUMMARY_FILE"
echo "TextCraft批量评估 - 参数网格搜索" | tee -a "$SUMMARY_FILE"
echo "================================================================================" | tee -a "$SUMMARY_FILE"
echo "时间: $(date)" | tee -a "$SUMMARY_FILE"
echo "模型: $MODEL_PATH" | tee -a "$SUMMARY_FILE"
echo "数据: $DATA_PATH" | tee -a "$SUMMARY_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$SUMMARY_FILE"
echo "每任务采样次数: $NUM_SAMPLES_PER_TASK" | tee -a "$SUMMARY_FILE"
echo "参数组合数量: ${#PARAM_COMBINATIONS[@]}" | tee -a "$SUMMARY_FILE"
echo "批次输出目录: $BATCH_DIR" | tee -a "$SUMMARY_FILE"
echo "================================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# 创建CSV表头
echo "experiment_id,temperature,top_p,description,task_level_success_rate,sample_level_success_rate,avg_reward,total_samples,timestamp,output_dir" > "$RESULTS_CSV"

# 初始化计数器
total_experiments=${#PARAM_COMBINATIONS[@]}
current_experiment=0

# 切换到工作目录
cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

# 循环测试每组参数
for param_combo in "${PARAM_COMBINATIONS[@]}"; do
    current_experiment=$((current_experiment + 1))
    
    # 解析参数
    IFS=',' read -r TEMPERATURE TOP_P DESCRIPTION <<< "$param_combo"
    
    # 创建实验ID
    EXP_ID="exp_$(printf '%02d' $current_experiment)_${DESCRIPTION}"
    EXP_OUTPUT_DIR="$BATCH_DIR/$EXP_ID"
    mkdir -p "$EXP_OUTPUT_DIR"
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "================================================================================" | tee -a "$SUMMARY_FILE"
    echo "实验 $current_experiment/$total_experiments: $EXP_ID" | tee -a "$SUMMARY_FILE"
    echo "================================================================================" | tee -a "$SUMMARY_FILE"
    echo "参数配置:" | tee -a "$SUMMARY_FILE"
    echo "  Temperature: $TEMPERATURE" | tee -a "$SUMMARY_FILE"
    echo "  Top-P: $TOP_P" | tee -a "$SUMMARY_FILE"
    echo "  Description: $DESCRIPTION" | tee -a "$SUMMARY_FILE"
    echo "  输出目录: $EXP_OUTPUT_DIR" | tee -a "$SUMMARY_FILE"
    echo "开始时间: $(date)" | tee -a "$SUMMARY_FILE"
    echo "================================================================================" | tee -a "$SUMMARY_FILE"
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 运行评估
    EXP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$EXP_OUTPUT_DIR/eval_${EXP_TIMESTAMP}.log"
    
    python examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_1.7b_vllm.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$EXP_OUTPUT_DIR" \
        --textcraft_server "$TEXTCRAFT_SERVER" \
        --max_samples "$MAX_SAMPLES" \
        --num_samples_per_task "$NUM_SAMPLES_PER_TASK" \
        --max_rounds "$MAX_ROUNDS" \
        --max_length "$MAX_LENGTH" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --seed "$SEED" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        2>&1 | tee "$LOG_FILE"
    
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "完成时间: $(date)" | tee -a "$SUMMARY_FILE"
    echo "耗时: ${DURATION}秒 ($(($DURATION / 60))分钟)" | tee -a "$SUMMARY_FILE"
    
    # 提取评估结果（从summary文件中）
    SUMMARY_PATH=$(ls -t "$EXP_OUTPUT_DIR"/eval_summary_*.txt 2>/dev/null | head -1)
    
    if [ -f "$SUMMARY_PATH" ]; then
        echo "提取评估指标..." | tee -a "$SUMMARY_FILE"
        
        # 提取Task-Level和Sample-Level的Success Rate
        TASK_SUCCESS=$(grep "Task-Level Metrics" -A 1 "$SUMMARY_PATH" | grep "Success Rate:" | awk '{print $3}' || echo "N/A")
        SAMPLE_SUCCESS=$(grep "Sample-Level Metrics" -A 1 "$SUMMARY_PATH" | grep "Success Rate:" | awk '{print $3}' || echo "N/A")
        AVG_REWARD=$(grep "Average Reward:" "$SUMMARY_PATH" | tail -1 | awk '{print $3}' || echo "N/A")
        TOTAL_SAMPLES=$(grep "Total samples:" "$SUMMARY_PATH" | awk '{print $3}' || echo "N/A")
        
        echo "  Task-Level Success Rate: $TASK_SUCCESS" | tee -a "$SUMMARY_FILE"
        echo "  Sample-Level Success Rate: $SAMPLE_SUCCESS" | tee -a "$SUMMARY_FILE"
        echo "  Average Reward: $AVG_REWARD" | tee -a "$SUMMARY_FILE"
        echo "  Total Samples: $TOTAL_SAMPLES" | tee -a "$SUMMARY_FILE"
        
        # 写入CSV
        echo "$EXP_ID,$TEMPERATURE,$TOP_P,$DESCRIPTION,$TASK_SUCCESS,$SAMPLE_SUCCESS,$AVG_REWARD,$TOTAL_SAMPLES,$EXP_TIMESTAMP,$EXP_OUTPUT_DIR" >> "$RESULTS_CSV"
    else
        echo "  警告: 未找到summary文件" | tee -a "$SUMMARY_FILE"
        echo "$EXP_ID,$TEMPERATURE,$TOP_P,$DESCRIPTION,N/A,N/A,N/A,N/A,$EXP_TIMESTAMP,$EXP_OUTPUT_DIR" >> "$RESULTS_CSV"
    fi
    
    echo "================================================================================" | tee -a "$SUMMARY_FILE"
done

# ============================================================================
# 汇总结果
# ============================================================================

echo "" | tee -a "$SUMMARY_FILE"
echo "================================================================================" | tee -a "$SUMMARY_FILE"
echo "批量评估完成汇总" | tee -a "$SUMMARY_FILE"
echo "================================================================================" | tee -a "$SUMMARY_FILE"
echo "完成时间: $(date)" | tee -a "$SUMMARY_FILE"
echo "总实验数: $total_experiments" | tee -a "$SUMMARY_FILE"
echo "结果保存在: $BATCH_DIR" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "结果汇总 (CSV格式):" | tee -a "$SUMMARY_FILE"
echo "-------------------------------------------------------------------------------" | tee -a "$SUMMARY_FILE"
cat "$RESULTS_CSV" | tee -a "$SUMMARY_FILE"
echo "================================================================================" | tee -a "$SUMMARY_FILE"

# 生成可视化的结果表格
echo "" | tee -a "$SUMMARY_FILE"
echo "结果对比表格:" | tee -a "$SUMMARY_FILE"
echo "-------------------------------------------------------------------------------" | tee -a "$SUMMARY_FILE"
printf "%-20s | %-8s | %-8s | %-15s | %-15s\n" "Experiment" "Temp" "Top-P" "Task Success" "Sample Success" | tee -a "$SUMMARY_FILE"
echo "-------------------------------------------------------------------------------" | tee -a "$SUMMARY_FILE"

tail -n +2 "$RESULTS_CSV" | while IFS=',' read -r exp_id temp top_p desc task_sr sample_sr reward samples timestamp outdir; do
    printf "%-20s | %-8s | %-8s | %-15s | %-15s\n" "$desc" "$temp" "$top_p" "$task_sr" "$sample_sr" | tee -a "$SUMMARY_FILE"
done

echo "================================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "✓ 批量评估完成！" | tee -a "$SUMMARY_FILE"
echo "✓ 汇总文件: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "✓ CSV结果: $RESULTS_CSV" | tee -a "$SUMMARY_FILE"
echo "✓ 详细结果: $BATCH_DIR" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# 显示GPU最终状态
echo "GPU最终状态:" | tee -a "$SUMMARY_FILE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$SUMMARY_FILE"

