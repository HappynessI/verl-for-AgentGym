#!/bin/bash
set -e

# ============================================================================
# 批量评估脚本 - 测试所有checkpoints和预训练模型（每个模型测试3次）
# ============================================================================
#
# 功能：
#   1. 评估所有SFT训练的checkpoints（每个3次）
#   2. 评估基线预训练模型（Qwen3-1.7B, Qwen3-4B-Instruct-2507, Qwen3-8B）（每个3次）
#   3. 使用固定随机种子（但由于环境等因素，每次结果可能略有不同）
#   4. 使用ADaPT格式配置参数
#   5. 生成统一的批量测试报告
#
# 输出：
#   - batch_test_[时间].log：详细运行日志
#   - batch_summary_[时间].txt：汇总报告
#   - [模型名]_run[1-3]/：各模型各次运行的详细结果文件夹
#   - [模型名]_run[1-3]_[时间]_results.jsonl：详细轨迹
#   - [模型名]_run[1-3]_[时间]_summary.txt：评估摘要
#
# 使用方法：
#   bash batch_eval_all_models.sh              # 跳过已评估的模型
#   bash batch_eval_all_models.sh --force      # 强制重新评估所有模型
#
# ============================================================================

# 解析命令行参数
FORCE_REEVAL=false
if [ "$1" = "--force" ]; then
    FORCE_REEVAL=true
    echo "⚠️  强制重新评估模式：将重新评估所有模型"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval/batch_eval_new"
BATCH_LOG="${BATCH_OUTPUT_DIR}/batch_test_${TIMESTAMP}.log"
BATCH_SUMMARY="${BATCH_OUTPUT_DIR}/batch_summary_${TIMESTAMP}.txt"

# 每个模型测试的次数
NUM_RUNS=3

# 创建输出目录
mkdir -p "${BATCH_OUTPUT_DIR}"

# 重定向所有输出到日志文件
exec 1> >(tee -a "${BATCH_LOG}")
exec 2>&1

echo "============================================================================"
echo "批量评估开始"
echo "时间: $(date)"
echo "输出目录: ${BATCH_OUTPUT_DIR}"
echo "============================================================================"
echo ""

# 固定随机种子
SEED=42
MAX_SAMPLES=100
TEXTCRAFT_SERVER="http://127.0.0.1:36002"
CUDA_DEVICE=3
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet"  # 使用测试集

# ADaPT格式参数
MAX_NEW_TOKENS=150
TEMPERATURE=0.0
TOP_P=1.0
DO_SAMPLE=""  # 空字符串表示不传--do_sample参数
MAX_ROUNDS=50

# 存储所有评估结果
declare -a RESULTS

# ============================================================================
# 函数：运行单个模型的单次评估
# ============================================================================
run_single_eval() {
    local MODEL_NAME="$1"
    local MODEL_PATH="$2"
    local PYTHON_SCRIPT="$3"
    local RUN_ID="$4"
    
    local SAFE_NAME=$(echo "${MODEL_NAME}" | sed 's/[\/:]/_/g')
    local MODEL_OUTPUT_DIR="${BATCH_OUTPUT_DIR}/${SAFE_NAME}"
    local EVAL_DIR="${MODEL_OUTPUT_DIR}/eval${RUN_ID}"
    
    echo "  → 运行 eval${RUN_ID}"
    
    # 检查是否已有完整评估结果（跳过逻辑）
    if [ "${FORCE_REEVAL}" = false ] && [ -d "${EVAL_DIR}" ]; then
        local LATEST_SUMMARY=$(ls -t ${EVAL_DIR}/eval_*_summary.txt 2>/dev/null | head -1)
        
        if [ -n "${LATEST_SUMMARY}" ]; then
            local TESTED_SAMPLES=$(grep "Total samples:" "${LATEST_SUMMARY}" | awk '{print $NF}')
            
            if [ "${TESTED_SAMPLES}" = "${MAX_SAMPLES}" ]; then
                local EXISTING_RATE=$(grep "Success Rate:" "${LATEST_SUMMARY}" | awk '{print $3}')
                echo "    ✓ eval${RUN_ID} 已完成 (成功率: ${EXISTING_RATE})，跳过"
                echo "${EXISTING_RATE}" > "${EVAL_DIR}/.success_rate"
                return 0
            fi
        fi
    fi
    
    # 创建eval目录
    mkdir -p "${EVAL_DIR}"
    local OUTPUT_PREFIX="${EVAL_DIR}/eval_${TIMESTAMP}"
    
    local START_TIME=$(date +%s)
    
    # 运行评估
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 "${PYTHON_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${EVAL_DIR}" \
        --textcraft_server "${TEXTCRAFT_SERVER}" \
        --max_samples ${MAX_SAMPLES} \
        --seed ${SEED} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --max_rounds ${MAX_ROUNDS} \
        ${DO_SAMPLE} \
        > "${OUTPUT_PREFIX}.log" 2>&1
    
    local EXIT_CODE=$?
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        # 查找并重命名结果文件
        local LATEST_RESULT=$(ls -t ${EVAL_DIR}/eval_results_*.jsonl 2>/dev/null | head -1)
        local LATEST_SUMMARY=$(ls -t ${EVAL_DIR}/eval_summary_*.txt 2>/dev/null | head -1)
        
        if [ -n "${LATEST_RESULT}" ]; then
            mv "${LATEST_RESULT}" "${OUTPUT_PREFIX}_results.jsonl"
            
            if [ -n "${LATEST_SUMMARY}" ]; then
                mv "${LATEST_SUMMARY}" "${OUTPUT_PREFIX}_summary.txt"
                local SUCCESS_RATE=$(grep "Success Rate" "${OUTPUT_PREFIX}_summary.txt" | awk '{print $NF}')
                echo "    ✓ eval${RUN_ID} 完成 (成功率: ${SUCCESS_RATE}, 耗时: ${DURATION}s)"
                echo "${SUCCESS_RATE}" > "${EVAL_DIR}/.success_rate"
                return 0
            fi
        fi
        echo "    ⚠ eval${RUN_ID} 完成但未找到结果文件"
        return 1
    else
        echo "    ❌ eval${RUN_ID} 失败 (退出码: ${EXIT_CODE})"
        return 1
    fi
}

# ============================================================================
# 函数：运行单个模型的完整评估（包含3次运行和汇总）
# ============================================================================
run_eval() {
    local MODEL_NAME="$1"
    local MODEL_PATH="$2"
    local PYTHON_SCRIPT="$3"
    
    echo ""
    echo "========================================================================"
    echo "评估模型: ${MODEL_NAME}"
    echo "模型路径: ${MODEL_PATH}"
    echo "========================================================================"
    
    # 检查模型路径是否存在
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "❌ 模型路径不存在，跳过: ${MODEL_PATH}"
        RESULTS+=("${MODEL_NAME}|SKIP|模型路径不存在")
        return 1
    fi
    
    local SAFE_NAME=$(echo "${MODEL_NAME}" | sed 's/[\/:]/_/g')
    local MODEL_OUTPUT_DIR="${BATCH_OUTPUT_DIR}/${SAFE_NAME}"
    mkdir -p "${MODEL_OUTPUT_DIR}"
    
    echo "开始时间: $(date)"
    local TOTAL_START_TIME=$(date +%s)
    
    # 运行3次评估
    local SUCCESS_COUNT=0
    for RUN_ID in $(seq 1 ${NUM_RUNS}); do
        if run_single_eval "${MODEL_NAME}" "${MODEL_PATH}" "${PYTHON_SCRIPT}" "${RUN_ID}"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    done
    
    local TOTAL_END_TIME=$(date +%s)
    local TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
    
    echo "结束时间: $(date)"
    echo "总耗时: ${TOTAL_DURATION}秒"
    
    # 生成汇总文件
    local SUMMARY_FILE="${MODEL_OUTPUT_DIR}/summary.txt"
    {
        echo "============================================================================"
        echo "模型评估汇总: ${MODEL_NAME}"
        echo "============================================================================"
        echo ""
        echo "评估时间: $(date)"
        echo "模型路径: ${MODEL_PATH}"
        echo "总耗时: ${TOTAL_DURATION}秒"
        echo "成功评估次数: ${SUCCESS_COUNT}/${NUM_RUNS}"
        echo ""
        echo "各次评估结果:"
        echo "----------------------------------------------------------------------------"
        
        local RATE_SUM=0
        local VALID_COUNT=0
        for RUN_ID in $(seq 1 ${NUM_RUNS}); do
            local EVAL_DIR="${MODEL_OUTPUT_DIR}/eval${RUN_ID}"
            if [ -f "${EVAL_DIR}/.success_rate" ]; then
                local RATE=$(cat "${EVAL_DIR}/.success_rate")
                echo "  eval${RUN_ID}: ${RATE}"
                # 计算平均值（假设格式是0.XX）
                RATE_NUM=$(echo "${RATE}" | sed 's/[^0-9.]//g')
                RATE_SUM=$(echo "${RATE_SUM} + ${RATE_NUM}" | bc -l)
                VALID_COUNT=$((VALID_COUNT + 1))
            else
                echo "  eval${RUN_ID}: 未完成或失败"
            fi
        done
        
        echo ""
        if [ ${VALID_COUNT} -gt 0 ]; then
            local AVG_RATE=$(echo "scale=4; ${RATE_SUM} / ${VALID_COUNT}" | bc -l)
            echo "平均成功率: ${AVG_RATE}"
            RESULTS+=("${MODEL_NAME}|SUCCESS|${AVG_RATE}(avg)|${TOTAL_DURATION}s")
        else
            echo "平均成功率: N/A (无有效结果)"
            RESULTS+=("${MODEL_NAME}|FAIL|N/A|${TOTAL_DURATION}s")
        fi
        
        echo ""
        echo "详细结果位置:"
        for RUN_ID in $(seq 1 ${NUM_RUNS}); do
            echo "  eval${RUN_ID}/: ${MODEL_OUTPUT_DIR}/eval${RUN_ID}/"
        done
        echo ""
        echo "============================================================================"
    } | tee "${SUMMARY_FILE}"
    
    echo "📊 汇总文件: ${SUMMARY_FILE}"
    echo ""
}

# ============================================================================
# 1. 评估所有SFT Checkpoints
# ============================================================================
echo ""
echo "========================================================================"
echo "第一阶段: 评估SFT Checkpoints"
echo "========================================================================"

CKPT_BASE_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/new_ckp"

if [ -d "${CKPT_BASE_DIR}" ]; then
    # 查找所有checkpoint（按global_step排序），使用huggingface子目录（FSDP转换后的完整HF格式）
    CHECKPOINTS=$(find ${CKPT_BASE_DIR} -type d -name "global_step_*" 2>/dev/null | sort -V | sed 's|$|/huggingface|')
    
    if [ -n "${CHECKPOINTS}" ]; then
        CKPT_COUNT=$(echo "${CHECKPOINTS}" | wc -l)
        echo "找到 ${CKPT_COUNT} 个checkpoints，每个测试 ${NUM_RUNS} 次"
        echo ""
        
        for CKPT_PATH in ${CHECKPOINTS}; do
            # 获取上一级目录名（global_step_XXX），而不是huggingface
            CKPT_NAME=$(basename "$(dirname "${CKPT_PATH}")")
            run_eval "SFT-${CKPT_NAME}" "${CKPT_PATH}" "/Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_1.7b.py"
        done
    else
        echo "未找到checkpoints"
    fi
else
    echo "Checkpoint目录不存在: ${CKPT_BASE_DIR}"
fi

# ============================================================================
# 2. 评估预训练模型
# ============================================================================
echo ""
echo "========================================================================"
echo "第二阶段: 评估预训练模型，每个测试 ${NUM_RUNS} 次"
echo "========================================================================"

# Qwen3-1.7B
run_eval "Qwen3-1.7B" \
    "/Data/public/Qwen3-1.7B" \
    "/Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_1.7b.py"

# Qwen3-4B-Instruct-2507
run_eval "Qwen3-4B-Instruct-2507" \
    "/Data/public/Qwen3-4B-Instruct-2507" \
    "/Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_1.7b.py"

# Qwen3-8B
run_eval "Qwen3-8B" \
    "/Data/public/Qwen3-8B" \
    "/Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_textcraft_qwen3_8b.py"

# ============================================================================
# 生成总结报告
# ============================================================================
echo ""
echo "========================================================================"
echo "生成总结报告"
echo "========================================================================"

{
    echo "============================================================================"
    echo "批量评估总结报告"
    echo "============================================================================"
    echo ""
    echo "评估时间: $(date)"
    echo "随机种子: ${SEED}"
    echo "数据集: ${DATA_PATH}"
    echo "测试样本数: ${MAX_SAMPLES}"
    echo "TextCraft服务器: ${TEXTCRAFT_SERVER}"
    echo ""
    echo "ADaPT配置参数:"
    echo "  - max_new_tokens: ${MAX_NEW_TOKENS}"
    echo "  - temperature: ${TEMPERATURE}"
    echo "  - top_p: ${TOP_P}"
    echo "  - max_rounds: ${MAX_ROUNDS}"
    echo "  - do_sample: False"
    echo ""
    echo "============================================================================"
    echo "评估结果"
    echo "============================================================================"
    echo ""
    printf "%-50s %-10s %-15s %-10s\n" "模型名称" "状态" "成功率" "耗时"
    echo "----------------------------------------------------------------------------"
    
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r name status rate duration <<< "$result"
        printf "%-50s %-10s %-15s %-10s\n" "$name" "$status" "$rate" "$duration"
    done
    
    echo ""
    echo "状态说明:"
    echo "  SUCCESS - 本次新评估成功"
    echo "  CACHED  - 使用已有评估结果（跳过）"
    echo "  FAIL    - 评估失败"
    echo "  SKIP    - 跳过（模型不存在等原因）"
    
    echo ""
    echo "============================================================================"
    echo "详细结果文件位置"
    echo "============================================================================"
    echo ""
    echo "输出目录: ${BATCH_OUTPUT_DIR}"
    echo "批量日志: ${BATCH_LOG}"
    echo "批量摘要: ${BATCH_SUMMARY}"
    echo ""
    echo "各模型结果文件夹:"
    for dir in ${BATCH_OUTPUT_DIR}/*/; do
        if [ -d "$dir" ]; then
            local MODEL_DIR_NAME=$(basename "$dir")
            echo "  ${MODEL_DIR_NAME}/"
            
            # 显示summary文件
            if [ -f "${dir}summary.txt" ]; then
                echo "    - summary.txt (汇总)"
            fi
            
            # 显示各eval子目录
            for eval_dir in ${dir}eval*/; do
                if [ -d "$eval_dir" ]; then
                    local EVAL_NAME=$(basename "$eval_dir")
                    echo "    - ${EVAL_NAME}/"
                    ls ${eval_dir}*.log ${eval_dir}*.jsonl ${eval_dir}*.txt 2>/dev/null | sed 's|.*/|      + |'
                fi
            done
        fi
    done
    echo ""
    echo "============================================================================"
    
} | tee "${BATCH_SUMMARY}"

echo ""
echo "============================================================================"
echo "批量评估完成！"
echo "============================================================================"
echo ""
echo "📊 总结报告: ${BATCH_SUMMARY}"
echo "📝 详细日志: ${BATCH_LOG}"
echo ""

