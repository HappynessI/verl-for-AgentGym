#!/bin/bash
# 批量转换FSDP checkpoints为HuggingFace格式
# 
# 使用方法:
#   bash convert_fsdp_checkpoints.sh

set -e

CKPT_BASE_DIR="/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/new_ckp"

echo "============================================================================"
echo "批量转换FSDP Checkpoints为HuggingFace格式"
echo "============================================================================"
echo "Checkpoint目录: ${CKPT_BASE_DIR}"
echo ""

cd /Data/wyh/verl
source ~/miniconda3/bin/activate verl

# 查找所有global_step_*目录
CHECKPOINTS=$(find ${CKPT_BASE_DIR} -maxdepth 1 -type d -name "global_step_*" 2>/dev/null | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ 未找到任何checkpoint"
    exit 1
fi

CKPT_COUNT=$(echo "$CHECKPOINTS" | wc -l)
echo "找到 ${CKPT_COUNT} 个checkpoints"
echo ""

for CKPT_DIR in $CHECKPOINTS; do
    CKPT_NAME=$(basename ${CKPT_DIR})
    TEMP_TARGET_DIR="${CKPT_DIR}/huggingface_merged"
    FINAL_TARGET_DIR="${CKPT_DIR}/huggingface"
    
    echo "========================================================================"
    echo "转换: ${CKPT_NAME}"
    echo "源目录: ${CKPT_DIR}"
    echo "目标目录: ${FINAL_TARGET_DIR}"
    echo "========================================================================"
    
    # 检查是否已转换（检查最终目录是否有完整模型）
    if [ -d "${FINAL_TARGET_DIR}" ] && \
       { [ -f "${FINAL_TARGET_DIR}/pytorch_model.bin" ] || \
         [ -f "${FINAL_TARGET_DIR}/model.safetensors" ] || \
         [ -f "${FINAL_TARGET_DIR}/model-00001-of-00002.safetensors" ]; }; then
        echo "✓ 已存在转换后的完整模型，跳过"
        echo ""
        continue
    fi
    
    # 检查FSDP文件是否存在
    if [ ! -f "${CKPT_DIR}/fsdp_config.json" ]; then
        echo "⚠ 未找到fsdp_config.json，跳过"
        echo ""
        continue
    fi
    
    # 转换到临时目录
    START_TIME=$(date +%s)
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${CKPT_DIR}" \
        --target_dir "${TEMP_TARGET_DIR}"
    
    CONVERT_EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ ${CONVERT_EXIT_CODE} -eq 0 ]; then
        echo "✓ 转换完成，耗时: ${DURATION}秒"
        
        # 验证转换后的模型文件存在
        if [ -f "${TEMP_TARGET_DIR}/config.json" ] && \
           { [ -f "${TEMP_TARGET_DIR}/pytorch_model.bin" ] || \
             [ -f "${TEMP_TARGET_DIR}/model.safetensors" ] || \
             [ -f "${TEMP_TARGET_DIR}/model-00001-of-00002.safetensors" ]; }; then
            
            echo "  验证转换结果..."
            
            # 删除旧的huggingface目录（只有config和tokenizer）
            if [ -d "${FINAL_TARGET_DIR}" ]; then
                echo "  删除旧的huggingface目录（仅config/tokenizer）..."
                rm -rf "${FINAL_TARGET_DIR}"
            fi
            
            # 重命名为最终目录
            echo "  重命名 huggingface_merged -> huggingface ..."
            mv "${TEMP_TARGET_DIR}" "${FINAL_TARGET_DIR}"
            
            # 删除FSDP分片文件以节省空间
            echo "  删除FSDP分片文件以节省空间..."
            rm -f "${CKPT_DIR}"/model_world_size_*.pt
            rm -f "${CKPT_DIR}"/optim_world_size_*.pt
            rm -f "${CKPT_DIR}"/extra_state_world_size_*.pt
            
            # 计算最终模型大小
            SAVED_SPACE=$(du -sh "${FINAL_TARGET_DIR}" | cut -f1)
            echo "  ✓ 转换完成，模型大小: ${SAVED_SPACE}"
        else
            echo "  ⚠ 转换后的模型文件验证失败，保留原始分片"
        fi
    else
        echo "✗ 转换失败（退出码: ${CONVERT_EXIT_CODE}），保留原始分片"
    fi
    echo ""
done

echo "============================================================================"
echo "所有checkpoint转换完成！"
echo "============================================================================"

