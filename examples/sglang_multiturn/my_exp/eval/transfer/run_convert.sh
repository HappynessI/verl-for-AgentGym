#!/bin/bash
# 转换textcraft轨迹格式的脚本
# 将adapt格式转换为box format格式

# 默认路径配置
INPUT_FILE="/Data/wyh/datasets/Sampling-Data/textcraft_次gemini-3-pro-preview_20251216_101619/textcraft_trajectories.jsonl"
OUTPUT_DIR="/Data/wyh/datasets/Sampling-Data/textcraft_次gemini-3-pro-preview_20251216_101619"
OUTPUT_FILE="${OUTPUT_DIR}/textcraft_trajectories_boxformat.jsonl"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --keep-failed)
            KEEP_FAILED="--keep-failed"
            shift
            ;;
        --no-replace-prompt)
            NO_REPLACE="--no-replace-prompt"
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -i, --input FILE       Input jsonl file (default: gemini轨迹文件)"
            echo "  -o, --output FILE      Output jsonl file (default: 同目录下_boxformat.jsonl)"
            echo "  --keep-failed          Keep failed trajectories"
            echo "  --no-replace-prompt    Do not replace system prompt"
            echo "  -v, --verbose          Verbose output"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "Textcraft轨迹格式转换"
echo "============================================"
echo "Input:  ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "============================================"

python3 "${SCRIPT_DIR}/convert_adapt_to_boxformat.py" \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}" \
    ${KEEP_FAILED} \
    ${NO_REPLACE} \
    ${VERBOSE}

echo "============================================"
echo "转换完成!"
echo "============================================"

