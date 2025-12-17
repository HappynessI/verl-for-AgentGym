#!/bin/bash
# TextCraft评估 - vLLM版本（支持后台运行）
# 
# 使用方法:
#   bash run_textcraft_eval_vllm_nohup.sh
#   
# 特点:
#   - 自动使用nohup后台运行
#   - 断开连接后继续执行
#   - 日志输出到指定文件

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NOHUP_LOG="/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval/logs/nohup_vllm_${TIMESTAMP}.log"

echo "========================================================================"
echo "启动TextCraft评估（后台运行模式）"
echo "========================================================================"
echo "时间: $(date)"
echo "日志: ${NOHUP_LOG}"
echo ""
echo "提示: 可以安全断开SSH连接，进程将继续运行"
echo "========================================================================"
echo ""

# 使用nohup后台运行
nohup bash /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/run_textcraft_eval_vllm.sh > "${NOHUP_LOG}" 2>&1 &

EVAL_PID=$!
echo "✓ 评估进程已启动"
echo "  PID: ${EVAL_PID}"
echo "  日志: ${NOHUP_LOG}"
echo ""
echo "监控方法:"
echo "  1. 查看实时日志: tail -f ${NOHUP_LOG}"
echo "  2. 检查进程状态: ps -p ${EVAL_PID}"
echo "  3. 查找所有eval进程: ps aux | grep eval_textcraft"
echo ""
echo "停止评估:"
echo "  kill ${EVAL_PID}"
echo ""

# 等待2秒确认进程启动
sleep 2
if ps -p ${EVAL_PID} > /dev/null; then
    echo "✓ 进程运行正常，可以安全断开连接"
else
    echo "✗ 进程启动失败，请查看日志: ${NOHUP_LOG}"
fi

