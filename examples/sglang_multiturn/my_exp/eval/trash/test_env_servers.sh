#!/bin/bash
# 测试所有AgentGym环境服务器的连接性

echo "========================================"
echo "Testing AgentGym Environment Servers"
echo "========================================"
echo ""

# 定义环境和端口
declare -A ENVS
ENVS=(
    ["webshop"]="36003"
    ["babyai"]="36001"
    ["alfworld"]="36002"
    ["sciworld"]="36004"
    ["sqlgym"]="36005"
    ["textcraft"]="36006"
    ["searchqa"]="36007"
)

# 测试每个环境
SUCCESS_COUNT=0
FAIL_COUNT=0

for env in "${!ENVS[@]}"; do
    port="${ENVS[$env]}"
    url="http://127.0.0.1:${port}"
    
    printf "%-15s (port %s): " "$env" "$port"
    
    if curl -s --connect-timeout 2 "$url/" > /dev/null 2>&1; then
        echo "✓ OK"
        ((SUCCESS_COUNT++))
    else
        echo "✗ FAILED"
        ((FAIL_COUNT++))
        echo "  To start: cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-$env"
        echo "            python -m uvicorn agentenv_$env.server:app --host 0.0.0.0 --port $port"
        echo ""
    fi
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Running:  $SUCCESS_COUNT / $((SUCCESS_COUNT + FAIL_COUNT))"
echo "Failed:   $FAIL_COUNT / $((SUCCESS_COUNT + FAIL_COUNT))"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓ All servers are running!"
    exit 0
else
    echo "✗ Some servers are not running."
    echo "Please start the failed servers before running evaluation."
    exit 1
fi

