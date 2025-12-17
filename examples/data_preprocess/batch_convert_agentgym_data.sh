#!/bin/bash
# Batch convert AgentGym-RL data to verl format

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Paths
AGENTGYM_DATA_DIR="/Data/wyh/datasets/AgentGym-RL-Data"
VERL_DATA_DIR="/Data/wyh/datasets/Verl-Data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environments to convert
ENVS=("webshop" "textcraft" "babyai" "alfworld" "sciworld" "searchqa")

echo "========================================"
echo "Converting AgentGym-RL Data to Verl Format"
echo "========================================"
echo ""

# Convert training data
echo "Converting training data..."
for env in "${ENVS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Processing: $env (train)"
    echo "----------------------------------------"
    
    input_file="$AGENTGYM_DATA_DIR/train/${env}_train.json"
    output_dir="$VERL_DATA_DIR/train/$env"
    
    if [ -f "$input_file" ]; then
        python3 "$SCRIPT_DIR/convert_agentgym_data.py" \
            --env="$env" \
            --input_file="$input_file" \
            --output_dir="$output_dir"
        echo "✓ $env train data converted"
    else
        echo "✗ $input_file not found, skipping"
    fi
done

echo ""
echo "========================================"
echo "Converting evaluation data..."
echo ""

# Convert evaluation data
for env in "${ENVS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Processing: $env (test)"
    echo "----------------------------------------"
    
    input_file="$AGENTGYM_DATA_DIR/eval/${env}_test.json"
    output_dir="$VERL_DATA_DIR/eval/$env"
    
    if [ -f "$input_file" ]; then
        python3 "$SCRIPT_DIR/convert_agentgym_data.py" \
            --env="$env" \
            --input_file="$input_file" \
            --output_dir="$output_dir"
        echo "✓ $env test data converted"
    else
        echo "✗ $input_file not found, skipping"
    fi
done

echo ""
echo "========================================"
echo "Conversion Complete!"
echo "========================================"
echo ""
echo "Output directory: $VERL_DATA_DIR"
echo ""
echo "Directory structure:"
tree -L 2 "$VERL_DATA_DIR" 2>/dev/null || ls -R "$VERL_DATA_DIR"

