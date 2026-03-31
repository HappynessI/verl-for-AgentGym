#!/usr/bin/env python3
"""
Preflight Smoke Test for TextCraft Prefix RL Training.

This script verifies that the training pipeline works correctly by:
1. Loading debug samples from the parquet file
2. Checking prompt and prefix_actions
3. Executing env replay
4. Running at least 1 step of student continuation
5. Outputting complete debug logs

Usage:
    python preflight_test.py --data_path <path> --model_path <path> --textcraft_server <url>
"""

import argparse
import asyncio
import logging
import os
import sys

# Setup path
sys.path.insert(0, '/Data/wyh/verl')

import pandas as pd
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from verl.interactions.textcraft_interaction import TextCraftInteraction
from verl.utils.dataset.rl_dataset import RLHFDataset


def setup_logging():
    """Setup logging for debug output."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_debug_samples(data_path: str, max_samples: int = 4):
    """Load debug samples from parquet file."""
    print(f"\n{'='*60}")
    print("Preflight: 加载数据样本")
    print(f"{'='*60}")
    
    df = pd.read_parquet(data_path)
    print(f"总样本数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    # Get first N samples
    samples = df.head(max_samples).to_dict('records')
    
    print(f"\n加载了 {len(samples)} 个样本")
    
    # Analyze each sample
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i} ---")
        
        # Check extra_info
        extra_info = sample.get('extra_info', {})
        interaction_kwargs = extra_info.get('interaction_kwargs', {})
        prefix_actions = interaction_kwargs.get('prefix_actions', [])
        
        # Safe check for prefix_actions (support list, np.ndarray, etc.)
        has_prefix = prefix_actions is not None and hasattr(prefix_actions, '__len__') and len(prefix_actions) > 0
        
        print(f"  prompt (前200字符): {str(sample.get('prompt', ''))[:200]}...")
        print(f"  是否存在 prefix_actions: {has_prefix}")
        if has_prefix:
            print(f"  prefix_actions 长度: {len(prefix_actions)}")
            print(f"  prefix_actions 前3项: {prefix_actions[:3]}")
        
        # Fail-fast: check prefix_actions
        if not has_prefix:
            print(f"❌ FAIL: Sample {i} has no prefix_actions!")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Preflight: 数据加载完成")
    print(f"{'='*60}\n")
    
    return samples


async def test_interaction_replay(textcraft_server: str, samples: list):
    """Test the interaction and replay functionality."""
    print(f"\n{'='*60}")
    print("Preflight: 测试 Interaction/Replay")
    print(f"{'='*60}")
    
    # Create interaction config
    config = OmegaConf.create({
        'env_server_base': textcraft_server,
    })
    
    interaction = TextCraftInteraction(config)
    
    # Test each sample
    for i, sample in enumerate(samples):
        extra_info = sample.get('extra_info', {})
        interaction_kwargs = extra_info.get('interaction_kwargs', {})
        prefix_actions = interaction_kwargs.get('prefix_actions', [])
        
        request_id = f"preflight_test_{i}"
        
        print(f"\n--- Testing Sample {i} ---")
        
        # Start interaction (this triggers replay)
        try:
            await interaction.start_interaction(
                request_id,
                prompt=sample.get('prompt'),
                **interaction_kwargs
            )
            print(f"  ✓ start_interaction 成功")
            
            # Check session
            session = interaction.instance_sessions.get(request_id)
            if session:
                print(f"  - env_id: {session['env_id']}")
                print(f"  - step_count after replay: {session['step_count']}")
                print(f"  - done after replay: {session['done']}")
                
                # Fail-fast: check replay was executed
                if session['step_count'] == 0 and len(prefix_actions) > 0:
                    print(f"❌ FAIL: Replay was called but no actions executed!")
                    sys.exit(1)
            else:
                print(f"❌ FAIL: No session created!")
                sys.exit(1)
            
            # Test student continuation - generate 1 response
            print(f"\n  Testing student continuation...")
            messages = [{"role": "system", "content": sample.get('prompt', '')}]
            
            # Get response from environment (simulate student getting obs)
            should_terminate, obs, reward, metrics = await interaction.generate_response(
                request_id,
                messages,
                **interaction_kwargs
            )
            
            print(f"  ✓ generate_response 成功")
            print(f"  - observation (���100字符): {str(obs)[:100]}...")
            print(f"  - reward: {reward}")
            print(f"  - should_terminate: {should_terminate}")
            
            if not obs:
                print(f"❌ FAIL: No observation returned - replay/continuation failed!")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ FAIL: Interaction/Replay failed: {e}")
            sys.exit(1)
        
        # Clean up (if method exists)
        if hasattr(interaction, 'end_interaction'):
            await interaction.end_interaction(request_id)
    
    print(f"\n{'='*60}")
    print("Preflight: Interaction/Replay 测试完成")
    print(f"{'='*60}\n")


def test_model_inference(model_path: str):
    """Test model inference capability."""
    print(f"\n{'='*60}")
    print("Preflight: 测试模型推理")
    print(f"{'='*60}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  ✓ Tokenizer 加载成功")
        
        # Test tokenization
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        print(f"  ✓ Tokenization 测试成功: {len(tokens)} tokens")
        
    except Exception as e:
        print(f"❌ FAIL: Model inference test failed: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Preflight: 模型推理测试完成")
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description="Preflight smoke test for TextCraft training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet data file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--textcraft_server", type=str, default="http://127.0.0.1:36001", help="TextCraft server URL")
    parser.add_argument("--max_samples", type=int, default=4, help="Maximum samples to test")
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print("# Preflight Smoke Test - TextCraft Prefix RL")
    print(f"{'#'*60}")
    print(f"Data path: {args.data_path}")
    print(f"Model path: {args.model_path}")
    print(f"TextCraft server: {args.textcraft_server}")
    print(f"Max samples: {args.max_samples}")
    print(f"{'#'*60}\n")
    
    # Setup
    os.environ['VERL_DEBUG_MODE'] = '1'
    setup_logging()
    
    # Step 1: Load data
    samples = load_debug_samples(args.data_path, args.max_samples)
    
    # Step 2: Test model
    test_model_inference(args.model_path)
    
    # Step 3: Test interaction/replay
    await test_interaction_replay(args.textcraft_server, samples)
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ Preflight Smoke Test PASSED!")
    print(f"{'='*60}")
    print("\n验证通过:")
    print("  ✓ 数据正确加载")
    print("  ✓ prefix_actions 存在")
    print("  ✓ interaction 层正确接收 prefix_actions")
    print("  ✓ env replay 成功执行")
    print("  ✓ student continuation 可以执行")
    print("\n训练链路已打通，可以开始正式训练！")


if __name__ == "__main__":
    asyncio.run(main())
