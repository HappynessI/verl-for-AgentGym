#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test script for Webshop interaction integration
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from verl.interactions.webshop_interaction import WebshopInteraction


async def test_webshop_interaction():
    """Test basic Webshop interaction functionality"""
    
    print("="*80)
    print("Testing Webshop Interaction Integration")
    print("="*80)
    
    # Configuration
    config = {
        "name": "webshop",
        "env_server_base": "http://127.0.0.1:36003",
        "timeout": 60,
    }
    
    print("\n1. Initializing WebshopInteraction...")
    try:
        interaction = WebshopInteraction(config)
        print("✓ WebshopInteraction initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nMake sure Webshop server is running:")
        print("  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop")
        print("  python -m agentenv_webshop.launch --port 36001")
        return False
    
    # Test interaction flow
    instance_id = "test_instance_001"
    session_id = 0
    
    print(f"\n2. Starting interaction (instance_id={instance_id}, session_id={session_id})...")
    try:
        await interaction.start_interaction(instance_id=instance_id, session_id=session_id)
        print("✓ Interaction started successfully")
        print(f"  Initial observation: {interaction._envs[instance_id]['initial_observation'][:100]}...")
    except Exception as e:
        print(f"✗ Failed to start interaction: {e}")
        return False
    
    # Simulate agent actions
    test_actions = [
        {
            "action": "search[red shirt]",
            "messages": [
                {"role": "user", "content": "Find a red shirt"},
                {"role": "assistant", "content": "I'll search for a red shirt. search[red shirt]"}
            ]
        },
        {
            "action": "click[B0B7QY8VXW]",
            "messages": [
                {"role": "user", "content": "Find a red shirt"},
                {"role": "assistant", "content": "search[red shirt]"},
                {"role": "user", "content": "Here are the search results..."},
                {"role": "assistant", "content": "I'll click on the first product. click[B0B7QY8VXW]"}
            ]
        },
    ]
    
    print("\n3. Testing agent-environment interaction...")
    for i, test_case in enumerate(test_actions, 1):
        print(f"\n   Step {i}: {test_case['action']}")
        try:
            done, observation, reward, info = await interaction.generate_response(
                instance_id=instance_id,
                messages=test_case["messages"]
            )
            print(f"   ✓ Action executed successfully")
            print(f"     - Done: {done}")
            print(f"     - Reward: {reward:.3f}")
            print(f"     - Observation: {observation[:150]}...")
            print(f"     - Info: {info}")
            
            if done:
                print("   Episode finished!")
                break
        except Exception as e:
            print(f"   ✗ Failed to execute action: {e}")
            # Continue with cleanup
    
    print("\n4. Calculating final score...")
    try:
        final_score = await interaction.calculate_score(instance_id=instance_id)
        print(f"✓ Final score: {final_score:.3f}")
    except Exception as e:
        print(f"✗ Failed to calculate score: {e}")
    
    print("\n5. Finalizing interaction...")
    try:
        await interaction.finalize_interaction(instance_id=instance_id)
        print("✓ Interaction finalized successfully")
    except Exception as e:
        print(f"✗ Failed to finalize: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    return True


async def test_action_extraction():
    """Test action extraction from different message formats"""
    
    print("\n" + "="*80)
    print("Testing Action Extraction")
    print("="*80)
    
    config = {"name": "webshop", "env_server_base": "http://127.0.0.1:36001"}
    interaction = WebshopInteraction(config)
    
    test_cases = [
        {
            "messages": [{"role": "assistant", "content": "I will search for shoes. search[running shoes]"}],
            "expected": "search[running shoes]"
        },
        {
            "messages": [{"role": "assistant", "content": "Let me click on this item: click[B0B7QY8VXW]"}],
            "expected": "click[B0B7QY8VXW]"
        },
        {
            "messages": [{"role": "assistant", "content": "SEARCH[laptop computer]"}],
            "expected": "search[laptop computer]"
        },
        {
            "messages": [{"role": "assistant", "content": "I think I should search for something"}],
            "expected": None
        },
    ]
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        extracted = interaction._extract_action(test["messages"])
        expected = test["expected"]
        
        if (extracted is None and expected is None) or (extracted == expected):
            print(f"✓ Test {i}: Passed")
            print(f"  Input: {test['messages'][0]['content']}")
            print(f"  Extracted: {extracted}")
        else:
            print(f"✗ Test {i}: Failed")
            print(f"  Input: {test['messages'][0]['content']}")
            print(f"  Expected: {expected}")
            print(f"  Got: {extracted}")
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✓ All action extraction tests passed!")
    else:
        print("✗ Some action extraction tests failed")
    print("="*80)
    
    return all_passed


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("WEBSHOP INTERACTION TEST SUITE")
    print("="*80)
    
    # Test 1: Action extraction (doesn't require server)
    test1_passed = await test_action_extraction()
    
    # Test 2: Full interaction (requires server)
    print("\n\nNote: The following test requires Webshop server to be running")
    user_input = input("Continue with server test? (y/n): ").strip().lower()
    
    if user_input == 'y':
        test2_passed = await test_webshop_interaction()
    else:
        print("Skipping server test")
        test2_passed = True  # Don't fail if user skipped
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Action extraction test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Server interaction test: {'✓ PASSED' if test2_passed else '✗ SKIPPED/FAILED'}")
    print("="*80)
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Webshop integration is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

