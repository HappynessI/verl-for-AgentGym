# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess the Webshop dataset for multi-turn interaction training
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


WEBSHOP_SYSTEM_PROMPT = """You are an expert shopping assistant in the WebShop environment. Your goal is to help users find and purchase the right product based on their requirements.

You can interact with the WebShop using two types of actions:
1. search[keywords] - Search for products using keywords
2. click[item] - Click on a button or item (e.g., "Back to Search", "Buy Now", product name, or option values)

Instructions:
- Carefully read the user's instruction about what to find
- Use search to find relevant products
- Click on products to view details
- Check if the product matches ALL requirements (color, size, features, price, etc.)
- Click options to select the right variant
- Click "Buy Now" when you find the perfect match
- If a product doesn't match, go back and continue searching

**CRITICAL RULES**: 
1. ALWAYS start by using search[keywords] - NEVER click a product without searching first
2. Only click products that appear in the current search results or page
3. Keep interacting until you successfully click "Buy Now"
4. Do NOT stop after just searching - you must click products and select options
5. Continue the interaction step by step until purchase is complete

Example interaction flow:
1. First, ALWAYS search for products using relevant keywords
2. Then click on a product from the search results to view details
3. Select the required options (color, size, etc.) by clicking them
4. Finally click "Buy Now" when all requirements are met

Example:
User: Find me men's shorts with color: navy, size: large, price < $50
Assistant: search[men's shorts navy]
Environment: [Shows search results with several products]
Assistant: click[<product_name_or_id_from_results>]
Environment: [Shows product details with options]
Assistant: click[navy]
Assistant: click[large]
Assistant: click[Buy Now]

Your response should contain only the action in the format: search[keywords] or click[item]
Do NOT include explanations, reasoning, or any other text - just the action."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, 
                       help="Path to the input Webshop JSON file (e.g., webshop_test.json)")
    parser.add_argument("--local_save_dir", default="~/data/webshop", 
                       help="Local directory to save preprocessed dataset")
    parser.add_argument("--hdfs_dir", default=None, 
                       help="Optional HDFS directory to copy the dataset")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to process (for testing)")

    args = parser.parse_args()

    # Load Webshop data
    print(f"Loading Webshop data from {args.input_file}")
    with open(args.input_file, 'r') as f:
        webshop_data = json.load(f)
    
    print(f"Loaded {len(webshop_data)} samples")
    
    if args.num_samples:
        webshop_data = webshop_data[:args.num_samples]
        print(f"Processing first {args.num_samples} samples")

    # Process data
    processed_data = []
    
    for idx, item in enumerate(webshop_data):
        # AgentGym data format: {"item_id": "webshop_5238"}
        # Extract session_id from item_id (e.g., "webshop_5238" -> 5238)
        if "item_id" in item:
            item_id = item["item_id"]
            # Extract numeric session_id from item_id
            if isinstance(item_id, str) and "_" in item_id:
                session_id = int(item_id.split("_")[-1])
            else:
                session_id = idx
        else:
            # Fallback for other formats with explicit task_id/session_id
            session_id = item.get("task_id", item.get("session_id", idx))
        
        # Note: We don't include specific instruction in the prompt
        # because it will be obtained from the environment server
        # when reset(session_id) is called
        data = {
            "data_source": "webshop",
            "prompt": [
                {
                    "role": "system",
                    "content": WEBSHOP_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "Please help me with my shopping task. I will provide you with the task description.",
                },
            ],
            "ability": "shopping",
            "reward_model": {
                "style": "interaction",  # Reward comes from environment interaction
                "ground_truth": "",  # Dummy field for naive reward manager compatibility
            },
            "extra_info": {
                "index": idx,
                "interaction_kwargs": {
                    "name": "webshop",  # Must match the name in webshop_interaction.yaml
                    "session_id": session_id,  # Webshop task ID
                },
            },
        }
        processed_data.append(data)
    
    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(processed_data)
    
    # Save dataset
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    output_file = os.path.join(local_save_dir, "train.parquet")
    dataset.to_parquet(output_file)
    print(f"Saved preprocessed dataset to {output_file}")
    
    # Optional: Copy to HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied dataset to HDFS: {args.hdfs_dir}")
    
    # Print sample
    print("\n" + "="*80)
    print("Sample data entry:")
    print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))
    print("="*80)

