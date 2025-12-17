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
Convert AgentGym-RL data format to verl format for all environments
"""

import argparse
import json
import os

import datasets


# Environment-specific system prompts
SYSTEM_PROMPTS = {
    "webshop": """You are an expert shopping assistant in the WebShop environment. Your goal is to help users find and purchase the right product based on their requirements.

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

Your response should contain only the action in the format: search[keywords] or click[item]""",
    
    "babyai": """You are an agent in the BabyAI environment. Your goal is to complete various navigation and object manipulation tasks.

You can use the following actions:
- move forward
- turn left
- turn right
- pick up object
- drop object
- toggle (open/close doors, switch lights)

Instructions:
- Read the task instruction carefully
- Navigate through the grid world efficiently
- Interact with objects as needed
- Complete the task goal

Your response should contain only the action.""",
    
    "alfworld": """You are an agent in the ALFWorld environment. Your goal is to complete household tasks by interacting with objects in various rooms.

You can use actions like:
- go to [location]
- take [object] from [location]
- put [object] in/on [location]
- open [object]
- close [object]
- toggle [object]
- clean [object] with [location]
- heat [object] with [location]
- cool [object] with [location]
- examine [object]
- inventory

Instructions:
- Read the task instruction carefully
- Plan your actions to complete the task
- Navigate between rooms as needed
- Manipulate objects correctly

Your response should contain only the action.""",
    
    "sciworld": """You are an agent in the ScienceWorld environment. Your goal is to complete scientific tasks by interacting with objects and conducting experiments.

You can use actions like:
- look around
- move to [location]
- take [object]
- put [object] in/on [location]
- open [object]
- close [object]
- activate [object]
- deactivate [object]
- use [object] on [target]
- read [object]
- focus on [object]

Instructions:
- Read the task instruction carefully
- Understand the scientific goal
- Plan your experiment steps
- Use equipment and materials correctly

Your response should contain only the action.""",
    
    "sqlgym": """You are an expert SQL assistant. Your goal is to write correct SQL queries to answer questions based on database schemas.

Instructions:
- Read the question carefully
- Understand the database schema
- Write correct SQL query
- Use proper SQL syntax
- Consider JOIN, WHERE, GROUP BY, ORDER BY as needed

Your response should contain only the SQL query.""",
    
    "textcraft": """You are an agent in the TextCraft environment. Your goal is to craft items by gathering resources and following recipes.

You can use actions like:
- move [direction]
- get [object]
- craft [item]
- inventory
- look

Instructions:
- Read the task instruction carefully
- Gather necessary resources
- Follow crafting recipes
- Complete the crafting goal

Your response should contain only the action.""",
    
    "searchqa": """You are a question answering assistant. Your goal is to answer questions by searching through documents and finding relevant information.

You can use actions like:
- search[keywords]
- read[document_id]
- answer[text]

Instructions:
- Read the question carefully
- Search for relevant information
- Read and analyze documents
- Provide accurate answers

Your response should contain only the action.""",
}

# Environment-specific user prompts
USER_PROMPTS = {
    "webshop": "Please help me with my shopping task. I will provide you with the task description.",
    "babyai": "Please help me complete the BabyAI task. I will provide you with the task description.",
    "alfworld": "Please help me complete the household task. I will provide you with the task description.",
    "sciworld": "Please help me complete the science task. I will provide you with the task description.",
    "sqlgym": "Please help me write a SQL query. I will provide you with the question and database schema.",
    "textcraft": "Please help me complete the crafting task. I will provide you with the task description.",
    "searchqa": "Please help me answer the question. I will provide you with the question.",
}

# Environment-specific abilities
ABILITIES = {
    "webshop": "shopping",
    "babyai": "navigation",
    "alfworld": "household",
    "sciworld": "science",
    "sqlgym": "sql",
    "textcraft": "crafting",
    "searchqa": "qa",
}


def extract_session_id(item_id, idx, env_name):
    """Extract session_id from item_id (e.g., 'webshop_5238' -> 5238)"""
    if isinstance(item_id, str) and "_" in item_id:
        try:
            return int(item_id.split("_")[-1])
        except ValueError:
            return idx
    elif isinstance(item_id, int):
        return item_id
    else:
        return idx


def convert_data(input_file, env_name, output_dir, num_samples=None):
    """Convert AgentGym-RL format to verl format"""
    
    print(f"Loading {env_name} data from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    if num_samples:
        data = data[:num_samples]
        print(f"Processing first {num_samples} samples")
    
    # Process data
    processed_data = []
    
    for idx, item in enumerate(data):
        # Extract session_id from item_id
        item_id = item.get("item_id", idx)
        session_id = extract_session_id(item_id, idx, env_name)
        
        # Create verl format
        data_entry = {
            "data_source": env_name,
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPTS[env_name],
                },
                {
                    "role": "user",
                    "content": USER_PROMPTS[env_name],
                },
            ],
            "ability": ABILITIES[env_name],
            "reward_model": {
                "style": "interaction",  # Reward comes from environment interaction
                "ground_truth": "",  # Dummy field for naive reward manager compatibility
            },
            "extra_info": {
                "index": idx,
                "interaction_kwargs": {
                    "name": env_name,  # Must match the name in interaction config
                    "session_id": session_id,  # Environment task ID
                },
            },
        }
        processed_data.append(data_entry)
    
    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(processed_data)
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output filename based on input filename
    input_basename = os.path.basename(input_file)
    if "test" in input_basename:
        output_filename = "test.parquet"
    else:
        output_filename = "train.parquet"
    
    output_file = os.path.join(output_dir, output_filename)
    dataset.to_parquet(output_file)
    print(f"Saved preprocessed dataset to {output_file}")
    
    # Print sample
    print("\n" + "="*80)
    print("Sample data entry:")
    print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))
    print("="*80)
    
    return len(processed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, 
                       choices=list(SYSTEM_PROMPTS.keys()),
                       help="Environment name")
    parser.add_argument("--input_file", required=True, 
                       help="Path to the input JSON file")
    parser.add_argument("--output_dir", required=True, 
                       help="Directory to save preprocessed dataset")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    convert_data(args.input_file, args.env, args.output_dir, args.num_samples)

