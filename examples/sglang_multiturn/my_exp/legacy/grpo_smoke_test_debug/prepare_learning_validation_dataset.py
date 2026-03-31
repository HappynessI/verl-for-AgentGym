import pandas as pd
import numpy as np
import os

INPUT_PATH = '/Data/wyh/datasets/Verl-Data/outputs/textcraft_old_logits/active/textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet'
OUTPUT_DIR = '/Data/wyh/datasets/Verl-Data/outputs/textcraft_grpo_prefix_smoke_test_canonicalized'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'textcraft_validated_prefix_history_canonicalized_with_turn_scores.parquet')

df = pd.read_parquet(INPUT_PATH)
print(f'Loaded {len(df)} rows')

# Add turn_scores to extra_info for each row
# Assign diverse rewards to ensure variance for GRPO advantage computation
# We cycle through different reward values so different samples get different rewards
reward_schedule = [1.0, 0.0, 0.5, -0.5, 0.8, 0.2, 1.0, 0.0, 0.5, -0.5]

turn_scores_values = []
for i in range(len(df)):
    reward_val = reward_schedule[i % len(reward_schedule)]
    turn_scores_values.append([float(reward_val)])

# Add turn_scores to extra_info
new_extra_info = []
for i, ei in enumerate(df['extra_info']):
    new_ei = dict(ei)  # shallow copy
    new_ei['turn_scores'] = turn_scores_values[i]
    new_extra_info.append(new_ei)

df_out = df.copy()
df_out['extra_info'] = new_extra_info

# Verify
for i in range(min(5, len(df_out))):
    ei = df_out['extra_info'].iloc[i]
    print(f'Row {i}: turn_scores = {ei["turn_scores"]}')

df_out.to_parquet(OUTPUT_PATH, index=False)
print(f'\nWritten to {OUTPUT_PATH}')

# Quick verify
df_verify = pd.read_parquet(OUTPUT_PATH)
print(f'Verified: {len(df_verify)} rows, columns: {list(df_verify.columns)}')
for i in range(5):
    ei = df_verify['extra_info'].iloc[i]
    print(f'  Row {i}: turn_scores = {ei["turn_scores"]}')

print(f'\nDataset statistics:')
print(f'  Total rows: {len(df_verify)}')
print(f'  Turn scores distribution:')
ts_values = [df_verify['extra_info'].iloc[i]['turn_scores'][0] for i in range(len(df_verify))]
print(f'    unique: {sorted(set(ts_values))}')
print(f'    mean: {np.mean(ts_values):.3f}')
print(f'    std: {np.std(ts_values):.3f}')
print(f'  prefix_token_count: mean={df_verify["prefix_token_count"].mean():.1f}, min={df_verify["prefix_token_count"].min()}, max={df_verify["prefix_token_count"].max()}')
