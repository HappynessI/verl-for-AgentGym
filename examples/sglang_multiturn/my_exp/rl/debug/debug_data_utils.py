#!/usr/bin/env python3
"""
Debug utilities for TextCraft prefix RL training.

This module provides:
1. Debug data subset generation from the main parquet file
2. Debug configuration helpers
"""

import os
import sys
import logging
import argparse
import pyarrow.parquet as pq
import pandas as pd

logger = logging.getLogger(__name__)


import numpy as np


def _is_non_empty_sequence(obj) -> bool:
    """
    Check if an object is a non-empty sequence (list or np.ndarray).
    
    Args:
        obj: Any object to check
        
    Returns:
        bool: True if it's a non-empty list or np.ndarray
    """
    if obj is None:
        return False
    if isinstance(obj, (list, np.ndarray)):
        return len(obj) > 0
    return False


def get_prefix_actions(row: dict) -> list:
    """
    Extract prefix_actions from a row with nested structure.
    
    Priority order:
    1. row['extra_info']['interaction_kwargs']['prefix_actions'] (nested)
    2. row['extra_info']['interaction_kwargs.prefix_actions'] (flattened dot notation)
    3. row['prefix_actions'] (flat column)
    
    Args:
        row: A dictionary representing a single row from the dataframe
        
    Returns:
        list: prefix_actions list, or empty list if not found
    """
    # Method 1: Nested structure - row['extra_info']['interaction_kwargs']['prefix_actions']
    try:
        extra_info = row.get('extra_info')
        if extra_info is not None and isinstance(extra_info, dict):
            interaction_kwargs = extra_info.get('interaction_kwargs')
            if interaction_kwargs is not None and isinstance(interaction_kwargs, dict):
                prefix_actions = interaction_kwargs.get('prefix_actions')
                if _is_non_empty_sequence(prefix_actions):
                    return list(prefix_actions)  # Convert to list for consistency
    except Exception:
        pass
    
    # Method 2: Flattened dot notation - row['extra_info.interaction_kwargs.prefix_actions']
    try:
        prefix_actions = row.get('extra_info.interaction_kwargs.prefix_actions')
        if _is_non_empty_sequence(prefix_actions):
            return list(prefix_actions)
    except Exception:
        pass
    
    # Method 3: Simple flat column - row['prefix_actions']
    try:
        prefix_actions = row.get('prefix_actions')
        if _is_non_empty_sequence(prefix_actions):
            return list(prefix_actions)
    except Exception:
        pass
    
    return []


def has_prefix_actions(row: dict) -> bool:
    """
    Check if a row has non-empty prefix_actions.
    
    Args:
        row: A dictionary representing a single row from the dataframe
        
    Returns:
        bool: True if prefix_actions exists and is non-empty
    """
    return len(get_prefix_actions(row)) > 0


def create_debug_subset(
    input_path: str,
    output_path: str,
    max_samples: int = 16,
    preferred_categories: list = None
) -> dict:
    """
    Create a debug subset from the main parquet file.
    
    Args:
        input_path: Path to the main parquet file
        output_path: Path to save the debug subset
        max_samples: Maximum number of samples to include
        preferred_categories: Preferred categories to prioritize (for sampling prefix_actions)
    
    Returns:
        dict: Statistics about the debug subset
    """
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    logger.info(f"Total samples in dataset: {len(df)}")
    
    # Analyze columns
    logger.info(f"Columns: {list(df.columns)}")
    
    # Convert rows to dict for nested access
    rows = df.to_dict('records')
    
    # Count samples with/without prefix_actions using the new nested-safe function
    samples_with_prefix = 0
    samples_without_prefix = 0
    
    for row in rows:
        if has_prefix_actions(row):
            samples_with_prefix += 1
        else:
            samples_without_prefix += 1
    
    logger.info(f"Samples with prefix_actions: {samples_with_prefix}")
    logger.info(f"Samples without prefix_actions: {samples_without_prefix}")
    
    # Select samples: prioritize those with prefix_actions
    if samples_with_prefix > 0:
        # First collect indices of samples with prefix_actions
        indices_with_prefix = []
        indices_without_prefix = []
        
        for idx, row in enumerate(rows):
            if has_prefix_actions(row):
                indices_with_prefix.append(idx)
            else:
                indices_without_prefix.append(idx)
        
        # Take all samples with prefix_actions first, then fill with others
        selected_indices = indices_with_prefix[:max_samples]
        
        if len(selected_indices) < max_samples:
            remaining = max_samples - len(selected_indices)
            selected_indices.extend(indices_without_prefix[:remaining])
        
        debug_df = df.iloc[selected_indices].reset_index(drop=True)
    else:
        # No prefix_actions found, just take first N samples
        logger.warning("No samples with prefix_actions found, using first N samples")
        debug_df = df.head(max_samples).reset_index(drop=True)
    
    # Save debug subset
    debug_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(debug_df)} debug samples to: {output_path}")
    
    # Print statistics - use the same nested-safe function
    debug_rows = debug_df.to_dict('records')
    debug_samples_with_prefix = sum(1 for row in debug_rows if has_prefix_actions(row))
    
    stats = {
        "total_samples": len(df),
        "debug_samples": len(debug_df),
        "samples_with_prefix_actions": debug_samples_with_prefix
    }
    
    logger.info(f"Debug subset stats: {stats}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Create debug subset for TextCraft training")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train subset command
    train_parser = subparsers.add_parser('train', help='Create debug train subset')
    train_parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    train_parser.add_argument("--output", type=str, required=True, help="Output debug parquet file")
    train_parser.add_argument("--max-samples", type=int, default=16, help="Maximum number of samples")
    
    # Val subset command
    val_parser = subparsers.add_parser('val', help='Create debug validation subset')
    val_parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    val_parser.add_argument("--output", type=str, required=True, help="Output debug parquet file")
    val_parser.add_argument("--max-samples", type=int, default=16, help="Maximum number of samples")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'train':
        create_debug_subset(args.input, args.output, args.max_samples)
    elif args.command == 'val':
        create_debug_val_subset(args.input, args.output, args.max_samples)
    else:
        parser.print_help()


def create_debug_val_subset(
    input_path: str,
    output_path: str,
    max_samples: int = 16,
    random_seed: int = 42
) -> dict:
    """
    Create a debug validation subset from the main parquet file.
    
    Unlike train subset, val subset:
    - Does NOT need prefix_actions
    - Just takes a random small sample from original val data
    
    Args:
        input_path: Path to the main parquet file
        output_path: Path to save the debug subset
        max_samples: Maximum number of samples to include
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Statistics about the debug subset
    """
    logger.info(f"Loading val data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    logger.info(f"Total samples in val dataset: {len(df)}")
    
    # Random sample - no need for prefix_actions
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)
        logger.info(f"Sampled {max_samples} random samples for debug val")
    else:
        logger.info(f"Dataset has fewer samples ({len(df)}) than max_samples ({max_samples}), using all")
    
    # Save debug subset
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} debug val samples to: {output_path}")
    
    stats = {
        "total_samples": len(df),
        "debug_samples": len(df),
    }
    
    logger.info(f"Debug val subset stats: {stats}")
    
    return stats


if __name__ == "__main__":
    main()