# Copyright 2024 wyh
# Turn-based Prefix RL for Multi-turn Agent Interactions
#
# 两种训练方式:
#   - 方式A (Full-Trajectory RL): 整个轨迹作为小模型自己的轨迹进行RL
#   - 方式B (Prefix-Guided RL): Prefix作为参考，只对rollout部分进行RL

from .algos.core_algos import (
    compute_turn_full_trajectory_advantage,
    compute_turn_prefix_guided_advantage,
)

__all__ = [
    "compute_turn_full_trajectory_advantage",
    "compute_turn_prefix_guided_advantage",
]

