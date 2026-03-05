#!/usr/bin/env python3
# Copyright 2024 wyh
# Turn-based Prefix RL 训练入口
"""
训练入口，解决两个关键问题：
1. 在 Ray worker 进程中注册自定义优势估计器（turn_full_trajectory 等）
2. 复用 verl.trainer.main_ppo 的完整训练流程

使用方式：
    python3 -m recipe.wyh_exp.main_train \
        --config-path=config \
        --config-name=train_config \
        actor_rollout_ref.model.path=/Data/public/Qwen3-0.6B \
        ...
"""

import hydra
import ray
from omegaconf import DictConfig

from verl.trainer.main_ppo import run_ppo, TaskRunner as BaseTaskRunner


class TurnPrefixTaskRunner(BaseTaskRunner):
    """
    扩展 TaskRunner，在 Ray worker 进程中注册自定义优势估计器。

    verl 的 compute_advantage() 在 TaskRunner.run() 中执行，
    而 TaskRunner 是 Ray remote actor，运行在独立进程中。
    必须在该进程中 import recipe.wyh_exp 才能让自定义估计器生效。
    """

    def run(self, config):
        # 在 Ray worker 进程中注册自定义 adv_estimator
        import recipe.wyh_exp  # noqa: F401

        from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
        print(f"[wyh_exp] Registered adv estimators: {list(ADV_ESTIMATOR_REGISTRY.keys())}")

        super().run(config)


@hydra.main(config_path="config", config_name="train_config", version_base=None)
def main(config: DictConfig):
    """入口函数，使用自定义 TaskRunner 启动训练"""
    task_runner_class = ray.remote(num_cpus=1)(TurnPrefixTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
