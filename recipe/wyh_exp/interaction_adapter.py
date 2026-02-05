# Copyright 2024 wyh
# Interaction Adapter for Turn-based Prefix RL
"""
AgentGym 环境适配器

支持从 prefix 轨迹的指定轮次继续环境交互（prefix 续写）

主要功能：
1. 从 prefix 历史恢复环境状态
2. 让小模型从指定轮次继续 rollout
3. 收集完整的交互轨迹
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrefixRolloutConfig:
    """Prefix Rollout 配置"""
    env_name: str = "textcraft"
    env_server_base: str = "http://localhost:8000"
    max_rounds: int = 50
    timeout: int = 30
    continue_from_prefix: bool = True


class TurnPrefixInteractionAdapter:
    """
    Turn-based Prefix 交互适配器
    
    封装 AgentGym 环境交互，支持：
    1. 从 prefix 历史恢复环境状态
    2. 执行 prefix 中的所有 action（重放）
    3. 让模型继续 rollout
    """
    
    def __init__(
        self,
        interaction_cls,  # AgentGym Interaction 类
        config: PrefixRolloutConfig,
    ):
        """
        初始化适配器
        
        Args:
            interaction_cls: AgentGym 的 Interaction 类（如 TextCraftInteraction）
            config: 配置对象
        """
        self.interaction_cls = interaction_cls
        self.config = config
        self.interaction = None
    
    def _create_interaction(self) -> Any:
        """创建环境交互实例"""
        if self.interaction is None:
            # 构建 interaction 配置
            interaction_config = {
                "env_server_base": self.config.env_server_base,
                "max_rounds": self.config.max_rounds,
                "timeout": self.config.timeout,
            }
            self.interaction = self.interaction_cls(interaction_config)
        return self.interaction
    
    async def continue_from_prefix(
        self,
        instance_id: str,
        prefix_turns: List[Dict[str, str]],
        task_kwargs: Dict[str, Any],
    ) -> Tuple[str, List[Dict]]:
        """
        从 prefix 历史继续交互
        
        Args:
            instance_id: 实例 ID
            prefix_turns: prefix 轮次列表，格式为:
                [
                    {"role": "observation", "content": "..."},
                    {"role": "think", "content": "..."},
                    {"role": "action", "content": "..."},
                    ...
                ]
            task_kwargs: 任务参数（用于创建环境）
        
        Returns:
            Tuple[str, List[Dict]]:
                - 当前 observation（供模型继续生成）
                - 重放的历史记录
        """
        interaction = self._create_interaction()
        
        # 1. 创建环境
        await interaction.start_interaction(instance_id, **task_kwargs)
        
        # 2. 提取 prefix 中的 actions 并重放
        replayed_history = []
        current_observation = ""
        
        for turn in prefix_turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "observation":
                # 记录初始或中间的 observation
                current_observation = content
                replayed_history.append({"role": "observation", "content": content})
            
            elif role == "action":
                # 执行 action
                try:
                    result = await interaction.step(instance_id, content)
                    
                    # 获取环境返回的 observation
                    env_observation = result.get("observation", "")
                    done = result.get("done", False)
                    reward = result.get("reward", 0.0)
                    
                    replayed_history.append({"role": "action", "content": content})
                    replayed_history.append({
                        "role": "observation", 
                        "content": env_observation,
                        "reward": reward,
                        "done": done,
                    })
                    
                    current_observation = env_observation
                    
                    if done:
                        logger.info(f"Instance {instance_id}: Environment done during prefix replay")
                        break
                        
                except Exception as e:
                    logger.error(f"Error replaying action for {instance_id}: {e}")
                    replayed_history.append({
                        "role": "error",
                        "content": str(e),
                    })
                    break
            
            elif role in ["think", "thought"]:
                # Think 不需要执行，只记录
                replayed_history.append({"role": "think", "content": content})
        
        return current_observation, replayed_history
    
    async def rollout_from_prefix(
        self,
        instance_id: str,
        prefix_turns: List[Dict[str, str]],
        model_generate_fn,  # 模型生成函数
        task_kwargs: Dict[str, Any],
        max_rollout_turns: int = 20,
    ) -> Dict[str, Any]:
        """
        从 prefix 开始进行完整的 rollout
        
        Args:
            instance_id: 实例 ID
            prefix_turns: prefix 轮次
            model_generate_fn: 模型生成函数，签名为:
                async def generate(observation: str, history: List[Dict]) -> str
            task_kwargs: 任务参数
            max_rollout_turns: 最大 rollout 轮数
        
        Returns:
            完整的 rollout 结果，包括 prefix 和新生成的部分
        """
        interaction = self._create_interaction()
        
        # 1. 从 prefix 继续
        current_obs, prefix_history = await self.continue_from_prefix(
            instance_id, prefix_turns, task_kwargs
        )
        
        # 检查 prefix 重放后环境是否已经结束
        if prefix_history and prefix_history[-1].get("done", False):
            return {
                "instance_id": instance_id,
                "prefix_turns": len(prefix_turns),
                "rollout_turns": 0,
                "total_turns": len(prefix_history),
                "history": prefix_history,
                "done": True,
                "reward": prefix_history[-1].get("reward", 0.0),
            }
        
        # 2. 模型继续 rollout
        rollout_history = []
        total_reward = 0.0
        done = False
        
        for turn_idx in range(max_rollout_turns):
            # 调用模型生成
            try:
                model_output = await model_generate_fn(
                    observation=current_obs,
                    history=prefix_history + rollout_history,
                )
                
                # 解析模型输出（假设包含 think 和 action）
                think_content, action_content = self._parse_model_output(model_output)
                
                if think_content:
                    rollout_history.append({"role": "think", "content": think_content})
                
                if action_content:
                    # 执行 action
                    result = await interaction.step(instance_id, action_content)
                    
                    env_observation = result.get("observation", "")
                    done = result.get("done", False)
                    reward = result.get("reward", 0.0)
                    total_reward += reward
                    
                    rollout_history.append({"role": "action", "content": action_content})
                    rollout_history.append({
                        "role": "observation",
                        "content": env_observation,
                        "reward": reward,
                        "done": done,
                    })
                    
                    current_obs = env_observation
                    
                    if done:
                        break
                else:
                    # 模型没有生成有效 action
                    logger.warning(f"Instance {instance_id}: No valid action generated")
                    break
                    
            except Exception as e:
                logger.error(f"Error during rollout for {instance_id}: {e}")
                rollout_history.append({"role": "error", "content": str(e)})
                break
        
        # 3. 关闭环境
        try:
            await interaction.close_interaction(instance_id)
        except:
            pass
        
        return {
            "instance_id": instance_id,
            "prefix_turns": len([t for t in prefix_turns if t.get("role") == "action"]),
            "rollout_turns": len([t for t in rollout_history if t.get("role") == "action"]),
            "total_turns": len(prefix_history) + len(rollout_history),
            "prefix_history": prefix_history,
            "rollout_history": rollout_history,
            "history": prefix_history + rollout_history,
            "done": done,
            "reward": total_reward,
        }
    
    def _parse_model_output(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析模型输出，提取 think 和 action
        
        这是一个简单的实现，实际使用时可能需要根据模型输出格式调整
        """
        import re
        
        think_content = None
        action_content = None
        
        # 提取 Think 部分
        think_match = re.search(r'Think:\s*(.*?)(?=Action:|$)', output, re.DOTALL | re.IGNORECASE)
        if think_match:
            think_content = think_match.group(1).strip()
        
        # 提取 Action 部分（支持 [[ ]] 格式）
        action_match = re.search(r'\[\[\s*(.*?)\s*\]\]', output, re.DOTALL)
        if action_match:
            action_content = action_match.group(1).strip()
        else:
            # 尝试其他格式
            action_match = re.search(r'Action:\s*(.*?)(?=\n|$)', output, re.IGNORECASE)
            if action_match:
                action_content = action_match.group(1).strip()
        
        return think_content, action_content


# 工厂函数：根据环境名称创建适配器
def create_interaction_adapter(
    env_name: str,
    config: Optional[Dict] = None,
) -> TurnPrefixInteractionAdapter:
    """
    创建环境交互适配器
    
    Args:
        env_name: 环境名称（textcraft, webshop, alfworld 等）
        config: 配置字典
    
    Returns:
        TurnPrefixInteractionAdapter 实例
    """
    # 导入对应的 Interaction 类
    interaction_cls = None
    
    try:
        if env_name == "textcraft":
            from verl.interactions.textcraft_interaction import TextCraftInteraction
            interaction_cls = TextCraftInteraction
        elif env_name == "webshop":
            from verl.interactions.webshop_interaction import WebShopInteraction
            interaction_cls = WebShopInteraction
        elif env_name == "alfworld":
            from verl.interactions.alfworld_interaction import AlfWorldInteraction
            interaction_cls = AlfWorldInteraction
        else:
            raise ValueError(f"Unknown environment: {env_name}")
    except ImportError as e:
        logger.error(f"Failed to import interaction class for {env_name}: {e}")
        raise
    
    # 构建配置
    adapter_config = PrefixRolloutConfig(env_name=env_name)
    if config:
        for key, value in config.items():
            if hasattr(adapter_config, key):
                setattr(adapter_config, key, value)
    
    return TurnPrefixInteractionAdapter(interaction_cls, adapter_config)

