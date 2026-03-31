"""SciWorld环境的Interaction实现"""

import re
import logging
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__name__)


class SciWorldInteraction(AgentGymBaseInteraction):
    """SciWorld环境交互类
    
    SciWorld action格式示例（科学实验环境）：
    - "open door to kitchen"
    - "move to kitchen"
    - "take thermometer from table"
    - "use thermometer on water"
    - "read thermometer"
    - "pour water from beaker to container"
    
    与 AgentGym 官方采样脚本保持一致:
    - 使用 'score' 字段作为 reward (累积完成度分数，非 step 级惩罚)
    - 使用 BaseAdapter.parse_react 解析 ReAct 格式的 action
    """
    
    # SciWorld 环境使用 'score' 字段作为 reward (累积完成度分数)
    reward_field = "score"
    
    # 使用与 AgentGym 官方相同的 action_format
    action_format = "react"
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "sciworld"
        self.max_rounds = 100  # 科学实验可能需要很多步骤
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        SciWorld环境特殊处理：
        1. reset需要data_idx参数
        2. 初始observation需要包含task_description
        """
        # 创建环境实例
        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json={})
            response.raise_for_status()
            data = response.json()
            
            # SciWorld返回 {"id": int}
            if isinstance(data, dict):
                env_id = data.get('id') or data.get('env_id')
            elif isinstance(data, int):
                env_id = data
            else:
                env_id = data
        except Exception as e:
            logger.error(f"Failed to create SciWorld environment: {e}")
            raise
        
        # 使用session_id作为data_idx来reset
        session_id = kwargs.get('session_id', 0)
        reset_url = f"{self.env_server_base}/reset"
        try:
            reset_response = await self._async_post(
                reset_url,
                json={"id": env_id, "data_idx": session_id}
            )
            reset_response.raise_for_status()
            reset_data = reset_response.json()
        except Exception as e:
            logger.error(f"Failed to reset SciWorld environment: {e}")
            raise
        
        # 构建完整的初始observation（包含任务描述）
        task_description = reset_data.get('task_description', '')
        observation = reset_data.get('observation', '')
        
        # 组合任务描述和初始观察
        if task_description:
            initial_observation = f"Task: {task_description}\n\n{observation}"
        else:
            initial_observation = observation
        
        # 保存session信息
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'done': reset_data.get('done', False),
            'step_count': 0,
            'initial_observation': initial_observation,
            'task_description': task_description,
            'kwargs': kwargs
        }
        
        logger.info(f"Started SciWorld interaction {instance_id} with env_id {env_id}, task: {task_description[:50]}...")
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取SciWorld action
        
        支持两种格式：
        1. ReAct格式: Thought:\n...\nAction:\naction
        2. [[ ]] 格式: Action: [[ action ]]
        """
        # 移除 </s> 后缀
        action = text
        if action.endswith("</s>"):
            action = action[:-5]
        
        # 先移除思考标签，避免干扰 action 提取
        action = re.sub(r'<think>.*?</think>', '', action, flags=re.DOTALL)
        
        # 优先尝试解析 [[ ]] 格式（你的prompt使用的格式）
        bracket_match = re.search(r'\[\[\s*(.+?)\s*\]\]', action, re.IGNORECASE | re.DOTALL)
        if bracket_match:
            extracted = bracket_match.group(1).strip()
            if extracted:
                logger.debug(f"Extracted action from [[ ]]: {extracted}")
                return extracted.lower()
        
        # 尝试使用 BaseAdapter.parse_react 解析 ReAct 格式
        try:
            from agentenv.controller.utils import BaseAdapter
            parsed = BaseAdapter.parse_react(action)
            action_str = parsed.action.strip()
            if action_str:
                return action_str.lower()
        except Exception as e:
            logger.debug(f"Failed to parse action with BaseAdapter.parse_react: {e}")
        
        # Fallback: 尝试直接提取 Action: 后面的内容（不带 [[ ]]）
        action_match = re.search(r'Action:\s*\n?\s*(.+)', action, re.IGNORECASE | re.DOTALL)
        if action_match:
            extracted = action_match.group(1).strip()
            # 移除 [[ ]] 括号
            extracted = re.sub(r'^\[\[\s*', '', extracted)
            extracted = re.sub(r'\s*\]\]$', '', extracted)
            if extracted:
                logger.debug(f"Extracted action from Action: {extracted}")
                return extracted.lower()
        
        # Fallback: 如果都失败，返回None
        logger.warning(f"Failed to extract action from: {action[:100]}...")
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Your response should use the following format:\n"
                "Thought:\nyour thoughts.\n\n"
                "Action:\n[[ your next action ]]\n\n"
                "IMPORTANT: Always wrap your action in [[ ]] brackets.\n"
                "Example: Action: [[ open door to kitchen ]]")
