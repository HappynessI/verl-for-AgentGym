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
    """
    
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
        
        支持的格式：
        1. [[ action ]] 格式（推荐）
        2. Action: action 格式
        3. 直接action
        """
        text = text.strip()
        
        # 移除chat template标记
        text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\|im_end\|>', '', text)
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 格式1: [[ action ]]
        action_matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if action_matches:
            action = action_matches[-1].strip()
            action = " ".join(action.split())
            if action:
                return action.lower()
        
        # 格式2: Action: action
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            action = re.sub(r'^\[\[\s*|\s*\]\]$', '', action)
            if action:
                return action.lower()
        
        # 格式3: 提取最后一行非空文本
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().lower()
            if line and len(line) < 150:
                # 排除思考内容
                if not line.startswith('think:') and not line.startswith('thought:'):
                    return line
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action wrapped in [[ ]].\n"
                "Example actions:\n"
                "- [[ move to kitchen ]]\n"
                "- [[ take thermometer from table ]]\n"
                "- [[ use thermometer on water ]]\n"
                "- [[ read thermometer ]]\n"
                "- [[ look around ]]")
