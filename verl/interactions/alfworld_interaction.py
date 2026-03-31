"""ALFWorld环境的Interaction实现"""

import re
import logging
from typing import Optional, Dict, Any
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__name__)


class ALFWorldInteraction(AgentGymBaseInteraction):
    """ALFWorld环境交互类
    
    ALFWorld action格式示例：
    - "go to shelf 1"
    - "take mug 1 from shelf 1"
    - "put mug 1 in/on coffeemachine 1"
    - "heat mug 1 with microwave 1"
    - "clean mug 1 with sinkbasin 1"
    - "use desklamp 1"
    - "examine mug 1"
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.env_name = "alfworld"
        self.max_rounds = 50  # ALFWorld任务可能较长
        # 默认使用 Text 模式
        self.world_type = config.get('world_type', 'Text')
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        启动ALFWorld交互
        
        Args:
            instance_id: 实例唯一标识
            **kwargs: 必须包含:
                - session_id: 对应ALFWorld的game索引
        """
        session_id = kwargs.get('session_id', 0)
        
        # 创建环境实例
        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json={})
            response.raise_for_status()
            data = response.json()
            env_id = data.get('env_id') or data.get('id')
            if not env_id:
                raise ValueError(f"No env_id found in response: {data}")
            logger.debug(f"Created ALFWorld environment with env_id={env_id}")
        except Exception as e:
            logger.error(f"Failed to create ALFWorld environment: {e}")
            raise
        
        # 重置环境获取初始observation
        # ALFWorld需要额外的game和world_type参数
        reset_url = f"{self.env_server_base}/reset"
        try:
            reset_data = {
                'id': env_id,
                'game': session_id,
                'world_type': self.world_type
            }
            logger.debug(f"ALFWorld reset request: {reset_data}")
            response = await self._async_post(reset_url, json=reset_data)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to reset ALFWorld environment {env_id}: {e}")
            raise
        
        # 保存session信息
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'done': False,
            'step_count': 0,
            'initial_observation': data.get('observation', ''),
            'kwargs': kwargs
        }
    
    def extract_action(self, text: str) -> Optional[str]:
        """        """
        text = text.strip()
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 优先提取 [[ ]] 格式的action，只取第一个
        bracket_matches = re.findall(r'\[\[([^\]]+)\]\]', text)
        if bracket_matches:
            # 只取第一个有效的action
            for match in bracket_matches:
                action = match.strip()
                if action:  # 忽略空action
                    return action.lower()
        
        # 提取"Action:"后的内容
        action_match = re.search(r'Action:\s*(.+)', text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            # 移除 [[ ]] 括号
            action = re.sub(r'^\[\[\s*', '', action)
            action = re.sub(r'\s*\]\]$', '', action)
            return action.lower()
        
        # 提取最后一行非空文本
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) < 100:  # ALFWorld action可以稍长
                # 移除 [[ ]] 括号
                line = re.sub(r'^\[\[\s*', '', line)
                line = re.sub(r'\s*\]\]$', '', line)
                return line.lower()
        
        return None
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions: go to <object>, take <object> from <receptacle>, "
                "put <object> in/on <receptacle>, examine <object>")

