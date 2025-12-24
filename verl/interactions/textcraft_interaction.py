"""TextCraft环境的Interaction实现"""

import re
import logging
import requests
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__name__)


class TextCraftInteraction(AgentGymBaseInteraction):
    """TextCraft环境交互类（文本版Minecraft）
    
    TextCraft action格式示例：
    - "craft(wood_pickaxe)"
    - "mine(stone)"
    - "get(wood)"
    - "goto(forest)"
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "textcraft"
        self.max_rounds = 100  # 制作任务可能需要很多步骤
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        TextCraft环境特殊处理：API返回的是'id'而非'env_id'
        """
        # 创建环境实例
        create_url = f"{self.env_server_base}/create"
        try:
            response = requests.post(create_url, json=kwargs, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            # TextCraft返回的是'id'字段，不是'env_id'
            # 注意：id可能为0，所以不能用 or 判断
            if 'id' in data:
                env_id = data['id']
            elif 'env_id' in data:
                env_id = data['env_id']
            else:
                raise ValueError(f"No env_id found in response: {data}")
        except Exception as e:
            logger.error(f"Failed to create TextCraft environment: {e}")
            raise
        
        # TextCraft的create接口已经返回了initial observation，不需要再调用reset
        # 保存session信息
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'done': data.get('done', False),
            'step_count': 0,
            'initial_observation': data.get('observation', ''),
            'kwargs': kwargs
        }
        
        logger.info(f"Started TextCraft interaction {instance_id} with env_id {env_id}")
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取TextCraft action
        
        新版格式：只提取 [[ ... ]] 中的内容
        格式示例：
        - Action: [[ inventory ]]
        - Action: [[ get 3 logs ]]
        - Action: [[ craft 4 stick using 2 oak planks ]]
        
        括号外的任何内容（包括Think:、幻觉等）都会被丢弃
        """
        text = text.strip()
        
        # 移除chat template标记
        text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\|im_end\|>', '', text)
        
        # 使用正则提取 [[ ... ]] 中的内容（最后一个匹配）
        # 匹配模式：[[ 后面跟任意字符（非贪婪），直到 ]]
        action_matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        
        if action_matches:
            # 取最后一个匹配（模型可能生成多个action，取最新的）
            action = action_matches[-1].strip()
            # 清理多余的空白字符
            action = " ".join(action.split())
            if action:
                logger.debug(f"Extracted action from [[ ]]: {action}")
                return action
        
        logger.warning(f"No [[ ]] format found in text: {text[:100]}...")
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions:\n"
                "- craft 1 blue dye using 1 lapis lazuli\n"
                "- get 9 slime ball\n"
                "- inventory")

