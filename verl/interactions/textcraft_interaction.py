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
        
        完全遵循ADaPT项目的实现逻辑（textcraft.py step方法）
        支持两种格式：
        1. ADaPT格式: "> command" 或最后一行非think的内容（从后往前找）
        2. ReAct格式: "Action:" 关键词
        """
        text = text.strip()
        
        # 移除chat template标记
        text = re.sub(r'^assistant\s*\n?', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\|im_end\|>', '', text)
        text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
        
        # === 方法1: ADaPT格式（从最后一行往回找）===
        lines = text.split('\n')
        extracted_action = ""
        for line in reversed(lines):  # 关键：从后往前遍历！
            line = line.strip()
            # 跳过空行和OK
            if not line or line == 'OK.' or line == 'OK':
                continue
            # 移除开头的 ">"
            if line.startswith('>'):
                cmd = line[1:].strip()
                # 跳过think:开头的命令
                if not cmd.lower().startswith('think:'):
                    extracted_action = cmd
                    break
            else:
                # 没有>前缀，检查是否是think:
                if not line.lower().startswith('think:'):
                    extracted_action = line
                    break
        
        if extracted_action:
            # 清理action字符串（完全遵循ADaPT的方式）
            action = re.sub(r"[^A-Za-z0-9, ]+", "", extracted_action)  # 替换为空字符串
            action = " ".join(action.split()).strip()
            if action:
                return action
        
        # === 方法2: ReAct格式回退（Action:关键词）===
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", text, re.DOTALL)
        if action_matches:
            action = action_matches[-1].strip()
            action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
            action = " ".join(action.split()).strip()
            if action:
                return action
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions:\n"
                "- craft 1 blue dye using 1 lapis lazuli\n"
                "- get 9 slime ball\n"
                "- inventory")

