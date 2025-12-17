"""BabyAI环境的Interaction实现"""

import re
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction


class BabyAIInteraction(AgentGymBaseInteraction):
    """BabyAI环境交互类
    
    BabyAI action格式示例：
    - "turn left"
    - "go forward"
    - "pick up"
    - "drop"
    - "toggle"
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "babyai"
        self.max_rounds = 50  # BabyAI可能需要更多步数
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取BabyAI action
        
        支持的格式：
        1. 直接action: "turn left"
        2. 带标签: <action>turn left</action>
        3. 带思考: <think>...</think>\nturn left
        """
        text = text.strip()
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 提取action标签内容
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        if action_match:
            return action_match.group(1).strip().lower()
        
        # 提取最后一行非空文本作为action
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().lower()
            if line and len(line) < 50:  # action通常很短
                return line
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Valid actions: turn left, turn right, go forward, pick up, drop, toggle")

