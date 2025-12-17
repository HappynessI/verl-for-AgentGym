"""SciWorld环境的Interaction实现"""

import re
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction


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
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取SciWorld action
        
        支持的格式：
        1. 直接action: "take thermometer from table"
        2. 带标签: Action: take thermometer
        3. 带思考: <think>...</think>\ntake thermometer
        """
        text = text.strip()
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 提取"Action:"后的内容
        action_match = re.search(r'Action:\s*(.+)', text, re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip().lower()
        
        # 提取最后一行非空文本
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().lower()
            if line and len(line) < 150:
                return line
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions: move to <location>, take <object> from <location>, "
                "use <object> on <target>, read <object>, pour <object> from/to <container>")

