"""ALFWorld环境的Interaction实现"""

import re
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction


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
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "alfworld"
        self.max_rounds = 50  # ALFWorld任务可能较长
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取ALFWorld action
        
        支持的格式：
        1. 直接action: "go to shelf 1"
        2. 带标签: Action: go to shelf 1
        3. 带思考: <think>...</think>\ngo to shelf 1
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
            if line and len(line) < 100:  # ALFWorld action可以稍长
                return line
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions: go to <object>, take <object> from <receptacle>, "
                "put <object> in/on <receptacle>, examine <object>")

