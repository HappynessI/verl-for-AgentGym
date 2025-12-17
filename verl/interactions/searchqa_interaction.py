"""SearchQA环境的Interaction实现"""

import re
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction


class SearchQAInteraction(AgentGymBaseInteraction):
    """SearchQA环境交互类（搜索问答）
    
    SearchQA action格式示例：
    - search[machine learning]
    - click[3]
    - answer[deep learning is a subset of machine learning]
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "searchqa"
        self.max_rounds = 15  # 搜索问答步数适中
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取SearchQA action
        
        支持的格式：
        1. 直接action: search[machine learning]
        2. 带标签: Action: search[machine learning]
        3. 带思考: <think>...</think>\nsearch[machine learning]
        """
        text = text.strip()
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 提取search[...], click[...], answer[...]格式
        action_match = re.search(r'\b(search|click|answer)\s*\[([^\]]+)\]', text, re.IGNORECASE)
        if action_match:
            action_type = action_match.group(1).lower()
            action_arg = action_match.group(2).strip()
            return f"{action_type}[{action_arg}]"
        
        # 提取"Action:"后的内容
        action_label_match = re.search(r'Action:\s*(.+)', text, re.IGNORECASE)
        if action_label_match:
            action_line = action_label_match.group(1).strip()
            # 再次尝试提取action格式
            action_match = re.search(r'\b(search|click|answer)\s*\[([^\]]+)\]', action_line, re.IGNORECASE)
            if action_match:
                action_type = action_match.group(1).lower()
                action_arg = action_match.group(2).strip()
                return f"{action_type}[{action_arg}]"
        
        # 提取最后一行包含action格式的文本
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            action_match = re.search(r'\b(search|click|answer)\s*\[([^\]]+)\]', line, re.IGNORECASE)
            if action_match:
                action_type = action_match.group(1).lower()
                action_arg = action_match.group(2).strip()
                return f"{action_type}[{action_arg}]"
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Valid actions: search[keywords], click[number], answer[your answer]")

