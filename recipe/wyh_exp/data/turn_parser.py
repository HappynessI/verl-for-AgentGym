# Copyright 2024 wyh
# Turn Parser for Multi-turn Agent Interactions
"""
Turn 解析器：处理多轮 Agent 交互数据

多轮交互格式:
    Observation → Think → Action → Observation → Think → Action → ... → Final Answer

每个 Turn 包含:
    - observation: 环境观察（来自环境或上一步的反馈）
    - think: 模型的思考过程（可选）
    - action: 模型执行的动作

示例数据格式:
{
    "prompt_id": "textcraft_001",
    "query": "Craft a wooden pickaxe",
    "turns": [
        {"role": "observation", "content": "You are in a forest with trees around..."},
        {"role": "think", "content": "I need to get wood first to craft a pickaxe..."},
        {"role": "action", "content": "[[ get 3 logs ]]"},
        {"role": "observation", "content": "You got 3 logs. Inventory: 3 logs"},
        {"role": "think", "content": "Now I need to craft planks from logs..."},
        {"role": "action", "content": "[[ craft 4 planks using 1 log ]]"},
        ...
    ],
    "reward": 1.0,
    "total_turns": 5
}
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TurnRole(Enum):
    """Turn 角色类型"""
    OBSERVATION = "observation"  # 环境观察/反馈
    THINK = "think"              # 思考过程
    ACTION = "action"            # 执行动作
    SYSTEM = "system"            # 系统提示
    USER = "user"                # 用户输入
    ASSISTANT = "assistant"      # 助手回复（通用）


@dataclass
class Turn:
    """单个 Turn 的数据结构"""
    role: TurnRole
    content: str
    turn_index: int = 0           # 在轨迹中的索引
    token_start: int = 0          # token 起始位置
    token_end: int = 0            # token 结束位置
    is_from_env: bool = False     # 是否来自环境（observation）
    
    def __post_init__(self):
        if isinstance(self.role, str):
            self.role = TurnRole(self.role)
    
    @property
    def is_model_generated(self) -> bool:
        """是否是模型生成的内容"""
        return self.role in [TurnRole.THINK, TurnRole.ACTION, TurnRole.ASSISTANT]
    
    @property
    def token_length(self) -> int:
        """Turn 的 token 长度"""
        return self.token_end - self.token_start


@dataclass
class Trajectory:
    """完整轨迹的数据结构"""
    prompt_id: str
    query: str
    turns: List[Turn] = field(default_factory=list)
    reward: float = 0.0
    total_tokens: int = 0
    
    # Prefix 相关
    prefix_turns: int = 0         # 前 N 个 turn 作为 prefix
    prefix_token_end: int = 0     # prefix 结束的 token 位置
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_turns(self) -> int:
        return len(self.turns)
    
    @property
    def num_model_turns(self) -> int:
        """模型生成的 turn 数量"""
        return sum(1 for t in self.turns if t.is_model_generated)
    
    def get_turn_boundaries(self) -> List[Tuple[int, int]]:
        """获取所有 turn 的 token 边界"""
        return [(t.token_start, t.token_end) for t in self.turns]
    
    def get_prefix_mask(self, total_length: int) -> List[int]:
        """生成 prefix mask（prefix 部分为 1，rollout 部分为 0）"""
        mask = [0] * total_length
        for i in range(self.prefix_token_end):
            if i < total_length:
                mask[i] = 1
        return mask
    
    def get_model_generated_mask(self, total_length: int) -> List[int]:
        """生成模型生成内容的 mask"""
        mask = [0] * total_length
        for turn in self.turns:
            if turn.is_model_generated:
                for i in range(turn.token_start, min(turn.token_end, total_length)):
                    mask[i] = 1
        return mask


class TurnParser:
    """Turn 解析器"""
    
    def __init__(
        self,
        tokenizer,
        observation_markers: List[str] = None,
        think_markers: List[str] = None,
        action_markers: List[str] = None,
    ):
        """
        初始化 Turn 解析器
        
        Args:
            tokenizer: HuggingFace tokenizer
            observation_markers: 识别 observation 的标记
            think_markers: 识别 think 的标记
            action_markers: 识别 action 的标记
        """
        self.tokenizer = tokenizer
        
        # 默认标记（可根据不同环境自定义）
        self.observation_markers = observation_markers or [
            "Observation:", "Obs:", "Environment:", "Result:", 
            "You see", "You are", "Inventory:"
        ]
        self.think_markers = think_markers or [
            "Think:", "Thought:", "Reasoning:", "Analysis:",
            "I need to", "Let me", "First,", "Now,"
        ]
        self.action_markers = action_markers or [
            "Action:", "[[", "Act:", "Execute:",
            ">>> ", "```"
        ]
    
    def parse_messages(
        self,
        messages: List[Dict[str, str]],
        prefix_turns: int = 0,
    ) -> Trajectory:
        """
        从消息列表解析轨迹
        
        Args:
            messages: 消息列表，格式为 [{"role": "...", "content": "..."}]
            prefix_turns: 前 N 个 turn 作为 prefix
        
        Returns:
            Trajectory 对象
        """
        turns = []
        current_token_pos = 0
        
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "assistant")
            content = msg.get("content", "")
            
            # 确定 turn 类型
            role = self._determine_role(role_str, content)
            
            # Tokenize 内容
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            token_length = len(tokens)
            
            turn = Turn(
                role=role,
                content=content,
                turn_index=idx,
                token_start=current_token_pos,
                token_end=current_token_pos + token_length,
                is_from_env=(role == TurnRole.OBSERVATION),
            )
            turns.append(turn)
            current_token_pos += token_length
        
        # 创建轨迹
        trajectory = Trajectory(
            prompt_id="",
            query="",
            turns=turns,
            total_tokens=current_token_pos,
            prefix_turns=prefix_turns,
        )
        
        # 计算 prefix 结束位置
        if prefix_turns > 0 and prefix_turns <= len(turns):
            trajectory.prefix_token_end = turns[prefix_turns - 1].token_end
        
        return trajectory
    
    def parse_text(
        self,
        text: str,
        prefix_turns: int = 0,
    ) -> Trajectory:
        """
        从原始文本解析轨迹（使用标记识别）
        
        Args:
            text: 原始文本
            prefix_turns: 前 N 个 turn 作为 prefix
        
        Returns:
            Trajectory 对象
        """
        # 简单的基于标记的分割
        segments = self._segment_text(text)
        
        turns = []
        current_token_pos = 0
        
        for idx, (role, content) in enumerate(segments):
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            token_length = len(tokens)
            
            turn = Turn(
                role=role,
                content=content,
                turn_index=idx,
                token_start=current_token_pos,
                token_end=current_token_pos + token_length,
                is_from_env=(role == TurnRole.OBSERVATION),
            )
            turns.append(turn)
            current_token_pos += token_length
        
        trajectory = Trajectory(
            prompt_id="",
            query="",
            turns=turns,
            total_tokens=current_token_pos,
            prefix_turns=prefix_turns,
        )
        
        if prefix_turns > 0 and prefix_turns <= len(turns):
            trajectory.prefix_token_end = turns[prefix_turns - 1].token_end
        
        return trajectory
    
    def _determine_role(self, role_str: str, content: str) -> TurnRole:
        """根据角色字符串和内容确定 Turn 类型"""
        role_str = role_str.lower()
        
        # 直接映射
        role_map = {
            "observation": TurnRole.OBSERVATION,
            "think": TurnRole.THINK,
            "thought": TurnRole.THINK,
            "action": TurnRole.ACTION,
            "system": TurnRole.SYSTEM,
            "user": TurnRole.USER,
            "assistant": TurnRole.ASSISTANT,
        }
        
        if role_str in role_map:
            return role_map[role_str]
        
        # 基于内容推断
        content_lower = content.lower()
        
        for marker in self.observation_markers:
            if marker.lower() in content_lower[:100]:
                return TurnRole.OBSERVATION
        
        for marker in self.action_markers:
            if marker.lower() in content_lower[:50]:
                return TurnRole.ACTION
        
        for marker in self.think_markers:
            if marker.lower() in content_lower[:50]:
                return TurnRole.THINK
        
        # 默认为 assistant
        return TurnRole.ASSISTANT
    
    def _segment_text(self, text: str) -> List[Tuple[TurnRole, str]]:
        """将文本分割为多个 segment"""
        # 这里使用简单的换行分割，可以根据需要改进
        segments = []
        current_role = TurnRole.ASSISTANT
        current_content = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # 检测角色变化
            new_role = self._detect_role_from_line(line)
            
            if new_role != current_role and current_content:
                segments.append((current_role, '\n'.join(current_content)))
                current_content = []
            
            current_role = new_role
            current_content.append(line)
        
        if current_content:
            segments.append((current_role, '\n'.join(current_content)))
        
        return segments
    
    def _detect_role_from_line(self, line: str) -> TurnRole:
        """从单行检测角色"""
        line_lower = line.lower()
        
        for marker in self.observation_markers:
            if line_lower.startswith(marker.lower()):
                return TurnRole.OBSERVATION
        
        for marker in self.action_markers:
            if marker.lower() in line_lower[:30]:
                return TurnRole.ACTION
        
        for marker in self.think_markers:
            if line_lower.startswith(marker.lower()):
                return TurnRole.THINK
        
        return TurnRole.ASSISTANT
    
    @staticmethod
    def create_prefix_variants(
        trajectory: Trajectory,
        min_prefix_turns: int = 1,
        max_prefix_turns: int = None,
    ) -> List[Trajectory]:
        """
        为一个轨迹创建多个不同 prefix 长度的变体
        
        Args:
            trajectory: 原始轨迹
            min_prefix_turns: 最小 prefix turn 数
            max_prefix_turns: 最大 prefix turn 数
        
        Returns:
            轨迹变体列表
        """
        if max_prefix_turns is None:
            max_prefix_turns = trajectory.num_turns - 1
        
        max_prefix_turns = min(max_prefix_turns, trajectory.num_turns - 1)
        
        variants = []
        for prefix_len in range(min_prefix_turns, max_prefix_turns + 1):
            variant = Trajectory(
                prompt_id=f"{trajectory.prompt_id}_prefix{prefix_len}",
                query=trajectory.query,
                turns=trajectory.turns.copy(),
                reward=trajectory.reward,
                total_tokens=trajectory.total_tokens,
                prefix_turns=prefix_len,
                metadata=trajectory.metadata.copy(),
            )
            
            if prefix_len <= len(trajectory.turns):
                variant.prefix_token_end = trajectory.turns[prefix_len - 1].token_end
            
            variants.append(variant)
        
        return variants

