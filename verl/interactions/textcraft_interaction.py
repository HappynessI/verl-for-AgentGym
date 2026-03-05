"""TextCraft环境的Interaction实现"""

import re
import logging
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
        支持 prefix_actions：创建环境后 replay 指定的 actions 来同步状态
        """
        prefix_actions = kwargs.pop('prefix_actions', None)
        if prefix_actions is not None and not isinstance(prefix_actions, list):
            prefix_actions = list(prefix_actions)

        # 创建环境实例
        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json=kwargs)
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

        # Replay prefix actions to sync environment state
        if prefix_actions:
            await self._replay_prefix_actions(instance_id, prefix_actions)

    async def _replay_prefix_actions(self, instance_id: str, actions: list[str]) -> None:
        """Replay a sequence of actions to bring the environment to the post-prefix state."""
        session = self.instance_sessions.get(instance_id)
        if not session:
            raise ValueError(f"Instance {instance_id} not found for prefix replay")

        env_id = session['env_id']
        step_url = f"{self.env_server_base}/step"

        for i, action in enumerate(actions):
            if session['done']:
                logger.warning(f"[{instance_id}] Env terminated during prefix replay at action {i}/{len(actions)}")
                break
            try:
                response = await self._async_post(
                    step_url,
                    json=self._build_step_payload(env_id, action),
                )
                response.raise_for_status()
                data = response.json()
                session['step_count'] += 1
                session['done'] = data.get('done', False)
            except Exception as e:
                logger.error(f"[{instance_id}] Prefix replay failed at action {i}: {action!r} - {e}")
                raise

        logger.info(
            f"[{instance_id}] Replayed {len(actions)} prefix actions, "
            f"step_count={session['step_count']}, done={session['done']}"
        )
    
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
                # logger.debug(f"Extracted action from [[ ]]: {action}")
                return action
        
        # logger.warning(f"No [[ ]] format found in text: {text[:100]}...")
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions:\n"
                "- craft 1 blue dye using 1 lapis lazuli\n"
                "- get 9 slime ball\n"
                "- inventory")
