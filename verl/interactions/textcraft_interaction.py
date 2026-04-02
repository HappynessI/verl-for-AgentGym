"""TextCraft环境的Interaction实现"""

import os
import re
import logging
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Debug mode flag
DEBUG_MODE = os.getenv("VERL_DEBUG_MODE", "0") == "1" or os.getenv("DEBUG_MODE", "0") == "1"


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

    @staticmethod
    def _normalize_prompt_messages(prompt) -> list[dict]:
        if prompt is None:
            return []
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, list):
            return prompt
        return []

    @staticmethod
    def _extract_goal_and_commands_from_prompt(prompt) -> tuple[Optional[str], Optional[str]]:
        prompt_list = TextCraftInteraction._normalize_prompt_messages(prompt)
        for msg in prompt_list:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            goal_match = re.search(
                r"Goal:\s*craft\s+(.+?)\.?$",
                content,
                re.IGNORECASE | re.MULTILINE,
            )
            commands_match = re.search(
                r"Crafting commands:\n(.+?)\n\nGoal:\s*craft\s+.+?\.?$",
                content,
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            if goal_match:
                goal = goal_match.group(1).strip()
                commands = commands_match.group(1).strip() if commands_match else None
                return goal, commands
        return None, None
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        TextCraft环境特殊处理：API返回的是'id'而非'env_id'
        支持 prefix_actions：创建环境后 replay 指定的 actions 来同步状态
        """
        prefix_actions = kwargs.pop('prefix_actions', None)
        if prefix_actions is not None and not isinstance(prefix_actions, list):
            prefix_actions = list(prefix_actions)

        # 优先使用显式 goal；若 parquet 未写 goal，则尝试从 prompt 中兜底解析。
        expected_goal = kwargs.pop('goal', None)
        if expected_goal is None:
            prompt = kwargs.get('prompt')
            parsed_goal, _ = self._extract_goal_and_commands_from_prompt(prompt)
            if parsed_goal is not None:
                expected_goal = parsed_goal
                logger.warning(
                    f"[{instance_id}] goal not in interaction_kwargs - "
                    f"extracted from prompt: {expected_goal!r}"
                )

        parsed_commands = kwargs.pop('commands', None)
        if parsed_commands is None:
            _, parsed_commands = self._extract_goal_and_commands_from_prompt(kwargs.get('prompt'))

        session_id = kwargs.get('session_id')
        data_idx = kwargs.pop('data_idx', None)
        if data_idx is None and session_id is not None:
            try:
                data_idx = int(session_id)
            except (TypeError, ValueError):
                data_idx = None

        create_body = {}
        if expected_goal is not None:
            create_body['goal'] = expected_goal
        if parsed_commands:
            create_body['commands'] = parsed_commands
        if data_idx is not None:
            create_body['data_idx'] = data_idx

        # 创建环境实例
        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json=create_body)
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

        actual_obs = data.get('observation', '')
        actual_goal_in_obs = None
        match = re.search(r'Goal:\s*craft\s+(.+?)\.?$', actual_obs, re.IGNORECASE | re.MULTILINE)
        if match:
            actual_goal_in_obs = match.group(1).strip()

        if expected_goal is not None and actual_goal_in_obs is not None:
            def _norm(goal: str) -> str:
                return goal.lower().replace('_', ' ').replace("'", '').strip()

            if _norm(actual_goal_in_obs) != _norm(expected_goal):
                raise ValueError(
                    f"[{instance_id}] FAIL-FAST: Goal mismatch! "
                    f"expected_goal={expected_goal!r}, actual_goal_in_env={actual_goal_in_obs!r}"
                )
        elif expected_goal is None and actual_goal_in_obs is not None:
            logger.warning(
                f"[{instance_id}] No expected_goal provided - "
                f"server assigned goal={actual_goal_in_obs!r}, data_idx={data_idx}"
            )

        # TextCraft的create接口已经返回了initial observation，不需要再调用reset
        # 保存session信息
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'done': data.get('done', False),
            'step_count': 0,
            'initial_observation': actual_obs,
            'kwargs': kwargs,
            'expected_goal': expected_goal,
            'actual_goal_in_obs': actual_goal_in_obs,
            'data_idx': data_idx,
        }
        
        logger.info(f"Started TextCraft interaction {instance_id} with env_id {env_id}")

        # ==================== DEBUG: Replay 开始 ====================
        if DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"DEBUG: TextCraft start_interaction for {instance_id}")
            print(f"  - env_id: {env_id}")
            print(f"  - prefix_actions 数量: {len(prefix_actions) if prefix_actions else 0}")
            if prefix_actions:
                print(f"  - prefix_actions 前3项: {prefix_actions[:3]}")
            print(f"{'='*60}\n")

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

        # ==================== DEBUG: Replay 过程 ====================
        if DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"DEBUG: Replay 开始 for {instance_id}")
            print(f"  - 总共 {len(actions)} 个 actions 需要 replay")
            print(f"{'='*60}\n")

        for i, action in enumerate(actions):
            # ==================== DEBUG: 每个 replay step ====================
            if DEBUG_MODE:
                print(f"  [Replay Step {i+1}/{len(actions)}] action: {action}")

            if session['done']:
                if DEBUG_MODE:
                    print(f"    ⚠️  Environment done during replay at step {i+1}")
                logger.warning(f"[{instance_id}] Env terminated during prefix replay at action {i}/{len(actions)}")
                break
            try:
                response = await self._async_post(
                    step_url,
                    json=self._build_step_payload(env_id, action),
                )
                response.raise_for_status()
                data = response.json()
                
                obs = data.get('observation', '')[:200] if data.get('observation') else ''
                reward = data.get('reward')
                done = data.get('done', False)
                
                # 保存最新的 observation，用于 replay 结束后打印
                session['latest_observation'] = data.get('observation', '')
                
                if DEBUG_MODE:
                    print(f"    ✓ 执行成功")
                    print(f"    - observation (前100字符): {obs[:100]}...")
                    print(f"    - reward: {reward}")
                    print(f"    - done: {done}")

                session['step_count'] += 1
                session['done'] = done
            except Exception as e:
                if DEBUG_MODE:
                    print(f"    ❌ 执行失败: {e}")
                    print(f"❌ FAIL-FAST: Replay action failed!")
                logger.error(f"[{instance_id}] Prefix replay failed at action {i}: {action!r} - {e}")
                raise

        # ==================== DEBUG: Replay 结束 ====================
        if DEBUG_MODE:
            # 使用最新保存的 observation，而不是初始的
            final_obs = session.get('latest_observation', session.get('initial_observation', ''))
            final_obs_preview = final_obs[:300] if final_obs else ''
            print(f"\n{'='*60}")
            print(f"DEBUG: Replay 结束 for {instance_id}")
            print(f"  - replay 后的 step_count: {session['step_count']}")
            print(f"  - replay 后 environment done: {session['done']}")
            print(f"  - replay 后的真实 observation (前300字符):")
            print(f"    {final_obs_preview}...")
            print(f"{'='*60}\n")
            
            # Fail-fast: 检查 replay 是否成功
            if session['step_count'] == 0 and len(actions) > 0:
                print(f"❌ FAIL-FAST: Replay was called but no actions were executed!")
                raise ValueError(f"Sample {instance_id}: Replay executed but no actions ran!")

        logger.info(
            f"[{instance_id}] Replayed {len(actions)} prefix actions, "
            f"step_count={session['step_count']}, done={session['done']}"
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Close the remote TextCraft env before dropping local session state."""
        session = self.instance_sessions.get(instance_id)
        if not session:
            return

        env_id = session['env_id']
        close_url = f"{self.env_server_base}/close"
        try:
            response = await self._async_post(close_url, json={"id": env_id})
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to close TextCraft environment {env_id}: {e}")
        finally:
            self.instance_sessions.pop(instance_id, None)
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取TextCraft action
        
        支持两种格式：
        1. [[ ... ]] 格式（评估脚本要求）: Action: [[ inventory ]]
        2. Action: 格式（训练数据格式）: Action: \n inventory
        
        格式示例：
        - Action: [[ inventory ]]
        - Action: [[ get 3 logs ]]
        - Action: [[ craft 4 stick using 2 oak planks ]]
        - Action: \n inventory
        - Action: \n get 3 logs
        """
        text = text.strip()
        
        # 移除chat template标记
        text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\|im_end\|>', '', text)
        
        # 方法1: 提取 [[ ... ]] 格式
        action_matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        
        if action_matches:
            action = action_matches[-1].strip()
            action = " ".join(action.split())
            if action:
                return action
        
        # 方法2: 提取 Action:\nxxx 格式（训练数据格式）
        # 匹配 "Action:" 后面跟换行符和实际action内容
        action_match = re.search(r'Action:\s*\n\s*(.+?)(?:\n|$)', text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            action = " ".join(action.split())
            if action:
                return action
        
        # 方法3: 尝试匹配 "Action:" 后紧跟内容（无换行）
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            action = " ".join(action.split())
            if action:
                return action
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions:\n"
                "- craft 1 blue dye using 1 lapis lazuli\n"
                "- get 9 slime ball\n"
                "- inventory")
