from .environment import TextCraftEnv
from .crafting_tree import CraftingTree
import threading


class TextCraft_Wrapper:
    def __init__(self, minecraft_dir="agentenv_textcraft/"):
        self._max_id = 0
        self.env = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.ls = []
        self.crafting_tree = CraftingTree(minecraft_dir=minecraft_dir)
        self._lock = threading.Lock()
        self._log_every_n = 20
        self._log_counts = {}

    def _should_log(self, name: str) -> bool:
        with self._lock:
            count = self._log_counts.get(name, 0) + 1
            self._log_counts[name] = count
        return count == 1 or count % self._log_every_n == 0

    def _log_count(self, name: str) -> int:
        with self._lock:
            return self._log_counts.get(name, 0)

    def _normalize_goal(self, goal: str) -> str:
        if goal is None:
            return None
        goal = goal.strip()
        if not goal.startswith("minecraft:"):
            goal = "minecraft:" + goal.replace(" ", "_")
        return goal

    def create(self, commands: str = None, goal: str = None, data_idx: int = None):
        try:
            with self._lock:
                id = self._max_id
                self._max_id += 1
            normalized_goal = self._normalize_goal(goal)
            effective_data_idx = data_idx if data_idx is not None else id
            new_env = TextCraftEnv(
                crafting_tree=self.crafting_tree, commands=commands, goal=normalized_goal
            )
            ob, _ = new_env.reset(data_idx=effective_data_idx)
            if self._should_log("create"):
                print(
                    f"-------Env create [{self._log_count('create')}] "
                    f"id={id} (data_idx={effective_data_idx}, goal={goal})--------"
                )
            self.ls.append(id)
            payload = {"id": id, "observation": ob, "done": False, "reward": 0, "goal": goal}
            self.env[id] = new_env
            self.info[id] = {
                "observation": ob,
                "done": False,
                "reward": 0,
                "deleted": False,
                "goal": goal,
                "data_idx": effective_data_idx,
            }
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def step(self, id: int, action: str):
        try:
            self._check_id(id)
            (ob, reward, done, _, _) = self.env[id].step(action)
            payload = {"observation": ob, "reward": reward, "done": done}
            self.info[id].update(payload)
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def reset(self, id: int, data_idx: int):
        try:
            self._check_id(id)
            ob, _ = self.env[id].reset(data_idx=data_idx)
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.info[id].update(
                {"observation": ob, "done": False, "reward": 0, "deleted": False}
            )
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, id: int):
        if id not in self.info:
            raise NameError(f"The id {id} is not valid.")
        if self.info[id]["deleted"]:
            raise NameError(f"The task with environment {id} has been deleted.")
        
    def close(self, id):
        try:
            self.ls.remove(id)
            self.env[id].close() 
            del self.info[id] 
            del self.env[id] 
            if self._should_log("close_env"):
                print(f"-------Env close [{self._log_count('close_env')}] id={id}--------")
            return True
        except KeyError:
            print(f"--------Env {id} not exist--------")
            return False
        except Exception as e:
            print(f"Error while closing Env {id}: {e}")
            return False
    
    
    def __del__(self):
        for idx in self.ls:
            self.env[idx].close()
            print(f"-------Env {idx} closed--------")

server = TextCraft_Wrapper()
