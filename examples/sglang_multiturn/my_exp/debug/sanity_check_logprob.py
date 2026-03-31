"""
Sanity Check: Rollout LogProb vs Actor Recompute LogProb 一致性验证
=========================================================================

核心目标：验证 rollout_log_probs 与 actor recompute log_prob 是否一致

【适用范围限制】
本脚本仅适用于：
- 单轮对话场景
- 纯文本输入
- 无 padding 或最小 padding 假设
- 近似 actor 路径（不是完整 verl actor worker）

它不能直接替代完整训练链路的诊断。

【模型权重说明】
vLLM engine 和 transformers actor model 是从同一路径加载的两个独立实例：
- 最多只是"从同一 checkpoint 加载"，不是"共享同一份运行时权重"
- 在真实训练中使用 hybrid engine 才能实现真正的权重共享
- 离线验证无法完全复现训练时的权重同步行为

【position_ids / attention_mask 说明】
当前实现：
- attention_mask = ones_like(input_tensor)
- position_ids = arange(0, seq_len)

能验证：logits 提取位置、temperature 处理、log_softmax 逻辑
不能验证：多轮场景下的 attention mask、真实的 position_ids 生成逻辑

如果 baseline 仍异常，下一步应进一步对齐 attention_mask / position_ids / 真实 actor 路径

=========================================================================
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)

# 设备配置 - 默认使用 cuda:0
# 如果需要使用其他GPU，请在运行脚本时设置环境变量 CUDA_VISIBLE_DEVICES
GPU_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class SanityCheckConfig:
    """实验配置"""
    model_path: str = "/path/to/your/model"
    tensor_parallel_size: int = 1
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    max_tokens: int = 512
    debug: bool = True


class LogProbComparator:
    """
    比较 rollout 和 actor 侧 log_prob 的类
    
    【关键说明】
    - vLLM 和 actor model 是两个独立加载的模型实例
    - 使用与 verl 一致的 chat template 构建输入
    - 修正了 logits 切片位置（teacher-forcing 对齐）
    """
    
    def __init__(self, config: SanityCheckConfig):
        self.config = config
        self.vllm_engine = None
        self.actor_model = None
        self.tokenizer = None
    
    def init_models(self):
        """
        初始化模型
        
        注意：vLLM engine 和 transformers actor_model 是两个独立的模型实例，
        不是共享同一份运行时权重。
        """
        from vllm import LLM
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Initializing vLLM with model: {self.config.model_path}")
        self.vllm_engine = LLM(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            enforce_eager=True,  # 禁用 CUDA graph 以避免编译问题
        )
        
        # 获取 vLLM 使用的 tokenizer（与 vLLM 内部一致）
        self.tokenizer = self.vllm_engine.get_tokenizer()
        
        # 加载 actor 模型 - 独立实例，非共享权重
        logger.info(f"Loading actor model: {self.config.model_path}")
        self.actor_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.actor_model = self.actor_model.to(GPU_DEVICE)
        self.actor_model.eval()
    
    def build_prompt_token_ids(
        self, 
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """使用与 verl 一致的 chat template 构建 prompt_token_ids"""
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        return prompt_token_ids
    
    def generate_with_vllm(
        self, 
        prompt_token_ids: List[int],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_tokens: int = 256,
    ) -> Tuple[List[int], List[float]]:
        """用 vLLM 生成 response"""
        from vllm import SamplingParams
        from vllm.inputs.data import TokensPrompt
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            logprobs=0,
        )
        
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        outputs = self.vllm_engine.generate([prompt], sampling_params)
        
        token_ids = outputs[0].outputs[0].token_ids
        
        log_probs = None
        if outputs[0].outputs[0].logprobs is not None:
            log_probs = [
                logprobs[token_ids[i]].logprob 
                for i, logprobs in enumerate(outputs[0].outputs[0].logprobs)
            ]
        
        return token_ids, log_probs
    
    def compute_log_prob_actor_style(
        self,
        prompt_token_ids: List[int],
        response_token_ids: List[int],
        temperature: float,
    ) -> List[float]:
        """
        用 actor 方式 recompute log_prob
        
        【修正说明 - logits 切片位置】
        
        Teacher-forcing 模式下：
        - input_ids 位置 t 的 logits 预测的是 token[t+1]
        - 即 logits[t] 对应的是 next_token
        
        对于 input_ids = [p0, p1, ..., p_{prompt_len-1}, r0, r1, ..., r_{response_len-1}]：
        - 位置 prompt_len-1 的 logits 预测的是 r0（第一个 response token）
        - 位置 prompt_len 的 logits 预测的是 r1
        - ...
        - 位置 prompt_len+response_len-2 的 logits 预测的是 r_{response_len-1}
        
        所以切片应该是：
        logits[:, prompt_len - 1 : prompt_len + response_len - 1, :]
        
        与 dp_actor.py:298 一致：
        logits[:, -response_length - 1 : -1, :]
        这等价于 [prompt_len - 1 : prompt_len + response_len - 1]
        """
        # 构建 input_ids = prompt + response
        input_ids = prompt_token_ids + response_token_ids
        prompt_len = len(prompt_token_ids)
        response_len = len(response_token_ids)
        
        # 转换为 tensor
        input_tensor = torch.tensor([input_ids], device=self.actor_model.device)
        
        # attention_mask: 全 1（无 padding）
        # 适用场景：单轮纯文本，无 padding
        # 不适用：多轮、有 padding、特殊 attention 模式
        attention_mask = torch.ones_like(input_tensor)
        
        # position_ids: 从 0 开始
        # 适用场景：单轮纯文本
        # 不适用：多轮、特殊 position 编码
        position_ids = torch.arange(0, input_tensor.size(1), device=self.actor_model.device).unsqueeze(0)
        
        # Forward (teacher-forcing)
        with torch.no_grad():
            outputs = self.actor_model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # ===== 修正后的切片位置 =====
        # 与 dp_actor.py:298 一致
        # logits[:, -response_length - 1 : -1, :] 
        # = logits[:, prompt_len - 1 : prompt_len + response_len - 1, :]
        response_logits = logits[0, prompt_len - 1 : prompt_len + response_len - 1, :]
        
        # 验证切片长度
        assert response_logits.shape[0] == response_len, \
            f"Slice length mismatch: {response_logits.shape[0]} vs {response_len}"
        
        # 与 dp_actor.py:298 一致：先除 temperature，再 log_softmax
        response_logits = response_logits / temperature
        log_probs = torch.log_softmax(response_logits, dim=-1)
        
        # gather 对应 token 的 log_prob
        response_tensor = torch.tensor(response_token_ids, device=self.actor_model.device)
        selected_log_probs = log_probs[torch.arange(len(response_token_ids), device=log_probs.device), response_tensor]
        
        return selected_log_probs.cpu().tolist()
    
    def compare(
        self,
        rollout_log_probs: List[float],
        actor_log_probs: List[float],
    ) -> Dict[str, float]:
        """比较两种 log_prob 计算方式的差异"""
        rollout = torch.tensor(rollout_log_probs)
        actor = torch.tensor(actor_log_probs)
        
        diff = rollout - actor
        abs_diff = torch.abs(diff)
        
        valid_mask = torch.isfinite(rollout) & torch.isfinite(actor)
        
        metrics = {
            "num_tokens": len(rollout_log_probs),
            "num_valid_tokens": valid_mask.sum().item(),
            "mean_diff": diff[valid_mask].mean().item() if valid_mask.any() else 0.0,
            "mean_abs_diff": abs_diff[valid_mask].mean().item() if valid_mask.any() else 0.0,
            "max_abs_diff": abs_diff[valid_mask].max().item() if valid_mask.any() else 0.0,
            "std_diff": diff[valid_mask].std().item() if valid_mask.any() else 0.0,
        }
        
        if valid_mask.sum() > 1:
            corr = torch.corrcoef(torch.stack([rollout[valid_mask], actor[valid_mask]]))
            metrics["pearson_corr"] = corr[0, 1].item()
        else:
            metrics["pearson_corr"] = 0.0
        
        rollout_mean = rollout[valid_mask].mean()
        actor_mean = actor[valid_mask].mean()
        
        if rollout_mean.isfinite() and actor_mean.isfinite():
            rollout_ppl = torch.exp(-rollout_mean).item()
            actor_ppl = torch.exp(-actor_mean).item()
            
            metrics["rollout_ppl"] = rollout_ppl
            metrics["actor_ppl"] = actor_ppl
            metrics["log_ppl_diff"] = (-rollout_mean - (-actor_mean)).item()
            metrics["ppl_ratio"] = rollout_ppl / actor_ppl if actor_ppl > 0 else float('inf')
        else:
            metrics["rollout_ppl"] = float('nan')
            metrics["actor_ppl"] = float('nan')
            metrics["log_ppl_diff"] = float('nan')
            metrics["ppl_ratio"] = float('nan')
        
        return metrics


async def run_single_experiment(
    comparator: LogProbComparator,
    messages: List[Dict[str, str]],
    experiment_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    """运行单次实验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"  temperature={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={repetition_penalty}")
    logger.info(f"{'='*60}")
    
    # Step 1: 构建 prompt_token_ids
    prompt_token_ids = comparator.build_prompt_token_ids(messages)
    logger.info(f"Prompt token ids length: {len(prompt_token_ids)}")
    
    # Step 2: vLLM 生成
    token_ids, rollout_log_probs = comparator.generate_with_vllm(
        prompt_token_ids, temperature, top_p, top_k, repetition_penalty
    )
    logger.info(f"Generated {len(token_ids)} tokens")
    logger.info(f"Rollout log_probs (first 5): {rollout_log_probs[:5]}")
    
    # Step 3: Actor 侧重算
    actor_log_probs = comparator.compute_log_prob_actor_style(
        prompt_token_ids, token_ids, temperature
    )
    logger.info(f"Actor log_probs (first 5): {actor_log_probs[:5]}")
    
    # Step 4: 比较
    metrics = comparator.compare(rollout_log_probs, actor_log_probs)
    
    logger.info(f"\nResults:")
    logger.info(f"  mean_diff:      {metrics['mean_diff']:.4f}")
    logger.info(f"  mean_abs_diff: {metrics['mean_abs_diff']:.4f}")
    logger.info(f"  max_abs_diff:  {metrics['max_abs_diff']:.4f}")
    logger.info(f"  pearson_corr:  {metrics['pearson_corr']:.4f}")
    logger.info(f"  log_ppl_diff:  {metrics['log_ppl_diff']:.4f}")
    logger.info(f"  ppl_ratio:     {metrics['ppl_ratio']:.4f}")
    
    return {
        "experiment": experiment_name,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "metrics": metrics,
    }


async def main():
    """主函数"""
    # 配置
    config = SanityCheckConfig(
        model_path="/Data/public/Qwen3-1.7B",
        tensor_parallel_size=1,
    )
    
    # 测试消息
    messages = [
        {"role": "user", "content": "Write a short story about a robot learning to dance."}
    ]
    
    # 初始化
    comparator = LogProbComparator(config)
    comparator.init_models()
    
    # ======== 实验 1: baseline ========
    exp1 = await run_single_experiment(
        comparator, messages,
        "baseline",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
    )
    
    # ======== 实验 2: train_config ========
    exp2 = await run_single_experiment(
        comparator, messages,
        "train_config",
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )
    
    # 汇总
    results = [exp1, exp2]
    
    # 保存结果到文件
    import os
    from datetime import datetime
    
    logs_dir = "/Data/wyh/verl/examples/sglang_multiturn/my_exp/debug/logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(logs_dir, f"result_{timestamp}.txt")
    
    with open(result_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Experiment':<20} {'mean_abs_diff':<12} {'pearson_corr':<12} {'log_ppl_diff':<12} {'ppl_ratio':<10}\n")
        f.write("-"*80 + "\n")
        for r in results:
            m = r["metrics"]
            f.write(f"{r['experiment']:<20} {m['mean_abs_diff']:<12.4f} {m['pearson_corr']:<12.4f} {m['log_ppl_diff']:<12.4f} {m['ppl_ratio']:<10.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
