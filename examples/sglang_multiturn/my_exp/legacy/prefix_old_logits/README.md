# SFT Old Logprob 预处理方案（完整轨迹级）

## 一、方案概述

### 旧方案 vs 新方案

| | 旧方案（已废弃） | 新方案（当前） |
|---|---|---|
| 输入 | `prefix_history_canonicalized.parquet`（已切分的 prefix 数据） | `textcraft_trajectories.jsonl`（完整 teacher 轨迹） |
| 输出 | 扩展到 prefix 的 parquet | sidecar parquet |
| 缓存设计 | 只服务当前 fixed cut | 支持任意 prefix cut 策略 |
| 标识字段 | `task_id` + `index` | `item_id` + `sample_idx` |
| 关键字段 | `sequence_log_probs`, `assistant_prefix_mask` | `sequence_old_logprobs`, `assistant_turn_spans` |

### 为什么不再用旧的 prefix parquet

**根本原因**：旧方案的缓存设计和"当前 fixed cut"绑死了。

- 旧方案输入是已经按 `fixed_ratio_0.4` 切分好的 prefix 数据
- 如果换成 entropy peak、cumulative 50% 等其他切分策略，需要重新跑一遍 old logprob 计算

**新方案的优势**：
- 输入是完整的 teacher trajectory
- 缓存的是完整轨迹级别的 old logprob
- 后续训练时可以根据不同 cut 策略从完整缓存中切出需要的 prefix
- 一次预处理，多种 cut 策略复用

---

## 二、输入输出

### 输入

```
/Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl
```

每条样本是完整的 teacher 尝试，包含：
- `conversations`：完整对话历史
- `item_id`：任务唯一标识
- `sample_idx`：第几次尝试
- `success`：是否成功
- `reward`：奖励值

### 输出

Sidecar parquet 文件，例如：
```
textcraft_trajectories_old_logprobs.parquet
```

**Join Key**：`item_id + sample_idx`（唯一标识）

---

## 三、实际输出字段

| 字段 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `item_id` | str | scalar | 任务唯一标识 |
| `sample_idx` | int | scalar | 第几次尝试（配合 item_id） |
| `token_length` | int | scalar | 完整序列 token 长度 |
| `old_logprob_length` | int | scalar | = token_length - 1 |
| `assistant_token_count` | int | scalar | assistant token 总数 |
| `assistant_turn_count` | int | scalar | assistant turn 总数 |
| `success` | int | scalar | 原始成功标记 |
| `reward` | float | scalar | 原始奖励值 |
| `sequence_old_logprobs` | List[float] | `(token_length - 1,)` | **per-token scalar old logprob** |
| `assistant_mask` | List[float] | `(token_length,)` | assistant token 位置标识 |
| `assistant_turn_spans` | List[Dict] | `[{"start": int, "end": int, "turn_idx": int}, ...]` | 每个 assistant turn 的起止位置 |

---

## 四、Span / Logprob 坐标对齐

这是最关键的契约说明。

### Token 坐标 vs Logprob 坐标

- **`assistant_turn_spans`**：原始 token 坐标，定义在完整 `input_ids` 上，区间为 `[start, end)`
- **`sequence_old_logprobs`**：next-token 坐标，长度是 `token_length - 1`

### 对齐规则

```python
# input_ids[i] 对应的 old logprob 存储在 sequence_old_logprobs[i-1]
# 即：sequence_old_logprobs[i] 对应 input_ids[i+1] 的 old logprob

# 例如：
# input_ids = [bos, A, B, C, D, eos]  # token 坐标 0,1,2,3,4,5
# sequence_old_logprobs = [logp(A), logp(B), logp(C), logp(D), logp(eos)]  # 长度 5
#                               索引 0      1      2      3      4
```

### 从 Turn Span 映射到 Old Logprob Span

如果某个 assistant turn 的 token span 是 `[start, end)`，则：

```python
# token span [start, end) 映射到 old_logprob 索引 [start-1, end-1)
# 需要处理边界：start=0 时，start-1 = -1，需要 clip 到 0

# 伪代码：
old_logprob_start = max(0, start - 1)
old_logprob_end = end - 1  # 不需要 clip，因为 end <= token_length，end-1 <= token_length-1
old_logprob_span = sequence_old_logprobs[old_logprob_start:old_logprob_end]
```

**重要**：不要直接用 `assistant_mask[1:]` 过滤，这会选出**整条完整轨迹**的所有 assistant token，不是当前 cut 后的 prefix assistant token。

正确做法是：
1. 先根据 cut 策略（如 fixed_ratio_0.4）从 `assistant_turn_spans` 恢复 prefix span
2. 再把 prefix span 映射到 old_logprob 坐标
3. 最后提取对应的 old logprob

### 边界条件

- 当 `start = 0` 时（第一个 token 就是 assistant），`start - 1 = -1`，需要 clip 到 0
- `end` 不会超过 `token_length`，所以 `end - 1` 不会超过 `token_length - 1`

---

## 五、Tokenization 对齐保证

### 核心实现：真实 tokenization 差分

脚本使用**完全真实**的 tokenization 来推导 assistant turn spans：

```python
def compute_token_spans_by_diff(tokenizer, conversations):
    """
    逐条增量构造消息前缀，用真实 tokenization 差分确定 span
    """
    spans = []
    cumulative_length = 0
    
    for i in range(len(conversations)):
        # 对前 i+1 条消息做真实 tokenization
        _, current_length = tokenize_conversations(tokenizer, conversations[:i+1])
        
        # 差分得到第 i 条消息的 span
        start_pos = cumulative_length
        end_pos = current_length
        role = conversations[i]["role"]
        
        spans.append((start_pos, end_pos, role))
        cumulative_length = current_length
    
    return spans
```

**关键点**：
- 必须完全复用同一个 `apply_chat_template + tokenizer(...)` 路径
- assistant turn spans 来源于真实 tokenization 差分，而不是手工估算

---

## 六、后续训练如何消费这个 Sidecar

### 1. Join 侧car

```python
# 用 (item_id, sample_idx) 把 sidecar 和训练样本 join
train_df = train_parquet.merge(
    sidecar_df,
    on=["item_id", "sample_idx"],
    how="left"
)
```

### 2. 从完整缓存中恢复 prefix span

根据 cut 策略（fixed_ratio、entropy peak、cumulative 等），从 `assistant_turn_spans` 中恢复 prefix 的 token 范围：

```python
# 示例：fixed_ratio = 0.4（turn-based）
# 设共有 n 个 assistant turns
# 第 i 个 assistant turn 的相对位置是 q(i) = i / (n - 1)
# 选择第一个满足 q(i) >= fixed_ratio 的 assistant turn 作为 cut point

n = len(assistant_turn_spans)

if n == 1:
    # 只有一个 turn，直接取它
    target_turn_idx = 0
else:
    # 找到第一个满足 q(i) >= fixed_ratio 的 turn
    target_turn_idx = 0
    for i in range(n):
        if i / (n - 1) >= fixed_ratio:
            target_turn_idx = i
            break

# prefix span
prefix_start = assistant_turn_spans[0]["start"]
prefix_end = assistant_turn_spans[target_turn_idx]["end"]

# 例如：n=5, fixed_ratio=0.4
# - turn 0: q=0/4=0.0
# - turn 1: q=1/4=0.25
# - turn 2: q=2/4=0.5 >= 0.4 -> cut here
```

### 3. 映射到 old_logprob 坐标

将 prefix token span 映射到 `sequence_old_logprobs` 坐标：

```python
# 合并所有 prefix span
prefix_start = min(span["start"] for span in prefix_spans)
prefix_end = max(span["end"] for span in prefix_spans)

# 映射到 old_logprob 坐标
old_logprob_start = max(0, prefix_start - 1)
old_logprob_end = prefix_end - 1

# 提取 prefix old logprob
prefix_old_logprobs = sequence_old_logprobs[old_logprob_start:old_logprob_end]
```

**重要**：不要直接用 `assistant_mask[1:]` 过滤来获取 prefix old logprob，这会选出整条完整轨迹的所有 assistant token，不是当前 prefix 的。

### 4. 训练消费

```python
# unified GRPO loss 示例
loss = -ratio * advantages * log_probs  # 其中 log_probs 来自 prefix_old_logprobs
```

---

## 七、预处理脚本使用说明

### 输入

```bash
python precompute_sft_old_logprobs_from_trajectories.py \
    --input_path /Data/wyh/datasets/Sampling-Data/textcraft_MiniMax-M2.1_20260307_150412/textcraft_trajectories.jsonl \
    --output_path /path/to/textcraft_trajectories_old_logprobs.parquet \
    --model_path /Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface \
    --device cuda \
    --max_samples 10  # 可选，用于调试
```

### 或使用运行脚本

```bash
bash run_precompute_from_trajectories.sh
```

---

## 八、静态校验（fail-fast）

脚本在处理过程中会自动校验以下条件，**任何一项不满足都会直接报错退出**：

1. `len(sequence_old_logprobs) == token_length - 1`
2. `len(assistant_mask) == token_length`
3. `assistant_turn_spans` 不越界（0 <= start < end <= token_length）
4. `assistant_turn_spans` 之间不重叠
5. `assistant_token_count > 0`
6. `(item_id, sample_idx)` 唯一（可用于后续 join）

---

## 九、字段命名与训练代码的对应

| 预处理字段 | 训练代码中的对应 |
|-----------|----------------|
| `sequence_old_logprobs` | 完整序列的 per-token scalar old logprob |
| `assistant_mask` | 用于过滤 assistant token |
| `assistant_turn_spans` | 用于恢复不同 cut 策略的 prefix span |
| `item_id` + `sample_idx` | join key |
| `token_length` | 用于验证 |
| `assistant_token_count` | 统计信息 |

---

## 十、示例输出

处理完成后，第一条样本的典型输出：

```
item_id: textcraft_31
sample_idx: 0
token_length: 1024
len(sequence_old_logprobs): 1023
assistant_token_count: 256
assistant_turn_count: 4
len(assistant_mask): 1024
assistant_turn_spans: [{'start': 12, 'end': 80, 'turn_idx': 1}, {'start': 150, 'end': 400, 'turn_idx': 2}, ...]

Mask 和 Log-probs 对齐验证:
  len(assistant_mask) == token_length: True
  len(sequence_old_logprobs) == token_length - 1: True
  assistant_turn_spans 不越界: True
  assistant_turn_spans 不重叠: True

Assistant token logprob 示例:
  assistant_token_count: 256
  assistant logprob 前5个: [-0.12, -0.34, -0.56, -0.78, -0.90]
  assistant logprob 均值: -1.2345
```