# TextCraft Entropy-Based Prefix 切分全流程说明

日期：2026-04-06

## 1. 文档目的

这份文档说明 `/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix` 目录下两类基于 entropy 的 prefix 切分方案，重点回答四个问题：

1. 用哪个模型计算 entropy。
2. 两类 entropy 切分方案各自如何定义。
3. 从 stage1 到 stage7 的完整数据流如何衔接。
4. 第二种“平滑 + 累积 entropy”方案在确定切分点之后，如何继续派生出 `intersection_777` 和 `max_coverage` 两类最终数据集。

这里的“两个方案”指的是 scorer 维度上的两类方法：

1. `raw_topk`
2. `change_topk`

`interaction_user` 和 `interaction_assistant` 是信号域选择，不是另外两种 scorer。当前最重要、后续又被继续拆分成 rank 数据集的是：

1. `entropy_change_topk_w11_interaction_assistant_k3`

## 2. 目录与产物总览

核心脚本目录：

1. `scripts/01_split_teacher_jsonl.py`
2. `scripts/02_compute_teacher_entropy.py`
3. `scripts/03_merge_entropy_shards.py`
4. `scripts/05_export_entropy_prefix_candidates.py`
5. `scripts/06_replay_validate_entropy_candidates.py`
6. `scripts/07_canonicalize_entropy_validated.py`
7. `scripts/08_build_entropy_training_dataset.py`
8. `scripts/09_merge_entropy_training_shards.py`
9. `scripts/10_audit_entropy_release.py`
10. `scripts/common.py`

核心阶段产物：

1. `stage1_entropy/textcraft_teacher_entropy_step200.parquet`
2. `stage2_splits/prefix_candidates_entropy_topk.parquet`
3. `stage3_replay_validation/*_validated.parquet`
4. `stage4_canonicalized/*_validated_canonicalized.parquet`
5. `stage6_training_build/textcraft_prefix_*_step200.prompt_space_recomputed.full.parquet`
6. `stage7_audit_release/textcraft_prefix_*_step200.audited.parquet`

第二种方案后续重点目录：

1. `stage7_audit_release/change_topk_w11_interaction_assistant_k3_rank_split_intersection_777`
2. `stage7_audit_release/change_topk_w11_interaction_assistant_k3_rank_split_max_coverage`

## 3. 用于计算 Entropy 的模型

### 3.1 模型身份

entropy 不是用当前 RL 训练中的 online policy 即时计算的，而是先离线用一个固定 teacher 模型在完整 teacher 轨迹上做 teacher-forcing forward。

当前脚本中默认模型路径定义在 `scripts/common.py`：

`/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface`

因此，当前 entropy sidecar 使用的模型是：

1. `Qwen3-1.7B SFT`
2. checkpoint 为 `global_step_200`
3. tokenizer 与 model 都从该 HuggingFace 导出目录加载

### 3.2 计算方式

实现入口是 `scripts/02_compute_teacher_entropy.py`，核心计算在 `scripts/common.py` 的 `compute_token_entropy_batch()`。

对一条完整轨迹的 token 序列 `x_0, x_1, ..., x_T`，脚本先前向得到 `logits[:, :-1, :]`，再计算每个位置的 next-token predictive entropy：

`H_t = - sum_v p(v | x_<t) log p(v | x_<t)`

代码里用的等价形式是：

`entropy = logsumexp(logits) - sum(probs * logits)`

这里的含义是：

1. entropy 衡量的是“在当前 teacher-forcing 历史下，模型对下一个 token 的预测不确定性”。
2. 它不是环境状态熵，也不是 PPO/GRPO 意义上的 advantage 或 value uncertainty。
3. 它是纯语言模型层面的 next-token predictive uncertainty。

### 3.3 token 对齐口径

stage1 会保存字段：

1. `sequence_entropies`
2. `message_stats`
3. `assistant_message_stats`
4. `interaction_assistant_message_stats`
5. `user_message_stats`
6. `interaction_user_message_stats`

其中坐标约定是：

1. `sequence_entropies[i]` 对齐到完整轨迹中的 `token_position = i + 1`
2. message span 通过 tokenizer offset mapping 从 chat template 文本里精确映射出来
3. 后续所有 token 级切分，都是在这个完整轨迹 token 坐标系里进行

### 3.4 warmup 的处理

脚本会在 `common.py` 中识别 warmup message：

1. warmup user：配方介绍和环境说明
2. warmup assistant：`OK. I'll follow your instructions...`

后续 `interaction_*` 域会自动排除 warmup，只保留真实交互部分。

## 4. 全流程数据链路

### 4.1 stage1: 计算 entropy sidecar

1. `01_split_teacher_jsonl.py`
   把 `new_prefix_rl/stage0_teacher/teacher_normalized.jsonl` 按 `char_length` 均衡切成 shard。
2. `02_compute_teacher_entropy.py`
   对每个 shard 跑 teacher-forcing，输出整条轨迹的 per-token entropy。
3. `03_merge_entropy_shards.py`
   把 shard parquet 合并成 `textcraft_teacher_entropy_step200.parquet`。

这一步的本质是生成 keyed sidecar，不直接产出 prefix cut。

### 4.2 stage2: 导出 entropy-based cut candidates

入口脚本是 `05_export_entropy_prefix_candidates.py`。

输入：

1. stage1 entropy sidecar
2. `new_prefix_rl/stage0_teacher/teacher_normalized.parquet`

输出：

1. `stage2_splits/prefix_candidates_entropy_topk.parquet`

这一步会给每个样本导出若干 `candidate_uid`，并直接写出：

1. `cut_turn_idx`
2. `candidate_rank`
3. `prefix_messages`
4. `continuation_messages`
5. `prefix_actions`
6. `source_token_entropy`
7. `smoothed_entropy`
8. `cumulative_entropy`
9. `change_score`
10. `selection_score`

### 4.3 stage3: replay validation

入口脚本是 `06_replay_validate_entropy_candidates.py`。

它会把 `prefix_actions` 在本地 TextCraft server 上重放，比较：

1. cut 时刻 observation 是否一致
2. cut 之后的首个 assistant action 执行后 observation 是否一致

结果分成四类，本次全量 stage3 共 `17596` 条候选，实际占比如下：

1. `validated`：回放成功执行，且 cut 时刻以及可比较的下一步 observation 都和原始轨迹一致，占 `12524 / 17596 = 71.18%`。
2. `mismatch`：回放成功执行，也存在可比较字段，但回放得到的 observation 与原始轨迹不一致，占 `226 / 17596 = 1.28%`；例如原始轨迹里 cut 后 observation 是 `Inventory: wood, stone`，而回放后变成了另一组 inventory，或者执行 continuation 的首个 action 后 `Got ...` / `Crafted ...` 行不一致。
3. `unverifiable`：回放成功执行，但双方 observation 中没有足够共享的结构化字段可做严格比对，占 `4846 / 17596 = 27.54%`；例如某一步只出现自由文本提示，或回放侧与原始侧都没有可被脚本抽取的 `Inventory:` / `Got ...` / `Crafted ...` 字段，因此无法判断它们是否真正一致。
4. `error`：环境创建或 step 请求本身抛异常，导致这条候选没有完成有效回放，占 `0 / 17596 = 0.00%`。

只有 `validated` 会继续进入后面的训练数据链路。

### 4.3.1 为什么会出现 `unverifiable`

`unverifiable` 的主要来源不是“这条样本一定有问题”，而是当前 replay validator 的比对规则比较保守。

当前 `06_replay_validate_entropy_candidates.py` 只会从 observation 中抽取三类结构化字段：

1. `Inventory:`
2. `Got ...`
3. `Crafted ...`

只有当回放侧和原始侧至少共享其中一类字段时，脚本才会继续做严格比对；否则就会判成 `unverifiable`。

这意味着很多文本上看起来一致的样本，仍然可能因为“不属于这三类字段”而被归入 `unverifiable`。例如：

1. `Could not find enough items to craft ...`
2. `Could not find ...`
3. 只有自由文本提示、没有 `Inventory/Got/Crafted` 结构字段的 observation

在本次全量 stage3 的 `4846` 条 `unverifiable` 中，有 `3172` 条其实连 cut 时刻的原始 observation 字符串都是完全一致的，只是当前验证器没有把这类文本纳入可验证字段集合。

### 4.3.2 为什么会出现 `mismatch`

`mismatch` 更像是真正的“回放状态与原轨迹跑偏”，但这批数据里它主要不是 entropy 切点本身的问题，而是 replay 前的 action 解析存在失真。

当前 `prefix_actions` 是通过 `common.py` 中的 `extract_action()` 从 assistant message 文本里提取的，这个解析器对异常格式不够鲁棒，最常见的问题是：

1. 一个 assistant message 里如果写了多行 action，解析器会把它们拼成一个动作字符串，例如把 `get 2 wheat` 和 `get 1 cocoa beans` 拼成 `get 2 wheat get 1 cocoa beans`
2. assistant 的 `<think>` 段里如果提前出现了 `Action:` 文本，解析器可能会误抓 `<think>` 中的内容，而不是最终真正执行的 action block

这会导致 stage3 回放时执行了错误的 prefix action，进而让 cut 时刻的环境状态与原轨迹不一致，于是被标成 `mismatch`。

从实际数据看，这个问题非常集中：

1. 全部四个策略合计 `226` 条 `mismatch` 中，有 `222` 条都包含“多动作被串成一个 action”的前缀动作字符串
2. 对当前重点策略 `entropy_change_topk_w11_interaction_assistant_k3`，`50` 条 `mismatch` 中有 `49` 条属于这种多动作串接问题
3. 该策略剩下的 `1` 条 `mismatch` 也是 action 解析问题：assistant 在 `<think>` 中先写了一个 `Action:`，而真正最终输出的 action 是后面的 `inventory`，但解析器抓成了前者，导致回放偏离原轨迹

因此，当前 `mismatch` 的主因应理解为：

1. replay action extraction 不够鲁棒
2. 不是 entropy 切分方法本身系统性失效

### 4.3.3 多行 action 在实际系统里是如何被解析的

需要把“模型回复解析”和“环境执行解析”分开看，因为它们是两层不同逻辑。

第一层是 AgentGym 上层的回复解析。在 `agentenv/agentenv/envs/textcraft.py` 中，ReAct 模式会先从模型回复里提取 `Action:` 字段：

1. 如果回复里出现多个 `Action:` 标签，上层会直接返回 `Error: Only one 'Action' is allowed per response. Please adjust your response.`，这时根本不会调用底层 TextCraft 环境。
2. 如果只有一个 `Action:` 标签，但后面写了多行命令，例如 `Action:\nget 2 wheat\nget 1 cocoa beans`，上层通常只会提取第一条，也就是 `get 2 wheat`，后面的命令不会被顺序执行。

第二层才是 TextCraft 环境本身的 `/step` 解析。在 `agentenv_textcraft/environment.py` 中，环境只接收一个已经提取好的 `action: str`，然后用正则去匹配：

1. `craft ... using ...`
2. `get N item`
3. `inventory`

环境没有“逐行执行多条命令”的机制，因此它的语义始终是“一次 step 只执行一条 action”。

这会带来两个直接后果：

1. 如果上层把多行 action 截成第一行，那么实际执行的就只有第一条命令。
2. 如果某个不够鲁棒的 parser 把多行 action 压平成一个字符串，例如把 `get 2 wheat` 和 `get 1 cocoa beans` 拼成 `get 2 wheat get 1 cocoa beans`，那么环境不会把它拆成两步，而是把整串当成一个 `get` 动作去解析，进而把 `wheat get 1 cocoa beans` 当成 item 名称，通常会导致非法动作或错误状态回放。

因此，从系统设计上讲，TextCraft 当前并不支持“一次回复实际执行多条 action”；多行 action 要么在上层被拒绝，要么只执行第一条，要么被错误拼接后导致 `mismatch`。

### 4.4 stage4: canonicalization

入口脚本是 `07_canonicalize_entropy_validated.py`。

这里会把 `validated` 候选重写成训练 prompt，prompt 结构与当前 Prefix-GRPO 主实验一致：

1. `system`
2. canonicalized `prefix history`
3. `cut-state user observation`

其中：

1. warmup message 被过滤
2. assistant prefix history 被规范为 `Action: [[ ... ]]` 形式
3. `continuation_messages` 的第一个 user message 作为 cut-state observation 追加到 prompt 尾部

### 4.5 stage6: 在 canonicalized prompt 上重算 old logprob

入口脚本是 `08_build_entropy_training_dataset.py`。

这里不复用早期 full-trajectory old-logprob sidecar，而是直接在当前 canonicalized prompt 上重新做 teacher-forcing，补齐训练真正需要的字段：

1. `assistant_prefix_old_log_probs`
2. `prefix_mask`
3. `prefix_token_count`
4. `assistant_prefix_span`
5. `prefix_coordinate_system = canonicalized_prompt`

这一步非常关键，因为当前主实验约定里：

1. `assistant_prefix_span`
2. `prefix_mask`
3. `assistant_prefix_old_log_probs`

都必须定义在 canonicalized prompt-space 上，而不是原始 full trajectory 上。

### 4.6 stage7: audit and release

入口脚本是 `10_audit_entropy_release.py`。

审计内容包括：

1. 必备字段是否齐全
2. `candidate_uid` 是否唯一
3. `(item_id, sample_idx, strategy, cut_turn_idx)` 是否重复
4. `assistant_prefix_old_log_probs` 与 `prefix_mask` 长度是否一致
5. `prefix_token_count` 是否等于 `sum(prefix_mask)`
6. 根据当前 `prompt` 反向重建的 prefix span / mask / blocks 是否与存档一致
7. 是否和 stage3 drop set 有重叠

通过后才会在 `stage7_audit_release` 下写出 `.audited.parquet`。

## 5. 两类 Entropy Prefix 切分方案

## 5.1 共同前置：domain token 的构建

两类 scorer 都先经过 `build_domain_tokens()`。

该函数会把 `message_stats` 展开成按 `token_position` 排序的 token 序列，并为每个 token 计算它对应的切点：

1. 如果 token 来自 assistant message，则映射到当前 assistant turn
2. 如果 token 来自 user message，则映射到前一个 assistant turn

代码中这个规则被明确写成：

1. `mapping_mode = current_or_previous_assistant`

所以，这里的切分单位始终是 assistant turn，而不是原始 token 本身。

`cut_turn_idx` 的语义是：

1. 以 assistant turn 为单位的 0-based index
2. 后续 `split_messages_at_cut_turn()` 会把该 assistant turn 以及它之前的所有 message 放入 prefix
3. 剩余 message 放入 continuation

### 5.2 方案一：`raw_topk`

定义最直接：

1. 在选定 domain 上取 token 级 `raw_entropy`
2. 直接按 `raw_entropy` 排序
3. 选择 top-k token
4. 但如果多个 token 映射到同一个 `cut_turn_idx`，只保留分数最高的那个

当前默认参数：

1. `top_k = 3`
2. `min_domain_gap = 0`

因此每个样本最终通常得到 `2` 到 `3` 个唯一切点，而不是机械固定 `3` 个。

### 5.3 方案二：`change_topk`

这是第二种、也是本文重点解释的方案。它不是直接用原始 entropy 排序，而是先平滑，再把“平滑熵的变化强度”当作选点信号。

当前默认配置：

1. `change_window = 11`
2. `top_k = 3`
3. 重要策略实例：`entropy_change_topk_w11_interaction_assistant_k3`

## 6. 第二种方案的精确定义

### 6.1 输入序列

先从选定 domain 里取出按 `token_position` 排序后的 entropy 序列：

`e_0, e_1, ..., e_{n-1}`

这里的 domain 在当前重点数据集里是：

1. `interaction_assistant`

即：

1. 只看 assistant token
2. 排除 warmup assistant

### 6.2 平滑

脚本使用 `moving_average(values, window=11)` 做中心滑动平均。令 `r = 5`，则：

`hat_e_t = average(e_{max(0, t-r)} ... e_{min(n-1, t+r)})`

也就是说：

1. 每个位置看左右各 5 个 token
2. 边界位置自动截断窗口
3. 输出保存为 `smoothed_entropy`

### 6.3 累积

在平滑后做前缀累积：

`C_t = sum_{i <= t} hat_e_i`

输出保存为 `cumulative_entropy`。

这里要特别说明：

1. `cumulative_entropy` 被保存到了 stage2 候选里
2. 但当前代码真正用于排序的不是 `C_t` 本身
3. 当前实现把它解释成“累计曲线斜率变化”的辅助量，实际排名信号是平滑熵的一阶差分绝对值

### 6.4 实际排序分数

当前代码的定义是：

1. `change_score[0] = 0`
2. `change_score[t] = |hat_e_t - hat_e_{t-1}|`，当 `t > 0`

也就是：

`change_score = |Delta hat_e_t|`

这和“对累计曲线 `C_t` 取二阶变化强度”在离散直觉上是一致的，因为：

1. `Delta C_t = hat_e_t`
2. `Delta^2 C_t = Delta hat_e_t`

最终用于排序的字段是：

1. `selection_score = change_score`

这里还需要特别强调一点：

1. 当前实现只看变化幅度，不看变化方向
2. 如果平滑熵从高到低下降，例如 `hat_e_t - hat_e_{t-1} < 0`，代码会先得到负的一阶差分
3. 但由于后面取了绝对值 `abs(...)`，这类“熵降低”的变化同样会被记为高分候选
4. 因此当前方法不会区分“熵突然升高”和“熵突然降低”，只关心“跳变是否足够大”

所以第二种方案更准确的描述应该是：

1. 先做窗口为 11 的平滑
2. 再把平滑熵的跳变强度当作切点信号
3. `cumulative_entropy` 主要用于保存累计曲线，帮助解释该信号来自“累计曲线的斜率变化”

不能把当前实现误写成“直接按 cumulative entropy 最大的点切”，因为代码不是这么做的。

### 6.5 从 token 到 candidate rank

对第二种方案，单个样本的候选生成顺序是：

1. 构造 domain token 列表
2. 计算每个 token 的 `smoothed_entropy`
3. 计算每个 token 的 `cumulative_entropy`
4. 计算每个 token 的 `change_score`
5. 按 `selection_score = change_score` 降序排序
6. 如果多个 token 映射到同一个 `cut_turn_idx`，只保留分数最高的那个
7. 最多保留 `top_k = 3` 个唯一 assistant turn
8. 依排序结果赋予 `candidate_rank = 1/2/3`

因此：

1. `rank1` 是该样本中变化最强的合法切点
2. `rank2` 是第二强且 turn 不重复的切点
3. `rank3` 是第三强且 turn 不重复的切点

## 7. Prefix 的具体切分逻辑

一旦 `cut_turn_idx` 被确定，真正的 prefix 切分是统一的，不区分 `raw_topk` 还是 `change_topk`。

具体由 `common.py` 的 `split_messages_at_cut_turn()` 完成：

1. 找出所有 assistant message 的 message index
2. 用 `cut_turn_idx` 选中对应 assistant message
3. 把该 assistant message 及其之前所有 message 放入 `prefix_messages`
4. 把之后所有 message 放入 `continuation_messages`

也就是说，切分点语义是：

1. prefix 以某个 assistant turn 结束
2. continuation 从这之后的下一条 message 开始
3. 在 stage4 canonicalization 时，会再把 continuation 的第一条 user message 作为 cut-state observation 放回 prompt

同时，`extract_actions_from_messages()` 会从 `prefix_messages` 中抽取 assistant action，形成：

1. `prefix_actions`

这是 stage3 replay validation 直接使用的动作前缀。

## 8. 当前四组具体策略

同一套 stage2 exporter 当前一次性导出四组策略：

1. `entropy_raw_topk_interaction_user_k3`
2. `entropy_change_topk_w11_interaction_user_k3`
3. `entropy_raw_topk_interaction_assistant_k3`
4. `entropy_change_topk_w11_interaction_assistant_k3`

它们共享同一套模型与流程，只在两个维度上不同：

1. scorer：`raw_topk` 或 `change_topk`
2. domain：`interaction_user` 或 `interaction_assistant`

当前最值得关注的是 assistant 域，尤其是第二种方案，因为它后续被继续整理成多个 stage7 rank 数据集。

## 9. 第二种方案的 stage3 到 stage7 主结果

下面只列最关键的 `entropy_change_topk_w11_interaction_assistant_k3`：

stage3 validated：

1. `rows = 3197`
2. `unique_sample_uid = 1295`
3. `candidate_rank_counts = {1: 1083, 2: 1086, 3: 1028}`

stage7 audited 基表：

1. `stage7_audit_release/textcraft_prefix_entropy_change_topk_w11_interaction_assistant_k3_step200.audited.parquet`
2. `rows = 3197`
3. `unique_sample_uid = 1295`
4. `candidate_rank_counts = {1: 1083, 2: 1086, 3: 1028}`

这份 `.audited.parquet` 是后面两个 rank-split 目录的共同来源基表。

## 10. 第二种方案确定切分点之后的两类后处理数据集

这一部分最容易混淆。

这里的“两类方案”不是重新计算 cut point 的新算法，而是在第二种方案已经产出并通过审计之后，对已确定的 `rank1/rank2/rank3` 候选做两种不同的数据集整理方式：

1. `intersection_777`
2. `max_coverage`

### 10.1 一个重要事实

在 `scripts/`、`manifests/` 和当前目录文档中，确实没有找到这两个目录的专门 checked-in 生成脚本。

但是，继续追到 2026-04-07 凌晨的 Codex 对话历史后，可以从日志里恢复出实际执行过的生成命令，时间是 `2026-04-07 00:39:24 CST`，记录位置是 `/home/wyh/.codex/log/codex-tui.log`。

这条命令不是仓库内脚本，而是一段临时执行的 `pyarrow` 处理代码；它直接从基表：

`textcraft_prefix_entropy_change_topk_w11_interaction_assistant_k3_step200.audited.parquet`

生成了两个后处理目录：

1. `change_topk_w11_interaction_assistant_k3_rank_split_max_coverage`
2. `change_topk_w11_interaction_assistant_k3_rank_split_intersection_777`

其核心逻辑可以概括为：

1. 读入基表，并按 `candidate_rank in {1, 2, 3}` 分别过滤出三个 rank 子表
2. 每个 rank 子表原样写入 `max_coverage` 目录
3. 分别收集三个 rank 子表的 `sample_uid` 集合，并计算交集 `S1 ∩ S2 ∩ S3`
4. 在每个 rank 子表内进一步过滤 `sample_uid in (S1 ∩ S2 ∩ S3)`，再写入 `intersection_777` 目录

也就是说，这里的两类后处理数据不是通过新的切分算法生成的，而是对同一个 audited 基表做了两层不同的集合过滤。

下面的 10.2 到 10.7，不再只是“从产物反推”的定义，而是和这条实际生成命令完全一致的集合化表达。

### 10.2 `rank` 的定义

这里的 `rank` 指的是 `candidate_rank`。

它不是：

1. prefix 的长度分桶
2. cut 点在轨迹中的先后顺序
3. 全数据集范围内的全局排名

它的准确含义是：

1. 对同一个 `sample_uid`，先把候选切点按 `selection_score` 从高到低排序
2. 分数最高的候选记为 `rank1`
3. 分数第二高的候选记为 `rank2`
4. 分数第三高的候选记为 `rank3`

当前这批数据之所以最多只有 `rank1/rank2/rank3`，是因为候选导出时配置的是 `top_k = 3`，而不是因为方法本身只能有 3 个 rank。

对第二种 `change_topk` 方案，`rank` 的排序依据不是 prefix 长度，而是：

1. 先对 token 级 entropy 做 `window = 11` 平滑
2. 再计算 `change_score = |Delta smoothed_entropy|`
3. 最终按 `change_score` 排序选 top-k

如果两个候选的分数完全相同，脚本才会进一步按更靠前的 `token_position` 做 tie-break。

因此，`rank1` 只表示“这个样本内部最强的候选切点”，不表示“切得最早”或“prefix 最短”；真正决定切分位置的是 `cut_turn_idx`。

### 10.3 为什么一个样本会同时拥有多个 rank

同一个样本通常不只对应一个候选切点。

流程是：

1. 先把 domain 内的很多 token 映射到 assistant turn 对应的 `cut_turn_idx`
2. 再按分数排序，从高到低挑出 top-k 个候选
3. 但如果多个 token 最终映射到同一个 `cut_turn_idx`，只保留该切点里分数最高的那个
4. 留下来的不同切点，分别标成 `rank1`、`rank2`、`rank3`

所以，一个 `sample_uid` 最终会展开成多条候选记录，每条记录都有：

1. 同一个 `sample_uid`
2. 不同的 `candidate_rank`
3. 可能不同的 `cut_turn_idx`
4. 对应不同的 `candidate_uid = sample_uid + strategy + rank`

一个最直观的例子是：

1. 某条轨迹一共有 6 个 assistant turn
2. 候选排序后，分数最高的切点落在 `cut_turn_idx = 4`，于是它是 `rank1`
3. 分数第二高的切点落在 `cut_turn_idx = 1`，于是它是 `rank2`
4. 分数第三高的切点落在 `cut_turn_idx = 3`，于是它是 `rank3`

这说明：

1. `rank1` 的 prefix 可能比 `rank2` 更长
2. `rank2` 也可能比 `rank3` 更短
3. `rank` 和长度没有单调对应关系

### 10.4 为什么 `S1`、`S2`、`S3` 不是同一批样本

对基表 `textcraft_prefix_entropy_change_topk_w11_interaction_assistant_k3_step200.audited.parquet`，定义：

1. `S1 = {sample_uid | 样本拥有 validated 的 rank1 切点}`
2. `S2 = {sample_uid | 样本拥有 validated 的 rank2 切点}`
3. `S3 = {sample_uid | 样本拥有 validated 的 rank3 切点}`

这里最关键的一点是：`rank1/rank2/rank3` 的编号在 replay 验证之前就已经固定了，后续不会因为某个 rank 失败而重新补位、重新编号或向前挪动。

stage3 replay 验证会把每个候选分别判成：

1. `validated`
2. `mismatch`
3. `unverifiable`
4. `error`

后续 stage4 只继续读取 `*_validated.parquet`，也就是只保留验证通过的候选。

因此，如果某个样本初始时有 `rank1/rank2/rank3` 三个候选，但 replay 结果是：

1. `rank1 = validated`
2. `rank2 = mismatch`
3. `rank3 = validated`

那么最终结果就是：

1. 这个样本会出现在 `S1`
2. 这个样本不会出现在 `S2`
3. 这个样本会出现在 `S3`
4. `rank3` 仍然叫 `rank3`，不会自动补位成新的 `rank2`

这就是为什么 `S1`、`S2`、`S3` 不是完全相同的集合。

实际规模是：

1. `|S1| = 1083`
2. `|S2| = 1086`
3. `|S3| = 1028`

样本按“最终拥有几个 validated rank”统计为：

1. 只有 1 个合法 rank：`170`
2. 有 2 个合法 rank：`348`
3. 同时有 3 个合法 rank：`777`

按具体组合拆开是：

1. 仅 rank1：`51`
2. 仅 rank2：`55`
3. 仅 rank3：`64`
4. rank1 + rank2：`161`
5. rank1 + rank3：`94`
6. rank2 + rank3：`93`
7. rank1 + rank2 + rank3：`777`

### 10.5 `intersection_777`

先定义交集：

`I = S1 ∩ S2 ∩ S3`

实际大小：

`|I| = 777`

`intersection_777` 目录里的三个文件，等价于：

1. rank1 文件 = 基表中过滤 `candidate_rank = 1` 且 `sample_uid in I`
2. rank2 文件 = 基表中过滤 `candidate_rank = 2` 且 `sample_uid in I`
3. rank3 文件 = 基表中过滤 `candidate_rank = 3` 且 `sample_uid in I`

这意味着：

1. 这里只保留那些同时拥有 `validated rank1`、`validated rank2`、`validated rank3` 的样本
2. 三个文件拥有完全相同的 `sample_uid` 集合
3. 每个样本在三个文件中各保留一个合法切点
4. 三个文件之间可以做严格配对比较

实际规模：

1. `rank1.intersection_777.audited.parquet`: `777` 行
2. `rank2.intersection_777.audited.parquet`: `777` 行
3. `rank3.intersection_777.audited.parquet`: `777` 行

任务覆盖：

1. `unique_task_id = 312`

这个方案的用途是：

1. 控制样本集合完全一致
2. 只让切点 rank 发生变化
3. 适合做 rank1 vs rank2 vs rank3 的严格对照实验

它的代价是：

1. 只保留 777 个样本
2. 任何缺少某一个 validated rank 的样本都会被整体丢掉
3. 任务覆盖从 364/361/362 降到 312

### 10.6 `max_coverage`

`max_coverage` 目录里的三个文件，等价于：

1. rank1 文件 = 基表中过滤 `candidate_rank = 1`
2. rank2 文件 = 基表中过滤 `candidate_rank = 2`
3. rank3 文件 = 基表中过滤 `candidate_rank = 3`

也就是说，它就是“每个 rank 各自保留全部可用 validated 样本”。

这里不存在额外的交集约束；某个样本只要某个 rank 通过了验证，就会出现在对应 rank 文件中，即使它的其他 rank 失败了也不影响保留。

实际规模：

1. `rank1.max_coverage.audited.parquet`: `1083` 行
2. `rank2.max_coverage.audited.parquet`: `1086` 行
3. `rank3.max_coverage.audited.parquet`: `1028` 行

任务覆盖：

1. rank1：`364`
2. rank2：`361`
3. rank3：`362`

这个方案的特点是：

1. 每个 rank 单独看时覆盖最大
2. 不强制三个 rank 使用同一批样本
3. 更适合“每个 rank 独立训练，追求样本量最大化”的场景

代价是：

1. rank1/rank2/rank3 的样本集合不完全相同
2. 不能直接把三个 rank 的结果解释成“同一批样本只改切点”的严格对照

### 10.7 两类后处理方案的关系

从产物比对结果可以精确确认：

1. `intersection_777` 的每个 rank 文件，都是对应 `max_coverage` 同 rank 文件的真子集
2. 同 rank 下，`intersection_777` 与 `max_coverage` 的重叠 `candidate_uid` 都是 `777`
3. `max_coverage` 才是每个 rank 的全集
4. `intersection_777` 是为“同样本对比”额外施加的交集约束

如果从“某个 rank 失败以后会怎样”这个角度看，两类方案的差别更直观：

1. 在 `max_coverage` 里，失败的 rank 会被删除，但同一样本的其他 validated rank 仍然保留
2. 在 `intersection_777` 里，只要三档 rank 里缺任何一个，这个样本就不会进入最终交集数据集

因此，如果写成集合表达式：

1. `rank_r.max_coverage = Base[candidate_rank = r]`
2. `rank_r.intersection_777 = Base[candidate_rank = r and sample_uid in (S1 ∩ S2 ∩ S3)]`

这就是这两个目录最准确的定义。

## 11. 应该如何理解这两类后处理方案

如果目标是：

1. 比较 rank1、rank2、rank3 哪个切得更合适
2. 希望三组训练只在切点上不同，样本完全相同

就应该用：

1. `intersection_777`

如果目标是：

1. 单独训练某一个 rank
2. 希望该 rank 拿到尽可能多的合法样本

就应该用：

1. `max_coverage`

两者不是竞争关系，而是服务于不同实验问题。

## 12. 当前文档的核心结论

1. 两类 entropy 切分方案使用的是同一个离线 teacher 模型：`Qwen3-1.7B SFT global_step_200`。
2. `raw_topk` 直接按 token 原始 entropy 选点。
3. `change_topk` 先做 `window=11` 平滑，再记录累计 entropy，但真正用于排序的是 `|Delta smoothed_entropy|`，因此熵升高和熵降低都会被当作有效变化。
4. token 不直接成为切点，必须先映射到 assistant turn，映射规则是 `current_or_previous_assistant`。
5. `cut_turn_idx` 一旦确定，就统一按 assistant turn 切成 `prefix_messages` 和 `continuation_messages`。
6. 第二种方案当前最重要的基表是 `textcraft_prefix_entropy_change_topk_w11_interaction_assistant_k3_step200.audited.parquet`。
7. 当前的 `rank1/rank2/rank3` 是同一个样本内部按 `selection_score` 排出的 top-3 候选，不对应 prefix 长度，也不对应切点先后。
8. `max_coverage` 是按 rank 从基表直接切开，各 rank 保留全部合法样本。
9. `intersection_777` 是先取同时拥有 rank1/rank2/rank3 三个合法切点的 777 个样本，再按 rank 切开。
10. `intersection_777` 适合做同样本对照，`max_coverage` 适合做单 rank 最大覆盖训练。
