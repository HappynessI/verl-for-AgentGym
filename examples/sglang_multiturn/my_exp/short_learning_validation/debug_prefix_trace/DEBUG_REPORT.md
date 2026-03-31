# Prefix Trace Debug Report

## Run Info

- **Timestamp**: 2026-03-24 14:47 (run started at 13:47)
- **Script**: `run_debug.sh`
- **GPU**: 0
- **Log file**: `logs/debug_run_20260324_134741.log`
- **Exit code**: 0 (hydra exit 0, but Ray task threw exception)
- **algorithm.optimize_prefix_tokens**: true
- **algorithm.adv_estimator**: turn_full_trajectory
- **train_batch_size**: 2
- **ppo_mini_batch_size**: 2
- **rollout.n**: 1
- **total_training_steps**: 1
- **Data**: `textcraft_validated_prefix_history_canonicalized_with_prefix_old_logprobs_step200_v2.parquet`
- **Parquet rows**: 1093; all pass `max_prompt_length=4096` filter

---

## COMPLETE TRACEBACK

```
ray.exceptions.RayTaskError(TypeError): ray::TurnPrefixTaskRunner.run()
  File "recipe/wyh_exp/main_train.py", line 47, in main
    run_ppo(config, task_runner_class=task_runner_class)
  File "verl/trainer/main_ppo.py", line 96, in run_ppo
    ray.get(runner.run.remote(config))
  File "verl/trainer/ppo/ray_trainer.py", line 947, in fit
    trainer.fit()
  File "verl/trainer/ppo/ray_trainer.py", line 1670, in fit
    cached_prefix_logprobs_full_seq = torch.from_numpy(
        np.array(cached_prefix_logprobs_tensor)
    ).float()
TypeError: can't convert np.ndarray of type numpy.object_.
The only supported types are: float64, float32, float16, complex64,
complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
```

Crash location: `ray_trainer.py:1670` — inside the `optimize_prefix_tokens=True`
branch, immediately after `DEBUG_1592` prints the dtype as `object`.

---

## DEBUG_EXPAND_BEFORE (ray_trainer.py ~line 1423)

### assistant_prefix_old_log_probs

```
key=assistant_prefix_old_log_probs, orig_len=2, orig_val type=numpy.ndarray,
dtype=object, shape=(2,)
  RAGGED object array: 5 per-sample lens=[128, 57], ALL_SAME=False
  ALL per-sample lens: [128, 57], unique=['128', '57']
  np.concatenate CRASHED:
    all the input array dimensions except for the concatenation axis must
    match exactly, but along dimension 1, the array at index 0 has size 128
    and the array at index 1 has size 57
  This means the expand code CANNOT handle ragged arrays!
  Per-sample lengths: [128, 57]
```

### prefix_token_count

```
key=prefix_token_count, orig_len=2, orig_val type=numpy.ndarray, dtype=object,
shape=(2,)
  RAGGED object array: 5 per-sample lens=['SCALAR(int)', 'SCALAR(int)'],
  ALL_SAME=True
  ALL per-sample lens: ['SCALAR(int)', 'SCALAR(int)'],
  unique=['SCALAR(int)']
  after concat: shape=(2,), dtype=int64  ← SAME-length → safe
```

### prefix_mask

```
key=prefix_mask, orig_len=2, orig_val type=numpy.ndarray, dtype=object,
shape=(2,)
  RAGGED object array: 5 per-sample lens=[1054, 965], ALL_SAME=False
  ALL per-sample lens: [1054, 965], unique=['1054', '965']
  np.concatenate CRASHED:
    along dimension 1, the array at index 0 has size 1054 and the array
    at index 1 has size 965
  This means the expand code CANNOT handle ragged arrays!
  Per-sample lengths: [1054, 965]
```

### assistant_prefix_span

```
key=assistant_prefix_span, orig_len=2, orig_val type=numpy.ndarray,
dtype=object, shape=(2,)
  RAGGED object array: 5 per-sample lens=[2, 2], ALL_SAME=True
  ALL per-sample lens: [2, 2], unique=['2']
  after concat: shape=(2, 2), dtype=int64  ← SAME-length → safe
```

### raw_prompt

```
key=raw_prompt, orig_len=2, orig_val type=numpy.ndarray, dtype=object,
shape=(2,)
  RAGGED object array: 5 per-sample lens=[6, 4], ALL_SAME=False
  ALL per-sample lens: [6, 4], unique=['4', '6']
  np.concatenate CRASHED:
    along dimension 1, the array at index 0 has size 6 and the array
    at index 1 has size 4
  This means the expand code CANNOT handle ragged arrays!
  Per-sample lengths: [6, 4]
```

---

## DEBUG_EXPAND_AFTER (ray_trainer.py ~line 1469)

```
assistant_prefix_old_log_probs:
  type=ndarray, dtype=object, shape=(2,)
  per-sample lens=[128, 57]  ← RAGGED preserved (n_rollouts=1 bypass kept orig_val)

prefix_token_count:
  type=ndarray, dtype=int64, shape=(2,)  ← SAFE: same-length → concat succeeded

prefix_mask:
  type=ndarray, dtype=object, shape=(2,)
  per-sample lens=[1054, 965]  ← RAGGED preserved

assistant_prefix_span:
  type=ndarray, dtype=int64, shape=(2, 2)  ← SAFE: same-length

raw_prompt:
  type=ndarray, dtype=object, shape=(2,)
  per-sample lens=[6, 4]  ← RAGGED preserved
```

Note: For `n_rollouts=1`, the TEMP FIX bypass keeps `orig_val` unchanged.
For `n_rollouts>1` the `raise` would fire and the run would abort at expand.

---

## DEBUG_RESTORE (ray_trainer.py ~line 1556)

```
Copying key=assistant_prefix_old_log_probs from gen_batch.non_tensor_batch
  → batch.batch
  value is ragged object array: shape=(2,), first_5_lens=[128, 57]
  AFTER assignment: batch.batch['assistant_prefix_old_log_probs']
    type=ndarray, is_Tensor=False, is_ndarray=True, dtype=object, shape=(2,)

Copying key=prefix_token_count from gen_batch.non_tensor_batch → batch.batch
  (no ragged info — all scalars)
  AFTER assignment: batch.batch['prefix_token_count']
    type=Tensor, is_Tensor=True, dtype=torch.int64, shape=torch.Size([2])

Copying key=prefix_mask from gen_batch.non_tensor_batch → batch.batch
  value is ragged object array: shape=(2,), first_5_lens=[1054, 965]
  AFTER assignment: batch.batch['prefix_mask']
    type=ndarray, is_Tensor=False, is_ndarray=True, dtype=object, shape=(2,)

Copying key=assistant_prefix_span from gen_batch.non_tensor_batch → batch.batch
  AFTER assignment: batch.batch['assistant_prefix_span']
    type=Tensor, is_Tensor=True, dtype=torch.int64, shape=torch.Size([2, 2])

Copying key=raw_prompt from gen_batch.non_tensor_batch → batch.batch
  value is ragged object array: shape=(2,), first_5_lens=[6, 4]
  AFTER assignment: batch.batch['raw_prompt']
    type=ndarray, is_Tensor=False, is_ndarray=True, dtype=object, shape=(2,)
```

---

## DEBUG_AFTER_UNION / DEBUG_AFTER

After `batch = batch.union(gen_batch_output)`:

```
Restored prefix keys to batch.batch: [
  'assistant_prefix_old_log_probs', 'prefix_token_count', 'prefix_mask',
  'assistant_prefix_span', 'raw_prompt'
]
batch.batch.keys() after union = [
  'assistant_prefix_old_log_probs', 'prefix_token_count', 'prefix_mask',
  'assistant_prefix_span', 'raw_prompt',  ← restored
  'prompts', 'responses', 'response_mask', 'input_ids',           ← from gen_output
  'attention_mask', 'position_ids', 'rm_scores'
]

assistant_prefix_old_log_probs: type=numpy.ndarray, is_Tensor=False, shape=(2,)
  is np.ndarray, shape=(2,), dtype=object

prefix_token_count: type=torch.Tensor, is_Tensor=True, shape=torch.Size([2])

prefix_mask: type=numpy.ndarray, is_Tensor=False, shape=(2,)

assistant_prefix_span: type=torch.Tensor, is_Tensor=True, shape=torch.Size([2, 2])

raw_prompt: type=numpy.ndarray, is_Tensor=False, shape=(2,)
```

Key observation: `union` does NOT convert numpy object arrays to Tensors.
The ragged keys stay as `numpy.ndarray` dtype=object.

---

## DEBUG_PREFIX_TOKEN_COUNT (ray_trainer.py ~line 1579)

```
gen_batch.non_tensor_batch['prefix_token_count']:
  type=ndarray, dtype=int64, shape=(2,)
  first 5 values: [128, 57]  ← SAME as assistant_prefix_old_log_probs lengths!

batch.batch['prefix_token_count'] after restore:
  type=Tensor, dtype=torch.int64, shape=torch.Size([2])
  first 5 values: [128, 57]  ← values unchanged; TensorDict auto-converted

key insight:
  prefix_token_count values [128, 57] EXACTLY match the
  assistant_prefix_old_log_probs per-sample lengths [128, 57].
  → prefix_token_count IS a reliable source of per-sample cached lengths.
  → It is already a flat int array, auto-converted to Tensor by TensorDict.
```

---

## DEBUG_1592 (ray_trainer.py ~line 1651, inside optimize_prefix_tokens branch)

```
prefix_logprobs_key=assistant_prefix_old_log_probs
cached_prefix_logprobs_tensor: type=numpy.ndarray, is_Tensor=False
cached_prefix_logprobs_tensor: dtype=object
cached_prefix_logprobs_tensor: shape=(2,)
cached_prefix_logprobs_tensor: len=2, first_elem_type=list, first_elem_len=128
  ← elements are raw Python lists (len=128), NOT numpy arrays
  ← first sample has 128 cached logprobs; second has 57

torch.from_numpy FAILED:
  can't convert np.ndarray of type numpy.object_.
  The only supported types are: float64, float32, float16, complex64,
  complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
```

**This is the FATAL crash point.** The ragged structure persisted from parquet
through every pipeline stage, and `torch.from_numpy` cannot handle object arrays.

---

## SUMMARY OF KEY FACTS

| # | Fact | Source |
|---|------|--------|
| 1 | `assistant_prefix_old_log_probs` per-sample lengths in batch: [128, 57] | DEBUG_EXPAND_BEFORE |
| 2 | `prefix_mask` per-sample lengths: [1054, 965] | DEBUG_EXPAND_BEFORE |
| 3 | `raw_prompt` per-sample lengths: [6, 4] | DEBUG_EXPAND_BEFORE |
| 4 | All three are stored as `numpy.ndarray dtype=object` in parquet | Parquet schema |
| 5 | `np.concatenate` fails on any key with unequal per-sample lengths | DEBUG_EXPAND_BEFORE |
| 6 | `prefix_token_count` is scalar (int) → safe for concat | DEBUG_EXPAND_BEFORE |
| 7 | `assistant_prefix_span` elements are same-length → safe for concat | DEBUG_EXPAND_BEFORE |
| 8 | After TEMP FIX (n_rollouts=1), ragged structure preserved | DEBUG_EXPAND_AFTER |
| 9 | After restore, ragged keys remain as `numpy.ndarray dtype=object` | DEBUG_RESTORE |
| 10 | `prefix_token_count` and `assistant_prefix_span` become `torch.Tensor` after restore | DEBUG_RESTORE |
| 11 | `union` does NOT convert object arrays to Tensors | DEBUG_AFTER_UNION |
| 12 | At 1592: elements are raw Python `list`, not numpy arrays | DEBUG_1592 |
| 13 | `prefix_token_count` values [128, 57] exactly match per-sample lengths | DEBUG_PREFIX_TOKEN_COUNT |
| 14 | `torch.from_numpy` cannot convert object array → crashes at line 1670 | DEBUG_1592 + traceback |
| 15 | `DEBUG_1054` (actor side) NOT reached — trainer crashes first | No DEBUG_1054 in log |
| 16 | `DEBUG_1592 AFTER conversion` NOT reached — conversion itself crashes | No DEBUG_AFTER in log for 1592 |

---

## WHAT IS KNOWN vs UNKNOWN

### Known (confirmed by evidence)
- `assistant_prefix_old_log_probs` is a ragged numpy object array throughout the entire trainer pipeline
- It is never converted to a Tensor before reaching line 1670
- `torch.from_numpy` fails on object arrays → the crash is deterministic
- `prefix_token_count` stores per-sample lengths and matches `assistant_prefix_old_log_probs` lengths exactly
- `prefix_token_count` is auto-converted to `torch.Tensor` by TensorDict at restore time
- The `np.concatenate` expand code cannot handle ragged arrays with different per-sample lengths

### Unknown (needs code path to reach actor)
- Whether `pad_offset` in dp_actor's line 1054 is computed correctly given ragged inputs
- The actual semantics of the "compact prompt length" (prefix_mask.shape[1])
- Whether `prefix_token_count` survives the `model_inputs` conversion in dp_actor
- The per-sample prefix_mask ones count vs the stored `prefix_mask` list length discrepancy
  (prefix_mask[0] len=1054 in gen_batch, but assistant_prefix_old_log_probs[0] len=128)

---

## CODE FILES WITH DEBUG PRINTS

- `verl/trainer/ppo/ray_trainer.py`
  - Lines ~1423-1460: DEBUG_EXPAND_BEFORE / DEBUG_EXPAND_AFTER
  - Lines ~1531-1549: DEBUG_RESTORE / DEBUG_PREFIX_TOKEN_COUNT (trainer side)
  - Lines ~1559-1577: DEBUG_AFTER_UNION / DEBUG_AFTER
  - Lines ~1651-1673: DEBUG_1592
  - Lines ~1476: TEMP FIX for n_rollouts=1 bypass

- `verl/workers/actor/dp_actor.py`
  - Lines ~614-648: DEBUG_1054 (length-related variables)
  - Lines ~649-658: DEBUG_PREFIX_TOKEN_COUNT (actor side)
  - Line ~456: prefix_token_count added to select_keys
