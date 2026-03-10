set -e

MODEL_PATH=${MODEL_PATH:-"/workspace/models/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/workspace/datasets/Verl-Data/train/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/outputs/textcraft_grpo_tis"}

GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=${NUM_GPUS:-8}

NUM_EPOCHS=${NUM_EPOCHS:-200}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}

ROLLOUT_IS="sequence"
ROLLOUT_IS_THRESHOLD=2.0
ROLLOUT_RS="none"

ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-0.8}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ENFORCE_EAGER=false
FREE_CACHE_ENGINE=true
CALCULATE_LOG_PROBS=true

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=16384
ROLLOUT_PROMPT_LENGTH=16384
MAX_MODEL_LEN=20480
PPO_MAX_TOKEN_LEN=24576
MAX_NUM_BATCHED_TOKENS=16384

TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_grpo_tis_${TIMESTAMP}"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "TextCraft GRPO+TIS Docker Version" | tee "$LOG_FILE"

cd /workspace/verl

export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_LOGGING_LEVEL=WARNING

ROLLOUT_CORR_ARGS="algorithm.rollout_correction.bypass_mode=true \
algorithm.rollout_correction.use_policy_gradient=true \
algorithm.rollout_correction.rollout_is=$ROLLOUT_IS \
algorithm.rollout_correction.rollout_is_threshold=$ROLLOUT_IS_THRESHOLD"

python3 -m verl.trainer.main_ppo \
    --config-path='/workspace/verl/examples/sglang_multiturn/config' \
    --config-name='textcraft_grpo_train.docker' \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.calculate_log_probs=$CALCULATE_LOG_PROBS \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=textcraft_grpo_tis \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=disable \
    $ROLLOUT_CORR_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo "Done! Log: $LOG_FILE"
