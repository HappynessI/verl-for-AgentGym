#!/bin/bash
set -eo pipefail

MODEL_PATH=${MODEL_PATH:-"/Data/public/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/Data/wyh/datasets/Verl-Data/outputs/textcraft_drpo"}
INTERACTION_CONFIG=${INTERACTION_CONFIG:-"/Data/wyh/verl/examples/sglang_multiturn/config/interaction_config/textcraft_interaction.yaml"}
VERL_ROOT=${VERL_ROOT:-"/Data/wyh/verl"}
CONFIG_ROOT=${CONFIG_ROOT:-"/Data/wyh/verl/examples/sglang_multiturn/config"}
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}

NUM_GPUS=${NUM_GPUS:-2}
if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

DELTA=${DELTA:-1e-4}
BETA=${BETA:-1e3}
TAU=${TAU:-10}
LAMBDA=${LAMBDA:-0.1}

NUM_EPOCHS=${NUM_EPOCHS:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_EPOCHS=${PPO_EPOCHS:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-3e-6}
SAVE_FREQ=${SAVE_FREQ:-500}
TEST_FREQ=${TEST_FREQ:-500}
ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-2048}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-512}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-30}
MAX_USER_TURNS=${MAX_USER_TURNS:-30}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_N=${VAL_N:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
METRICS_CSV_FREQ=${METRICS_CSV_FREQ:-50}
METRICS_CSV_FILENAME=${METRICS_CSV_FILENAME:-training_metrics.csv}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_drpo_${TIMESTAMP}"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "错误: GPU_IDS中的GPU数量($GPU_COUNT)与NUM_GPUS($NUM_GPUS)不一致！" | tee "$LOG_FILE"
    exit 1
fi

echo "检查TextCraft服务器..." | tee -a "$LOG_FILE"
SERVER_RESPONSE=$(curl -s "$TEXTCRAFT_SERVER/" 2>&1)
if [[ "$SERVER_RESPONSE" == *"TextCraft"* ]]; then
    echo "✓ TextCraft服务器正常运行" | tee -a "$LOG_FILE"
else
    echo "警告: TextCraft服务器 ($TEXTCRAFT_SERVER) 未运行！" | tee -a "$LOG_FILE"
    echo "请先启动服务器：" | tee -a "$LOG_FILE"
    echo "  cd /Data/wyh/AgentGym-RL/AgentGym/agentenv-textcraft" | tee -a "$LOG_FILE"
    echo "  textcraft --host 0.0.0.0 --port 36001" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

cd "$VERL_ROOT"
source ~/miniconda3/bin/activate verl

VERL_CONFIG_ROOT="${VERL_ROOT}/verl/trainer/config"
if [ ! -d "$VERL_CONFIG_ROOT" ]; then
    echo "错误: verl 配置目录不存在: $VERL_CONFIG_ROOT" | tee -a "$LOG_FILE"
    exit 1
fi

echo "verl 配置路径: $VERL_CONFIG_ROOT" | tee -a "$LOG_FILE"
echo "Interaction Config: $INTERACTION_CONFIG" | tee -a "$LOG_FILE"
echo "Metrics CSV: $OUTPUT_DIR/metrics/$METRICS_CSV_FILENAME (every $METRICS_CSV_FREQ steps)" | tee -a "$LOG_FILE"
echo "DRPO: delta=$DELTA, beta=$BETA, tau=$TAU, Lambda=$LAMBDA" | tee -a "$LOG_FILE"
echo "Current training defaults: prompt=$MAX_PROMPT_LENGTH, cumulative_response=$MAX_RESPONSE_LENGTH, per_turn_max=$ROLLOUT_MAX_TOKENS, max_model_len=$MAX_MODEL_LEN" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE:-4}

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_ROOT}" \
    --config-name='textcraft_drpo_train' \
    hydra.searchpath=[file://${VERL_CONFIG_ROOT},file://${CONFIG_ROOT}] \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_PATH" \
    data.val_files="$DATA_PATH" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.delta=$DELTA \
    actor_rollout_ref.actor.beta=$BETA \
    actor_rollout_ref.actor.tau=$TAU \
    actor_rollout_ref.actor.Lambda=$LAMBDA \
    'actor_rollout_ref.actor.policy_loss.loss_type=drpo' \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.ppo_kl_type=kl \
    actor_rollout_ref.actor.ppo_kl_coef=0.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$ROLLOUT_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_tokens=$ROLLOUT_MAX_TOKENS \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=false \
    trainer.default_local_dir="$OUTPUT_DIR" \
    +trainer.metrics_csv_freq=$METRICS_CSV_FREQ \
    +trainer.metrics_csv_filename="$METRICS_CSV_FILENAME" \
    trainer.project_name=textcraft_drpo \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.resume_mode=disable \
    2>&1 | tee -a "$LOG_FILE"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "错误: 训练失败，退出码: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "训练完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
