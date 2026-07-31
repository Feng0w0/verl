#!/usr/bin/env bash
set -euo pipefail

if [[ "${DEBUG_SHELL:-0}" == "1" ]]; then
    set -x
fi

# Required user-provided paths. Keep machine- and user-specific paths out of
# this example so it can be shared safely.
: "${HF_MODEL_PATH:?Set HF_MODEL_PATH to the Kimi-K2.5 model directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet file}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet file}"

# Source the Ascend environment only when explicit scripts are provided.
# These variables may be omitted when the environment is already initialized.
if [[ -n "${ASCEND_TOOLKIT_ENV_SCRIPT:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_TOOLKIT_ENV_SCRIPT}"
fi
if [[ -n "${ATB_ENV_SCRIPT:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ATB_ENV_SCRIPT}"
fi

# Optional source checkouts. Installed packages can be used instead by leaving
# these variables unset.
for source_root in \
    "${MEGATRON_BRIDGE_ROOT:-}" \
    "${MEGATRON_LM_ROOT:-}" \
    "${MINDSPEED_ROOT:-}"; do
    if [[ -n "${source_root}" ]]; then
        PYTHONPATH="${source_root}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
done
export PYTHONPATH

# Automatically use the default network interface unless explicitly set.
NETWORK_INTERFACE=${NETWORK_INTERFACE:-}
if [[ -z "${NETWORK_INTERFACE}" ]] && command -v ip >/dev/null 2>&1; then
    NETWORK_INTERFACE=$(ip route show default | awk 'NR == 1 {print $5}')
fi
if [[ -n "${NETWORK_INTERFACE}" ]]; then
    export HCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export GLOO_SOCKET_IFNAME="${NETWORK_INTERFACE}"
fi

export GLOO_PORT=${GLOO_PORT:-29501}
export GLOO_TIMEOUT_SECONDS=${GLOO_TIMEOUT_SECONDS:-120}
export MASTER_PORT=${MASTER_PORT:-29501}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:512}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}

# Stopping Ray can disrupt other work on a shared server, so it is opt-in.
if [[ "${STOP_EXISTING_RAY:-0}" == "1" ]]; then
    ray stop --force || true
fi

########################### Quick Config ###########################

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_kimi_k2_5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-kimi_k2_5_megatron_npu}
LOG_DIR=${LOG_DIR:-logs}

# Actor/reference parallelism. EP=16 requires a compatible world size.
TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-16}
ETP=${ETP:-1}

# vLLM rollout parallelism.
GEN_TP=${GEN_TP:-16}
GEN_DP=${GEN_DP:-1}
GEN_EP=${GEN_EP:-1}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-16}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
PROMPT_LENGTH=${PROMPT_LENGTH:-512}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}
ROLLOUT_N=${ROLLOUT_N:-1}
PPO_MAX_TOKEN_LEN=$(((PROMPT_LENGTH + RESPONSE_LENGTH) * 2))

########################### Parameter Arrays ###########################

DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${PROMPT_LENGTH}
    data.max_response_length=${RESPONSE_LENGTH}
    data.truncation=error
    data.filter_overlong_prompts=True
    data.trust_remote_code=True
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]'
    actor_rollout_ref.actor.checkpoint.strict=False
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.data_parallel_size=${GEN_DP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16}
    +actor_rollout_ref.rollout.limit_images=1
)

REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    ++actor_rollout_ref.ref.megatron.override_transformer_config.moe_grouped_gemm=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.device=npu
    trainer.save_freq=-1
    trainer.val_before_train=False
    trainer.test_freq=5
    trainer.total_epochs=${TOTAL_EPOCHS:-2}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-100}
)

########################### Launch ###########################

mkdir -p "${LOG_DIR}"
START_TIME=$(date +%Y%m%d_%H%M%S)

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer.yaml \
    "${DATA_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/kimi_k2_5_megatron_npu_${START_TIME}.log"
