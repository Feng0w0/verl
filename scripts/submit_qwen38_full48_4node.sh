#!/usr/bin/env bash
# Submit from 172.27.2.127 after the four-node Ray cluster is running.
# Usage: bash submit_qwen38_full48_4node.sh [Hydra overrides...]
set -euo pipefail

PROJECT=/mnt/share/h00943455/Qwen3.8-Flash-Next/verl
CONDA_ENV=${CONDA_ENV:-/mnt/share/h00943455/conda_envs/verl-qwen3.8-flash-veomni}
CANN_ROOT=${CANN_ROOT:-/mnt/share/cann/CANN9.2.0}
HEAD_ADDR=${HEAD_ADDR:-172.27.2.132:26379}

# ===================================== User settings =====================================
# model_path=${model_path:-/mnt/weight/Qwen3.8-Flash-Next}
model_path=${model_path:-/mnt/share/h00943455/Qwen3.8-Flash-Next-8layer-frozen-ple}
data_path=${data_path:-/mnt/share/datasets/geo3k}
npu_count=${npu_count:-16}
fsdp_size=${fsdp_size:-32}
expert_parallel_size=${expert_parallel_size:-32}
ple_parallel_size=${ple_parallel_size:-8}
infer_tp=${infer_tp:-8}
infer_dp=${infer_dp:-4}
infer_ep=${infer_ep:-32}
max_prompt_length=${max_prompt_length:-1024}
max_response_length=${max_response_length:-2048}
max_model_len=$((max_prompt_length + max_response_length))
total_training_steps=${total_training_steps:-100}
exp_name=${exp_name:-qwen38-full48-4node64npu-gbs32-n5-geo3k-ple-trainable-$(date +%Y%m%d_%H%M%S)}
run_dir=${run_dir:-$PROJECT/artifacts/$exp_name}

# ===================================== Environment =====================================
export PATH="$CONDA_ENV/bin:$PATH" CONDA_PREFIX="$CONDA_ENV"
set +u
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi=1
set -u
export LD_PRELOAD="$CONDA_ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="/mnt/share/h00943455/Qwen3.8-Flash-Next/vllm-ascend:/mnt/share/h00943455/Qwen3.8-Flash-Next/VeOmni:$PROJECT:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export HCCL_SOCKET_IFNAME=data0.172 GLOO_SOCKET_IFNAME=data0.172
export HCCL_IF_IP="$(ip -4 -o addr show dev data0.172 | awk '{print $4}' | cut -d/ -f1)"
export HCCL_CONNECT_TIMEOUT=1200 HCCL_EXEC_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_ASCEND_SYNC_BEFORE_CAMEM_SLEEP=1
export RAY_USAGE_STATS_ENABLED=0 VLLM_ASCEND_ENABLE_NZ=0 VLLM_USE_V1=1 VLLM_PLE_CPU_OFFLOAD=0
export OMP_NUM_THREADS=4 VLLM_LOGGING_LEVEL=INFO TRITON_CACHE_DIR="/tmp/triton-qwen38-$exp_name"
export VERL_FILE_LOGGER_ROOT="$run_dir/logs"
cd "$PROJECT"

# ===================================== Parameter arrays =====================================
CONFIG=(
  "--config-path=./config"
  "--config-name=ppo_trainer"
  "model_engine=veomni"
)

ALGORITHM=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=false"
)

DATA=(
  "data.train_files=$data_path/train.parquet"
  "data.val_files=$data_path/test.parquet"
  "data.train_batch_size=32"
  "data.max_prompt_length=$max_prompt_length"
  "data.max_response_length=$max_response_length"
  "data.filter_overlong_prompts=true"
  "data.truncation=error"
)

ACTOR=(
  "actor_rollout_ref.model.path=$model_path"
  "actor_rollout_ref.model.trust_remote_code=true"
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.model.use_remove_padding=true"
  "actor_rollout_ref.actor.optim.lr=1e-5"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=true"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.use_kl_loss=false"
  "actor_rollout_ref.actor.entropy_coeff=0"
)

ACTOR_VEOMNI=(
  "actor_rollout_ref.actor.veomni.fsdp_size=$fsdp_size"
  "actor_rollout_ref.actor.veomni.expert_parallel_size=$expert_parallel_size"
  "actor_rollout_ref.actor.veomni.ple_parallel_size=$ple_parallel_size"
  "actor_rollout_ref.actor.veomni.broadcast_model_weights_from_rank0=false"
  "actor_rollout_ref.actor.veomni.ep_sharded_stream_load=true"
  "actor_rollout_ref.actor.veomni.mixed_precision=true"
  "actor_rollout_ref.actor.veomni.param_offload=true"
  "actor_rollout_ref.actor.veomni.optimizer_offload=true"
  "actor_rollout_ref.actor.veomni.use_torch_compile=false"
  "actor_rollout_ref.actor.veomni.forward_prefetch=false"
  "actor_rollout_ref.actor.veomni.attn_implementation=sdpa"
  "actor_rollout_ref.actor.veomni.moe_implementation=fused_npu"
  "actor_rollout_ref.actor.veomni.freeze_ple_embeddings=false"
)

ROLLOUT=(
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp"
  "actor_rollout_ref.rollout.data_parallel_size=$infer_dp"
  "actor_rollout_ref.rollout.expert_parallel_size=$infer_ep"
  # "actor_rollout_ref.rollout.agent.num_workers=4"
  "actor_rollout_ref.rollout.n=5"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.80"
  "actor_rollout_ref.rollout.free_cache_engine=true"
  "+actor_rollout_ref.rollout.enable_sleep_mode=true"
  "actor_rollout_ref.rollout.enforce_eager=false"
  "actor_rollout_ref.rollout.enable_prefix_caching=false"
  "actor_rollout_ref.rollout.enable_chunked_prefill=false"
  "actor_rollout_ref.rollout.max_model_len=$max_model_len"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$max_model_len"
  "actor_rollout_ref.rollout.max_num_seqs=8"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true"
  "actor_rollout_ref.rollout.ignore_eos=true"
  # "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[8,16,32,64,128]"
  # "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY"
)

ROUTER_REPLAY=(
    "actor_rollout_ref.actor.veomni.router_replay.mode=R3"
    "actor_rollout_ref.rollout.enable_rollout_routing_replay=True"
)

TRAINER=(
  "trainer.device=npu"
  "trainer.n_gpus_per_node=$npu_count"
  "trainer.nnodes=2"
  "trainer.logger=[console,file]"
  "trainer.project_name=qwen38_veomni"
  "trainer.experiment_name=$exp_name"
  "trainer.val_before_train=false"
  "trainer.test_freq=-1"
  "trainer.save_freq=10"
  "trainer.max_actor_ckpt_to_keep=1"
  "trainer.resume_mode=auto"
  "trainer.total_training_steps=$total_training_steps"
  "trainer.total_epochs=100"
  "trainer.default_local_dir=$run_dir/checkpoints"
  "trainer.rollout_data_dir=$run_dir/rollouts"
)

RAY=(
  "+ray_kwargs.ray_init.address=$HEAD_ADDR"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
)

# ===================================== Launch =====================================
mkdir -p "$VERL_FILE_LOGGER_ROOT"
python -m verl.trainer.main_ppo \
  "${CONFIG[@]}" \
  "${ALGORITHM[@]}" \
  "${DATA[@]}" \
  "${ACTOR[@]}" \
  "${ACTOR_VEOMNI[@]}" \
  "${ROLLOUT[@]}" \
  "${TRAINER[@]}" \
  "${RAY[@]}" \
  "${ROUTER_REPLAY[@]}" \
  "hydra.run.dir=$run_dir/hydra" \
  "$@" 2>&1 | tee "$VERL_FILE_LOGGER_ROOT/train_$(date +%Y%m%d_%H%M%S).log"
