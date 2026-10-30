#!/usr/bin/env bash
# Run on one Linux Ascend host; no pre-started Ray cluster is required.
# Usage: bash submit_qwen38_small_model.sh [Hydra overrides...]
# Default: 16 NPUs. For 8 NPUs: npu_count=8 bash submit_qwen38_small_model.sh
# Optional NIC selection: COMM_IFNAME=eth0 bash submit_qwen38_small_model.sh
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/share/h00943455/Qwen3.8-Flash-Next-New}
PROJECT=${PROJECT:-$REPO_ROOT/verl}
VLLM_ASCEND_PATH=${VLLM_ASCEND_PATH:-$REPO_ROOT/vllm-ascend}
VEOMNI_PATH=${VEOMNI_PATH:-$REPO_ROOT/VeOmni}
CONDA_ENV=${CONDA_ENV:-/mnt/share/h00943455/conda_envs/verl-qwen3.8-flash-veomni-16384}
CANN_ROOT=${CANN_ROOT:-/mnt/share/cann/CANN_9_2_0_B020}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ===================================== User settings =====================================
model_path=${model_path:-/mnt/share/w00878026/Qwen3.8/VeOmni/qwen4-exp-sft-internal/output_hf_ckpt/step500}
data_path=${data_path:-/mnt/share/h00943455/Qwen3.8-Flash-Next-New/verl/artifacts/geo3k-sft-rl}
npu_count=${npu_count:-16}
[[ $npu_count =~ ^[1-9][0-9]*$ ]] || die 'npu_count must be a positive integer.'

# Keep the original 16-NPU topology; scale the defaults with the local NPU count.
fsdp_size=${fsdp_size:-$npu_count}
expert_parallel_size=${expert_parallel_size:-$fsdp_size}
infer_tp=${infer_tp:-$((npu_count >= 8 ? 8 : npu_count))}
[[ $infer_tp =~ ^[1-9][0-9]*$ ]] || die 'infer_tp must be a positive integer.'
infer_dp=${infer_dp:-$((npu_count / infer_tp))}
infer_ep=${infer_ep:-$((infer_tp * infer_dp))}
ple_parallel_size=${ple_parallel_size:-$infer_tp}
max_prompt_length=${max_prompt_length:-1024}
max_response_length=${max_response_length:-2048}
max_model_len=$((max_prompt_length + max_response_length))
total_training_steps=${total_training_steps:-100}
exp_name=${exp_name:-qwen38-full48-1node${npu_count}npu-gbs8-n2-geo3k-$(date +%Y%m%d_%H%M%S)}
run_dir=${run_dir:-$PROJECT/artifacts/$exp_name}

for name in fsdp_size expert_parallel_size ple_parallel_size infer_dp infer_ep; do
  [[ ${!name} =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer."
done
(( npu_count % fsdp_size == 0 )) || die 'fsdp_size must divide npu_count.'
(( fsdp_size % expert_parallel_size == 0 )) || die 'expert_parallel_size must divide fsdp_size.'
(( fsdp_size % ple_parallel_size == 0 )) || die 'ple_parallel_size must divide fsdp_size.'
infer_world_size=$((infer_tp * infer_dp))
(( npu_count % infer_world_size == 0 )) || die 'infer_tp * infer_dp must divide npu_count.'
(( infer_ep == 1 || infer_ep == infer_world_size )) || die 'Use infer_ep=1 (disabled) or infer_ep=infer_tp*infer_dp.'

ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 "$((npu_count - 1))")}
[[ $ASCEND_RT_VISIBLE_DEVICES =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'ASCEND_RT_VISIBLE_DEVICES must be comma-separated device IDs.'
IFS=, read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
(( ${#visible_devices[@]} == npu_count )) || die 'The visible device count must equal npu_count.'
export ASCEND_RT_VISIBLE_DEVICES

# ===================================== Environment =====================================
[[ -d $PROJECT ]] || die "PROJECT does not exist: $PROJECT"
[[ -x $CONDA_ENV/bin/python ]] || die "Python does not exist: $CONDA_ENV/bin/python"
[[ -f $CANN_ROOT/cann/set_env.sh ]] || die "CANN environment script does not exist: $CANN_ROOT/cann/set_env.sh"
[[ -f $CANN_ROOT/nnal/atb/set_env.sh ]] || die "ATB environment script does not exist: $CANN_ROOT/nnal/atb/set_env.sh"

export PATH="$CONDA_ENV/bin:$PATH" CONDA_PREFIX="$CONDA_ENV"
set +u
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi=1
set -u
export LD_PRELOAD="$CONDA_ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$VLLM_ASCEND_PATH:$VEOMNI_PATH:$PROJECT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}

# Clear network settings inherited from the former multi-node launch environment.
# By default HCCL/Gloo select their own local interfaces. No fixed IP or NIC.
unset RAY_ADDRESS HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME HCCL_IF_IP
local_ip=
if [[ -n ${COMM_IFNAME:-} ]]; then
  command -v ip >/dev/null || die 'COMM_IFNAME requires the ip command (iproute2).'
  local_ip=$(ip -4 -o addr show dev "$COMM_IFNAME" | awk 'NR == 1 {split($4, a, "/"); print a[1]}')
  [[ -n $local_ip ]] || die "No IPv4 address found on $COMM_IFNAME."
  export HCCL_SOCKET_IFNAME="=$COMM_IFNAME"
  export GLOO_SOCKET_IFNAME="$COMM_IFNAME"
  export HCCL_IF_IP="$local_ip"
fi

export HCCL_CONNECT_TIMEOUT=1200 HCCL_EXEC_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RAY_USAGE_STATS_ENABLED=0
export VLLM_USE_V2_MODEL_RUNNER=0
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
  "data.train_batch_size=8"
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
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.ppo_mini_batch_size=8"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=true"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_model_len"
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
  "actor_rollout_ref.actor.veomni.enable_autocast=false"
  "actor_rollout_ref.actor.veomni.param_offload=true"
  "actor_rollout_ref.actor.veomni.optimizer_offload=true"
  "actor_rollout_ref.actor.veomni.use_torch_compile=false"
  "actor_rollout_ref.actor.veomni.forward_prefetch=false"
  "actor_rollout_ref.actor.veomni.attn_implementation=sdpa"
  "actor_rollout_ref.actor.veomni.moe_implementation=fused_npu"
  "actor_rollout_ref.actor.veomni.freeze_ple_embeddings=true"
)

ROLLOUT=(
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.load_format=safetensors"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp"
  "actor_rollout_ref.rollout.data_parallel_size=$infer_dp"
  "actor_rollout_ref.rollout.expert_parallel_size=$infer_ep"
  "actor_rollout_ref.rollout.n=2"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_model_len"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
  "actor_rollout_ref.rollout.free_cache_engine=true"
  "+actor_rollout_ref.rollout.enable_sleep_mode=true"
  "actor_rollout_ref.rollout.enforce_eager=false"
  "actor_rollout_ref.rollout.enable_prefix_caching=true"
  "actor_rollout_ref.rollout.enable_chunked_prefill=true"
  "actor_rollout_ref.rollout.max_model_len=$max_model_len"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$max_model_len"
  "actor_rollout_ref.rollout.max_num_seqs=128"
  "actor_rollout_ref.rollout.ignore_eos=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.vit_attention_backend=fia"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.vit_fp32_output_projection=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.compile_mm_encoder=false"
  "++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false"
)

ROUTER_REPLAY=(
  "actor_rollout_ref.actor.veomni.router_replay.mode=R3"
  "actor_rollout_ref.rollout.enable_rollout_routing_replay=True"
)

TRAINER=(
  "trainer.device=npu"
  "trainer.n_gpus_per_node=$npu_count"
  "trainer.nnodes=1"
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
  "++ray_kwargs.ray_init.address=local"
  "++ray_kwargs.ray_init.resources.NPU=$npu_count"
  "++ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER='0'"
)
if [[ -n $local_ip ]]; then
  RAY+=("++ray_kwargs.ray_init._node_ip_address=$local_ip")
fi

# ===================================== Launch =====================================
mkdir -p "$VERL_FILE_LOGGER_ROOT"
printf 'Single-node: NPUs=%s, FSDP=%s, EP=%s, PLE=%s, rollout TP=%s DP=%s EP=%s\n' \
  "$npu_count" "$fsdp_size" "$expert_parallel_size" "$ple_parallel_size" "$infer_tp" "$infer_dp" "$infer_ep"
printf 'Communication NIC: %s; Ray address: local; output: %s\n' "${COMM_IFNAME:-automatic}" "$run_dir"
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
