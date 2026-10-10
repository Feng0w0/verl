#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/h00943455/Qwen3.8-Flash-Next/verl-qwen38-veomni
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7,14,15}
export VLLM_PLE_CPU_OFFLOAD=0
export VERL_FILE_LOGGER_ROOT="$PROJECT/artifacts/qwen38-grpo-8layer-8npu/logs"
bash "$PROJECT/scripts/run_qwen38_grpo_4npu.sh" \
  ray_kwargs.ray_init.num_cpus=48 \
  ray_kwargs.ray_init._temp_dir=/tmp/ray-qwen38-grpo-8layer \
  actor_rollout_ref.model.path=/mnt/weight/Qwen3.8-Flash-Next-8layer-frozen-ple \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.veomni.enable_autocast=false \
  actor_rollout_ref.actor.veomni.fsdp_size=8 \
  actor_rollout_ref.actor.veomni.expert_parallel_size=8 \
  actor_rollout_ref.actor.veomni.ple_parallel_size=8 \
  actor_rollout_ref.actor.veomni.freeze_ple_embeddings=true \
  actor_rollout_ref.actor.veomni.frozen_ple_dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.32 \
  trainer.n_gpus_per_node=8 \
  'trainer.logger=[console,file]' \
  trainer.experiment_name=8layer_frozen_ple_grpo_8npu_gc_no_autocast \
  trainer.default_local_dir="$PROJECT/artifacts/qwen38-grpo-8layer-8npu/checkpoints" \
  trainer.rollout_data_dir="$PROJECT/artifacts/qwen38-grpo-8layer-8npu/rollouts" \
  "$@"
