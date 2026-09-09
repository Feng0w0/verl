"""Initialize the 4-layer Qwen3.8-Flash-Next actor with VeOmni on four NPUs.

Run with: torchrun --standalone --nproc_per_node=4 scripts/smoke_qwen38_flash_next_4npu.py
"""

import os

import torch
import torch.distributed as dist

from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, VeOmniEngineConfig, VeOmniOptimizerConfig
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead


MODEL_PATH = "/mnt/weight/Qwen3.8-Flash-Next-4layer"


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"This smoke test requires exactly 4 ranks, got {world_size}.")

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        model_config = HFModelConfig(
            path=MODEL_PATH,
            load_tokenizer=False,
            trust_remote_code=True,
            enable_gradient_checkpointing=True,
            use_remove_padding=True,
        )
        engine_config = VeOmniEngineConfig(
            fsdp_size=4,
            expert_parallel_size=4,
            ple_parallel_size=4,
            broadcast_model_weights_from_rank0=False,
            ep_sharded_stream_load=True,
            param_offload=True,
            optimizer_offload=True,
            mixed_precision=False,
            init_device="meta",
            enable_full_shard=True,
            use_torch_compile=False,
            attn_implementation="sdpa",
            moe_implementation="fused",
        )
        optimizer_config = VeOmniOptimizerConfig(
            lr=1e-6,
            total_training_steps=1,
        )
        engine = VeOmniEngineWithLMHead(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=CheckpointConfig(),
        )
        engine.initialize()
        local_params = sum(parameter.numel() for parameter in engine.module.parameters())
        print(
            f"Qwen3.8-Flash-Next VeOmni initialization succeeded: "
            f"rank={dist.get_rank()} local_parameters={local_params}",
            flush=True,
        )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
