"""Four-NPU forward/backward diagnostic for the reduced Qwen3.8 model."""

import os
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, VeOmniEngineConfig, VeOmniOptimizerConfig
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/weight/Qwen3.8-Flash-Next-3layer-grpo")


def log(stage, **fields):
    torch.npu.synchronize()
    print(
        dict(
            stage=stage,
            rank=dist.get_rank(),
            time=time.time(),
            allocated_gib=torch.npu.memory_allocated() / 2**30,
            **fields,
        ),
        flush=True,
    )


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        engine = VeOmniEngineWithLMHead(
            HFModelConfig(
                path=MODEL_PATH, load_tokenizer=False, trust_remote_code=True, enable_gradient_checkpointing=True
            ),
            VeOmniEngineConfig(
                fsdp_size=4,
                expert_parallel_size=4,
                ple_parallel_size=1,
                broadcast_model_weights_from_rank0=False,
                ep_sharded_stream_load=True,
                mixed_precision=True,
                init_device="meta",
                enable_full_shard=True,
                param_offload=False,
                optimizer_offload=False,
                use_torch_compile=False,
                attn_implementation="sdpa",
                moe_implementation="fused_npu",
            ),
            VeOmniOptimizerConfig(lr=1e-5, total_training_steps=2),
            CheckpointConfig(),
        )
        engine.initialize()
        log("initialized")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokens = tokenizer.encode("Calculate 1 + 1. The answer is 2.", add_special_tokens=False)[:16]
        input_ids = torch.tensor([tokens], device="npu")
        position_ids = torch.arange(input_ids.shape[-1], device="npu").unsqueeze(0)
        engine.module.train()
        log("before_forward", length=len(tokens))
        with torch.autocast("npu", dtype=torch.bfloat16):
            output = engine.module(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
            logits = output.logits
            assert torch.isfinite(logits).all(), "Non-finite logits"
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(), input_ids[:, 1:].reshape(-1)
            )
        log("forward_ok", loss=float(loss.detach()))
        loss.backward()
        log("backward_ok")
        norm = engine.module.clip_grad_norm_(1.0)
        log("grad_clip_ok", norm=float(norm))
        assert torch.isfinite(norm), "Non-finite gradient"
        log("before_optimizer")
        engine.optimizer.step()
        engine.optimizer.zero_grad()
        log("optimizer_ok")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
