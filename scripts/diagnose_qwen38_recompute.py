"""Standalone VeOmni recomputation diagnosis; no Verl/Ray/vLLM imports."""

import argparse
import json
import os
import re
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from veomni.arguments import MixedPrecisionConfig, OpsImplementationConfig
from veomni.distributed import parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.auto import build_foundation_model

PHASE = "init"


def emit(event, **fields):
    if dist.get_rank() == 0:
        print(json.dumps({"event": event, "phase": PHASE, **fields}), flush=True)


def tensors(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: tensors(item) for key, item in value.items() if isinstance(item, torch.Tensor)}
    return type(value).__name__


def trace_forward(module, name):
    original = module.forward

    def traced(*args, **kwargs):
        emit("enter", module=name, inputs=tensors(args), kwargs=tensors(kwargs))
        result = original(*args, **kwargs)
        emit("exit", module=name, outputs=tensors(result))
        return result

    module.forward = traced


def main():
    global PHASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/weight/Qwen3.8-Flash-Next-4layer")
    parser.add_argument("--ple-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--fsdp-output-bf16", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    torch.manual_seed(1234)
    world = dist.get_world_size()
    try:
        parallel_state.init_parallel_state(
            dp_size=world, dp_replicate_size=1, dp_shard_size=world,
            extra_parallel_sizes=(world, world),
            extra_parallel_placement_innermost=(False, False),
            extra_parallel_names=("ep", "ple"), ulysses_size=1, dp_mode="fsdp2",
        )
        ops = OpsImplementationConfig(attn_implementation="sdpa", moe_implementation="fused_npu")
        model = build_foundation_model(
            config_path=args.model, weights_path=args.model, torch_dtype="float32",
            attn_implementation="sdpa", ops_implementation=ops, init_device="meta",
        )
        index = json.loads((Path(args.model) / "model.safetensors.index.json").read_text())["weight_map"]
        model = build_parallelize_model(
            model, init_device="meta", weights_path=args.model, enable_full_shard=True,
            mixed_precision=MixedPrecisionConfig(
                enable=True, output_dtype="bfloat16" if args.fsdp_output_bf16 else None,
            ),
            enable_gradient_checkpointing=not args.no_checkpoint,
            enable_fsdp_offload=False, basic_modules=list(getattr(model, "_no_split_modules", None) or []),
            enable_reentrant=False, enable_forward_prefetch=False,
            broadcast_model_weights_from_rank0=False, ep_sharded_stream_load=True,
            fqn_to_index_mapping=index,
        )
        table_count = 0
        for name, param in model.named_parameters():
            if re.search(r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$", name):
                param.requires_grad_(False)
                if args.ple_dtype == "bfloat16":
                    converted = torch.nn.Parameter(param.to(torch.bfloat16), requires_grad=False)
                    torch.utils.swap_tensors(param, converted)
                table_count += 1
        emit("initialized", tables=table_count, ple_dtype=args.ple_dtype)
        for name, module in model.named_modules():
            if re.search(r"\.layers\.\d+$", name) or any(
                name.endswith(suffix) for suffix in (
                    ".ple", ".ple.key_proj", ".ple.value_proj", ".ple.norm_key",
                    ".ple.norm_query", ".ple.norm_conv", ".ple.conv1d",
                    ".attn_hyper_connection", ".mlp_hyper_connection",
                )
            ):
                trace_forward(module, name)
        model.train()
        ids = (torch.arange(73, device="npu") + 100).unsqueeze(0)
        position_ids = torch.arange(73, device="npu").unsqueeze(0)
        PHASE = "forward"
        with torch.autocast("npu", dtype=torch.bfloat16, enabled=not args.no_autocast):
            output = model(input_ids=ids, position_ids=position_ids,
                           attention_mask=torch.ones_like(ids), use_cache=False)
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].reshape(-1, output.logits.shape[-1]).float(), ids[:, 1:].reshape(-1)
            )
        emit("loss", value=float(loss.detach()))
        PHASE = "backward"
        loss.backward()
        torch.npu.synchronize()
        emit("backward_ok", allocated_gib=torch.npu.memory_allocated() / 2**30)
        dist.barrier()
    except Exception as error:
        emit("failure", error=str(error), kind=type(error).__name__)
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
