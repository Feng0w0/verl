"""Diagnostic controls for NPU autocast and FSDP2 checkpoint boundaries."""
import functools
import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer


class Producer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x).float()


class Consumer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.phase = "forward"

    def forward(self, x):
        print(json.dumps({"phase": self.phase, "consumer_input": str(x.dtype)}), flush=True)
        return self.linear(x * x)


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        x = torch.randn(1, 4, 16, device="npu", dtype=torch.bfloat16)
        for enabled in (False, True):
            with torch.autocast("npu", dtype=torch.bfloat16, enabled=enabled):
                product = x * x
                summed = product.sum(dim=-1, keepdim=True)
                gate = summed / 4
                absolute = gate.abs()
                clamped = absolute.clamp_min(1e-6)
                rooted = clamped.sqrt()
                signed = rooted * gate.sign()
                sigmoid = torch.sigmoid(signed)
                gated = sigmoid * x
            values = dict(product=product, summed=summed, gate=gate, absolute=absolute,
                          clamped=clamped, rooted=rooted, signed=signed, sigmoid=sigmoid, gated=gated)
            print(json.dumps({"autocast": enabled, "dtypes": {
                k: str(v.dtype) for k, v in values.items()
            }}), flush=True)
        for output_dtype in (None, torch.bfloat16):
            torch.manual_seed(123)
            producer = Producer().npu().train()
            consumer = Consumer().npu().train()
            consumer.gradient_checkpointing = True
            consumer._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=False)
            fully_shard(producer, mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, output_dtype=output_dtype))
            fully_shard(consumer, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
            x = torch.randn(2, 16, device="npu", dtype=torch.bfloat16, requires_grad=True)
            try:
                with torch.autocast("npu", dtype=torch.bfloat16):
                    boundary = producer(x)
                    loss = consumer(boundary).float().square().mean()
                consumer.phase = "recompute"
                loss.backward()
                result = {"status": "PASS", "grad_finite": bool(torch.isfinite(x.grad).all())}
            except Exception as error:
                result = {"status": type(error).__name__, "error": str(error)}
            print(json.dumps({"producer_output_policy": str(output_dtype),
                              "boundary_dtype": str(boundary.dtype), **result}), flush=True)
            if output_dtype is None:
                assert result["status"] == "CheckpointError", result
            else:
                assert result["status"] == "PASS" and result["grad_finite"], result
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
