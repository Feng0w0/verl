"""Isolate HF checkpoint/FSDP2 input-cast ordering without VeOmni or Verl."""

import functools
import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer


class Block(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.phase = "forward"

    def forward(self, x):
        print(json.dumps({"phase": self.phase, "input": str(x.dtype)}), flush=True)
        return self.linear(x * x)


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        for enabled, dtype in [(True, torch.float32), (True, torch.bfloat16), (False, torch.float32)]:
            torch.manual_seed(123)
            block = Block().npu().train()
            block.gradient_checkpointing = enabled
            block._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=False)
            fully_shard(block, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
            x = torch.randn(2, 16, device="npu", dtype=dtype, requires_grad=True)
            try:
                with torch.autocast("npu", dtype=torch.bfloat16):
                    loss = block(x).float().square().mean()
                block.phase = "recompute"
                loss.backward()
                result = {"status": "PASS", "grad_finite": bool(torch.isfinite(x.grad).all())}
            except Exception as error:
                result = {"status": type(error).__name__, "error": str(error)}
            print(json.dumps({"checkpoint": enabled, "original_input": str(dtype), **result}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
