"""Frozen Qwen4-Exp PLE tables: optimizer exclusion and mutation guards."""

import re

import torch

_PLE_TABLE = re.compile(r"(?:^|\.)ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")


def is_ple_table(name: str) -> bool:
    return _PLE_TABLE.search(name) is not None


def prepare_ple_storage(model: torch.nn.Module, dtype: str | None) -> None:
    """Set table storage before freezing; leave all trainable weights unchanged."""
    if dtype is None:
        return
    if dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported frozen PLE dtype: {dtype}")
    for name, param in model.named_parameters():
        if is_ple_table(name):
            # DTensor .data assignment changes wrapper dtype but leaves its
            # local shard unchanged. Swap the complete tensor implementation.
            converted = torch.nn.Parameter(param.to(dtype=getattr(torch, dtype)), requires_grad=param.requires_grad)
            torch.utils.swap_tensors(param, converted)


def _optimizer_parameter_ids(optimizer):
    if optimizer is None:
        return set()
    children = optimizer if isinstance(optimizer, dict) else getattr(optimizer, "optimizers_dict", None)
    if children is not None:
        return set().union(*(_optimizer_parameter_ids(child) for child in children.values()))
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


class FrozenPLEGuard:
    def __init__(self, model: torch.nn.Module, expected_device: str | None = None):
        if getattr(model.config, "model_type", None) != "qwen4_exp":
            raise ValueError("freeze_ple_embeddings currently supports only qwen4_exp")
        self.parameters = {}
        self.local_bytes = 0
        for name, param in model.named_parameters():
            if not is_ple_table(name):
                continue
            param.requires_grad_(False)
            param.grad = None
            local = param.to_local() if hasattr(param, "to_local") else param
            if expected_device is not None and local.device.type != expected_device:
                raise RuntimeError(f"PLE table {name} is on {local.device}, expected {expected_device}")
            self.parameters[name] = (param, param._version, local._version, local.device)
            self.local_bytes += local.numel() * local.element_size()
        if not self.parameters:
            raise ValueError("freeze_ple_embeddings requested but no PLE table parameters were found")

    def verify(self, model, optimizer=None):
        current = dict(model.named_parameters())
        optimized = _optimizer_parameter_ids(optimizer)
        for name, (original, parameter_version, version, device) in self.parameters.items():
            param = current[name]
            local = param.to_local() if hasattr(param, "to_local") else param
            if (
                param is not original
                or param.requires_grad
                or param.grad is not None
                or param._version != parameter_version
                or local._version != version
                or local.device != device
                or id(param) in optimized
            ):
                raise RuntimeError(f"Frozen PLE invariant violated: {name}")
