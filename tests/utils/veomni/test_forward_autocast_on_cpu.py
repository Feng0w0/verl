"""Test the actual forward AMP context without importing accelerator workers."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import ContextManager

import pytest
import torch


@pytest.fixture
def context_method():
    source = Path("verl/workers/engine/fsdp/transformer_impl.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_forward_autocast_context"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(torch=torch, nullcontext=nullcontext, ContextManager=ContextManager, get_device_name=lambda: "cpu")
    exec(compile(ast.fix_missing_locations(module), "<forward-amp-context>", "exec"), namespace)
    return namespace["_get_forward_autocast_context"]


@pytest.mark.parametrize("flag", [None, True])
def test_default_preserves_bf16_autocast(context_method, flag):
    config = SimpleNamespace() if flag is None else SimpleNamespace(enable_autocast=flag)
    engine = SimpleNamespace(engine_config=config)
    with context_method(engine):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_disabled_overrides_inherited_context_without_changing_fsdp(context_method):
    config = SimpleNamespace(enable_autocast=False, mixed_precision=True)
    engine = SimpleNamespace(engine_config=config, _autocast_dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with context_method(engine):
            assert not torch.is_autocast_enabled("cpu")
            assert config.mixed_precision
            assert engine._autocast_dtype == torch.bfloat16
        assert torch.is_autocast_enabled("cpu")


def test_fp32_keeps_existing_noop_behavior(context_method):
    engine = SimpleNamespace(engine_config=SimpleNamespace(), _autocast_dtype=torch.float32)
    with context_method(engine):
        assert not torch.is_autocast_enabled("cpu")
