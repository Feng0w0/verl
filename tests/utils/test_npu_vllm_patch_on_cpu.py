# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock


def _load_npu_vllm_patch_module():
    module_path = Path(__file__).parents[2] / "verl" / "utils" / "vllm" / "npu_vllm_patch.py"
    spec = importlib.util.spec_from_file_location("test_npu_vllm_patch", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_npu_vllm_patches_accepts_modular_fused_moe(monkeypatch):
    npu_vllm_patch = _load_npu_vllm_patch_module()
    monkeypatch.setattr(npu_vllm_patch, "is_torch_npu_available", lambda check_device=False: True)
    rotary_patch = Mock()
    monkeypatch.setattr(npu_vllm_patch, "patch_vllm013_rotary_emb", rotary_patch)

    vllm = ModuleType("vllm")
    model_executor = ModuleType("vllm.model_executor")
    layers = ModuleType("vllm.model_executor.layers")
    fused_moe = ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe.FusedMoEFactory = lambda *args, **kwargs: None
    layers.fused_moe = fused_moe

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fused_moe)

    npu_vllm_patch.apply_npu_vllm_patches()

    rotary_patch.assert_called_once_with()


def test_modular_ascend_moe_reload_restores_layout_once(monkeypatch):
    import torch

    patch = _load_npu_vllm_patch_module()
    monkeypatch.setattr(patch, "is_torch_npu_available", lambda check_device=False: True)
    ascend_module = ModuleType("vllm_ascend.ops.fused_moe.fused_moe")

    class AscendMethod:
        pass

    ascend_module.AscendUnquantizedFusedMoEMethod = AscendMethod
    monkeypatch.setitem(sys.modules, ascend_module.__name__, ascend_module)

    checkpoint_w13 = torch.arange(2 * 8 * 6).reshape(2, 8, 6).float()
    checkpoint_w2 = torch.arange(2 * 6 * 4).reshape(2, 6, 4).float()
    layer = torch.nn.Module()
    layer.quant_method = AscendMethod()
    layer.w13_weight = torch.nn.Parameter(checkpoint_w13.transpose(1, 2).contiguous())
    layer.w2_weight = torch.nn.Parameter(checkpoint_w2.transpose(1, 2).contiguous())
    alias = torch.nn.Module()
    alias.quant_method = layer.quant_method
    alias.w13_weight = layer.w13_weight
    alias.w2_weight = layer.w2_weight
    other = torch.nn.Module()
    other.quant_method = object()
    other.w13_weight = torch.nn.Parameter(torch.zeros(2, 6, 8))
    model = torch.nn.ModuleList([layer, alias, other])
    for _ in range(2):
        assert patch.prepare_npu_moe_weights_for_reload(model) == 2
        torch.testing.assert_close(layer.w13_weight, checkpoint_w13)
        torch.testing.assert_close(layer.w2_weight, checkpoint_w2)
        assert other.w13_weight.shape == (2, 6, 8)
        # Emulate Ascend's post-load processing between reloads.
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()


def test_moe_reload_is_noop_without_npu(monkeypatch):
    patch = _load_npu_vllm_patch_module()
    monkeypatch.setattr(patch, "is_torch_npu_available", lambda check_device=False: False)
    model = Mock()
    assert patch.prepare_npu_moe_weights_for_reload(model) == 0
    model.modules.assert_not_called()
