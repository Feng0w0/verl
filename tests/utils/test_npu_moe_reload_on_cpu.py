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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("legacy_module", [False, True])
def test_two_weight_reloads_preserve_captured_storage(monkeypatch, legacy_module):
    path = Path(__file__).parents[2] / "verl/utils/vllm/npu_vllm_patch.py"
    spec = importlib.util.spec_from_file_location("npu_reload_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "is_torch_npu_available", lambda **kwargs: True)

    class Unquantized:
        pass

    method_module = SimpleNamespace(AscendUnquantizedFusedMoEMethod=Unquantized)
    imports = []

    def load_module(name):
        imports.append(name)
        return method_module if legacy_module or name.endswith("routed_experts") else SimpleNamespace()

    monkeypatch.setattr(module, "import_module", load_module)
    model = torch.nn.Module()
    model.expert = torch.nn.Module()
    model.expert.quant_method = Unquantized()
    originals = {}
    pointers = {}
    for name, shape in [("w13_weight", (2, 3, 7)), ("w2_weight", (2, 5, 3))]:
        original = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
        originals[name] = original
        parameter = torch.nn.Parameter(original.transpose(1, 2).contiguous())
        model.expert.register_parameter(name, parameter)
        pointers[name] = parameter.data_ptr()
    # Shared expert storage must be transposed only once per stream.
    model.alias = torch.nn.Module()
    model.alias.quant_method = Unquantized()
    model.alias.w13_weight = model.expert.w13_weight
    model.alias.w2_weight = model.expert.w2_weight

    with torch.no_grad():
        for step in [1, 2]:
            assert module.prepare_npu_moe_weights_for_reload(model) == 2
            for name, original in originals.items():
                parameter = getattr(model.expert, name)
                parameter.copy_(original + step * 100)
                parameter.data = parameter.data.transpose(1, 2).contiguous()
                torch.testing.assert_close(parameter, (original + step * 100).transpose(1, 2))
                assert parameter.data_ptr() == pointers[name]
    assert any(name.endswith("routed_experts") for name in imports) is not legacy_module
