"""Freeze only the PLE lookup table and reject accidental updates."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.veomni.frozen_ple import FrozenPLEGuard, is_ple_table


def model_with_ple():
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    model.ple = torch.nn.Module()
    model.ple.ple_embedding = torch.nn.Module()
    model.ple.ple_embedding.ngram_embedding = torch.nn.ModuleDict({"shard_0": torch.nn.Embedding(8, 4)})
    model.ple.key_proj = torch.nn.Linear(4, 2)
    return model


def test_frozen_bf16_storage_preserves_checkpoint_values():
    from verl.utils.veomni.frozen_ple import prepare_ple_storage

    model = model_with_ple()
    table = model.ple.ple_embedding.ngram_embedding.shard_0.weight
    expected = table.detach().to(torch.bfloat16)
    table.data = expected.float()
    prepare_ple_storage(model, "bfloat16")
    assert table.dtype == torch.bfloat16
    torch.testing.assert_close(table, expected, rtol=0, atol=0)
    assert model.ple.key_proj.weight.dtype == torch.float32


def test_frozen_storage_can_be_prepared_on_meta():
    from verl.utils.veomni.frozen_ple import prepare_ple_storage

    model = model_with_ple().to("meta")
    prepare_ple_storage(model, "bfloat16")
    assert model.ple.ple_embedding.ngram_embedding.shard_0.weight.dtype == torch.bfloat16
    assert model.ple.key_proj.weight.dtype == torch.float32


def test_only_lookup_table_is_frozen_and_unchanged_after_adam():
    model = model_with_ple()
    guard = FrozenPLEGuard(model)
    table = model.ple.ple_embedding.ngram_embedding.shard_0.weight
    before = table.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    output = model.ple.key_proj(model.ple.ple_embedding.ngram_embedding.shard_0(torch.tensor([1, 3])))
    output.square().sum().backward()
    assert model.ple.key_proj.weight.grad is not None
    optimizer.step()
    guard.verify(model, optimizer)
    torch.testing.assert_close(table, before, rtol=0, atol=0)
    assert table.grad is None


def test_guard_rejects_accidental_mutation():
    model = model_with_ple()
    guard = FrozenPLEGuard(model)
    with torch.no_grad():
        model.ple.ple_embedding.ngram_embedding.shard_0.weight.add_(1)
    with pytest.raises(RuntimeError, match="Frozen PLE invariant"):
        guard.verify(model)


def test_guard_rejects_table_in_optimizer():
    model = model_with_ple()
    guard = FrozenPLEGuard(model)
    with pytest.raises(RuntimeError, match="Frozen PLE invariant"):
        guard.verify(model, torch.optim.AdamW(model.parameters()))


def test_table_name_matching_does_not_freeze_projections():
    assert is_ple_table("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight")
    assert not is_ple_table("model.language_model.layers.1.ple.key_proj.weight")
    assert not is_ple_table("model.embed_tokens.weight")


def test_missing_table_fails_closed():
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type="qwen4_exp")
    with pytest.raises(ValueError, match="no PLE table"):
        FrozenPLEGuard(model)


def test_multi_optimizer_container_checks_every_child():
    model = model_with_ple()
    guard = FrozenPLEGuard(model)
    trainable = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    container = SimpleNamespace(optimizers_dict={"dense": trainable})
    guard.verify(model, container)
    table = model.ple.ple_embedding.ngram_embedding.shard_0.weight
    container.optimizers_dict["ple"] = torch.optim.AdamW([table])
    with pytest.raises(RuntimeError, match="Frozen PLE invariant"):
        guard.verify(model, container)


def test_dtensor_storage_conversion_and_freeze(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from verl.utils.veomni.frozen_ple import prepare_ple_storage

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ple",))
        model = model_with_ple()
        embedding = model.ple.ple_embedding.ngram_embedding.shard_0
        expected = embedding.weight.detach().to(torch.bfloat16)
        embedding.weight = torch.nn.Parameter(distribute_tensor(expected.float(), mesh, [Shard(0)]))
        original = embedding.weight
        prepare_ple_storage(model, "bfloat16")
        assert embedding.weight is original
        assert embedding.weight.dtype == torch.bfloat16
        assert embedding.weight.to_local().dtype == torch.bfloat16
        torch.testing.assert_close(embedding.weight.to_local(), expected, rtol=0, atol=0)
        guard = FrozenPLEGuard(model, expected_device="cpu")
        assert guard.local_bytes == expected.numel() * 2
        guard.verify(model)
        with torch.no_grad():
            embedding.weight.add_(1)
        with pytest.raises(RuntimeError, match="Frozen PLE invariant"):
            guard.verify(model)
    finally:
        dist.destroy_process_group()


def test_wrong_storage_device_fails_closed():
    with pytest.raises(RuntimeError, match="expected npu"):
        FrozenPLEGuard(model_with_ple(), expected_device="npu")
