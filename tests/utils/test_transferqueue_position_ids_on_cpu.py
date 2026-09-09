"""Regression coverage for TransferQueue mRoPE sequence-axis preservation."""

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import _normalize_tq_position_ids


@pytest.mark.parametrize("lengths", [(73, 73), (4, 4), (71, 73)])
def test_tq_mrope_survives_engine_micro_batch_slicing(lengths):
    samples = [torch.arange(length).expand(4, -1).clone() + 100 * i for i, length in enumerate(lengths)]
    layout = torch.jagged if lengths[0] == lengths[1] else torch.strided
    raw = torch.nested.as_nested_tensor(samples, layout=layout)
    td = TensorDict({"position_ids": raw}, batch_size=[2])
    _normalize_tq_position_ids(td)
    tu.maybe_fix_3d_position_ids(td)
    for chunk, expected in zip(tu.chunk_tensordict(td, 2), samples, strict=True):
        torch.testing.assert_close(chunk["position_ids"].values(), expected)


def test_canonical_mrope_is_unchanged():
    values = torch.arange(8).expand(4, -1).clone()
    positions = torch.nested.nested_tensor_from_jagged(values, offsets=torch.tensor([0, 4, 8]), jagged_dim=2)
    td = TensorDict({"position_ids": positions}, batch_size=[2])
    _normalize_tq_position_ids(td)
    assert td["position_ids"] is positions


def test_text_positions_are_unchanged():
    positions = torch.nested.as_nested_tensor([torch.arange(7), torch.arange(9)], layout=torch.jagged)
    td = TensorDict({"position_ids": positions}, batch_size=[2])
    _normalize_tq_position_ids(td)
    assert td["position_ids"] is positions
