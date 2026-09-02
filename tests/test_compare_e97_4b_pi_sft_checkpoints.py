import pytest
import torch

from scripts.compare_e97_4b_pi_sft_checkpoints import compare


def test_recursive_checkpoint_state_comparison_is_bitwise():
    left = {"state": {0: {"z": torch.tensor([1, 2], dtype=torch.bfloat16)}},
            "param_groups": [{"lr": 1e-6, "params": [0]}]}
    right = {"state": {0: {"z": torch.tensor([1, 2], dtype=torch.bfloat16)}},
             "param_groups": [{"lr": 1e-6, "params": [0]}]}
    counts = {"tensors": 0, "tensor_elements": 0, "scalars": 0}
    compare(left, right, "optimizer", counts)
    assert counts["tensors"] == 1
    assert counts["tensor_elements"] == 2


def test_recursive_checkpoint_state_comparison_rejects_one_bit_difference():
    counts = {"tensors": 0, "tensor_elements": 0, "scalars": 0}
    with pytest.raises(RuntimeError, match="tensor values differ"):
        compare(
            {"weight": torch.tensor([1], dtype=torch.int32)},
            {"weight": torch.tensor([0], dtype=torch.int32)},
            "model", counts)
