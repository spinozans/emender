import pytest
import torch

from ndm.models.e88_fla_hybrid import E88FLAHybrid
from ndm.triton.e97_chunked_autograd import _launch_policy


def test_e97_chunked_requires_delta_linear_triton_path():
    with pytest.raises(ValueError, match="use_chunked_e97 requires"):
        E88FLAHybrid(
            dim=64,
            n_heads=2,
            n_state=16,
            expansion=1.0,
            use_gate=True,
            gate_activation="silu",
            use_split_edit=True,
            use_triton=True,
            linear_state=True,
            raw_write=True,
            use_chunked_e97=True,
        )


def test_e97_launch_policy_rejects_unsupported_chunk_size():
    x = torch.empty(1)
    with pytest.raises(ValueError, match="chunk_size"):
        _launch_policy(x, chunk_size=48, backward=False)


def test_e97_launch_policy_defaults_to_frontier_debug_profile(monkeypatch):
    monkeypatch.delenv("E97_TRITON_NUM_WARPS", raising=False)
    monkeypatch.delenv("E97_TRITON_NUM_STAGES", raising=False)
    x = torch.empty(1)
    assert _launch_policy(x, chunk_size=32, backward=True) == {
        "num_warps": 4,
        "num_stages": 1,
    }
