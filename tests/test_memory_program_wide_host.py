import pytest
import torch
from torch import nn
from experiments.expressivity_tasks.x47_xsa_attention.wide_host import copies, expand_into


class Double(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.embed = nn.Embedding(256, 4 * r)
        self.norm = nn.RMSNorm(4 * r, eps=1e-5)
        self.mix = nn.Linear(4 * r, 8 * r)
        self.compress = nn.Linear(8 * r, 3)
        self.expand = nn.Linear(3, 4 * r)
        self.conv = nn.Conv1d(4 * r, 4 * r, 2, groups=4 * r, bias=False)
        self.head = nn.Linear(4 * r, 256)

    def forward(self, ids):
        x = self.embed(ids)
        x = x + self.expand(self.compress(torch.nn.functional.gelu(self.mix(self.norm(x)))))
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x, self.head(x)


def test_dense_expansion_preserves_bottleneck_channels_conv_and_unreplicated_head():
    torch.manual_seed(7681)
    a = Double(1)
    b = Double(4)
    report = expand_into(a, b, 4)
    ids = torch.randint(256, (3, 9))
    x, p = a(ids)
    y, q = b(ids)
    torch.testing.assert_close(y, x.repeat(1, 1, 4))
    torch.testing.assert_close(q, p)
    assert (
        report["parameters_verified"] == sum(v.numel() for v in b.parameters())
        and report["tensors"] == report["distinct_storages"]
    )


def test_replica_refuses_unknown_rule_inexact_geometry_missing_keys_and_alias():
    with pytest.raises(ValueError, match="unsupported"):
        copies(torch.ones(1), torch.ones(4), nn.Identity(), "weight", 4)
    with pytest.raises(ValueError, match="geometry"):
        copies(torch.ones(2, 3), torch.ones(8, 7), nn.Linear(7, 8), "weight", 4)
    a = Double(1)
    b = Double(4)
    b.register_buffer("extra", torch.ones(1))
    with pytest.raises(ValueError, match="coverage"):
        expand_into(a, b, 4)
    with pytest.raises(ValueError, match="independent"):
        expand_into(a, a, 4)
