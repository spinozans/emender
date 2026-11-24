import copy
import pytest
import torch
from torch import nn
from ndm.memory_program.host_vjp import HostVJP


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 5)
        self.out = nn.Linear(5, 259)

    def forward(self, x):
        return self.out(self.embed(x).cumsum(dim=1).tanh())


def test_direct_and_rematerialized_shared_host_all_microbatches_and_separate_calls():
    torch.manual_seed(90)
    a = Tiny()
    b = copy.deepcopy(a)
    x = torch.randint(256, (5, 7))
    outs = []
    for net, remat in [(a, False), (b, True)]:
        tx = HostVJP(net, width=3, microbatch=2, remat=remat)
        h, p = tx.stage(x[:1])
        h2, p2 = tx.stage(x[1:])
        out = (h.sum() + p2.mean()) * (h2.sum() + p.mean())
        outs.append(out.item())
        tx.backward(out)
        assert (
            tx.work["forward_tokens"] == 35
            and tx.work["recompute_tokens"] == (35 if remat else 0)
            and tx.state == "done"
            and not tx.rows
        )
        with pytest.raises(ValueError, match="closed"):
            tx.backward(out)
    assert outs[0] == outs[1]
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_stale_transaction_fails_without_partial_gradients():
    net = Tiny()
    tx = HostVJP(net, width=3, microbatch=1, remat=True)
    h, p = tx.stage(torch.tensor([[0, 30], [255, 0]]))
    with torch.no_grad():
        net.out.weight.add_(1)
    with pytest.raises(ValueError, match="version"):
        tx.backward((h.sum() + p.sum()))
    assert tx.state == "aborted" and not tx.rows and all(p.grad is None for p in net.parameters())
    with pytest.raises(ValueError, match="closed"):
        tx.stage(torch.tensor([[0]]))


def test_stochastic_stateful_or_contaminated_hosts_rejected():
    for m in (nn.Dropout(0.1), nn.BatchNorm1d(259)):
        with pytest.raises(ValueError, match="stochastic or stateful"):
            HostVJP(nn.Sequential(Tiny(), m), width=3, microbatch=1, remat=True)

    class Noise(Tiny):
        def forward(self, x):
            return super().forward(x) + torch.rand(1)

    tx = HostVJP(Noise(), width=3, microbatch=2, remat=True)
    with pytest.raises(ValueError, match="RNG"):
        tx.stage(torch.tensor([[255]]))
    net = Tiny()
    net.out.weight.grad = torch.zeros_like(net.out.weight)
    with pytest.raises(ValueError, match="fresh"):
        HostVJP(net, width=3, microbatch=2, remat=True)


def test_exact_recompute_and_missing_branch_gradient_fail_closed():
    class Drift(Tiny):
        def forward(self, x):
            return super().forward(x) + int(torch.is_grad_enabled())

    for net, reason in [(Drift(), "changed packet"), (Tiny(), "cotangent")]:
        tx = HostVJP(net, width=3, microbatch=1, remat=True)
        h, _ = tx.stage(torch.tensor([[0, 255]]))
        if reason == "cotangent":
            tx.stage(torch.tensor([[30]]))
        with pytest.raises(ValueError, match=reason):
            tx.backward(h.mean())
        assert tx.state == "aborted" and all(p.grad is None for p in net.parameters())
