"""hetero-kernel: stream-overlap must be numerically identical to the sequential
sub-block execution (forward exact, fwd+bwd grads at bf16 noise) for every blended
composition. This guards the throughput optimization (running the latency-bound
gdn2_nonlin_shell scan on a side CUDA stream concurrently with the chunked bulk)
against silent numerical drift."""
from __future__ import annotations
import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for stream-overlap parity"),
]
from ndm.models.typed_head_mixture import TypedHeadMixtureLayer, TYPE_NAMES


def _relerr(a, b):
    return (a - b).abs().max().item() / b.abs().max().clamp_min(1e-8).item()


def _logits(types):
    lg = [-9.0] * len(TYPE_NAMES)
    for t in types:
        lg[TYPE_NAMES.index(t)] = 2.0
    return lg


def _build(types, overlap, dim=512, nh=16, seed=0, e97_state_nonlin='tanh',
           use_chunked_e97_delta=True):
    torch.manual_seed(seed)
    return TypedHeadMixtureLayer(
        dim=dim, n_state=32, n_heads=nh, head_type_logits=_logits(types),
        shell_state_nonlin='tanh', shell_state_chunk=16, shell_fused=True,
        e97_state_nonlin=e97_state_nonlin, use_chunked_e97_delta=use_chunked_e97_delta,
        e97_chunk_size=16, overlap_streams=overlap,
    ).cuda().bfloat16()


@pytest.mark.parametrize("types", [
    ['gdn2_recall', 'gdn2_nonlin_shell'],
    ['gdn2_recall', 'e97_delta', 'gdn2_nonlin_shell'],
    ['gdn2_recall', 'e97_raw', 'e97_delta', 'gdn2_nonlin_shell'],
])
def test_overlap_matches_sequential_fwd_bwd(types):
    seq = _build(types, overlap=False)
    ovl = _build(types, overlap=True)
    ovl.load_state_dict(seq.state_dict())
    assert seq.alloc['counts']['gdn2_nonlin_shell'] > 0  # overlap path is exercised
    x = torch.randn(2, 96, 512, device='cuda', dtype=torch.bfloat16)
    x0 = x.clone().requires_grad_(True)
    x1 = x.clone().requires_grad_(True)
    o0 = seq(x0)
    o1 = ovl(x1)
    g = torch.randn_like(o0)
    o0.backward(g)
    o1.backward(g)
    torch.cuda.synchronize()
    assert _relerr(o1.float(), o0.float()) < 1e-2, "overlap forward drifted"
    assert _relerr(x1.grad.float(), x0.grad.float()) < 5e-2, "overlap dx drifted"
    for (n, p0), (_, p1) in zip(seq.named_parameters(), ovl.named_parameters()):
        if p0.grad is not None and p1.grad is not None:
            assert _relerr(p1.grad.float(), p0.grad.float()) < 5e-2, f"overlap grad drift {n}"


@pytest.mark.parametrize("types", [
    ['gdn2_recall', 'e97_delta'],
    ['gdn2_recall', 'e97_raw', 'e97_delta'],
])
def test_overlap_seq_split_edit_matches_sequential(types):
    """The depth-capability head is e97_delta with per-step tanh on split-edit, which
    runs the SEQUENTIAL T-scan (no chunked kernel for nonlinear state). hetero-cma
    extends the side-stream overlap to cover it. Overlap must be numerically identical
    to running it on the main stream."""
    seq = _build(types, overlap=False, e97_state_nonlin='tanh', use_chunked_e97_delta=False)
    ovl = _build(types, overlap=True, e97_state_nonlin='tanh', use_chunked_e97_delta=False)
    ovl.load_state_dict(seq.state_dict())
    assert seq.alloc['counts']['e97_delta'] > 0
    assert ovl._e97_delta_is_seq() and ovl._overlap_active(torch.empty(1, device='cuda'))
    x = torch.randn(2, 96, 512, device='cuda', dtype=torch.bfloat16)
    x0 = x.clone().requires_grad_(True)
    x1 = x.clone().requires_grad_(True)
    o0 = seq(x0)
    o1 = ovl(x1)
    g = torch.randn_like(o0)
    o0.backward(g)
    o1.backward(g)
    torch.cuda.synchronize()
    assert _relerr(o1.float(), o0.float()) < 1e-2, "seq-split-edit overlap forward drifted"
    assert _relerr(x1.grad.float(), x0.grad.float()) < 5e-2, "seq-split-edit overlap dx drifted"
