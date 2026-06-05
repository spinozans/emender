"""Correctness gate for the UNIFIED parameterized matrix-recurrence cell.

The Triton forward+backward kernels must match the pure-PyTorch reference
recurrence (whose gradients come from autograd) to <1e-4 relative error, across
random inputs/shapes/knobs and ALL phi modes, including the un-cribbed
lambda>=1 regime.

Run: pytest tests/test_unified_cell.py -v   (requires CUDA + Triton)
"""
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="CUDA required for Triton unified-cell tests"),
]

from ndm.triton.unified_cell_forward import (
    unified_cell_forward, unified_cell_torch_reference,
    PHI_IDENTITY, PHI_TANH, PHI_GAMMA_MIX, PHI_RELU, PHI_SOFTPLUS,
)
from ndm.triton.unified_cell_backward import unified_cell

PHI_MODES = [
    (PHI_IDENTITY, 'identity'),
    (PHI_TANH, 'tanh'),
    (PHI_GAMMA_MIX, 'gamma_mix'),
    (PHI_RELU, 'relu'),
    (PHI_SOFTPLUS, 'softplus'),
]

TOL = 1e-4


def _make(T, B, H, N, V, seed, lam_hi=1.3, requires_grad=False, device='cuda',
          gates=False):
    g = torch.Generator(device=device).manual_seed(seed)
    S0 = 0.05 * torch.randn(B, H, N, V, device=device, generator=g)
    k = torch.randn(T, B, H, N, device=device, generator=g)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = torch.randn(T, B, H, N, device=device, generator=g)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = 0.3 * torch.randn(T, B, H, V, device=device, generator=g)
    # lambda spans (0.3, lam_hi) -> includes the un-cribbed >=1 regime
    lam = 0.3 + (lam_hi - 0.3) * torch.rand(T, B, H, device=device, generator=g)
    beta = torch.rand(T, B, H, device=device, generator=g)
    igain = 0.5 + torch.rand(T, B, H, device=device, generator=g)
    gamma = torch.rand(H, device=device, generator=g)
    ts = [S0, k, v, q, lam, beta, igain, gamma]
    if gates:
        # E97 split gates: erase b in (0,1)^N, value w in (0,1)^V (sigmoid of randn)
        b_gate = torch.sigmoid(torch.randn(T, B, H, N, device=device, generator=g))
        w_gate = torch.sigmoid(torch.randn(T, B, H, V, device=device, generator=g))
        ts += [b_gate, w_gate]
    if requires_grad:
        ts = [t.detach().clone().requires_grad_(True) for t in ts]
    return ts


def _relerr(a, b):
    return (a - b).abs().max().item() / b.abs().max().clamp_min(1e-8).item()


@pytest.mark.parametrize("phi_mode,name", PHI_MODES)
@pytest.mark.parametrize("shape", [(32, 2, 4, 24, 20), (64, 1, 8, 16, 16), (16, 3, 5, 32, 32)])
def test_forward_matches_reference(phi_mode, name, shape):
    T, B, H, N, V = shape
    S0, k, v, q, lam, beta, igain, gamma = _make(T, B, H, N, V, seed=hash((phi_mode, shape)) & 0xffff)
    out, S_final, S_ckpt = unified_cell_forward(S0, k, v, q, lam, beta, igain, gamma, phi_mode=phi_mode)
    out_ref, S_ref, ckpt_dense = unified_cell_torch_reference(S0, k, v, q, lam, beta, igain, gamma, phi_mode=phi_mode)
    assert _relerr(out, out_ref) < TOL, f"{name}: out relerr too high"
    assert _relerr(S_final, S_ref) < TOL, f"{name}: S_final relerr too high"
    # sparse checkpoint (interval 16) must match the dense subsample.
    ckpt_int = 16
    if T % ckpt_int == 0:
        for slot in range(S_ckpt.shape[0]):
            dense_idx = 0 if slot == 0 else slot * ckpt_int
            assert _relerr(S_ckpt[slot], ckpt_dense[dense_idx]) < TOL


@pytest.mark.parametrize("phi_mode,name", PHI_MODES)
def test_backward_matches_autograd(phi_mode, name):
    T, B, H, N, V = 48, 3, 5, 24, 20
    seed = 1000 + phi_mode
    tri = _make(T, B, H, N, V, seed=seed, requires_grad=True)
    ref = _make(T, B, H, N, V, seed=seed, requires_grad=True)
    S0, k, v, q, lam, beta, igain, gamma = tri
    out, _ = unified_cell(k, v, q, lam, beta, igain, gamma, S0, phi_mode=phi_mode)
    go = torch.randn_like(out)
    out.backward(go)
    S0r, kr, vr, qr, lamr, betar, igr, gammar = ref
    out2, _, _ = unified_cell_torch_reference(S0r, kr, vr, qr, lamr, betar, igr, gammar, phi_mode=phi_mode)
    out2.backward(go)

    def cmp(a, b):
        if b is None or (b is not None and b.abs().max() == 0):
            return 0.0 if (a is None or a.abs().max() < 1e-6) else float('inf')
        return _relerr(a, b)

    pairs = [('out', out, out2), ('dk', k.grad, kr.grad), ('dv', v.grad, vr.grad),
             ('dq', q.grad, qr.grad), ('dlam', lam.grad, lamr.grad),
             ('dbeta', beta.grad, betar.grad), ('dig', igain.grad, igr.grad),
             ('dgamma', gamma.grad, gammar.grad), ('dS0', S0.grad, S0r.grad)]
    for nm, a, b in pairs:
        e = cmp(a, b)
        assert e < TOL, f"{name}: grad {nm} relerr {e:.2e} >= {TOL}"


def test_e88_equivalence():
    """E88's recurrence is the unified cell at lambda=decay, beta=1, igain=1, tanh."""
    from ndm.triton.e88_triton_forward import e88_torch_reference
    T, B, H, N, V = 32, 2, 4, 16, 16
    device = 'cuda'
    g = torch.Generator(device=device).manual_seed(123)
    S0 = 0.05 * torch.randn(B, H, N, V, device=device, generator=g)
    k = torch.randn(T, B, H, N, device=device, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    q = torch.randn(T, B, H, N, device=device, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    v = 0.3 * torch.randn(T, B, H, V, device=device, generator=g)
    decay = 0.3 + 0.6 * torch.rand(T, B, H, device=device, generator=g)
    # E88 reference
    out_e88, S_e88, _ = e88_torch_reference(S0, k, v, q, decay, linear_state=False)
    # Unified cell: beta=1, igain=1, tanh
    one = torch.ones_like(decay)
    gamma = torch.zeros(H, device=device)
    out_u, S_u, _ = unified_cell_torch_reference(S0, k, v, q, decay, one, one, gamma, phi_mode=PHI_TANH)
    assert _relerr(out_u, out_e88) < TOL
    assert _relerr(S_u, S_e88) < TOL


def test_high_lambda_stability():
    """lambda>1 with tanh self-bounds (latch corner): |S| stays <= 1, no NaN/Inf."""
    T, B, H, N, V = 64, 2, 4, 16, 16
    S0, k, v, q, lam, beta, igain, gamma = _make(T, B, H, N, V, seed=42, lam_hi=1.5)
    lam = torch.full_like(lam, 1.3)   # runaway gain
    beta = torch.zeros_like(beta)     # pure accumulation
    out, S_final, _ = unified_cell_forward(S0, k, v, q, lam, beta, igain, gamma, phi_mode=PHI_TANH)
    assert torch.isfinite(out).all() and torch.isfinite(S_final).all()
    assert S_final.abs().max() <= 1.0 + 1e-4


# ===========================================================================
# E97 SPLIT-GATE (E98): decoupled erase/read (b*k) and value-write (w*v) gates.
# pre = lambda*S - beta*k((b*k)^T S) + i*k(w*v)^T ;  b=w=1 reduces to the E88-based
# unified cell. These re-validate the split-gated kernel fwd+bwd <1e-4 over all
# phi modes incl lambda>=1, and ADD an E97-equivalence test matching the
# e88_fla_hybrid use_split_edit recurrence.
# ===========================================================================
@pytest.mark.parametrize("phi_mode,name", PHI_MODES)
@pytest.mark.parametrize("shape", [(32, 2, 4, 24, 20), (64, 1, 8, 16, 16), (16, 3, 5, 32, 32)])
def test_split_forward_matches_reference(phi_mode, name, shape):
    T, B, H, N, V = shape
    S0, k, v, q, lam, beta, igain, gamma, b_gate, w_gate = _make(
        T, B, H, N, V, seed=hash((phi_mode, shape, 'split')) & 0xffff, gates=True)
    out, S_final, S_ckpt = unified_cell_forward(
        S0, k, v, q, lam, beta, igain, gamma, phi_mode=phi_mode,
        b_gate=b_gate, w_gate=w_gate)
    out_ref, S_ref, ckpt_dense = unified_cell_torch_reference(
        S0, k, v, q, lam, beta, igain, gamma, phi_mode=phi_mode,
        b_gate=b_gate, w_gate=w_gate)
    assert _relerr(out, out_ref) < TOL, f"{name}: split out relerr too high"
    assert _relerr(S_final, S_ref) < TOL, f"{name}: split S_final relerr too high"
    ckpt_int = 16
    if T % ckpt_int == 0:
        for slot in range(S_ckpt.shape[0]):
            dense_idx = 0 if slot == 0 else slot * ckpt_int
            assert _relerr(S_ckpt[slot], ckpt_dense[dense_idx]) < TOL


@pytest.mark.parametrize("phi_mode,name", PHI_MODES)
def test_split_backward_matches_autograd(phi_mode, name):
    T, B, H, N, V = 48, 3, 5, 24, 20
    seed = 2000 + phi_mode
    tri = _make(T, B, H, N, V, seed=seed, requires_grad=True, gates=True)
    ref = _make(T, B, H, N, V, seed=seed, requires_grad=True, gates=True)
    S0, k, v, q, lam, beta, igain, gamma, b_gate, w_gate = tri
    out, _ = unified_cell(k, v, q, lam, beta, igain, gamma, S0, phi_mode=phi_mode,
                          b_gate=b_gate, w_gate=w_gate)
    go = torch.randn_like(out)
    out.backward(go)
    S0r, kr, vr, qr, lamr, betar, igr, gammar, bgr, wgr = ref
    out2, _, _ = unified_cell_torch_reference(
        S0r, kr, vr, qr, lamr, betar, igr, gammar, phi_mode=phi_mode,
        b_gate=bgr, w_gate=wgr)
    out2.backward(go)

    def cmp(a, b):
        if b is None or (b is not None and b.abs().max() == 0):
            return 0.0 if (a is None or a.abs().max() < 1e-6) else float('inf')
        return _relerr(a, b)

    pairs = [('out', out, out2), ('dk', k.grad, kr.grad), ('dv', v.grad, vr.grad),
             ('dq', q.grad, qr.grad), ('db', b_gate.grad, bgr.grad),
             ('dw', w_gate.grad, wgr.grad), ('dlam', lam.grad, lamr.grad),
             ('dbeta', beta.grad, betar.grad), ('dig', igain.grad, igr.grad),
             ('dgamma', gamma.grad, gammar.grad), ('dS0', S0.grad, S0r.grad)]
    for nm, a, b in pairs:
        e = cmp(a, b)
        assert e < TOL, f"{name}: split grad {nm} relerr {e:.2e} >= {TOL}"


def test_split_reduces_to_unified_when_gates_one():
    """b_gate=w_gate=1 must reproduce the E88-based unified cell EXACTLY."""
    T, B, H, N, V = 32, 2, 4, 16, 16
    S0, k, v, q, lam, beta, igain, gamma = _make(T, B, H, N, V, seed=7, lam_hi=1.5)
    ones_b = torch.ones(T, B, H, N, device='cuda')
    ones_w = torch.ones(T, B, H, V, device='cuda')
    out_plain, S_plain, _ = unified_cell_forward(
        S0, k, v, q, lam, beta, igain, gamma, phi_mode=PHI_GAMMA_MIX)
    out_split, S_split, _ = unified_cell_forward(
        S0, k, v, q, lam, beta, igain, gamma, phi_mode=PHI_GAMMA_MIX,
        b_gate=ones_b, w_gate=ones_w)
    assert _relerr(out_split, out_plain) < TOL
    assert _relerr(S_split, S_plain) < TOL


def _e97_reference_recurrence(S0, k, v, q, decay, b_gate, w_gate):
    """Faithful copy of the e88_fla_hybrid use_split_edit=True PyTorch recurrence
    (ndm/models/e88_fla_hybrid.py, the non-fused loop, use_l2_norm already applied
    upstream, raw_write=False, pos_eigval_clamp=False, no write_gate):

        read_key    = k * b ;  write_value = v * w
        retrieved   = (read_key)^T S
        delta       = write_value - retrieved
        S           = tanh(decay*S + delta (x) k)        # outer key = ungated k
        out         = S^T q
    k,q assumed L2-normalized. decay: [T,B,H]. Returns out [T,B,H,V], S_final.
    """
    T, B, H, N = k.shape
    S = S0.to(torch.float32).clone()
    outs = []
    for t in range(T):
        k_t = k[t].float(); q_t = q[t].float(); v_t = v[t].float()
        read_key = k_t * b_gate[t].float()                    # b*k
        write_value = v_t * w_gate[t].float()                 # w*v
        decay_t = decay[t].float().unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
        retrieved = torch.einsum('bhnv,bhn->bhv', S, read_key)  # (b*k)^T S
        delta = write_value - retrieved
        outer = torch.einsum('bhv,bhn->bhnv', delta, k_t)      # outer key = ungated k
        S = torch.tanh(decay_t * S + outer)
        outs.append(torch.einsum('bhnv,bhn->bhv', S, q_t))
    return torch.stack(outs, dim=0), S


def test_e97_equivalence():
    """E97 (split-edit E88) = unified split-gated cell at lambda=decay in (0,1),
    beta=1, igain=1, phi=tanh, split-gate ON. Matches the e88_fla_hybrid
    use_split_edit recurrence to <1e-4 (fwd + backward, Triton kernel)."""
    T, B, H, N, V = 32, 2, 4, 16, 16
    device = 'cuda'
    g = torch.Generator(device=device).manual_seed(97)
    S0 = 0.05 * torch.randn(B, H, N, V, device=device, generator=g)
    k = torch.randn(T, B, H, N, device=device, generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    q = torch.randn(T, B, H, N, device=device, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    v = 0.3 * torch.randn(T, B, H, V, device=device, generator=g)
    decay = 0.3 + 0.6 * torch.rand(T, B, H, device=device, generator=g)  # in (0,1)
    b_gate = torch.sigmoid(torch.randn(T, B, H, N, device=device, generator=g))
    w_gate = torch.sigmoid(torch.randn(T, B, H, V, device=device, generator=g))

    # E97 reference recurrence (the e88_fla_hybrid split-edit path)
    out_e97, S_e97 = _e97_reference_recurrence(S0, k, v, q, decay, b_gate, w_gate)

    # Unified split-gated cell: beta=1, igain=1, tanh, lambda=decay, split ON.
    one = torch.ones_like(decay)
    gamma = torch.zeros(H, device=device)
    # forward (Triton)
    out_u, S_u, _ = unified_cell_forward(
        S0, k, v, q, decay, one, one, gamma, phi_mode=PHI_TANH,
        b_gate=b_gate, w_gate=w_gate)
    assert _relerr(out_u, out_e97) < TOL, "E97 forward mismatch"
    assert _relerr(S_u, S_e97) < TOL, "E97 S_final mismatch"

    # backward equivalence: gradients of the Triton split cell vs autograd through
    # the E97 reference, for a random scalar objective.
    tens = {n: t.detach().clone().requires_grad_(True)
            for n, t in dict(k=k, v=v, q=q, decay=decay, b=b_gate, w=w_gate).items()}
    out_t, _ = unified_cell(
        tens['k'], tens['v'], tens['q'], tens['decay'], one, one, gamma,
        S0, phi_mode=PHI_TANH, b_gate=tens['b'], w_gate=tens['w'])
    go = torch.randn_like(out_t)
    out_t.backward(go)
    rens = {n: t.detach().clone().requires_grad_(True)
            for n, t in dict(k=k, v=v, q=q, decay=decay, b=b_gate, w=w_gate).items()}
    out_r, _ = _e97_reference_recurrence(
        S0, rens['k'], rens['v'], rens['q'], rens['decay'], rens['b'], rens['w'])
    out_r.backward(go)
    for nm in ('k', 'v', 'q', 'decay', 'b', 'w'):
        e = _relerr(tens[nm].grad, rens[nm].grad)
        assert e < TOL, f"E97 grad {nm} relerr {e:.2e} >= {TOL}"


# === E98 SIXTH CORNER: gated-delta backbone (input-dependent gated decay) =====
def test_gated_delta_preset_is_clean_overwrite_corner():
    """The gated-delta PRESET realises the GDN clean-overwrite operating point in
    the E98 cell: beta=1 (full delta), identity phi, along-key eig (lambda-beta)
    ~= 0, head_norm on, and input-dependent decay enabled."""
    from ndm.models.unified_cell import UnifiedCellLayer, PRESETS
    m = UnifiedCellLayer(dim=128, n_state=16, n_heads=8, knob_mode='pinned',
                         preset='gated-delta', split_gate=True).to('cuda')
    kv = m.knob_values()
    assert torch.allclose(kv['beta'], torch.ones(8), atol=1e-3), "beta must be ~1 (full delta)"
    assert kv['eig_along'].abs().max() < 0.05, "along-key eig must be ~0 (clean overwrite)"
    assert m.phi_mode == PHI_IDENTITY, "gated-delta phi must be identity (linear state)"
    assert m.decay_gate and m.decay_gate_proj is not None, "decay gate must be enabled"
    assert m.head_norm, "gated-delta uses head_norm readout stabiliser"
    assert PRESETS['gated-delta']['decay_gate'] is True


def test_gated_delta_decay_is_input_dependent():
    """The decay lambda_t must genuinely depend on the input (Mamba/GDN gated
    decay), NOT be a fixed per-head scalar: two different inputs must induce
    different lambda_t, and lambda_t must stay in (0, lam_max)."""
    from ndm.models.unified_cell import UnifiedCellLayer
    torch.manual_seed(0)
    m = UnifiedCellLayer(dim=128, n_state=16, n_heads=8, knob_mode='pinned',
                         preset='gated-delta', split_gate=True).to('cuda')
    # randomise the decay gate so it is not a constant map
    with torch.no_grad():
        m.decay_gate_proj.weight.normal_(0, 1.0)
    B, T, D, H = 2, 8, 128, 8
    x = torch.randn(B, T, D, device='cuda')
    lam_t = m.lam_max * torch.sigmoid(m.decay_gate_proj(x))  # [B,T,H]
    assert (lam_t > 0).all() and (lam_t <= m.lam_max + 1e-6).all(), "lambda_t out of (0,lam_max]"
    # input-dependence: variance across the time axis is non-trivial
    assert lam_t.std(dim=1).max().item() > 1e-3, "lambda_t does not vary with input"
    # forward + backward routes gradient into the decay gate
    x2 = torch.randn(B, T, D, device='cuda', requires_grad=True)
    y = m(x2); y.float().pow(2).mean().backward()
    assert m.decay_gate_proj.weight.grad is not None
    assert m.decay_gate_proj.weight.grad.norm().item() > 0


def test_spread6_places_all_six_corners():
    """spread-6 init partitions the 8+ heads across all six corners, including a
    gated-delta head at eig ~= 0 (the new backbone) alongside the leaky head."""
    from ndm.models.unified_cell import UnifiedCellLayer, SPREAD_CORNERS_6, SPEC_CORNERS
    m = UnifiedCellLayer(dim=128, n_state=16, n_heads=12, knob_mode='learned',
                         phi='gamma_mix', lam_max=1.5, spread_init=True,
                         n_spread_corners=6, split_gate=True).to('cuda')
    kv = m.knob_values()
    lam, beta, gam = kv['lambda'], kv['beta'], kv['gamma']
    # gated-delta is corner index 5 -> heads 5, 11 with round-robin over 6 corners
    gl, gb, gg = SPEC_CORNERS['gated-delta']
    for h in (5, 11):
        assert abs(float(lam[h]) - gl) < 0.02 and abs(float(beta[h]) - gb) < 0.02, \
            f"head {h} should init at gated-delta corner"
        assert abs(float(kv['eig_along'][h])) < 0.05, "gated-delta head eig must be ~0"
    assert SPREAD_CORNERS_6[5] == 'gated-delta'
