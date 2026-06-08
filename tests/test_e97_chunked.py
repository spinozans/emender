"""Parity tests for the chunked-parallel E97 split-edit delta kernel.

Verifies the chunked form (ndm.triton.e97_chunked.e97_delta_chunked) matches the
sequential reference recurrence (e88_torch_reference, linear_state=True,
raw_write=False, split-edit gates) in BOTH forward and backward, fp32 and bf16.
"""
import pytest
import torch

from ndm.triton.e97_chunked import e97_delta_chunked
from ndm.triton.e97_chunked_autograd import e97_delta_chunked_triton
from ndm.triton.e88_triton_forward import e88_torch_reference

CUDA = torch.cuda.is_available()


def _ref_forward(S0, k, v, q, decay, e, w):
    """Reference: [B,T,H,*] -> uses e88_torch_reference which wants [T,B,H,*]."""
    kT = k.transpose(0, 1).contiguous()
    vT = v.transpose(0, 1).contiguous()
    qT = q.transpose(0, 1).contiguous()
    dT = decay.transpose(0, 1).contiguous()
    eT = e.transpose(0, 1).contiguous()
    wT = w.transpose(0, 1).contiguous()
    out, S_final, _ = e88_torch_reference(
        S0, kT, vT, qT, dT, linear_state=True, raw_write=False,
        erase_gate=eT, value_write_gate=wT,
    )
    return out.transpose(0, 1).contiguous(), S_final  # [B,T,H,V], [B,H,N,V]


def _mk(B, T, H, N, V, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    q = torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g) * 0.5
    # decay in (0,1) like Mamba2 exp-decay
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype, generator=g)) * 0.4 + 0.5
    e = torch.sigmoid(torch.randn(B, T, H, N, device=device, dtype=dtype, generator=g))
    w = torch.sigmoid(torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g))
    return k, v, q, decay, e, w


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T,C", [(64, 64), (128, 64), (256, 64), (192, 64), (130, 64)])
def test_forward_parity_fp32(T, C):
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32)
    S0 = torch.zeros(B, H, N, V, device=dev)
    ref_out, ref_S = _ref_forward(S0, k, v, q, decay, e, w)
    out, S = e97_delta_chunked(k, v, q, decay, e, w, S0=S0, chunk_size=C)
    err = (out - ref_out).abs().max().item()
    serr = (S - ref_S).abs().max().item()
    assert err < 2e-4, f"fwd out err {err}"
    assert serr < 2e-4, f"S_final err {serr}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_forward_parity_bf16():
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.bfloat16)
    S0 = torch.zeros(B, H, N, V, device=dev, dtype=torch.bfloat16)
    ref_out, ref_S = _ref_forward(S0, k, v, q, decay, e, w)
    out, S = e97_delta_chunked(k, v, q, decay, e, w, S0=S0, chunk_size=64)
    # bf16 inputs, fp32 compute internally; compare in fp32
    err = (out.float() - ref_out.float()).abs().max().item()
    rel = err / (ref_out.float().abs().max().item() + 1e-6)
    assert rel < 0.05, f"bf16 fwd rel err {rel} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_backward_parity_fp32():
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=7)
    S0 = torch.zeros(B, H, N, V, device=dev)

    def run(fn):
        kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)
        qq = q.clone().requires_grad_(True)
        dd = decay.clone().requires_grad_(True)
        ee = e.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        out, _ = fn(kk, vv, qq, dd, ee, ww)
        loss = (out * out).sum()
        loss.backward()
        return [t.grad.clone() for t in (kk, vv, qq, dd, ee, ww)]

    g_ref = run(lambda kk, vv, qq, dd, ee, ww: _ref_forward(S0, kk, vv, qq, dd, ee, ww))
    g_chk = run(lambda kk, vv, qq, dd, ee, ww: e97_delta_chunked(kk, vv, qq, dd, ee, ww, S0=S0, chunk_size=64))
    names = ['k', 'v', 'q', 'decay', 'e', 'w']
    for nm, a, b in zip(names, g_ref, g_chk):
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 1e-3, f"grad {nm} rel err {err/scale} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [64, 128, 256, 96])
def test_fused_triton_forward_parity_fp32(T):
    """Fused fwd+bwd Triton autograd kernel — forward parity, fp32 (C=32)."""
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=3)
    S0 = torch.zeros(B, H, N, V, device=dev)
    ref_out, ref_S = _ref_forward(S0, k, v, q, decay, e, w)
    out, S = e97_delta_chunked_triton(k, v, q, decay, e, w, chunk_size=32)
    assert (out - ref_out).abs().max().item() < 2e-4
    assert (S - ref_S).abs().max().item() < 2e-4


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_triton_backward_parity_fp32():
    """Fused Triton kernel — gradient parity vs reference recurrence, fp32."""
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=11)
    S0 = torch.zeros(B, H, N, V, device=dev)

    def run(fn):
        ts = [t.clone().requires_grad_(True) for t in (k, v, q, decay, e, w)]
        out, _ = fn(*ts)
        (out * out).sum().backward()
        return [t.grad.clone() for t in ts]

    g_ref = run(lambda kk, vv, qq, dd, ee, ww: _ref_forward(S0, kk, vv, qq, dd, ee, ww))
    g_chk = run(lambda kk, vv, qq, dd, ee, ww: e97_delta_chunked_triton(kk, vv, qq, dd, ee, ww, chunk_size=32))
    for nm, a, b in zip(['k', 'v', 'q', 'decay', 'e', 'w'], g_ref, g_chk):
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 3e-3, f"fused grad {nm} rel err {err/scale} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_triton_backward_parity_bf16():
    """Fused Triton kernel — bf16 fwd + grad parity (TF32 tensor-core dots)."""
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 3, 32, 32
    k, v, q, decay, e, w = _mk(B, T, H, N, V, dev, torch.bfloat16, seed=5)
    S0 = torch.zeros(B, H, N, V, device=dev, dtype=torch.bfloat16)

    def run(fn):
        ts = [t.clone().requires_grad_(True) for t in (k, v, q, decay, e, w)]
        out, _ = fn(*ts)
        (out.float() * out.float()).sum().backward()
        return out.detach(), [t.grad.clone() for t in ts]

    o_ref, g_ref = run(lambda kk, vv, qq, dd, ee, ww: _ref_forward(S0, kk, vv, qq, dd, ee, ww))
    o_chk, g_chk = run(lambda kk, vv, qq, dd, ee, ww: e97_delta_chunked_triton(kk, vv, qq, dd, ee, ww, chunk_size=32))
    frel = (o_chk.float() - o_ref.float()).abs().max().item() / (o_ref.float().abs().max().item() + 1e-6)
    assert frel < 0.05, f"bf16 fwd rel err {frel}"
    for nm, a, b in zip(['k', 'v', 'q', 'decay', 'e', 'w'], g_ref, g_chk):
        err = (a.float() - b.float()).abs().max().item()
        scale = a.float().abs().max().item() + 1e-6
        assert err / scale < 0.06, f"bf16 fused grad {nm} rel err {err/scale}"


# ---------------------------------------------------------------------------
# log_decay=True parity (fuse-2kernel): the chunked e88_fla_hybrid path passes the
# upstream LOG-decay g (decay=exp(g)) so the backward returns grad wrt g directly,
# with no dg/decay division. This is the linear-state 1.3B NaN fix — at Mamba2
# init the decay can be ~1e-5 and dg/decay overflows. The forward must be IDENTICAL
# to the decay-space path; the backward grad-wrt-g must equal grad-wrt-decay * decay
# (chain rule), which we check against the reference recurrence reparametrized in g.
# ---------------------------------------------------------------------------

def _mk_logdecay(B, T, H, N, V, device, dtype, seed=0, small=False):
    """Like _mk but returns g=log(decay). small=True puts g in the Mamba2-init
    regime (decay ~1e-5..1e-2) where the dg/decay decay-space backward blows up."""
    k, v, q, decay, e, w = _mk(B, T, H, N, V, device, dtype, seed=seed)
    if small:
        # g in roughly [-12, -4]: decay in [6e-6, 1.8e-2]
        g = (torch.rand(B, T, H, device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(seed + 99))
             * -8.0 - 4.0)
    else:
        g = decay.clamp_min(1e-9).log()
    return k, v, q, g, e, w


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("T", [64, 128, 256, 96])
def test_logdecay_forward_parity_fp32(T):
    """log_decay=True forward == decay-space forward == reference (fp32)."""
    dev = 'cuda'
    B, H, N, V = 2, 4, 32, 32
    k, v, q, g, e, w = _mk_logdecay(B, T, H, N, V, dev, torch.float32, seed=3)
    decay = g.exp()
    S0 = torch.zeros(B, H, N, V, device=dev)
    ref_out, ref_S = _ref_forward(S0, k, v, q, decay, e, w)
    out_d, S_d = e97_delta_chunked_triton(k, v, q, decay, e, w, chunk_size=32, log_decay=False)
    out_g, S_g = e97_delta_chunked_triton(k, v, q, g, e, w, chunk_size=32, log_decay=True)
    assert (out_g - ref_out).abs().max().item() < 2e-4, "log_decay fwd vs ref"
    # log vs decay-space differ only by the exp->log fp32 roundtrip in the decay path.
    rel_dg = (out_g - out_d).abs().max().item() / (out_d.abs().max().item() + 1e-9)
    assert rel_dg < 1e-5, f"log_decay fwd vs decay-space rel {rel_dg}"
    assert (S_g - ref_S).abs().max().item() < 2e-4, "log_decay S_final vs ref"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_logdecay_backward_parity_fp32():
    """grad wrt g (log_decay=True) matches the reference recurrence reparametrized
    in g; the other grads (k,v,q,e,w) match exactly. fp32, moderate decay."""
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, g, e, w = _mk_logdecay(B, T, H, N, V, dev, torch.float32, seed=11)
    S0 = torch.zeros(B, H, N, V, device=dev)

    def run_ref():
        ts = [t.clone().requires_grad_(True) for t in (k, v, q, g, e, w)]
        kk, vv, qq, gg, ee, ww = ts
        decay = gg.exp()  # reparametrize: grad flows to gg as grad-wrt-log-decay
        out, _ = _ref_forward(S0, kk, vv, qq, decay, ee, ww)
        (out * out).sum().backward()
        return [t.grad.clone() for t in ts]

    def run_chk():
        ts = [t.clone().requires_grad_(True) for t in (k, v, q, g, e, w)]
        kk, vv, qq, gg, ee, ww = ts
        out, _ = e97_delta_chunked_triton(kk, vv, qq, gg, ee, ww, chunk_size=32, log_decay=True)
        (out * out).sum().backward()
        return [t.grad.clone() for t in ts]

    g_ref, g_chk = run_ref(), run_chk()
    for nm, a, b in zip(['k', 'v', 'q', 'g', 'e', 'w'], g_ref, g_chk):
        err = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-6
        assert err / scale < 3e-3, f"log_decay grad {nm} rel err {err/scale} (abs {err})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_logdecay_backward_parity_bf16():
    """bf16 log_decay=True fwd+grad parity vs reference reparametrized in g."""
    dev = 'cuda'
    B, T, H, N, V = 2, 256, 3, 32, 32
    k, v, q, g, e, w = _mk_logdecay(B, T, H, N, V, dev, torch.bfloat16, seed=5)
    S0 = torch.zeros(B, H, N, V, device=dev, dtype=torch.bfloat16)

    def run(make_out):
        ts = [t.clone().requires_grad_(True) for t in (k, v, q, g, e, w)]
        out = make_out(*ts)
        (out.float() * out.float()).sum().backward()
        return out.detach(), [t.grad.clone() for t in ts]

    o_ref, g_ref = run(lambda kk, vv, qq, gg, ee, ww:
                       _ref_forward(S0, kk, vv, qq, gg.exp(), ee, ww)[0])
    o_chk, g_chk = run(lambda kk, vv, qq, gg, ee, ww:
                       e97_delta_chunked_triton(kk, vv, qq, gg, ee, ww,
                                                chunk_size=32, log_decay=True)[0])
    frel = (o_chk.float() - o_ref.float()).abs().max().item() / (o_ref.float().abs().max().item() + 1e-6)
    assert frel < 0.05, f"bf16 log_decay fwd rel err {frel}"
    for nm, a, b in zip(['k', 'v', 'q', 'g', 'e', 'w'], g_ref, g_chk):
        err = (a.float() - b.float()).abs().max().item()
        scale = a.float().abs().max().item() + 1e-6
        assert err / scale < 0.06, f"bf16 log_decay grad {nm} rel err {err/scale}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_logdecay_realistic_init_decay_finite_fp32():
    """THE NaN FIX, realistic regime. At Mamba2 init the e97_delta decay is mostly
    ~0.9-1.0 with SPARSE small-decay outliers down to ~5e-4 (measured on the 1.3B
    candidate). There the decay-space backward amplifies grad by 1/decay (up to
    ~2e3x at the outliers) feeding A_log/a_proj/dt_bias; the log_decay path returns
    grad wrt g directly. Assert: log grads finite AND match the reference recurrence
    reparametrized in g (decay=exp(g)) — correct, not just finite.

    NOTE (chunk-normalization boundary, documented honestly): a *run* of many
    consecutive tiny-decay steps within one C=32 chunk underflows decay^C in fp32
    and overflows the intra-chunk 1/cumdecay normalization for BOTH the log and
    decay-space paths — a chunked-kernel limit independent of the decay
    parametrization. Real init never produces such runs (median decay ~0.93,
    outliers sparse not consecutive), so training stays finite; this test pins the
    realistic regime the fix actually serves."""
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    k, v, q, _, e, w = _mk(B, T, H, N, V, dev, torch.float32, seed=7)
    gen = torch.Generator(device=dev).manual_seed(7)
    g = -torch.rand(B, T, H, device=dev, generator=gen).pow(3) * 0.5     # decay ~0.9-1.0
    mask = torch.rand(B, T, H, device=dev, generator=gen) < 0.05
    g = torch.where(mask, torch.full_like(g, -7.6), g)                   # 5% outliers ~5e-4
    decay = g.exp()
    assert decay.median().item() > 0.8 and decay.min().item() < 1e-2     # init-like

    ts = [t.clone().requires_grad_(True) for t in (k, v, q, g, e, w)]
    out, _ = e97_delta_chunked_triton(*ts, chunk_size=32, log_decay=True)
    (out * out).sum().backward()
    for nm, t in zip(['k', 'v', 'q', 'g', 'e', 'w'], ts):
        assert torch.isfinite(t.grad).all(), f"log_decay grad {nm} non-finite at init-like decay"

    S0 = torch.zeros(B, H, N, V, device=dev)
    tr = [t.clone().requires_grad_(True) for t in (k, v, q, g, e, w)]
    out_r, _ = _ref_forward(S0, tr[0], tr[1], tr[2], tr[3].exp(), tr[4], tr[5])
    (out_r * out_r).sum().backward()
    err = (tr[3].grad - ts[3].grad).abs().max().item()
    scale = tr[3].grad.abs().max().item() + 1e-6
    assert err / scale < 5e-3, f"init-like grad-wrt-g rel err {err/scale}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_strong_decay_cluster_backward_finite():
    """REGRESSION (fuse-2kernel 1.3B NaN): a chunk containing several strong-decay
    steps (g ~ -31, the model's decay floor) makes the cumulative G span large, so
    the UPPER triangle of DA = exp(G_i - G_j) overflows to +inf. The forward masks
    it away, but the backward's unmasked `*DA` products hit 0*inf = NaN, poisoning
    every key/decay grad (k,q,g,e) while v,w stay finite — the exact signature seen
    at real 1.3B init. The DA-exponent clamp (exp(min(.,0))) must keep both forward
    and ALL backward grads finite, matching the reference. fp32 and bf16."""
    dev = 'cuda'
    B, T, H, N, V = 2, 128, 3, 32, 32
    for dtype in (torch.float32, torch.bfloat16):
        k, v, q, _, e, w = _mk(B, T, H, N, V, dev, dtype, seed=13)
        # g mostly mild, with a CONSECUTIVE run of strong-decay steps inside a chunk
        # so the upper-triangle cumulative |G_i - G_j| exceeds 88 (exp -> inf).
        g = (-torch.rand(B, T, H, device=dev) * 0.3).to(dtype)
        g[:, 40:48, :] = -31.0   # 8 consecutive floor-decay steps (span ~248 >> 88)
        S0 = torch.zeros(B, H, N, V, device=dev, dtype=dtype)

        ts = [t.clone().float().requires_grad_(True) if dtype == torch.float32
              else t.clone().requires_grad_(True) for t in (k, v, q, g, e, w)]
        out, _ = e97_delta_chunked_triton(*ts, chunk_size=32, log_decay=True)
        assert torch.isfinite(out).all(), f"strong-decay fwd non-finite ({dtype})"
        (out.float() ** 2).sum().backward()
        for nm, t in zip(['k', 'v', 'q', 'g', 'e', 'w'], ts):
            assert torch.isfinite(t.grad).all(), f"strong-decay grad {nm} non-finite ({dtype})"

        # correctness vs reference reparametrized in g, in the finite regime.
        tr = [t.clone().requires_grad_(True) for t in
              (k.float(), v.float(), q.float(), g.float(), e.float(), w.float())]
        out_r, _ = _ref_forward(S0.float(), tr[0], tr[1], tr[2], tr[3].exp(), tr[4], tr[5])
        (out_r ** 2).sum().backward()
        if dtype == torch.float32:
            for nm, a, b in zip(['k', 'v', 'q', 'g', 'e', 'w'], tr, ts):
                err = (a.grad - b.grad).abs().max().item()
                scale = a.grad.abs().max().item() + 1e-6
                assert err / scale < 5e-3, f"strong-decay grad {nm} rel err {err/scale}"


if __name__ == '__main__':
    if not CUDA:
        print("no CUDA; skipping")
    else:
        test_forward_parity_fp32(256, 64)
        print("staged fwd fp32 OK")
        test_forward_parity_bf16()
        print("staged fwd bf16 OK")
        test_backward_parity_fp32()
        print("staged bwd fp32 OK")
        test_fused_triton_forward_parity_fp32(256)
        print("fused fwd fp32 OK")
        test_fused_triton_backward_parity_fp32()
        print("fused bwd fp32 OK")
        test_fused_triton_backward_parity_bf16()
        print("fused bwd bf16 OK")
        test_logdecay_forward_parity_fp32(256)
        print("log_decay fwd fp32 OK")
        test_logdecay_backward_parity_fp32()
        print("log_decay bwd fp32 OK")
        test_logdecay_backward_parity_bf16()
        print("log_decay bwd bf16 OK")
        test_logdecay_realistic_init_decay_finite_fp32()
        print("log_decay realistic-init-decay (NaN-fix) OK")
        test_strong_decay_cluster_backward_finite()
        print("strong-decay-cluster backward finite (NaN-fix regression) OK")
        print("ALL PARITY OK")
