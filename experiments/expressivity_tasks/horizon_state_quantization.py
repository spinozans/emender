"""X21E eval-only state perturbations; no production model/kernel changes."""
from __future__ import annotations

from contextlib import contextmanager
import types

import torch
import triton
import triton.language as tl

PRECISIONS = ('native', 'wrapper-fp32', 'bf16', 'fp8', 'int8', 'int4')


def quantize_state(state, precision):
    """Reference codec; integer scale is per example/head, not per tile."""
    if precision == 'wrapper-fp32':
        return state
    if precision == 'bf16':
        return state.to(torch.bfloat16).float()
    if precision == 'fp8':
        return state.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    if precision not in ('int8', 'int4'):
        raise ValueError(precision)
    qmax = 127 if precision == 'int8' else 7
    maximum = state.abs().amax((-2, -1), keepdim=True)
    scale = torch.where(maximum == 0, 1., maximum / qmax)
    return (state / scale).round().clamp(-qmax, qmax) * scale


@triton.jit
def _codec(s, MODE: tl.constexpr):
    if MODE == 1:
        s = s.to(tl.bfloat16).to(tl.float32)
    elif MODE == 2:
        s = tl.minimum(tl.maximum(s, -448.), 448.).to(tl.float8e4nv).to(tl.float32)
    elif MODE == 3 or MODE == 4:
        qmax: tl.constexpr = 127 if MODE == 3 else 7
        maximum = tl.max(tl.max(tl.abs(s), 0), 0)
        scale = tl.where(maximum == 0., 1., maximum / qmax)
        s = tl.extra.cuda.libdevice.nearbyint(s / scale) * scale
    return s


@triton.jit
def _e97_scan(K, V, Q, D, G, E, W, S0, OUT, FINAL,
              T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
              N: tl.constexpr, NV: tl.constexpr, MODE: tl.constexpr,
              NORMALIZE: tl.constexpr, SILU: tl.constexpr, GATE: tl.constexpr,
              BN: tl.constexpr, BV: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    n, v = tl.arange(0, BN), tl.arange(0, BV)
    mask = (n[:, None] < N) & (v[None, :] < NV)
    off = ((b * H + h) * N + n[:, None]) * NV + v[None, :]
    s = tl.load(S0 + off, mask, 0).to(tl.float32)
    for t in range(T):
        kn = ((b * T + t) * H + h) * N + n
        vv = ((b * T + t) * H + h) * NV + v
        k = tl.load(K + kn, n < N, 0).to(tl.float32)
        q = tl.load(Q + kn, n < N, 0).to(tl.float32)
        val = tl.load(V + vv, v < NV, 0).to(tl.float32)
        if SILU:
            k = k / (1. + tl.exp(-k))
            q = q / (1. + tl.exp(-q))
            val = val / (1. + tl.exp(-val))
        if NORMALIZE:
            k = k / (tl.sqrt(tl.sum(k * k, 0)) + 1.e-6)
            q = q / (tl.sqrt(tl.sum(q * q, 0)) + 1.e-6)
        e = tl.load(E + kn, n < N, 0).to(tl.float32)
        w = tl.load(W + vv, v < NV, 0).to(tl.float32)
        d = tl.load(D + (b * T + t) * H + h).to(tl.float32)
        retrieved = tl.sum(s * (k * e)[:, None], 0)
        delta = val * w - retrieved
        pre = d * s + k[:, None] * delta[None, :]
        s = 2. * tl.sigmoid(2. * pre) - 1.
        s = _codec(s, MODE)
        out = tl.sum(s * q[:, None], 0)
        if GATE:
            g = tl.load(G + vv, v < NV, 0).to(tl.float32)
            out = out * (g / (1. + tl.exp(-g)))
        tl.store(OUT + vv, out, v < NV)
    tl.store(FINAL + off, s, mask)


def mode(precision):
    return {'wrapper-fp32': 0, 'bf16': 1, 'fp8': 2, 'int8': 3, 'int4': 4}[precision]


def e97_recurrence(precision, training, k, v, q, decay, g=None, S0=None,
                   n_heads=None, apply_gate=True, normalize_kq=False,
                   checkpoint_interval=16, apply_silu_qkv=False,
                   raw_write=False, linear_state=False, **kwargs):
    if training or torch.is_grad_enabled() or raw_write or linear_state:
        raise ValueError('X21E quantization supports eval-only tanh split-edit')
    if kwargs.get('reset_before') is not None or kwargs.get('valid_mask') is not None:
        raise ValueError('packed sequences not supported')
    b, t, h, n = k.shape
    nv = v.shape[-1]
    k, v, q, decay, e, w = [a.contiguous() for a in
        (k, v, q, decay, kwargs['erase_gate'], kwargs['value_write_gate'])]
    if S0 is None:
        S0 = torch.zeros((b, h, n, nv), device=k.device, dtype=torch.float32)
    final = torch.empty_like(S0)
    out = torch.empty_like(v)
    _e97_scan[(b, h)](k, v, q, decay, g.contiguous() if g is not None else v,
        e, w, S0.contiguous(), out, final, t, b, h, n, nv, mode(precision),
        normalize_kq, apply_silu_qkv, apply_gate,
        triton.next_power_of_2(n), triton.next_power_of_2(nv), num_warps=4)
    return final, out


@triton.jit
def _gdn_step(K, V, Q, D, BETA, STATE, UPDATED, OUT, SCALE,
              H: tl.constexpr, N: tl.constexpr, NV: tl.constexpr,
              T: tl.constexpr, TIME, QSCALE: tl.constexpr,
              MODE: tl.constexpr, READ_ONLY: tl.constexpr,
              BN: tl.constexpr, BV: tl.constexpr):
    bh, tile = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    n, v = tl.arange(0, BN), tile * BV + tl.arange(0, BV)
    mask = (n[:, None] < N) & (v[None, :] < NV)
    off = (bh * N + n[:, None]) * NV + v[None, :]
    qoff = ((b * T + TIME) * H + h) * N + n
    q = tl.load(Q + qoff, n < N, 0).to(tl.float32)
    if READ_ONLY:
        s = tl.load(UPDATED + off, mask, 0)
        scale = tl.load(SCALE + bh)
        qmax: tl.constexpr = 127 if MODE == 3 else 7
        s = tl.extra.cuda.libdevice.nearbyint(s / scale) * scale
    else:
        s = tl.load(STATE + off, mask, 0)
        k = tl.load(K + qoff, n < N, 0).to(tl.float32)
        val = tl.load(V + ((b * T + TIME) * H + h) * NV + v, v < NV, 0).to(tl.float32)
        decay = tl.exp(tl.load(D + (b * T + TIME) * H + h))
        beta = tl.load(BETA + (b * T + TIME) * H + h)
        s = s * decay
        delta = (val - tl.sum(s * k[:, None], 0)) * beta
        s = s + k[:, None] * delta[None, :]
        if MODE == 1:
            s = s.to(tl.bfloat16).to(tl.float32)
        elif MODE == 2:
            s = tl.minimum(tl.maximum(s, -448.), 448.).to(tl.float8e4nv).to(tl.float32)
    if not READ_ONLY and (MODE == 3 or MODE == 4):
        tl.store(UPDATED + off, s, mask)
    else:
        tl.store(STATE + off, s, mask)
        out = tl.sum(s * (q * QSCALE)[:, None], 0)
        tl.store(OUT + ((b * T + TIME) * H + h) * NV + v, out, v < NV)


def gdn_recurrence(precision, q, k, v, g, beta, scale=None, initial_state=None,
                   output_final_state=False, use_qk_l2norm_in_kernel=False,
                   use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                   state_v_first=False, cu_seqlens=None, use_gate_in_kernel=False,
                   A_log=None, dt_bias=None, **kwargs):
    if torch.is_grad_enabled() or cu_seqlens is not None or kwargs:
        raise ValueError('X21E GDN wrapper requires independent eval-only sequences')
    import torch.nn.functional as F
    dtype = q.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, g, beta = [a.float() for a in (q, k, v, g, beta)]
        if use_qk_l2norm_in_kernel:
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.e-6)
        if use_gate_in_kernel:
            g = -A_log.float().exp() * F.softplus(g + dt_bias.float())
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid() * (2. if allow_neg_eigval else 1.)
        elif allow_neg_eigval:
            raise ValueError('negative eigenvalues require beta sigmoid')
        b, t, h, n = k.shape
        nv = v.shape[-1]
        state = torch.zeros((b, h, n, nv), device=q.device)
        if initial_state is not None:
            state.copy_(initial_state.transpose(-1, -2) if state_v_first else initial_state)
        updated = torch.empty_like(state)
        out = torch.empty_like(v)
        scales = torch.ones((b, h), device=q.device)
        args = (q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous())
        q, k, v, g, beta = args
        grid = (b * h, triton.cdiv(nv, 32))
        m = mode(precision)
        for time in range(t):
            _gdn_step[grid](k, v, q, g, beta, state, updated, out, scales,
                h, n, nv, t, time, n ** -.5 if scale is None else scale,
                m, False, triton.next_power_of_2(n), 32, num_warps=4)
            if m in (3, 4):
                maximum = updated.abs().amax((-2, -1))
                scales = torch.where(maximum == 0, 1., maximum / (127 if m == 3 else 7))
                _gdn_step[grid](k, v, q, g, beta, state, updated, out, scales,
                    h, n, nv, t, time, n ** -.5 if scale is None else scale,
                    m, True, triton.next_power_of_2(n), 32, num_warps=4)
        final = state.transpose(-1, -2).contiguous() if state_v_first else state
        return out.to(dtype), final if output_final_state else None


@contextmanager
def state_quantization(model, architecture, precision):
    """Restore every patched callable even on an infrastructure failure."""
    if precision == 'native':
        yield
        return
    if architecture == 'e97':
        from ndm.triton import e97_sequential
        original = e97_sequential.e97_split_edit_triton_apply
        def replacement(*args, **kwargs):
            return e97_recurrence(precision, *args, **kwargs)
        e97_sequential.e97_split_edit_triton_apply = replacement
        try:
            yield
        finally:
            e97_sequential.e97_split_edit_triton_apply = original
    else:
        originals = []
        try:
            for layer in model.layers:
                gdn = layer.gdn
                original = gdn.forward
                originals.append((gdn, original))
                fn = original.__func__
                namespace = dict(fn.__globals__)
                def replacement(*args, **kwargs):
                    return gdn_recurrence(precision, *args, **kwargs)
                for name in ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'):
                    if name not in namespace:
                        raise ValueError('unsupported GDN forward: ' + name)
                    namespace[name] = replacement
                gdn.forward = types.MethodType(types.FunctionType(fn.__code__, namespace,
                    fn.__name__, fn.__defaults__, fn.__closure__), gdn)
            yield
        finally:
            for gdn, original in originals:
                gdn.forward = original
