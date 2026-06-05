"""Triton FORWARD kernel for the UNIFIED parameterized matrix-recurrence cell.

ONE parameterized cell spans four recurrent capabilities by freeing the
self-loop gain ``lambda`` (gain/decay), the correction ``beta``, the input
gain ``i``, and the state nonlinearity ``phi`` (gamma).

Per head, with matrix state S of shape [N, V]:

    u_t   = k_t^T @ S_{t-1}                       # retrieve  [V]
    pre_t = lambda_t * S_{t-1}
            - beta_t * (k_t  outer  u_t)          # delta-correction along key
            + i_t    * (k_t  outer  v_t)          # input write
    S_t   = phi_gamma(pre_t)
    o_t   = S_t^T @ q_t                           # readout  [V]

where (a outer x)[n, v] = a[n] * x[v].

The linear operator on S along the N axis is  A_t = lambda_t I - beta_t k_t k_t^T,
so (with L2-normalized k) the along-key eigenvalue is ``lambda - beta`` and all
orthogonal directions decay/grow by ``lambda``. This single parameterization
recovers every capability corner (see the module docstring of
``ndm.models.unified_cell``):

    track   : lambda<1, beta>lambda, linear   (along-key eig < 0 -> reflection / S5)
    count   : lambda=1, beta=0,      linear   (pure integration; +mLSTM-norm)
    latch   : lambda>1, beta=0,      tanh     (bistable +/-1 attractors)
    nonlin  : lambda<1, beta mid,    tanh/relu (genuinely state-nonlinear phi)
    E88-base: lambda<1, beta=1,      tanh     (cribbed contractive == E88 recurrence)

Indeed E88's recurrence  S = tanh(decay*S + outer(k, v - S^T k))  is EXACTLY this
cell at  lambda=decay (clamped to (0,1)),  beta=1,  i=1,  phi=tanh.

phi modes (PHI_MODE constexpr):
    0 identity   phi(x) = x
    1 tanh       phi(x) = tanh(x)
    2 gamma_mix  phi(x) = (1-g)*x + g*tanh(x),  g = gamma[head] in [0,1]
    3 relu       phi(x) = max(x, 0)             (non-saturating)
    4 softplus   phi(x) = log(1+exp(x))         (non-saturating, smooth)

Layout (matches the E88 kernel convention; caller does the [B,T,..]->[T,B,..]
transpose):
    k, q     : [T, B, H, N]
    v        : [T, B, H, V]
    lam,beta,igain : [T, B, H]
    gamma    : [H]                   (per-head, used iff PHI_MODE==2)
    S0       : [B, H, N, V]
    out      : [T, B, H, V]
    S_final  : [B, H, N, V]
    S_ckpt   : [num_ckpts, B, H, N, V]   sparse, every CKPT_INTERVAL steps.

Sparse checkpointing mirrors the E88 kernel:
    S_ckpt[0] = S0,  S_ckpt[k>=1] = S after step (k*CKPT_INTERVAL - 1).
Requires T % CKPT_INTERVAL == 0. N, V <= 64.
"""
from __future__ import absolute_import

from typing import Tuple

import torch
import triton
import triton.language as tl


DEFAULT_CKPT_INTERVAL = 16

# phi mode codes (kept in sync with the PyTorch reference + backward kernel).
PHI_IDENTITY = 0
PHI_TANH = 1
PHI_GAMMA_MIX = 2
PHI_RELU = 3
PHI_SOFTPLUS = 4

PHI_NAME_TO_CODE = {
    'identity': PHI_IDENTITY, 'linear': PHI_IDENTITY,
    'tanh': PHI_TANH,
    'gamma_mix': PHI_GAMMA_MIX, 'gamma': PHI_GAMMA_MIX,
    'relu': PHI_RELU,
    'softplus': PHI_SOFTPLUS,
}


# ---------------------------------------------------------------------------
# Device-side phi (must match the PyTorch reference exactly).
# ---------------------------------------------------------------------------
@triton.jit
def _apply_phi(pre, phi_mode: tl.constexpr, gamma):
    """Elementwise state nonlinearity. ``gamma`` is a [BLOCK_H,1,1] tensor."""
    if phi_mode == 0:  # identity
        return pre
    elif phi_mode == 1:  # tanh  (stable via sigmoid)
        return 2.0 * tl.sigmoid(2.0 * pre) - 1.0
    elif phi_mode == 2:  # gamma_mix
        th = 2.0 * tl.sigmoid(2.0 * pre) - 1.0
        return (1.0 - gamma) * pre + gamma * th
    elif phi_mode == 3:  # relu
        return tl.maximum(pre, 0.0)
    else:  # softplus = log(1+exp(x)), numerically stable
        return tl.where(pre > 20.0, pre, tl.log(1.0 + tl.exp(tl.minimum(pre, 20.0))))


@triton.jit
def _unified_forward_kernel(
    K_ptr, V_ptr, Q_ptr,
    LAM_ptr, BETA_ptr, IG_ptr,   # [T, B, H]
    GAMMA_ptr,                   # [H]
    S0_ptr,
    Out_ptr, Sfinal_ptr, Sckpt_ptr,
    # strides
    sk_t, sk_b, sk_h, sk_n,
    sv_t, sv_b, sv_h, sv_v,
    sq_t, sq_b, sq_h, sq_n,
    sl_t, sl_b, sl_h,
    sbe_t, sbe_b, sbe_h,
    sig_t, sig_b, sig_h,
    s0_b, s0_h, s0_n, s0_v,
    so_t, so_b, so_h, so_v,
    sf_b, sf_h, sf_n, sf_v,
    sc_t, sc_b, sc_h, sc_n, sc_v,
    # sizes
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr,
    CKPT_INTERVAL: tl.constexpr,
    PHI_MODE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    hg = tl.program_id(1).to(tl.int64)

    h_idx = hg * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_idx < H
    n_idx = tl.arange(0, BLOCK_N)
    v_idx = tl.arange(0, BLOCK_V)
    n_mask = n_idx < N
    v_mask = v_idx < V

    mask_hnv = (h_mask[:, None, None] & n_mask[None, :, None] & v_mask[None, None, :])
    mask_hn = h_mask[:, None] & n_mask[None, :]
    mask_hv = h_mask[:, None] & v_mask[None, :]

    gamma = tl.load(GAMMA_ptr + h_idx, mask=h_mask, other=0.0).to(tl.float32)  # [BH]
    gamma_b = gamma[:, None, None]

    s0_off = (b * s0_b + h_idx[:, None, None] * s0_h
              + n_idx[None, :, None] * s0_n + v_idx[None, None, :] * s0_v)
    S = tl.load(S0_ptr + s0_off, mask=mask_hnv, other=0.0).to(tl.float32)

    sc0_off = (0 * sc_t + b * sc_b + h_idx[:, None, None] * sc_h
               + n_idx[None, :, None] * sc_n + v_idx[None, None, :] * sc_v)
    tl.store(Sckpt_ptr + sc0_off, S.to(Sckpt_ptr.dtype.element_ty), mask=mask_hnv)

    for t in range(T):
        t_i64 = tl.full([1], t, dtype=tl.int64)
        k_off = t_i64 * sk_t + b * sk_b + h_idx[:, None] * sk_h + n_idx[None, :] * sk_n
        q_off = t_i64 * sq_t + b * sq_b + h_idx[:, None] * sq_h + n_idx[None, :] * sq_n
        v_off = t_i64 * sv_t + b * sv_b + h_idx[:, None] * sv_h + v_idx[None, :] * sv_v
        l_off = t_i64 * sl_t + b * sl_b + h_idx * sl_h
        be_off = t_i64 * sbe_t + b * sbe_b + h_idx * sbe_h
        ig_off = t_i64 * sig_t + b * sig_b + h_idx * sig_h

        k_vec = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH,BN]
        q_vec = tl.load(Q_ptr + q_off, mask=mask_hn, other=0.0).to(tl.float32)
        v_vec = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)   # [BH,BV]
        lam = tl.load(LAM_ptr + l_off, mask=h_mask, other=0.0).to(tl.float32)    # [BH]
        beta = tl.load(BETA_ptr + be_off, mask=h_mask, other=0.0).to(tl.float32)
        igain = tl.load(IG_ptr + ig_off, mask=h_mask, other=0.0).to(tl.float32)

        # retrieve u = k^T S : [BH, BV]
        u_vec = tl.sum(S * k_vec[:, :, None], axis=1)
        # pre = lambda*S - beta * k outer u + i * k outer v
        corr = (igain[:, None] * v_vec) - (beta[:, None] * u_vec)            # [BH,BV]
        pre = lam[:, None, None] * S + k_vec[:, :, None] * corr[:, None, :]   # [BH,BN,BV]
        S = _apply_phi(pre, PHI_MODE, gamma_b)
        S = tl.where(mask_hnv, S, 0.0)

        # readout o = S^T q : [BH, BV]
        out_vec = tl.sum(S * q_vec[:, :, None], axis=1)
        out_off = t_i64 * so_t + b * so_b + h_idx[:, None] * so_h + v_idx[None, :] * so_v
        tl.store(Out_ptr + out_off, out_vec.to(Out_ptr.dtype.element_ty), mask=mask_hv)

        if ((t + 1) % CKPT_INTERVAL) == 0:
            slot_i64 = tl.full([1], (t + 1) // CKPT_INTERVAL, dtype=tl.int64)
            sc_off = (slot_i64 * sc_t + b * sc_b + h_idx[:, None, None] * sc_h
                      + n_idx[None, :, None] * sc_n + v_idx[None, None, :] * sc_v)
            tl.store(Sckpt_ptr + sc_off, S.to(Sckpt_ptr.dtype.element_ty), mask=mask_hnv)

    sf_off = (b * sf_b + h_idx[:, None, None] * sf_h
              + n_idx[None, :, None] * sf_n + v_idx[None, None, :] * sf_v)
    tl.store(Sfinal_ptr + sf_off, S.to(Sfinal_ptr.dtype.element_ty), mask=mask_hnv)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(p, 16)


def unified_cell_forward(
    S0: torch.Tensor,
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
    lam: torch.Tensor, beta: torch.Tensor, igain: torch.Tensor,
    gamma: torch.Tensor,
    phi_mode: int = PHI_TANH,
    ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
    block_h: int = 1,
    num_warps: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the unified-cell forward recurrence in Triton.

    Args:
        S0:    [B, H, N, V]
        k, q:  [T, B, H, N]
        v:     [T, B, H, V]
        lam, beta, igain: [T, B, H]  per-step per-head knobs
        gamma: [H]  per-head phi-mix coefficient (used iff phi_mode==2)
        phi_mode: state-nonlinearity code (see PHI_* constants)
    Returns:
        out      [T, B, H, V]
        S_final  [B, H, N, V]
        S_ckpt   [num_ckpts, B, H, N, V]   (sparse)
    """
    assert k.is_cuda
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    assert q.shape == (T, B, H, N)
    assert v.shape == (T, B, H, Vsz)
    for nm, tns in (('lam', lam), ('beta', beta), ('igain', igain)):
        assert tns.shape == (T, B, H), f"{nm} must be [T,B,H], got {tuple(tns.shape)}"
    assert gamma.shape == (H,), f"gamma must be [H], got {tuple(gamma.shape)}"
    assert S0.shape == (B, H, N, Vsz)

    BLOCK_N, BLOCK_V = _next_pow2(N), _next_pow2(Vsz)
    if BLOCK_N > 64 or BLOCK_V > 64:
        raise NotImplementedError(
            f"unified_cell_forward supports N,V<=64 (got N={N}, V={Vsz}).")
    if T % ckpt_interval != 0:
        raise NotImplementedError(
            f"requires T % ckpt_interval == 0 (T={T}, ckpt={ckpt_interval}).")

    def _c(x):
        return x if x.stride(-1) == 1 else x.contiguous()
    k_c, v_c, q_c = _c(k), _c(v), _c(q)
    lam_c, beta_c, ig_c = _c(lam), _c(beta), _c(igain)
    s0_c = _c(S0)
    gamma_c = gamma.contiguous().float()

    out_dtype = k_c.dtype
    out = torch.empty((T, B, H, Vsz), dtype=out_dtype, device=k.device)
    S_final = torch.empty_like(s0_c)
    num_ckpts = T // ckpt_interval + 1
    S_ckpt = torch.empty((num_ckpts, B, H, N, Vsz), dtype=out_dtype, device=k.device)

    strides = (
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        lam_c.stride(0), lam_c.stride(1), lam_c.stride(2),
        beta_c.stride(0), beta_c.stride(1), beta_c.stride(2),
        ig_c.stride(0), ig_c.stride(1), ig_c.stride(2),
        s0_c.stride(0), s0_c.stride(1), s0_c.stride(2), s0_c.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S_final.stride(0), S_final.stride(1), S_final.stride(2), S_final.stride(3),
        S_ckpt.stride(0), S_ckpt.stride(1), S_ckpt.stride(2), S_ckpt.stride(3), S_ckpt.stride(4),
    )
    grid = (B, (H + block_h - 1) // block_h)
    _unified_forward_kernel[grid](
        k_c, v_c, q_c, lam_c, beta_c, ig_c, gamma_c, s0_c,
        out, S_final, S_ckpt, *strides,
        T=T, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V, BLOCK_H=block_h,
        CKPT_INTERVAL=ckpt_interval, PHI_MODE=phi_mode,
        num_warps=num_warps,
    )
    return out, S_final, S_ckpt


# ---------------------------------------------------------------------------
# PyTorch reference (defines the exact recurrence; autograd gives gold grads).
# ---------------------------------------------------------------------------
def _phi_torch(pre: torch.Tensor, phi_mode: int, gamma: torch.Tensor) -> torch.Tensor:
    if phi_mode == PHI_IDENTITY:
        return pre
    if phi_mode == PHI_TANH:
        return torch.tanh(pre)
    if phi_mode == PHI_GAMMA_MIX:
        g = gamma.view(1, -1, 1, 1)  # [1,H,1,1] broadcast over [B,H,N,V]
        return (1.0 - g) * pre + g * torch.tanh(pre)
    if phi_mode == PHI_RELU:
        return torch.relu(pre)
    if phi_mode == PHI_SOFTPLUS:
        return torch.nn.functional.softplus(pre)
    raise ValueError(f"unknown phi_mode {phi_mode}")


def unified_cell_torch_reference(
    S0: torch.Tensor,
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
    lam: torch.Tensor, beta: torch.Tensor, igain: torch.Tensor,
    gamma: torch.Tensor,
    phi_mode: int = PHI_TANH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference. Differentiable (autograd) for gold gradients.

    Same [T,B,H,..] layout as the Triton wrapper. Returns out, S_final and the
    DENSE per-step checkpoint history (ckpt[0]=S0, ckpt[t+1]=S after step t).
    """
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    S = S0.to(torch.float32)
    outs = []
    ckpts = [S]
    for t in range(T):
        k_t = k[t].float()                       # [B,H,N]
        q_t = q[t].float()
        v_t = v[t].float()                       # [B,H,V]
        lam_t = lam[t].float().unsqueeze(-1).unsqueeze(-1)   # [B,H,1,1]
        beta_t = beta[t].float().unsqueeze(-1)               # [B,H,1]
        ig_t = igain[t].float().unsqueeze(-1)                # [B,H,1]

        u = torch.einsum('bhnv,bhn->bhv', S, k_t)            # [B,H,V]
        corr = ig_t * v_t - beta_t * u                       # [B,H,V]
        pre = lam_t * S + torch.einsum('bhn,bhv->bhnv', k_t, corr)
        S = _phi_torch(pre, phi_mode, gamma)
        outs.append(torch.einsum('bhnv,bhn->bhv', S, q_t))
        ckpts.append(S)
    out = torch.stack(outs, dim=0)
    return out, S, torch.stack(ckpts, dim=0)
