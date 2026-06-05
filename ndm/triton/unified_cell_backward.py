"""Triton BACKWARD kernel + autograd wrapper for the unified-cell recurrence.

Pairs with ``ndm.triton.unified_cell_forward``. The forward (run with
``ckpt_interval=1``) saves the DENSE input-state history ``S_prev`` where
``S_prev[t]`` is the state fed INTO step t (``S_prev[0]=S0``, ``S_prev[t]=S_{t-1}``).
The backward then does an exact O(T) reverse scan, recomputing ``u_t``,
``pre_t``, ``S_t`` and ``phi'(pre_t)`` per step from ``S_prev[t]`` and that
step's inputs — no segmented recompute, no approximation.

Derived gradients (per b,h; S is [N,V]):
    pre_t[n,v] = lam_t S_{t-1}[n,v] + k_t[n]*(ig_t v_t[v] - beta_t u_t[v])
    S_t = phi(pre_t),   o_t[v] = sum_n S_t[n,v] q_t[n]

Reverse scan with G = dL/dS_t (carry):
    G += outer(q_t, dOut_t)              # readout contribution: G[n,v]+=q_t[n] dOut_t[v]
    dQ_t[n]  = sum_v dOut_t[v] S_t[n,v]
    dP_t     = G * phi'(pre_t)
    w_t[v]   = sum_n dP_t[n,v] k_t[n]
    dV_t[v]  = ig_t * w_t[v]
    dIG_t    = sum_v w_t[v] v_t[v]
    dBETA_t  = - sum_v w_t[v] u_t[v]
    dLAM_t   = sum_{n,v} dP_t[n,v] S_{t-1}[n,v]
    dK_t[j]  = sum_v dP_t[j,v]*(ig_t v_t[v] - beta_t u_t[v]) - beta_t sum_v w_t[v] S_{t-1}[j,v]
    dGAMMA  += sum_{n,v} G[n,v]*(tanh(pre_t)-pre_t)        # phi_mode==2 only
    G_prev   = lam_t * dP_t - beta_t * outer(k_t, w_t)     # carry to S_{t-1}
After t=0, G == dL/dS0.
"""
from __future__ import absolute_import

import torch
import triton
import triton.language as tl

from .unified_cell_forward import (
    unified_cell_forward, unified_cell_torch_reference,
    _next_pow2, PHI_IDENTITY, PHI_TANH, PHI_GAMMA_MIX, PHI_RELU, PHI_SOFTPLUS,
    PHI_NAME_TO_CODE,
)


@triton.jit
def _unified_backward_kernel(
    K_ptr, V_ptr, Q_ptr,
    LAM_ptr, BETA_ptr, IG_ptr, GAMMA_ptr,
    Sprev_ptr,            # [T+1, B, H, N, V]  Sprev[t]=state into step t
    dOut_ptr,             # [T, B, H, V]
    dSfinal_ptr,          # [B, H, N, V]
    # grad outputs
    dK_ptr, dV_ptr, dQ_ptr,
    dLAM_ptr, dBETA_ptr, dIG_ptr,
    dGAMMA_ptr,           # [B, H] partial (summed over B in wrapper)
    dS0_ptr,              # [B, H, N, V]
    # strides
    sk_t, sk_b, sk_h, sk_n,
    sv_t, sv_b, sv_h, sv_v,
    sq_t, sq_b, sq_h, sq_n,
    sl_t, sl_b, sl_h,
    sbe_t, sbe_b, sbe_h,
    sig_t, sig_b, sig_h,
    sp_t, sp_b, sp_h, sp_n, sp_v,
    sdo_t, sdo_b, sdo_h, sdo_v,
    sds_b, sds_h, sds_n, sds_v,
    sdk_t, sdk_b, sdk_h, sdk_n,
    sdv_t, sdv_b, sdv_h, sdv_v,
    sdq_t, sdq_b, sdq_h, sdq_n,
    sdl_t, sdl_b, sdl_h,
    sdbe_t, sdbe_b, sdbe_h,
    sdig_t, sdig_b, sdig_h,
    sdg_b, sdg_h,
    sds0_b, sds0_h, sds0_n, sds0_v,
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr,
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

    gamma = tl.load(GAMMA_ptr + h_idx, mask=h_mask, other=0.0).to(tl.float32)
    gamma_b = gamma[:, None, None]

    # G = dL/dS_T  (from dSfinal); add readout each step in reverse.
    ds_off = (b * sds_b + h_idx[:, None, None] * sds_h
              + n_idx[None, :, None] * sds_n + v_idx[None, None, :] * sds_v)
    G = tl.load(dSfinal_ptr + ds_off, mask=mask_hnv, other=0.0).to(tl.float32)

    dgamma_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for ti in range(T):
        t = T - 1 - ti
        t_i64 = tl.full([1], t, dtype=tl.int64)
        tp_i64 = tl.full([1], t, dtype=tl.int64)  # Sprev slot t = state into step t

        # loads
        k_off = t_i64 * sk_t + b * sk_b + h_idx[:, None] * sk_h + n_idx[None, :] * sk_n
        q_off = t_i64 * sq_t + b * sq_b + h_idx[:, None] * sq_h + n_idx[None, :] * sq_n
        v_off = t_i64 * sv_t + b * sv_b + h_idx[:, None] * sv_h + v_idx[None, :] * sv_v
        l_off = t_i64 * sl_t + b * sl_b + h_idx * sl_h
        be_off = t_i64 * sbe_t + b * sbe_b + h_idx * sbe_h
        ig_off = t_i64 * sig_t + b * sig_b + h_idx * sig_h
        sp_off = (tp_i64 * sp_t + b * sp_b + h_idx[:, None, None] * sp_h
                  + n_idx[None, :, None] * sp_n + v_idx[None, None, :] * sp_v)

        k_vec = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH,BN]
        q_vec = tl.load(Q_ptr + q_off, mask=mask_hn, other=0.0).to(tl.float32)
        v_vec = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)   # [BH,BV]
        lam = tl.load(LAM_ptr + l_off, mask=h_mask, other=0.0).to(tl.float32)
        beta = tl.load(BETA_ptr + be_off, mask=h_mask, other=0.0).to(tl.float32)
        igain = tl.load(IG_ptr + ig_off, mask=h_mask, other=0.0).to(tl.float32)
        Sprev = tl.load(Sprev_ptr + sp_off, mask=mask_hnv, other=0.0).to(tl.float32)  # S_{t-1}

        # recompute forward quantities
        u_vec = tl.sum(Sprev * k_vec[:, :, None], axis=1)                  # [BH,BV]
        corr = (igain[:, None] * v_vec) - (beta[:, None] * u_vec)         # [BH,BV]
        pre = lam[:, None, None] * Sprev + k_vec[:, :, None] * corr[:, None, :]
        # S_t = phi(pre) ; phi'(pre)
        if PHI_MODE == 0:
            S_t = pre
            dphi = tl.full(pre.shape, 1.0, tl.float32)
        elif PHI_MODE == 1:
            S_t = 2.0 * tl.sigmoid(2.0 * pre) - 1.0
            dphi = 1.0 - S_t * S_t
        elif PHI_MODE == 2:
            th = 2.0 * tl.sigmoid(2.0 * pre) - 1.0
            S_t = (1.0 - gamma_b) * pre + gamma_b * th
            dphi = (1.0 - gamma_b) + gamma_b * (1.0 - th * th)
        elif PHI_MODE == 3:
            S_t = tl.maximum(pre, 0.0)
            dphi = (pre > 0.0).to(tl.float32)
        else:
            sig = tl.sigmoid(pre)
            S_t = tl.where(pre > 20.0, pre, tl.log(1.0 + tl.exp(tl.minimum(pre, 20.0))))
            dphi = sig

        # readout grad contribution
        do_off = t_i64 * sdo_t + b * sdo_b + h_idx[:, None] * sdo_h + v_idx[None, :] * sdo_v
        dout = tl.load(dOut_ptr + do_off, mask=mask_hv, other=0.0).to(tl.float32)  # [BH,BV]
        # dQ_t[n] = sum_v dout[v] S_t[n,v]
        dq = tl.sum(S_t * dout[:, None, :], axis=2)                       # [BH,BN]
        dq_off = t_i64 * sdq_t + b * sdq_b + h_idx[:, None] * sdq_h + n_idx[None, :] * sdq_n
        tl.store(dQ_ptr + dq_off, dq.to(dQ_ptr.dtype.element_ty), mask=mask_hn)

        # G += outer(q, dout)
        G = G + q_vec[:, :, None] * dout[:, None, :]
        # dGAMMA accumulation (mode 2): G * (tanh(pre) - pre)
        if PHI_MODE == 2:
            th2 = 2.0 * tl.sigmoid(2.0 * pre) - 1.0
            dgamma_acc += tl.sum(tl.sum(G * (th2 - pre), axis=2), axis=1)

        dP = G * dphi                                                     # [BH,BN,BV]
        w = tl.sum(dP * k_vec[:, :, None], axis=1)                       # [BH,BV]

        # dV, dIG, dBETA, dLAM
        dv = igain[:, None] * w
        dv_off = t_i64 * sdv_t + b * sdv_b + h_idx[:, None] * sdv_h + v_idx[None, :] * sdv_v
        tl.store(dV_ptr + dv_off, dv.to(dV_ptr.dtype.element_ty), mask=mask_hv)

        dig = tl.sum(w * v_vec, axis=1)                                   # [BH]
        dbeta = -tl.sum(w * u_vec, axis=1)
        dlam = tl.sum(tl.sum(dP * Sprev, axis=2), axis=1)
        dl_off = t_i64 * sdl_t + b * sdl_b + h_idx * sdl_h
        dbe_off = t_i64 * sdbe_t + b * sdbe_b + h_idx * sdbe_h
        dig_off = t_i64 * sdig_t + b * sdig_b + h_idx * sdig_h
        tl.store(dLAM_ptr + dl_off, dlam.to(dLAM_ptr.dtype.element_ty), mask=h_mask)
        tl.store(dBETA_ptr + dbe_off, dbeta.to(dBETA_ptr.dtype.element_ty), mask=h_mask)
        tl.store(dIG_ptr + dig_off, dig.to(dIG_ptr.dtype.element_ty), mask=h_mask)

        # dK[j] = sum_v dP[j,v]*(ig v - beta u) - beta * sum_v w[v] Sprev[j,v]
        dk = tl.sum(dP * corr[:, None, :], axis=2) - beta[:, None] * tl.sum(Sprev * w[:, None, :], axis=2)
        dk_off = t_i64 * sdk_t + b * sdk_b + h_idx[:, None] * sdk_h + n_idx[None, :] * sdk_n
        tl.store(dK_ptr + dk_off, dk.to(dK_ptr.dtype.element_ty), mask=mask_hn)

        # carry: G_prev = lam*dP - beta * outer(k, w)
        G = lam[:, None, None] * dP - beta[:, None, None] * (k_vec[:, :, None] * w[:, None, :])

    # G now = dL/dS0
    ds0_off = (b * sds0_b + h_idx[:, None, None] * sds0_h
               + n_idx[None, :, None] * sds0_n + v_idx[None, None, :] * sds0_v)
    tl.store(dS0_ptr + ds0_off, G.to(dS0_ptr.dtype.element_ty), mask=mask_hnv)

    # dGAMMA partial [B,H]
    dg_off = b * sdg_b + h_idx * sdg_h
    tl.store(dGAMMA_ptr + dg_off, dgamma_acc.to(dGAMMA_ptr.dtype.element_ty), mask=h_mask)


def unified_cell_backward(
    k, v, q, lam, beta, igain, gamma, S_prev_dense,
    d_out, d_Sfinal, phi_mode, block_h=1, num_warps=2,
):
    """Compute grads. ``S_prev_dense`` is [T+1,B,H,N,V] (forward ckpt_interval=1)."""
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    BLOCK_N, BLOCK_V = _next_pow2(N), _next_pow2(Vsz)

    def _c(x):
        return x if x.stride(-1) == 1 else x.contiguous()
    k, v, q = _c(k), _c(v), _c(q)
    lam, beta, igain = _c(lam), _c(beta), _c(igain)
    sp = _c(S_prev_dense)
    d_out = _c(d_out)
    d_Sfinal = _c(d_Sfinal)
    gamma_c = gamma.contiguous().float()

    f = torch.float32
    dK = torch.empty((T, B, H, N), device=k.device, dtype=f)
    dV = torch.empty((T, B, H, Vsz), device=k.device, dtype=f)
    dQ = torch.empty((T, B, H, N), device=k.device, dtype=f)
    dLAM = torch.empty((T, B, H), device=k.device, dtype=f)
    dBETA = torch.empty((T, B, H), device=k.device, dtype=f)
    dIG = torch.empty((T, B, H), device=k.device, dtype=f)
    dGAMMA_partial = torch.empty((B, H), device=k.device, dtype=f)
    dS0 = torch.empty((B, H, N, Vsz), device=k.device, dtype=f)

    strides = (
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        lam.stride(0), lam.stride(1), lam.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        igain.stride(0), igain.stride(1), igain.stride(2),
        sp.stride(0), sp.stride(1), sp.stride(2), sp.stride(3), sp.stride(4),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        d_Sfinal.stride(0), d_Sfinal.stride(1), d_Sfinal.stride(2), d_Sfinal.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dLAM.stride(0), dLAM.stride(1), dLAM.stride(2),
        dBETA.stride(0), dBETA.stride(1), dBETA.stride(2),
        dIG.stride(0), dIG.stride(1), dIG.stride(2),
        dGAMMA_partial.stride(0), dGAMMA_partial.stride(1),
        dS0.stride(0), dS0.stride(1), dS0.stride(2), dS0.stride(3),
    )
    grid = (B, (H + block_h - 1) // block_h)
    _unified_backward_kernel[grid](
        k, v, q, lam, beta, igain, gamma_c, sp, d_out, d_Sfinal,
        dK, dV, dQ, dLAM, dBETA, dIG, dGAMMA_partial, dS0,
        *strides,
        T=T, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V, BLOCK_H=block_h,
        PHI_MODE=phi_mode, num_warps=num_warps,
    )
    dGAMMA = dGAMMA_partial.sum(dim=0)  # [H]
    return dK, dV, dQ, dLAM, dBETA, dIG, dGAMMA, dS0


class UnifiedCellFunction(torch.autograd.Function):
    """Autograd-wrapped unified cell: Triton fwd (dense ckpt) + Triton bwd."""

    @staticmethod
    def forward(ctx, k, v, q, lam, beta, igain, gamma, S0, phi_mode):
        out, S_final, S_prev_dense = unified_cell_forward(
            S0, k, v, q, lam, beta, igain, gamma,
            phi_mode=phi_mode, ckpt_interval=1,
        )
        ctx.save_for_backward(k, v, q, lam, beta, igain, gamma, S_prev_dense)
        ctx.phi_mode = phi_mode
        ctx.s0_dtype = S0.dtype
        return out, S_final

    @staticmethod
    def backward(ctx, d_out, d_Sfinal):
        k, v, q, lam, beta, igain, gamma, S_prev_dense = ctx.saved_tensors
        if d_Sfinal is None:
            d_Sfinal = torch.zeros_like(S_prev_dense[0])
        dK, dV, dQ, dLAM, dBETA, dIG, dGAMMA, dS0 = unified_cell_backward(
            k, v, q, lam, beta, igain, gamma, S_prev_dense,
            d_out.contiguous(), d_Sfinal.contiguous(), ctx.phi_mode,
        )
        # match input dtypes
        return (dK.to(k.dtype), dV.to(v.dtype), dQ.to(q.dtype),
                dLAM.to(lam.dtype), dBETA.to(beta.dtype), dIG.to(igain.dtype),
                dGAMMA.to(gamma.dtype), dS0.to(ctx.s0_dtype), None)


def unified_cell(k, v, q, lam, beta, igain, gamma, S0, phi_mode=PHI_TANH):
    """Differentiable unified-cell recurrence (Triton fwd+bwd).

    k,q: [T,B,H,N]  v: [T,B,H,V]  lam,beta,igain: [T,B,H]  gamma: [H]  S0: [B,H,N,V]
    Returns (out [T,B,H,V], S_final [B,H,N,V]).
    """
    return UnifiedCellFunction.apply(k, v, q, lam, beta, igain, gamma, S0, phi_mode)
