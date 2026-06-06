"""Triton BACKWARD kernel + autograd wrapper for the unified-cell recurrence.

Pairs with ``ndm.triton.unified_cell_forward``. The forward (run with
``ckpt_interval=1``) saves the DENSE input-state history ``S_prev`` where
``S_prev[t]`` is the state fed INTO step t (``S_prev[0]=S0``, ``S_prev[t]=S_{t-1}``).
The backward then does an exact O(T) reverse scan, recomputing ``u_t``,
``pre_t``, ``S_t`` and ``phi'(pre_t)`` per step from ``S_prev[t]`` and that
step's inputs — no segmented recompute, no approximation.

Derived gradients (per b,h; S is [N,V]).  E97 split-gate: read/erase key
bk_t = b_t*k_t, write value wv_t = w_t*v_t, write (outer) key the ungated k_t:
    u_t[v]     = sum_n bk_t[n] S_{t-1}[n,v]               # (b*k)^T S
    pre_t[n,v] = lam_t S_{t-1}[n,v] + k_t[n]*(ig_t wv_t[v] - beta_t u_t[v])
    S_t = phi(pre_t),   o_t[v] = sum_n S_t[n,v] q_t[n]

Reverse scan with G = dL/dS_t (carry):
    G += outer(q_t, dOut_t)              # readout contribution: G[n,v]+=q_t[n] dOut_t[v]
    dQ_t[n]  = sum_v dOut_t[v] S_t[n,v]
    dP_t     = G * phi'(pre_t)
    pk_t[v]  = sum_n dP_t[n,v] k_t[n]                     # project onto WRITE key k
    skp_t[n] = sum_v pk_t[v] S_{t-1}[n,v]
    dV_t[v]  = ig_t * pk_t[v] * w_t[v]
    dW_t[v]  = ig_t * pk_t[v] * v_t[v]                    # value-gate grad
    dIG_t    = sum_v pk_t[v] wv_t[v]
    dBETA_t  = - sum_v pk_t[v] u_t[v]
    dLAM_t   = sum_{n,v} dP_t[n,v] S_{t-1}[n,v]
    dK_t[j]  = sum_v dP_t[j,v]*(ig_t wv_t[v] - beta_t u_t[v]) - beta_t b_t[j] skp_t[j]
    dB_t[j]  = - beta_t k_t[j] skp_t[j]                   # erase-gate grad
    dGAMMA  += sum_{n,v} G[n,v]*(tanh(pre_t)-pre_t)        # phi_mode==2 only
    G_prev   = lam_t * dP_t - beta_t * outer(bk_t, pk_t)  # carry to S_{t-1}
After t=0, G == dL/dS0.  b_t=w_t=1 recovers the E88-based unified gradients.
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
    BG_ptr, WG_ptr,       # E97 split gates: b erase [T,B,H,N], w value [T,B,H,V]
    LAM_ptr, BETA_ptr, IG_ptr, GAMMA_ptr,
    Sprev_ptr,            # [T+1, B, H, N, V]  Sprev[t]=state into step t
    dOut_ptr,             # [T, B, H, V]
    dSfinal_ptr,          # [B, H, N, V]
    # grad outputs
    dK_ptr, dV_ptr, dQ_ptr,
    dBG_ptr, dWG_ptr,     # split-gate grads: [T,B,H,N], [T,B,H,V]
    dLAM_ptr, dBETA_ptr, dIG_ptr,
    dGAMMA_ptr,           # [B, H] partial (summed over B in wrapper)
    dS0_ptr,              # [B, H, N, V]
    # strides
    sk_t, sk_b, sk_h, sk_n,
    sv_t, sv_b, sv_h, sv_v,
    sq_t, sq_b, sq_h, sq_n,
    sbg_t, sbg_b, sbg_h, sbg_n,
    swg_t, swg_b, swg_h, swg_v,
    sl_t, sl_b, sl_h,
    sbe_t, sbe_b, sbe_h,
    sig_t, sig_b, sig_h,
    sp_t, sp_b, sp_h, sp_n, sp_v,
    sdo_t, sdo_b, sdo_h, sdo_v,
    sds_b, sds_h, sds_n, sds_v,
    sdk_t, sdk_b, sdk_h, sdk_n,
    sdv_t, sdv_b, sdv_h, sdv_v,
    sdq_t, sdq_b, sdq_h, sdq_n,
    sdbg_t, sdbg_b, sdbg_h, sdbg_n,
    sdwg_t, sdwg_b, sdwg_h, sdwg_v,
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
        bg_off = t_i64 * sbg_t + b * sbg_b + h_idx[:, None] * sbg_h + n_idx[None, :] * sbg_n
        wg_off = t_i64 * swg_t + b * swg_b + h_idx[:, None] * swg_h + v_idx[None, :] * swg_v
        l_off = t_i64 * sl_t + b * sl_b + h_idx * sl_h
        be_off = t_i64 * sbe_t + b * sbe_b + h_idx * sbe_h
        ig_off = t_i64 * sig_t + b * sig_b + h_idx * sig_h
        sp_off = (tp_i64 * sp_t + b * sp_b + h_idx[:, None, None] * sp_h
                  + n_idx[None, :, None] * sp_n + v_idx[None, None, :] * sp_v)

        k_vec = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH,BN]
        q_vec = tl.load(Q_ptr + q_off, mask=mask_hn, other=0.0).to(tl.float32)
        v_vec = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)   # [BH,BV]
        bg_vec = tl.load(BG_ptr + bg_off, mask=mask_hn, other=0.0).to(tl.float32)  # erase b [BH,BN]
        wg_vec = tl.load(WG_ptr + wg_off, mask=mask_hv, other=0.0).to(tl.float32)  # value  w [BH,BV]
        lam = tl.load(LAM_ptr + l_off, mask=h_mask, other=0.0).to(tl.float32)
        beta = tl.load(BETA_ptr + be_off, mask=h_mask, other=0.0).to(tl.float32)
        igain = tl.load(IG_ptr + ig_off, mask=h_mask, other=0.0).to(tl.float32)
        Sprev = tl.load(Sprev_ptr + sp_off, mask=mask_hnv, other=0.0).to(tl.float32)  # S_{t-1}

        # recompute forward quantities (E97 split-gate)
        bk_vec = bg_vec * k_vec                                            # [BH,BN] read/erase key
        wv_vec = wg_vec * v_vec                                            # [BH,BV] write value
        u_vec = tl.sum(Sprev * bk_vec[:, :, None], axis=1)                # [BH,BV] = (b*k)^T S
        corr = (igain[:, None] * wv_vec) - (beta[:, None] * u_vec)        # [BH,BV]
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
        pk = tl.sum(dP * k_vec[:, :, None], axis=1)                       # [BH,BV] project on WRITE key k
        # skp[j] = sum_v pk[v] Sprev[j,v]  -> shared by dK (erase path) and dB
        skp = tl.sum(Sprev * pk[:, None, :], axis=2)                     # [BH,BN]

        # dV[v] = ig * pk[v] * w[v] ;  dW[v] = ig * pk[v] * v[v]
        dv = igain[:, None] * pk * wg_vec
        dv_off = t_i64 * sdv_t + b * sdv_b + h_idx[:, None] * sdv_h + v_idx[None, :] * sdv_v
        tl.store(dV_ptr + dv_off, dv.to(dV_ptr.dtype.element_ty), mask=mask_hv)
        dwg = igain[:, None] * pk * v_vec
        dwg_off = t_i64 * sdwg_t + b * sdwg_b + h_idx[:, None] * sdwg_h + v_idx[None, :] * sdwg_v
        tl.store(dWG_ptr + dwg_off, dwg.to(dWG_ptr.dtype.element_ty), mask=mask_hv)

        # dIG = sum_v pk[v] wv[v] ;  dBETA = -sum_v pk[v] u[v] ;  dLAM = sum dP*Sprev
        dig = tl.sum(pk * wv_vec, axis=1)                                 # [BH]
        dbeta = -tl.sum(pk * u_vec, axis=1)
        dlam = tl.sum(tl.sum(dP * Sprev, axis=2), axis=1)
        dl_off = t_i64 * sdl_t + b * sdl_b + h_idx * sdl_h
        dbe_off = t_i64 * sdbe_t + b * sdbe_b + h_idx * sdbe_h
        dig_off = t_i64 * sdig_t + b * sdig_b + h_idx * sdig_h
        tl.store(dLAM_ptr + dl_off, dlam.to(dLAM_ptr.dtype.element_ty), mask=h_mask)
        tl.store(dBETA_ptr + dbe_off, dbeta.to(dBETA_ptr.dtype.element_ty), mask=h_mask)
        tl.store(dIG_ptr + dig_off, dig.to(dIG_ptr.dtype.element_ty), mask=h_mask)

        # dK[j] = sum_v dP[j,v]*corr[v] - beta * b[j] * skp[j]
        dk = tl.sum(dP * corr[:, None, :], axis=2) - beta[:, None] * bg_vec * skp
        dk_off = t_i64 * sdk_t + b * sdk_b + h_idx[:, None] * sdk_h + n_idx[None, :] * sdk_n
        tl.store(dK_ptr + dk_off, dk.to(dK_ptr.dtype.element_ty), mask=mask_hn)
        # dB[j] = -beta * k[j] * skp[j]   (erase-gate grad)
        dbg = -beta[:, None] * k_vec * skp
        dbg_off = t_i64 * sdbg_t + b * sdbg_b + h_idx[:, None] * sdbg_h + n_idx[None, :] * sdbg_n
        tl.store(dBG_ptr + dbg_off, dbg.to(dBG_ptr.dtype.element_ty), mask=mask_hn)

        # carry: G_prev = lam*dP - beta * outer(b*k, pk)
        G = lam[:, None, None] * dP - beta[:, None, None] * (bk_vec[:, :, None] * pk[:, None, :])

    # G now = dL/dS0
    ds0_off = (b * sds0_b + h_idx[:, None, None] * sds0_h
               + n_idx[None, :, None] * sds0_n + v_idx[None, None, :] * sds0_v)
    tl.store(dS0_ptr + ds0_off, G.to(dS0_ptr.dtype.element_ty), mask=mask_hnv)

    # dGAMMA partial [B,H]
    dg_off = b * sdg_b + h_idx * sdg_h
    tl.store(dGAMMA_ptr + dg_off, dgamma_acc.to(dGAMMA_ptr.dtype.element_ty), mask=h_mask)


def unified_cell_backward(
    k, v, q, b_gate, w_gate, lam, beta, igain, gamma, S_prev_dense,
    d_out, d_Sfinal, phi_mode, block_h=1, num_warps=2,
):
    """Compute grads. ``S_prev_dense`` is [T+1,B,H,N,V] (forward ckpt_interval=1).

    ``b_gate`` [T,B,H,N] and ``w_gate`` [T,B,H,V] are the E97 split gates (all-ones
    recovers the E88-based unified recurrence). Returns dB/dW alongside the rest.
    """
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    BLOCK_N, BLOCK_V = _next_pow2(N), _next_pow2(Vsz)

    def _c(x):
        return x if x.stride(-1) == 1 else x.contiguous()
    k, v, q = _c(k), _c(v), _c(q)
    bg, wg = _c(b_gate), _c(w_gate)
    lam, beta, igain = _c(lam), _c(beta), _c(igain)
    sp = _c(S_prev_dense)
    d_out = _c(d_out)
    d_Sfinal = _c(d_Sfinal)
    gamma_c = gamma.contiguous().float()

    f = torch.float32
    dK = torch.empty((T, B, H, N), device=k.device, dtype=f)
    dV = torch.empty((T, B, H, Vsz), device=k.device, dtype=f)
    dQ = torch.empty((T, B, H, N), device=k.device, dtype=f)
    dBG = torch.empty((T, B, H, N), device=k.device, dtype=f)
    dWG = torch.empty((T, B, H, Vsz), device=k.device, dtype=f)
    dLAM = torch.empty((T, B, H), device=k.device, dtype=f)
    dBETA = torch.empty((T, B, H), device=k.device, dtype=f)
    dIG = torch.empty((T, B, H), device=k.device, dtype=f)
    dGAMMA_partial = torch.empty((B, H), device=k.device, dtype=f)
    dS0 = torch.empty((B, H, N, Vsz), device=k.device, dtype=f)

    strides = (
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        bg.stride(0), bg.stride(1), bg.stride(2), bg.stride(3),
        wg.stride(0), wg.stride(1), wg.stride(2), wg.stride(3),
        lam.stride(0), lam.stride(1), lam.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        igain.stride(0), igain.stride(1), igain.stride(2),
        sp.stride(0), sp.stride(1), sp.stride(2), sp.stride(3), sp.stride(4),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        d_Sfinal.stride(0), d_Sfinal.stride(1), d_Sfinal.stride(2), d_Sfinal.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dBG.stride(0), dBG.stride(1), dBG.stride(2), dBG.stride(3),
        dWG.stride(0), dWG.stride(1), dWG.stride(2), dWG.stride(3),
        dLAM.stride(0), dLAM.stride(1), dLAM.stride(2),
        dBETA.stride(0), dBETA.stride(1), dBETA.stride(2),
        dIG.stride(0), dIG.stride(1), dIG.stride(2),
        dGAMMA_partial.stride(0), dGAMMA_partial.stride(1),
        dS0.stride(0), dS0.stride(1), dS0.stride(2), dS0.stride(3),
    )
    grid = (B, (H + block_h - 1) // block_h)
    _unified_backward_kernel[grid](
        k, v, q, bg, wg, lam, beta, igain, gamma_c, sp, d_out, d_Sfinal,
        dK, dV, dQ, dBG, dWG, dLAM, dBETA, dIG, dGAMMA_partial, dS0,
        *strides,
        T=T, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V, BLOCK_H=block_h,
        PHI_MODE=phi_mode, num_warps=num_warps,
    )
    dGAMMA = dGAMMA_partial.sum(dim=0)  # [H]
    return dK, dV, dQ, dBG, dWG, dLAM, dBETA, dIG, dGAMMA, dS0


class UnifiedCellFunction(torch.autograd.Function):
    """Autograd-wrapped unified cell: Triton fwd (dense ckpt) + Triton bwd.

    ``b_gate``/``w_gate`` are the E97 split gates [T,B,H,N]/[T,B,H,V]. The wrapper
    (``unified_cell``) defaults them to all-ones when omitted, recovering the
    E88-based unified recurrence exactly.
    """

    @staticmethod
    def forward(ctx, k, v, q, b_gate, w_gate, lam, beta, igain, gamma, S0, phi_mode):
        out, S_final, S_prev_dense = unified_cell_forward(
            S0, k, v, q, lam, beta, igain, gamma,
            phi_mode=phi_mode, b_gate=b_gate, w_gate=w_gate, ckpt_interval=1,
        )
        ctx.save_for_backward(k, v, q, b_gate, w_gate, lam, beta, igain, gamma, S_prev_dense)
        ctx.phi_mode = phi_mode
        ctx.s0_dtype = S0.dtype
        return out, S_final

    @staticmethod
    def backward(ctx, d_out, d_Sfinal):
        k, v, q, b_gate, w_gate, lam, beta, igain, gamma, S_prev_dense = ctx.saved_tensors
        if d_Sfinal is None:
            d_Sfinal = torch.zeros_like(S_prev_dense[0])
        dK, dV, dQ, dBG, dWG, dLAM, dBETA, dIG, dGAMMA, dS0 = unified_cell_backward(
            k, v, q, b_gate, w_gate, lam, beta, igain, gamma, S_prev_dense,
            d_out.contiguous(), d_Sfinal.contiguous(), ctx.phi_mode,
        )
        # match input dtypes
        return (dK.to(k.dtype), dV.to(v.dtype), dQ.to(q.dtype),
                dBG.to(b_gate.dtype), dWG.to(w_gate.dtype),
                dLAM.to(lam.dtype), dBETA.to(beta.dtype), dIG.to(igain.dtype),
                dGAMMA.to(gamma.dtype), dS0.to(ctx.s0_dtype), None)


def unified_cell(k, v, q, lam, beta, igain, gamma, S0, phi_mode=PHI_TANH,
                 b_gate=None, w_gate=None):
    """Differentiable unified-cell recurrence (Triton fwd+bwd).

    k,q: [T,B,H,N]  v: [T,B,H,V]  lam,beta,igain: [T,B,H]  gamma: [H]  S0: [B,H,N,V]
    b_gate: [T,B,H,N] E97 erase gate (read along b*k); None == all-ones.
    w_gate: [T,B,H,V] E97 value-write gate (write w*v); None == all-ones.
    Returns (out [T,B,H,V], S_final [B,H,N,V]).
    """
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    if b_gate is None:
        b_gate = torch.ones((T, B, H, N), dtype=k.dtype, device=k.device)
    if w_gate is None:
        w_gate = torch.ones((T, B, H, Vsz), dtype=v.dtype, device=v.device)
    return UnifiedCellFunction.apply(k, v, q, b_gate, w_gate, lam, beta, igain, gamma, S0, phi_mode)
