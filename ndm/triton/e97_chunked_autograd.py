"""Fused chunked-parallel E97 split-edit delta — autograd.Function (fwd + bwd).

This is the throughput-flip artifact. The forward fuses the chunked delta scan
into one Triton program per (batch, head) (see ``e97_chunked_fwd_kernel`` for the
derivation); the backward is a SECOND fused Triton kernel that walks the chunks in
REVERSE, threading the state-gradient ``dS`` across chunks in registers and
recomputing each chunk's forward intermediates (the WY transform ``Tmat`` via
Newton-Schulz, the score matrices, the per-step ``Delta``) so the only thing
saved between fwd and bwd is the small per-chunk entry state. Every heavy step is
a ``tl.dot`` on tensor cores; the ``[C,C]`` intermediates never touch HBM. This is
the same fwd+bwd structure FLA's chunk_gated_delta_rule uses to hit ~97% util, so
the E97 split-edit cell can train at GDN-2-class throughput.

Recurrence (per b,h; linear state S [N,V]):
    read_key_t  = e_t (.) k_t ;  write_val_t = w_t (.) v_t
    delta_t     = write_val_t - S_{t-1}^T read_key_t
    S_t         = decay_t * S_{t-1} + k_t delta_t^T
    out_t       = S_t^T q_t
matching ``e88_torch_reference(linear_state=True, raw_write=False, split-edit)``.

Scope: N, V <= 64; linear state; S0 = 0. bf16 inputs run the TF32 tensor-core path
(``allow_tf32``); fp32 inputs run exact. Parity (fwd + grads, fp32 and bf16) is
checked in ``tests/test_e97_chunked.py`` against the reference recurrence.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(16, p)


# ---------------------------------------------------------------------------
# Forward kernel — same math as e97_chunked_fwd_kernel but also writes the
# per-chunk ENTRY state S_entry[c] (needed, and only needed, by the backward).
# ---------------------------------------------------------------------------
@triton.jit
def _e97_fwd_save_kernel(
    K_ptr, Q_ptr, U_ptr, P_ptr, G_ptr,   # [B,T,H,*]  (U=e*k, P=w*v, G=log decay [B,T,H])
    Out_ptr,                              # [B,T,H,V]
    Sentry_ptr,                           # [B,H,NC,N,V] entry state per chunk
    Sfinal_ptr,                           # [B,H,N,V]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NC: tl.constexpr,
    BN: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NEWTON_STEPS: tl.constexpr, ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    cidx = tl.arange(0, BC)
    nidx = tl.arange(0, BN)
    vidx = tl.arange(0, BV)
    c_mask = cidx < C
    n_mask = nidx < N
    v_mask = vidx < V
    lower_incl = (cidx[:, None] >= cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    lower_strict = (cidx[:, None] > cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    eyeC = (cidx[:, None] == cidx[None, :]).to(tl.float32) * c_mask[:, None]

    S = tl.zeros([BN, BV], dtype=tl.float32)
    for c in range(NC):
        t0 = c * C
        kn_off = ((b * T + (t0 + cidx[:, None])) * H + h) * N + nidx[None, :]
        kn_mask = c_mask[:, None] & n_mask[None, :]
        K = tl.load(K_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        Q = tl.load(Q_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        U = tl.load(U_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        pv_off = ((b * T + (t0 + cidx[:, None])) * H + h) * V + vidx[None, :]
        pv_mask = c_mask[:, None] & v_mask[None, :]
        P = tl.load(P_ptr + pv_off, mask=pv_mask, other=0.0).to(tl.float32)
        g_off = (b * T + (t0 + cidx)) * H + h
        g = tl.load(G_ptr + g_off, mask=c_mask, other=0.0).to(tl.float32)

        G = tl.cumsum(g, axis=0)
        gamma = tl.exp(G)
        decay_prev = tl.exp(G - g)
        inv_decay = tl.exp(-g)
        G_last = tl.sum(tl.where(c_mask, g, 0.0))

        # save entry state for this chunk (before update)
        se_off = (((b * H + h) * NC + c) * N + nidx[:, None]) * V + vidx[None, :]
        se_mask = n_mask[:, None] & v_mask[None, :]
        tl.store(Sentry_ptr + se_off, S.to(Sentry_ptr.dtype.element_ty), mask=se_mask)

        Kt = tl.trans(K)
        QK = tl.dot(Q, Kt, allow_tf32=ALLOW_TF32)
        KU = tl.dot(U, Kt, allow_tf32=ALLOW_TF32)
        # DA is only ever consumed in the LOWER triangle (A: lower_incl, M:
        # lower_strict), where G_i - G_j <= 0 so exp <= 1. The UPPER triangle is
        # always masked away, but with strong decay (g very negative) G_i - G_j can
        # exceed ~88 there and exp() overflows to +inf. The forward's tl.where drops
        # it, but the backward's UNMASKED `*DA` products then hit 0*inf = NaN. Clamp
        # the exponent to <= 0 so the (discarded) upper entries are a finite 1.0;
        # the used lower triangle is bit-identical. (fuse-2kernel 1.3B NaN fix.)
        DA = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0))
        A = tl.where(lower_incl, DA * QK, 0.0)
        M = tl.where(lower_strict, DA * inv_decay[:, None] * KU, 0.0)

        L = eyeC + M
        X = eyeC - M
        for _ in range(NEWTON_STEPS):
            LX = tl.dot(L, X, allow_tf32=ALLOW_TF32)
            X = tl.dot(X, 2.0 * eyeC - LX, allow_tf32=ALLOW_TF32)
        Tmat = X

        US = tl.dot(U, S, allow_tf32=ALLOW_TF32)
        RHS = P - decay_prev[:, None] * US
        Delta = tl.dot(Tmat, RHS, allow_tf32=ALLOW_TF32)

        QS = tl.dot(Q, S, allow_tf32=ALLOW_TF32)
        out = gamma[:, None] * QS + tl.dot(A, Delta, allow_tf32=ALLOW_TF32)
        tl.store(Out_ptr + pv_off, out.to(Out_ptr.dtype.element_ty), mask=pv_mask)

        kfac = tl.exp(G_last - G)
        Kscaled = K * kfac[:, None]
        SdK = tl.dot(tl.trans(Kscaled), Delta, allow_tf32=ALLOW_TF32)
        S = tl.exp(G_last) * S + SdK

    sf_off = ((b * H + h) * N + nidx[:, None]) * V + vidx[None, :]
    sf_mask = n_mask[:, None] & v_mask[None, :]
    tl.store(Sfinal_ptr + sf_off, S.to(Sfinal_ptr.dtype.element_ty), mask=sf_mask)


# ---------------------------------------------------------------------------
# Backward kernel — reverse chunk scan, fully fused. Recomputes the forward
# per-chunk quantities, then applies the chunked VJP. Produces gradients for
# k, q, v, decay, erase(e), write(w). dS is threaded in registers (the only
# sequential axis), initialised from dS_final at the last chunk.
# ---------------------------------------------------------------------------
@triton.jit
def _e97_bwd_kernel(
    K_ptr, Q_ptr, E_ptr, Vv_ptr, W_ptr, G_ptr,   # [B,T,H,*] raw inputs (e=erase, w=write gate)
    Sentry_ptr,                                   # [B,H,NC,N,V]
    dOut_ptr,                                     # [B,T,H,V]
    dSfinal_ptr,                                  # [B,H,N,V]
    dK_ptr, dQ_ptr, dE_ptr, dV_ptr, dW_ptr, dG_ptr,  # outputs (dG = grad wrt decay)
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NC: tl.constexpr,
    BN: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NEWTON_STEPS: tl.constexpr, ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    cidx = tl.arange(0, BC)
    nidx = tl.arange(0, BN)
    vidx = tl.arange(0, BV)
    c_mask = cidx < C
    n_mask = nidx < N
    v_mask = vidx < V
    lower_incl = (cidx[:, None] >= cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    lower_strict = (cidx[:, None] > cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    upper_incl = (cidx[:, None] <= cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    eyeC = (cidx[:, None] == cidx[None, :]).to(tl.float32) * c_mask[:, None]

    # dS = grad wrt state leaving the last chunk (= S_final)
    sf_off = ((b * H + h) * N + nidx[:, None]) * V + vidx[None, :]
    sf_mask = n_mask[:, None] & v_mask[None, :]
    dS = tl.load(dSfinal_ptr + sf_off, mask=sf_mask, other=0.0).to(tl.float32)

    for cc in range(NC):
        c = NC - 1 - cc
        t0 = c * C
        kn_off = ((b * T + (t0 + cidx[:, None])) * H + h) * N + nidx[None, :]
        kn_mask = c_mask[:, None] & n_mask[None, :]
        K = tl.load(K_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        Q = tl.load(Q_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        E = tl.load(E_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        pv_off = ((b * T + (t0 + cidx[:, None])) * H + h) * V + vidx[None, :]
        pv_mask = c_mask[:, None] & v_mask[None, :]
        Vv = tl.load(Vv_ptr + pv_off, mask=pv_mask, other=0.0).to(tl.float32)
        Wg = tl.load(W_ptr + pv_off, mask=pv_mask, other=0.0).to(tl.float32)
        g_off = (b * T + (t0 + cidx)) * H + h
        g = tl.load(G_ptr + g_off, mask=c_mask, other=0.0).to(tl.float32)
        dOut = tl.load(dOut_ptr + pv_off, mask=pv_mask, other=0.0).to(tl.float32)

        U = K * E
        P = Vv * Wg

        # ---- recompute forward intermediates ----
        G = tl.cumsum(g, axis=0)
        gamma = tl.exp(G)
        decay_prev = tl.exp(G - g)
        inv_decay = tl.exp(-g)
        G_last = tl.sum(tl.where(c_mask, g, 0.0))
        gamma_last = tl.exp(G_last)
        kfac = tl.exp(G_last - G)

        se_off = (((b * H + h) * NC + c) * N + nidx[:, None]) * V + vidx[None, :]
        se_mask = n_mask[:, None] & v_mask[None, :]
        S0c = tl.load(Sentry_ptr + se_off, mask=se_mask, other=0.0).to(tl.float32)

        # Invert (I+M) first, keeping minimal SMEM live across Newton: only L,X,LX.
        # The score matrices (DA,QK,KU,A) are cheap to recompute AFTER Newton, so
        # we do NOT keep them alive across the iteration (this is the SMEM budget
        # fix that lets C=64 tiles fit the 100KB/SM limit).
        Kt = tl.trans(K)
        # Clamp the exponent to <= 0 (only the lower triangle of DA is used; the
        # masked upper triangle would otherwise overflow to +inf at strong decay
        # and poison the backward's unmasked `*DA` products). See forward kernel.
        DA = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0))
        M = tl.where(lower_strict, DA * inv_decay[:, None] * tl.dot(U, Kt, allow_tf32=ALLOW_TF32), 0.0)
        L = eyeC + M
        X = eyeC - M
        for _ in range(NEWTON_STEPS):
            LX = tl.dot(L, X, allow_tf32=ALLOW_TF32)
            X = tl.dot(X, 2.0 * eyeC - LX, allow_tf32=ALLOW_TF32)
        Tmat = X
        # recompute score matrices for the VJP
        QK = tl.dot(Q, Kt, allow_tf32=ALLOW_TF32)
        KU = tl.dot(U, Kt, allow_tf32=ALLOW_TF32)
        A = tl.where(lower_incl, DA * QK, 0.0)

        Rmat = decay_prev[:, None] * U                         # [C,N]
        W_p = tl.dot(Tmat, P, allow_tf32=ALLOW_TF32)           # [C,V]
        W_u = tl.dot(Tmat, Rmat, allow_tf32=ALLOW_TF32)        # [C,N]
        Kscaled = K * kfac[:, None]                            # [C,N]
        Delta = W_p - tl.dot(W_u, S0c, allow_tf32=ALLOW_TF32)  # [C,V]
        QS0 = tl.dot(Q, S0c, allow_tf32=ALLOW_TF32)            # [C,V]

        # ---- VJP ----
        # out = gamma*QS0 + A@Delta ; S_next = gamma_last*S0c + Kscaled^T@Delta
        At = tl.trans(A)
        dDelta = tl.dot(At, dOut, allow_tf32=ALLOW_TF32) \
            + tl.dot(Kscaled, dS, allow_tf32=ALLOW_TF32)       # [C,V]
        dA = tl.where(lower_incl, tl.dot(dOut, tl.trans(Delta), allow_tf32=ALLOW_TF32), 0.0)

        dgamma = tl.sum(dOut * QS0, axis=1)                    # [C]
        dQS0 = gamma[:, None] * dOut                           # [C,V]
        dQ = tl.dot(dQS0, tl.trans(S0c), allow_tf32=ALLOW_TF32)   # [C,N]  (out term)
        dS0c = tl.dot(tl.trans(Q), dQS0, allow_tf32=ALLOW_TF32)   # [N,V]

        dgamma_last = tl.sum(tl.where(se_mask, dS * S0c, 0.0))
        dS0c += gamma_last * dS
        dKscaled = tl.dot(Delta, tl.trans(dS), allow_tf32=ALLOW_TF32)   # [C,N]

        # Delta = W_p - W_u @ S0c
        dW_p = dDelta
        dW_u = -tl.dot(dDelta, tl.trans(S0c), allow_tf32=ALLOW_TF32)    # [C,N]
        dS0c += -tl.dot(tl.trans(W_u), dDelta, allow_tf32=ALLOW_TF32)   # [N,V]

        # W_p = Tmat@P ; W_u = Tmat@Rmat
        dTmat = tl.dot(dW_p, tl.trans(P), allow_tf32=ALLOW_TF32) \
            + tl.dot(dW_u, tl.trans(Rmat), allow_tf32=ALLOW_TF32)       # [C,C]
        Tt = tl.trans(Tmat)
        dP = tl.dot(Tt, dW_p, allow_tf32=ALLOW_TF32)           # [C,V]
        dR = tl.dot(Tt, dW_u, allow_tf32=ALLOW_TF32)           # [C,N]
        dU = decay_prev[:, None] * dR                          # [C,N]  (from Rmat)
        ddecay_prev = tl.sum(dR * U, axis=1)                   # [C]

        # Tmat = (I+M)^{-1}: dL = -Tmat^T dTmat Tmat^T ; dM = strict_lower(dL)
        dL = -tl.dot(tl.dot(Tt, dTmat, allow_tf32=ALLOW_TF32), Tt, allow_tf32=ALLOW_TF32)
        dM = tl.where(lower_strict, dL, 0.0)

        # M = DA*inv_decay*KU (strict) ; A = DA*QK (incl). Consume each [C,C]
        # gradient (dQK, dKU) into its two dots immediately so they are not all
        # simultaneously live (keeps SMEM under the per-SM limit at C=64).
        d_invdecay = tl.sum(dM * DA * KU, axis=1)              # [C]
        # reduce dDA -> dG row/col contributions immediately, then free dDA
        DD = (dM * inv_decay[:, None] * KU + dA * QK) * DA     # [C,C] = dDA*DA
        dG_row = tl.sum(DD, axis=1)
        dG_col = tl.sum(DD, axis=0)

        dQK = dA * DA                                          # [C,C]
        dQ += tl.dot(dQK, K, allow_tf32=ALLOW_TF32)            # [C,N]
        dK = tl.dot(tl.trans(dQK), Q, allow_tf32=ALLOW_TF32)   # [C,N]

        dKU = dM * DA * inv_decay[:, None]                     # [C,C]
        dU += tl.dot(dKU, K, allow_tf32=ALLOW_TF32)            # [C,N]
        dK += tl.dot(tl.trans(dKU), U, allow_tf32=ALLOW_TF32)  # [C,N]

        # Kscaled = K*kfac
        dK += kfac[:, None] * dKscaled
        dkfac = tl.sum(dKscaled * K, axis=1)                   # [C]

        # ---- assemble dG (per-step log-decay grad) ----
        dG = dG_row - dG_col                                  # from DA (computed above)
        dG += dgamma * gamma                                  # gamma=exp(G)
        dG += ddecay_prev * decay_prev                        # decay_prev=exp(G-g)
        dG += -dkfac * kfac                                   # kfac=exp(G_last-G)
        dG_last = tl.sum(tl.where(c_mask, dkfac * kfac, 0.0)) + dgamma_last * gamma_last
        # fold dG_last into G[C-1] (== G_last)
        is_last = (cidx == (C - 1))
        dG += tl.where(is_last, dG_last, 0.0)
        # g enters G via inclusive cumsum: dg = rev_cumsum(dG) ; plus direct terms
        dg_cum = tl.sum(tl.where(upper_incl, dG[None, :], 0.0), axis=1)   # [C]
        dg_direct = -ddecay_prev * decay_prev - d_invdecay * inv_decay
        dg = dg_cum + dg_direct                                # [C]

        # ---- combine to raw input grads ----
        dk = dK + dU * E
        de = dU * K
        dv = dP * Wg
        dw = dP * Vv
        # Store the gradient wrt LOG-decay (dg) directly. The autograd.Function
        # converts to grad-wrt-decay (dg/decay) ONLY for legacy decay-space callers;
        # the log-decay caller (e88_fla_hybrid chunked path) consumes dg untouched.
        # Working in log space avoids the dg/decay blow-up as decay -> 0 (decay can
        # be ~1e-5 at Mamba2 init), which was the chunked-e97 backward-NaN that made
        # linear-state e97_delta untrainable at 1.3B (fuse-2kernel finding).

        # ---- store grads ----
        tl.store(dK_ptr + kn_off, dk.to(dK_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dQ_ptr + kn_off, dQ.to(dQ_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dE_ptr + kn_off, de.to(dE_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dV_ptr + pv_off, dv.to(dV_ptr.dtype.element_ty), mask=pv_mask)
        tl.store(dW_ptr + pv_off, dw.to(dW_ptr.dtype.element_ty), mask=pv_mask)
        tl.store(dG_ptr + g_off, dg.to(dG_ptr.dtype.element_ty), mask=c_mask)

        # propagate state grad to the previous chunk
        dS = dS0c


class E97DeltaChunkedFn(torch.autograd.Function):
    """Fused chunked E97 split-edit delta (linear state, S0=0). fwd+bwd Triton."""

    @staticmethod
    def forward(ctx, k, v, q, decay, erase, write, chunk_size, log_decay=False):
        # log_decay=True: `decay` IS the LOG-decay g (decay_t = exp(g_t)), and the
        # backward returns grad wrt g directly (no dg/decay division -> numerically
        # safe as decay -> 0). log_decay=False (legacy): `decay` is the decay in
        # (0,1]; backward returns grad wrt decay (dg/decay), exact for moderate decay.
        B, T, H, N = k.shape
        Vd = v.shape[-1]
        C = int(chunk_size)
        assert N <= 64 and Vd <= 64, "kernel supports N,V<=64"
        pad = (-T) % C
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
            erase = torch.nn.functional.pad(erase, (0, 0, 0, 0, 0, pad))
            write = torch.nn.functional.pad(write, (0, 0, 0, 0, 0, pad))
            # pad with the IDENTITY decay (decay=1 <=> g=0) so padded steps leave
            # the state unchanged regardless of decay parametrization.
            decay = torch.nn.functional.pad(decay, (0, 0, 0, pad),
                                            value=(0.0 if log_decay else 1.0))
        Tp = T + pad
        NC = Tp // C
        k = k.contiguous(); q = q.contiguous(); v = v.contiguous()
        erase = erase.contiguous(); write = write.contiguous()
        U = (k * erase).contiguous()
        Pw = (v * write).contiguous()
        glog = (decay if log_decay else decay.clamp_min(1e-9).log()).contiguous()

        out = torch.empty((B, Tp, H, Vd), device=k.device, dtype=k.dtype)
        S_entry = torch.empty((B, H, NC, N, Vd), device=k.device, dtype=torch.float32)
        S_final = torch.empty((B, H, N, Vd), device=k.device, dtype=k.dtype)
        BN, BV, BC = _next_pow2(N), _next_pow2(Vd), _next_pow2(C)
        newton = max(1, (C - 1).bit_length())
        allow_tf32 = k.dtype in (torch.bfloat16, torch.float16)
        _e97_fwd_save_kernel[(B * H,)](
            k, q, U, Pw, glog, out, S_entry, S_final,
            B=B, T=Tp, H=H, N=N, V=Vd, C=C, NC=NC,
            BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            num_warps=4,
        )
        ctx.save_for_backward(k, q, v, erase, write, glog, S_entry)
        ctx.shape = (B, T, Tp, H, N, Vd, C, NC, BN, BV, BC, newton, allow_tf32)
        ctx.log_decay = bool(log_decay)
        return out[:, :T].contiguous(), S_final

    @staticmethod
    def backward(ctx, dout, dSfinal):
        k, q, v, erase, write, glog, S_entry = ctx.saved_tensors
        (B, T, Tp, H, N, Vd, C, NC, BN, BV, BC, newton, allow_tf32) = ctx.shape
        if dout.shape[1] != Tp:
            dout = torch.nn.functional.pad(dout, (0, 0, 0, 0, 0, Tp - dout.shape[1]))
        dout = dout.contiguous()
        if dSfinal is None:
            dSfinal = torch.zeros((B, H, N, Vd), device=k.device, dtype=torch.float32)
        else:
            dSfinal = dSfinal.float().contiguous()
        dk = torch.empty_like(k)
        dq = torch.empty_like(q)
        de = torch.empty_like(erase)
        dv = torch.empty_like(v)
        dw = torch.empty_like(write)
        dgrad = torch.empty_like(glog)   # kernel writes dg = grad wrt LOG-decay
        _e97_bwd_kernel[(B * H,)](
            k, q, erase, v, write, glog, S_entry, dout, dSfinal,
            dk, dq, de, dv, dw, dgrad,
            B=B, T=Tp, H=H, N=N, V=Vd, C=C, NC=NC,
            BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )
        # dgrad = grad wrt log-decay g. For the log-decay caller, return it as-is
        # (the input WAS g). For the legacy decay caller, convert to grad-wrt-decay
        # via dg/decay (decay = exp(g)); exact and overflow-free for moderate decay.
        ddecay = dgrad if ctx.log_decay else dgrad / glog.exp().clamp_min(1e-9)
        sl = slice(0, T)
        return (dk[:, sl], dv[:, sl], dq[:, sl], ddecay[:, sl],
                de[:, sl], dw[:, sl], None, None)


# Default chunk for the fwd+bwd autograd path. The forward fits C=64 easily, but
# the fused backward holds ~6-7 [C,C] fp32 tiles live at peak; at C=64 that is
# 106 KB/SM, just over the 100 KB hard limit on Ada/Ampere (it fits on Hopper's
# 228 KB). C=32 keeps every tile at 4 KB, fits with wide margin, and already runs
# the fwd+bwd at 0.33-0.53x of GDN-2 (i.e. FASTER) with >=85% util at 1.3B dims —
# so we default to it. The forward-only fast path (e97_chunked_fwd_kernel) keeps
# C=64 for inference where there is no backward tile pressure.
_BWD_CHUNK_DEFAULT = 32


def e97_delta_chunked_triton(k, v, q, decay, erase_gate, value_write_gate,
                             chunk_size=_BWD_CHUNK_DEFAULT, log_decay=False):
    """Autograd-enabled fused chunked E97 split-edit delta. Returns (out, S_final).

    out: [B,T,H,V]; gradients flow to k, v, q, decay, erase_gate, value_write_gate.

    log_decay=True: `decay` is the LOG-decay g (decay_t = exp(g_t)); the backward
    returns grad wrt g directly with no dg/decay division — numerically safe when
    decay -> 0 (e.g. Mamba2-init decays ~1e-5). Prefer this form; the e88_fla_hybrid
    chunked path passes the upstream log-decay so the e97_delta decay parameters
    (A_log / a_proj / dt_bias) receive finite gradients at 1.3B scale.
    """
    return E97DeltaChunkedFn.apply(k, v, q, decay, erase_gate, value_write_gate,
                                   chunk_size, log_decay)
