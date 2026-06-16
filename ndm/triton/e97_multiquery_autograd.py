"""M2 — higher-rank MULTI-QUERY readout of the E97 matrix state (fused fwd+bwd).

This is the M2 mechanism from ``paper/review/STATE_AWARE_MLP_DESIGN.md`` §3: the
recurrence / state UPDATE is the **unchanged** chunked E97 split-edit delta (see
``e97_chunked_autograd.py``); only the READOUT changes. Today each head reads its
``[N,V]`` state with ONE query ``out = S^T q`` (a rank-1, ``V``-number read). M2
uses ``R`` queries/head producing ``R`` readout vectors — an ``R``-dim row-subspace
of each head's state — at essentially the cost of the recurrence (which is shared).

Why this is cheap and faithful: in the chunked kernel the query ``q`` is **purely a
readout** — it never enters the state update ``S``, the correction ``Delta``, the WY
inverse ``Tmat``, or the chunk-to-chunk state propagation. So R queries reuse ALL of
that shared work and only re-run the (cheap) readout matmuls per query:

    Delta_c, S_entry_c   computed ONCE per chunk (independent of every query)
    for r in 1..R:
        QK_r  = Q_r @ K^T               # [C,C]
        A_r   = tril(DA * QK_r)         # within-chunk read weights
        out_r = gamma * (Q_r @ S_entry) + A_r @ Delta_c     # [C,V]

The backward is the matching VJP, threaded into the SAME reverse-chunk replay as the
single-query kernel: the query-dependent gradients (dQ_r, and the Q-contributions to
dDelta / dK / dS_entry / the decay terms) are accumulated over ``r`` while the state
gradient ``dS`` is threaded across chunks exactly as before.

R=1 is byte-identical to ``e97_delta_chunked_triton`` by construction — the public
``e97_multiquery_chunked_triton`` wrapper routes R==1 straight to it (regression
guard), and the R=1 specialization of these kernels is verified equal in
``tests/test_e97_multiquery.py``.

Scope mirrors the single-query kernel: N, V <= 64; linear state; S0 = 0; bf16 inputs
run the TF32 tensor-core path. Parity (fwd + grads, fp32 and bf16) vs the eager
multi-query reference is in ``tests/test_e97_multiquery.py``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .e97_chunked_autograd import _next_pow2, _GLOG_FLOOR, e97_delta_chunked_triton


# ---------------------------------------------------------------------------
# Forward kernel — identical recurrence to ``_e97_fwd_save_kernel``; the ONLY
# change is the per-query readout loop (``for r in range(R)``) that emits R
# readouts per token from the SHARED chunk quantities (Delta, S_entry, DA).
# ---------------------------------------------------------------------------
@triton.jit
def _e97_fwd_mq_kernel(
    K_ptr, Q_ptr, U_ptr, P_ptr, G_ptr,   # K,U,P,G: [B,T,H,*]; Q: [B,T,H,R,N]
    Out_ptr,                              # [B,T,H,R,V]
    Sentry_ptr,                           # [B,H,NC,N,V]
    Sfinal_ptr,                           # [B,H,N,V]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
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
        KU = tl.dot(U, Kt, allow_tf32=ALLOW_TF32)
        # Clamp the (discarded) upper-triangle exponent to <= 0 so it cannot
        # overflow to +inf (it is always masked away; see _e97_fwd_save_kernel).
        DA = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0))
        M = tl.where(lower_strict, DA * inv_decay[:, None] * KU, 0.0)

        L = eyeC + M
        X = eyeC - M
        for _ in range(NEWTON_STEPS):
            LX = tl.dot(L, X, allow_tf32=ALLOW_TF32)
            X = tl.dot(X, 2.0 * eyeC - LX, allow_tf32=ALLOW_TF32)
        Tmat = X

        US = tl.dot(U, S, allow_tf32=ALLOW_TF32)
        RHS = P - decay_prev[:, None] * US
        Delta = tl.dot(Tmat, RHS, allow_tf32=ALLOW_TF32)   # [C,V] — shared by all queries

        # ----- multi-query readout (the M2 axis). All R reads share Delta/S/DA. -----
        for r in range(R):
            q_off = (((b * T + (t0 + cidx[:, None])) * H + h) * R + r) * N + nidx[None, :]
            Q = tl.load(Q_ptr + q_off, mask=kn_mask, other=0.0).to(tl.float32)
            QK = tl.dot(Q, Kt, allow_tf32=ALLOW_TF32)
            A = tl.where(lower_incl, DA * QK, 0.0)
            QS = tl.dot(Q, S, allow_tf32=ALLOW_TF32)
            out = gamma[:, None] * QS + tl.dot(A, Delta, allow_tf32=ALLOW_TF32)
            o_off = (((b * T + (t0 + cidx[:, None])) * H + h) * R + r) * V + vidx[None, :]
            tl.store(Out_ptr + o_off, out.to(Out_ptr.dtype.element_ty), mask=pv_mask)

        # state update (shared — independent of every query)
        kfac = tl.exp(G_last - G)
        Kscaled = K * kfac[:, None]
        SdK = tl.dot(tl.trans(Kscaled), Delta, allow_tf32=ALLOW_TF32)
        S = tl.exp(G_last) * S + SdK

    sf_off = ((b * H + h) * N + nidx[:, None]) * V + vidx[None, :]
    sf_mask = n_mask[:, None] & v_mask[None, :]
    tl.store(Sfinal_ptr + sf_off, S.to(Sfinal_ptr.dtype.element_ty), mask=sf_mask)


# ---------------------------------------------------------------------------
# Backward kernel — reverse chunk scan, fully fused. Same structure as
# ``_e97_bwd_kernel``; the query-dependent VJP terms are accumulated over the R
# queries while the shared recurrence VJP is computed once per chunk.
# ---------------------------------------------------------------------------
@triton.jit
def _e97_bwd_mq_kernel(
    K_ptr, Q_ptr, E_ptr, Vv_ptr, W_ptr, G_ptr,   # Q: [B,T,H,R,N]; rest [B,T,H,*]
    Sentry_ptr,                                   # [B,H,NC,N,V]
    dOut_ptr,                                     # [B,T,H,R,V]
    dSfinal_ptr,                                  # [B,H,N,V]
    dK_ptr, dQ_ptr, dE_ptr, dV_ptr, dW_ptr, dG_ptr,  # dQ: [B,T,H,R,N]; rest [B,T,H,*]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
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

    sf_off = ((b * H + h) * N + nidx[:, None]) * V + vidx[None, :]
    sf_mask = n_mask[:, None] & v_mask[None, :]
    dS = tl.load(dSfinal_ptr + sf_off, mask=sf_mask, other=0.0).to(tl.float32)

    for cc in range(NC):
        c = NC - 1 - cc
        t0 = c * C
        kn_off = ((b * T + (t0 + cidx[:, None])) * H + h) * N + nidx[None, :]
        kn_mask = c_mask[:, None] & n_mask[None, :]
        K = tl.load(K_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        E = tl.load(E_ptr + kn_off, mask=kn_mask, other=0.0).to(tl.float32)
        pv_off = ((b * T + (t0 + cidx[:, None])) * H + h) * V + vidx[None, :]
        pv_mask = c_mask[:, None] & v_mask[None, :]
        Vv = tl.load(Vv_ptr + pv_off, mask=pv_mask, other=0.0).to(tl.float32)
        Wg = tl.load(W_ptr + pv_off, mask=pv_mask, other=0.0).to(tl.float32)
        g_off = (b * T + (t0 + cidx)) * H + h
        g = tl.load(G_ptr + g_off, mask=c_mask, other=0.0).to(tl.float32)

        U = K * E
        P = Vv * Wg

        # ---- recompute forward intermediates (shared, query-independent) ----
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

        Kt = tl.trans(K)
        DA = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0))
        KU = tl.dot(U, Kt, allow_tf32=ALLOW_TF32)
        M = tl.where(lower_strict, DA * inv_decay[:, None] * KU, 0.0)
        L = eyeC + M
        X = eyeC - M
        for _ in range(NEWTON_STEPS):
            LX = tl.dot(L, X, allow_tf32=ALLOW_TF32)
            X = tl.dot(X, 2.0 * eyeC - LX, allow_tf32=ALLOW_TF32)
        Tmat = X

        Rmat = decay_prev[:, None] * U                         # [C,N]
        W_p = tl.dot(Tmat, P, allow_tf32=ALLOW_TF32)           # [C,V]
        W_u = tl.dot(Tmat, Rmat, allow_tf32=ALLOW_TF32)        # [C,N]
        Kscaled = K * kfac[:, None]                            # [C,N]
        Delta = W_p - tl.dot(W_u, S0c, allow_tf32=ALLOW_TF32)  # [C,V] — shared

        # ---- query-dependent VJP, accumulated over R queries ----
        # dDelta gets a per-query out-term A_r^T@dOut_r plus the shared state term
        # Kscaled@dS. dK accumulates the per-query dQK_r^T@Q_r. DD_A accumulates the
        # per-query dA_r*QK_r*DA (the A-side of the decay grad). dS0c_q and dgamma
        # accumulate the per-query out-term contributions.
        dDelta = tl.dot(Kscaled, dS, allow_tf32=ALLOW_TF32)    # [C,V] state term
        dgamma = tl.zeros([BC], dtype=tl.float32)
        dS0c_q = tl.zeros([BN, BV], dtype=tl.float32)
        dK = tl.zeros([BC, BN], dtype=tl.float32)
        DD_A = tl.zeros([BC, BC], dtype=tl.float32)
        for r in range(R):
            q_off = (((b * T + (t0 + cidx[:, None])) * H + h) * R + r) * N + nidx[None, :]
            Q = tl.load(Q_ptr + q_off, mask=kn_mask, other=0.0).to(tl.float32)
            do_off = (((b * T + (t0 + cidx[:, None])) * H + h) * R + r) * V + vidx[None, :]
            dOut = tl.load(dOut_ptr + do_off, mask=pv_mask, other=0.0).to(tl.float32)

            QK = tl.dot(Q, Kt, allow_tf32=ALLOW_TF32)
            A = tl.where(lower_incl, DA * QK, 0.0)
            QS0 = tl.dot(Q, S0c, allow_tf32=ALLOW_TF32)        # [C,V]

            At = tl.trans(A)
            dDelta += tl.dot(At, dOut, allow_tf32=ALLOW_TF32)
            dA = tl.where(lower_incl, tl.dot(dOut, tl.trans(Delta), allow_tf32=ALLOW_TF32), 0.0)

            dgamma += tl.sum(dOut * QS0, axis=1)               # [C]
            dQS0 = gamma[:, None] * dOut                       # [C,V]
            dQ = tl.dot(dQS0, tl.trans(S0c), allow_tf32=ALLOW_TF32)   # [C,N] out term
            dS0c_q += tl.dot(tl.trans(Q), dQS0, allow_tf32=ALLOW_TF32)

            dQK = dA * DA                                      # [C,C]
            dQ += tl.dot(dQK, K, allow_tf32=ALLOW_TF32)        # [C,N]
            dK += tl.dot(tl.trans(dQK), Q, allow_tf32=ALLOW_TF32)
            DD_A += dA * QK * DA                               # [C,C]

            tl.store(dQ_ptr + q_off, dQ.to(dQ_ptr.dtype.element_ty), mask=kn_mask)

        # ---- shared recurrence VJP (computed once per chunk) ----
        dgamma_last = tl.sum(tl.where(se_mask, dS * S0c, 0.0))
        dS0c = dS0c_q + gamma_last * dS
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
        dU = decay_prev[:, None] * dR                          # [C,N] (from Rmat)
        ddecay_prev = tl.sum(dR * U, axis=1)                   # [C]

        # Tmat = (I+M)^{-1}: dL = -Tmat^T dTmat Tmat^T ; dM = strict_lower(dL)
        dL = -tl.dot(tl.dot(Tt, dTmat, allow_tf32=ALLOW_TF32), Tt, allow_tf32=ALLOW_TF32)
        dM = tl.where(lower_strict, dL, 0.0)

        d_invdecay = tl.sum(dM * DA * KU, axis=1)              # [C]
        # DD = dDA*DA. The A-side (dA*QK) was already folded into DD_A inside the
        # query loop; here we add the M-side (dM*inv_decay*KU).
        DD = dM * inv_decay[:, None] * KU * DA + DD_A          # [C,C]
        dG_row = tl.sum(DD, axis=1)
        dG_col = tl.sum(DD, axis=0)

        # dK already holds sum_r dQK_r^T@Q_r from the query loop; add the dKU side.
        dKU = dM * DA * inv_decay[:, None]                     # [C,C]
        dU += tl.dot(dKU, K, allow_tf32=ALLOW_TF32)            # [C,N]
        dK += tl.dot(tl.trans(dKU), U, allow_tf32=ALLOW_TF32)  # [C,N]

        # Kscaled = K*kfac
        dK += kfac[:, None] * dKscaled
        dkfac = tl.sum(dKscaled * K, axis=1)                   # [C]

        # ---- assemble dG (per-step log-decay grad) ----
        dG = dG_row - dG_col
        dG += dgamma * gamma                                  # gamma=exp(G)
        dG += ddecay_prev * decay_prev                        # decay_prev=exp(G-g)
        dG += -dkfac * kfac                                   # kfac=exp(G_last-G)
        dG_last = tl.sum(tl.where(c_mask, dkfac * kfac, 0.0)) + dgamma_last * gamma_last
        is_last = (cidx == (C - 1))
        dG += tl.where(is_last, dG_last, 0.0)
        dg_cum = tl.sum(tl.where(upper_incl, dG[None, :], 0.0), axis=1)   # [C]
        dg_direct = -ddecay_prev * decay_prev - d_invdecay * inv_decay
        dg = dg_cum + dg_direct                                # [C]

        # ---- combine to raw input grads ----
        dk = dK + dU * E
        de = dU * K
        dv = dP * Wg
        dw = dP * Vv

        tl.store(dK_ptr + kn_off, dk.to(dK_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dE_ptr + kn_off, de.to(dE_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dV_ptr + pv_off, dv.to(dV_ptr.dtype.element_ty), mask=pv_mask)
        tl.store(dW_ptr + pv_off, dw.to(dW_ptr.dtype.element_ty), mask=pv_mask)
        tl.store(dG_ptr + g_off, dg.to(dG_ptr.dtype.element_ty), mask=c_mask)

        dS = dS0c


_BWD_CHUNK_DEFAULT = 32


class E97MultiQueryChunkedFn(torch.autograd.Function):
    """Fused chunked E97 split-edit delta with R-query readout (linear state, S0=0).

    q has shape [B,T,H,R,N]; out has shape [B,T,H,R,V]. The recurrence is identical
    to ``E97DeltaChunkedFn``; only the readout is rank-R.
    """

    @staticmethod
    def forward(ctx, k, v, q, decay, erase, write, chunk_size, log_decay=False):
        B, T, H, N = k.shape
        Vd = v.shape[-1]
        Rq = q.shape[3]
        assert q.shape == (B, T, H, Rq, N), f"q must be [B,T,H,R,N], got {tuple(q.shape)}"
        C = int(chunk_size)
        assert N <= 64 and Vd <= 64, "kernel supports N,V<=64"
        pad = (-T) % C
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
            erase = torch.nn.functional.pad(erase, (0, 0, 0, 0, 0, pad))
            write = torch.nn.functional.pad(write, (0, 0, 0, 0, 0, pad))
            decay = torch.nn.functional.pad(decay, (0, 0, 0, pad),
                                            value=(0.0 if log_decay else 1.0))
        Tp = T + pad
        NC = Tp // C
        k = k.contiguous(); q = q.contiguous(); v = v.contiguous()
        erase = erase.contiguous(); write = write.contiguous()
        U = (k * erase).contiguous()
        Pw = (v * write).contiguous()
        glog = (decay if log_decay else decay.clamp_min(1e-9).log())
        clamp_mask = glog < _GLOG_FLOOR
        glog = glog.clamp_min(_GLOG_FLOOR).contiguous()

        out = torch.empty((B, Tp, H, Rq, Vd), device=k.device, dtype=k.dtype)
        S_entry = torch.empty((B, H, NC, N, Vd), device=k.device, dtype=torch.float32)
        S_final = torch.empty((B, H, N, Vd), device=k.device, dtype=k.dtype)
        BN, BV, BC = _next_pow2(N), _next_pow2(Vd), _next_pow2(C)
        newton = max(1, (C - 1).bit_length())
        allow_tf32 = k.dtype in (torch.bfloat16, torch.float16)
        _e97_fwd_mq_kernel[(B * H,)](
            k, q, U, Pw, glog, out, S_entry, S_final,
            B=B, T=Tp, H=H, N=N, V=Vd, R=Rq, C=C, NC=NC,
            BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            num_warps=4,
        )
        ctx.save_for_backward(k, q, v, erase, write, glog, S_entry, clamp_mask)
        ctx.shape = (B, T, Tp, H, N, Vd, Rq, C, NC, BN, BV, BC, newton, allow_tf32)
        ctx.log_decay = bool(log_decay)
        return out[:, :T].contiguous(), S_final

    @staticmethod
    def backward(ctx, dout, dSfinal):
        k, q, v, erase, write, glog, S_entry, clamp_mask = ctx.saved_tensors
        (B, T, Tp, H, N, Vd, Rq, C, NC, BN, BV, BC, newton, allow_tf32) = ctx.shape
        if dout.shape[1] != Tp:
            dout = torch.nn.functional.pad(dout, (0, 0, 0, 0, 0, 0, 0, Tp - dout.shape[1]))
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
        dgrad = torch.empty_like(glog)
        _e97_bwd_mq_kernel[(B * H,)](
            k, q, erase, v, write, glog, S_entry, dout, dSfinal,
            dk, dq, de, dv, dw, dgrad,
            B=B, T=Tp, H=H, N=N, V=Vd, R=Rq, C=C, NC=NC,
            BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            num_warps=4, num_stages=1,
        )
        dgrad = dgrad.masked_fill(clamp_mask, 0.0)
        ddecay = dgrad if ctx.log_decay else dgrad / glog.exp().clamp_min(1e-9)
        sl = slice(0, T)
        return (dk[:, sl], dv[:, sl], dq[:, sl], ddecay[:, sl],
                de[:, sl], dw[:, sl], None, None)


def e97_multiquery_chunked_triton(k, v, q, decay, erase_gate, value_write_gate,
                                  chunk_size=_BWD_CHUNK_DEFAULT, log_decay=False):
    """Autograd-enabled fused chunked E97 with rank-R multi-query readout.

    Args:
        k, decay, erase_gate: [B,T,H,N] / [B,T,H] / [B,T,H,N]
        v, value_write_gate:  [B,T,H,V]
        q:                    [B,T,H,R,N]  (R queries per head)
    Returns:
        out:     [B,T,H,R,V]  (R readout vectors per head)
        S_final: [B,H,N,V]

    R==1 routes to the single-query ``e97_delta_chunked_triton`` so the M2 path
    reduces BYTE-IDENTICALLY to the current E97 readout (regression guard).
    """
    Rq = q.shape[3]
    if Rq == 1:
        out, S_final = e97_delta_chunked_triton(
            k, v, q[:, :, :, 0], decay, erase_gate, value_write_gate,
            chunk_size=chunk_size, log_decay=log_decay,
        )
        return out.unsqueeze(3), S_final
    return E97MultiQueryChunkedFn.apply(k, v, q, decay, erase_gate, value_write_gate,
                                        chunk_size, log_decay)
