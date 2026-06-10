"""Fused chunked-parallel `refit` (inner-optimization / TTT) WRITE cell — autograd.

`refit` is the inner-optimization head-type of TTT_WRITE_SPEC.md: the per-token
state update is one inner-optimizer step on the inner reconstruction loss
``L_inner(S; k,v) = ½‖v − Sᵀk‖²``, generalized along TWO chunkable axes —
**inner momentum** (heavy-ball / Titans surprise EMA) and the **exposed inner-step
count K** (the chunk Newton–Schulz iteration count). The recurrence (per b,h;
linear fast-weight state ``S [N,V]``, momentum buffer ``M [N,V]``):

    u_t  = e_t ⊙ k_t                       (read key,  erase gate)
    p_t  = w_t ⊙ v_t                       (write val, write gate)
    δ_t  = p_t − S_{t-1}ᵀ u_t              (surprise / residual)
    r_t  = k_t δ_tᵀ                         (per-token surprise outer product)
    M_t  = μ_t M_{t-1} + r_t               (inner momentum buffer)
    S_t  = g_t S_{t-1} + M_t               (decayed state + accumulated momentum)
    out_t = S_tᵀ q_t

Setting ``μ_t ≡ 0`` collapses ``M_t = r_t`` ⇒ ``S_t = g_t S_{t-1} + k_t δ_tᵀ``,
which is EXACTLY the gated-delta / e97 split-edit write (``e88_torch_reference``,
``linear_state=True, raw_write=False``) — the **delta-rule = one-inner-step special
case**, recovered bit-identically here via the ``HAS_MOM=False`` constexpr fast path
(the kernel then compiles to the e97 chunked delta path with zero momentum overhead).

The chunkability is real (TTT_WRITE_SPEC §3): the joint ``(S,M)`` recurrence is an
upper-triangular 2×2 ``(g,μ)`` companion per channel, hence affine in the state, hence
chunk-parallelizable. The intra-chunk coupling that delta has as a single decay twiddle
``exp(G_{t-1}−G_j)`` becomes a TWO-twiddle product summed over the intermediate index,
``Φ[t,l] = Σ_{j=l}^{t-1} exp(G_{t-1}−G_j) exp(Gm_j−Gm_l)``, which is exactly the matmul
``Φ = A1 @ A2`` of the decay-twiddle ``A1`` and the momentum-twiddle ``A2`` — both
``[C,C]`` tensor-core dots, the same nilpotent UT structure e97 already solves. At
``μ→0`` ``A2→I`` and ``Φ→A1`` recovers delta. See TTT_WRITE_SPEC §4.

Two fused ``@triton.jit`` kernels: ``_refit_fwd_kernel`` (forward, saves per-chunk
entry ``S`` and ``M``) and ``_refit_bwd_kernel`` (reverse chunk walk, threads both
state grads ``dS`` and ``dM`` in registers, recomputes forward intermediates). No
torch fallback in the hot path. Parity (fwd + grads, fp32 and bf16) is checked in
``tests/test_refit_chunked.py`` against the eager heavy-ball recurrence
``refit_eager_reference`` and, at ``μ=0``, against the e97 delta reference.

Scope: N, V ≤ 64; linear state; S0 = M0 = 0. ``NEWTON_STEPS = K`` exposed (default
``ceil(log2 C)`` = exact chunk refit; smaller = truncated-Neumann approximate inner
solve). bf16 inputs run the TF32 tensor-core path; fp32 inputs run exact.
"""
from __future__ import annotations

import os as _os

import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(16, p)


# Floor for the per-step LOG-gates (g = log decay, gm = log μ) fed to the kernel.
# Mirrors e97's _GLOG_FLOOR: inv_decay = exp(-g) appears in the intra-chunk score
# matrix and overflows fp32 once g < ~-88; at g = -30 the per-step decay is already
# exp(-30) ~ 9e-14 (numerically "forget everything in one step"), so flooring here
# costs zero modeling range while keeping exp(-g) <= ~1e13. The same floor on log μ
# keeps the momentum twiddles finite; μ ~ exp(-30) is "no momentum" already.
_GLOG_FLOOR = float(_os.environ.get('REFIT_GLOG_FLOOR', '-30.0'))


# ---------------------------------------------------------------------------
# Forward kernel. One Triton program per (batch, head); sequential chunk loop
# carries (S, M) in registers and writes the per-chunk ENTRY (S, M) for the bwd.
# ---------------------------------------------------------------------------
@triton.jit
def _refit_fwd_kernel(
    K_ptr, Q_ptr, U_ptr, P_ptr, G_ptr, GM_ptr,   # [B,T,H,*] (U=e*k, P=w*v, G=log g, GM=log μ)
    Out_ptr,                                      # [B,T,H,V]
    Sentry_ptr, Mentry_ptr,                       # [B,H,NC,N,V] entry S and M per chunk
    Sfinal_ptr,                                   # [B,H,N,V]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NC: tl.constexpr,
    BN: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NEWTON_STEPS: tl.constexpr, ALLOW_TF32: tl.constexpr, HAS_MOM: tl.constexpr,
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

    S = tl.zeros([BN, BV], dtype=tl.float32)
    M = tl.zeros([BN, BV], dtype=tl.float32)
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
        gamma_last = tl.exp(G_last)
        # DA[i,j] = exp(G_i - G_j); clamp exponent <= 0 so the masked-away upper
        # triangle is a finite 1.0 (used lower triangle is exact). See e97 note.
        DA = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0))

        # save entry (S, M) for this chunk (before update)
        se_off = (((b * H + h) * NC + c) * N + nidx[:, None]) * V + vidx[None, :]
        se_mask = n_mask[:, None] & v_mask[None, :]
        tl.store(Sentry_ptr + se_off, S.to(Sentry_ptr.dtype.element_ty), mask=se_mask)
        if HAS_MOM:
            tl.store(Mentry_ptr + se_off, M.to(Mentry_ptr.dtype.element_ty), mask=se_mask)

        Kt = tl.trans(K)
        QK = tl.dot(Q, Kt, allow_tf32=ALLOW_TF32)
        KU = tl.dot(U, Kt, allow_tf32=ALLOW_TF32)   # KU2[t,i] = u_t · k_i

        # A1[t,j] = exp(G_{t-1}-G_j) strict-lower (= e97 decay score factor).
        A1 = tl.where(lower_strict, DA * inv_decay[:, None], 0.0)
        # B1[t,j] = exp(G_t-G_j) lower-incl (= e97 read score factor).
        B1 = tl.where(lower_incl, DA, 0.0)

        if HAS_MOM:
            gm = tl.load(GM_ptr + g_off, mask=c_mask, other=0.0).to(tl.float32)
            Gm = tl.cumsum(gm, axis=0)
            gammam = tl.exp(Gm)
            Gm_last = tl.sum(tl.where(c_mask, gm, 0.0))
            gammam_last = tl.exp(Gm_last)
            DAm = tl.exp(tl.minimum(Gm[:, None] - Gm[None, :], 0.0))
            A2 = tl.where(lower_incl, DAm, 0.0)             # μ-twiddle, lower-incl
            Phi = tl.dot(A1, A2, allow_tf32=ALLOW_TF32)     # strict-lower coupling
            Phiout = tl.dot(B1, A2, allow_tf32=ALLOW_TF32)  # lower-incl coupling
            psi = tl.sum(A1 * gammam[None, :], axis=1)      # [C]
            psiout = tl.sum(B1 * gammam[None, :], axis=1)   # [C]
        else:
            Phi = A1
            Phiout = B1
            psi = tl.zeros([BC], dtype=tl.float32)
            psiout = tl.zeros([BC], dtype=tl.float32)

        Mscore = tl.where(lower_strict, Phi * KU, 0.0)
        Aout = tl.where(lower_incl, Phiout * QK, 0.0)

        # Tmat = (I + Mscore)^{-1} via NEWTON_STEPS Newton-Schulz iterations
        # (the explicit inner-step knob K). K = ceil(log2 C) => exact refit.
        L = eyeC + Mscore
        X = eyeC - Mscore
        for _ in range(NEWTON_STEPS):
            LX = tl.dot(L, X, allow_tf32=ALLOW_TF32)
            X = tl.dot(X, 2.0 * eyeC - LX, allow_tf32=ALLOW_TF32)
        Tmat = X

        US = tl.dot(U, S, allow_tf32=ALLOW_TF32)
        QS = tl.dot(Q, S, allow_tf32=ALLOW_TF32)
        RHS = P - decay_prev[:, None] * US
        out = gamma[:, None] * QS
        if HAS_MOM:
            UM = tl.dot(U, M, allow_tf32=ALLOW_TF32)
            QM = tl.dot(Q, M, allow_tf32=ALLOW_TF32)
            RHS = RHS - psi[:, None] * UM
            out = out + psiout[:, None] * QM
        Delta = tl.dot(Tmat, RHS, allow_tf32=ALLOW_TF32)
        out = out + tl.dot(Aout, Delta, allow_tf32=ALLOW_TF32)
        tl.store(Out_ptr + pv_off, out.to(Out_ptr.dtype.element_ty), mask=pv_mask)

        # ---- cross-chunk carry of (S, M). S_new uses the OLD M (entry M); compute
        #      S_new BEFORE overwriting M. ----
        kfacG = tl.exp(tl.minimum(G_last - G, 0.0))         # exp(G_last-G_j) <= 1
        if HAS_MOM:
            # w_i = sum_{j>=i} exp((G_last-G_j)+(Gm_j-Gm_i))  (j>=i upper-incl)
            Eij = (G_last - G[None, :]) + (Gm[None, :] - Gm[:, None])
            Wm = tl.where(upper_incl, tl.exp(tl.minimum(Eij, 0.0)), 0.0)
            w = tl.sum(Wm, axis=1)
            Scs = tl.sum(kfacG * gammam)
            Ksc = K * w[:, None]
            # S_new = gamma_last*S_old + Scs*M_old + (K*w)^T @ Delta
            S_new = (gamma_last * S + Scs * M
                     + tl.dot(tl.trans(Ksc), Delta, allow_tf32=ALLOW_TF32))
            # M_new = gammam_last*M_old + (K*kmfac)^T @ Delta
            kmfac = tl.exp(tl.minimum(Gm_last - Gm, 0.0))    # exp(Gm_last-Gm_i) <= 1
            Kmsc = K * kmfac[:, None]
            M = gammam_last * M + tl.dot(tl.trans(Kmsc), Delta, allow_tf32=ALLOW_TF32)
            S = S_new
        else:
            Kscaled = K * kfacG[:, None]
            S = gamma_last * S + tl.dot(tl.trans(Kscaled), Delta, allow_tf32=ALLOW_TF32)

    sf_off = ((b * H + h) * N + nidx[:, None]) * V + vidx[None, :]
    sf_mask = n_mask[:, None] & v_mask[None, :]
    tl.store(Sfinal_ptr + sf_off, S.to(Sfinal_ptr.dtype.element_ty), mask=sf_mask)


# ---------------------------------------------------------------------------
# Backward kernel — reverse chunk walk, fully fused. Recomputes the forward
# per-chunk quantities, then applies the chunked VJP, threading BOTH state grads
# dS and dM in registers (init dS=dS_final, dM=0 at the last chunk). Produces
# grads for k, q, v(=P), erase(=U), write(=P), decay(=g log), mu(=gm log).
# ---------------------------------------------------------------------------
@triton.jit
def _refit_bwd_kernel(
    K_ptr, Q_ptr, E_ptr, Vv_ptr, W_ptr, G_ptr, GM_ptr,   # [B,T,H,*] raw inputs
    Sentry_ptr, Mentry_ptr,                              # [B,H,NC,N,V]
    dOut_ptr,                                            # [B,T,H,V]
    dSfinal_ptr,                                         # [B,H,N,V]
    dK_ptr, dQ_ptr, dE_ptr, dV_ptr, dW_ptr, dG_ptr, dGM_ptr,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NC: tl.constexpr,
    BN: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NEWTON_STEPS: tl.constexpr, ALLOW_TF32: tl.constexpr, HAS_MOM: tl.constexpr,
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
    dMs = tl.zeros([BN, BV], dtype=tl.float32)

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

        # ===== recompute forward intermediates =====
        G = tl.cumsum(g, axis=0)
        gamma = tl.exp(G)
        decay_prev = tl.exp(G - g)
        inv_decay = tl.exp(-g)
        G_last = tl.sum(tl.where(c_mask, g, 0.0))
        gamma_last = tl.exp(G_last)
        DA = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0))
        kfacG = tl.exp(tl.minimum(G_last - G, 0.0))

        se_off = (((b * H + h) * NC + c) * N + nidx[:, None]) * V + vidx[None, :]
        se_mask = n_mask[:, None] & v_mask[None, :]
        S0c = tl.load(Sentry_ptr + se_off, mask=se_mask, other=0.0).to(tl.float32)

        Kt = tl.trans(K)
        QK = tl.dot(Q, Kt, allow_tf32=ALLOW_TF32)
        KU = tl.dot(U, Kt, allow_tf32=ALLOW_TF32)
        A1 = tl.where(lower_strict, DA * inv_decay[:, None], 0.0)
        B1 = tl.where(lower_incl, DA, 0.0)

        if HAS_MOM:
            gm = tl.load(GM_ptr + g_off, mask=c_mask, other=0.0).to(tl.float32)
            Gm = tl.cumsum(gm, axis=0)
            gammam = tl.exp(Gm)
            Gm_last = tl.sum(tl.where(c_mask, gm, 0.0))
            gammam_last = tl.exp(Gm_last)
            DAm = tl.exp(tl.minimum(Gm[:, None] - Gm[None, :], 0.0))
            A2 = tl.where(lower_incl, DAm, 0.0)
            Phi = tl.dot(A1, A2, allow_tf32=ALLOW_TF32)
            Phiout = tl.dot(B1, A2, allow_tf32=ALLOW_TF32)
            psi = tl.sum(A1 * gammam[None, :], axis=1)
            psiout = tl.sum(B1 * gammam[None, :], axis=1)
            M0c = tl.load(Mentry_ptr + se_off, mask=se_mask, other=0.0).to(tl.float32)
            kmfac = tl.exp(tl.minimum(Gm_last - Gm, 0.0))
            Scs = tl.sum(kfacG * gammam)
            Eij = (G_last - G[None, :]) + (Gm[None, :] - Gm[:, None])
            Wm = tl.where(upper_incl, tl.exp(tl.minimum(Eij, 0.0)), 0.0)
            w = tl.sum(Wm, axis=1)
        else:
            Phi = A1
            Phiout = B1
            psi = tl.zeros([BC], dtype=tl.float32)
            psiout = tl.zeros([BC], dtype=tl.float32)
            M0c = tl.zeros([BN, BV], dtype=tl.float32)
            w = kfacG

        Mscore = tl.where(lower_strict, Phi * KU, 0.0)
        Aout = tl.where(lower_incl, Phiout * QK, 0.0)
        L = eyeC + Mscore
        X = eyeC - Mscore
        for _ in range(NEWTON_STEPS):
            LX = tl.dot(L, X, allow_tf32=ALLOW_TF32)
            X = tl.dot(X, 2.0 * eyeC - LX, allow_tf32=ALLOW_TF32)
        Tmat = X

        US = tl.dot(U, S0c, allow_tf32=ALLOW_TF32)
        QS0 = tl.dot(Q, S0c, allow_tf32=ALLOW_TF32)
        RHS = P - decay_prev[:, None] * US
        if HAS_MOM:
            UM = tl.dot(U, M0c, allow_tf32=ALLOW_TF32)
            QM0 = tl.dot(Q, M0c, allow_tf32=ALLOW_TF32)
            RHS = RHS - psi[:, None] * UM
        Delta = tl.dot(Tmat, RHS, allow_tf32=ALLOW_TF32)

        # ===== VJP =====
        dG = tl.zeros([BC], dtype=tl.float32)
        dGm = tl.zeros([BC], dtype=tl.float32)
        dG_last_s = 0.0
        dGm_last_s = 0.0
        dgamma = tl.zeros([BC], dtype=tl.float32)
        dgammam = tl.zeros([BC], dtype=tl.float32)
        dkfacG = tl.zeros([BC], dtype=tl.float32)

        # ---- (carry) S_new / M_new ----
        if HAS_MOM:
            Ksc = K * w[:, None]
            Kmsc = K * kmfac[:, None]
            dG_last_s += tl.sum(tl.where(se_mask, dS * S0c, 0.0)) * gamma_last
            dScs = tl.sum(tl.where(se_mask, dS * M0c, 0.0))
            dM0c = Scs * dS
            dKsc = tl.dot(Delta, tl.trans(dS), allow_tf32=ALLOW_TF32)
            dDelta = tl.dot(Ksc, dS, allow_tf32=ALLOW_TF32)
            dS0c = gamma_last * dS
            dK = w[:, None] * dKsc
            dw = tl.sum(dKsc * K, axis=1)
            # M_new
            dGm_last_s += tl.sum(tl.where(se_mask, dMs * M0c, 0.0)) * gammam_last
            dKmsc = tl.dot(Delta, tl.trans(dMs), allow_tf32=ALLOW_TF32)
            dDelta += tl.dot(Kmsc, dMs, allow_tf32=ALLOW_TF32)
            dM0c += gammam_last * dMs
            dK += kmfac[:, None] * dKmsc
            dkmfac = tl.sum(dKmsc * K, axis=1)
        else:
            Kscaled = K * kfacG[:, None]
            dG_last_s += tl.sum(tl.where(se_mask, dS * S0c, 0.0)) * gamma_last
            dKscaled = tl.dot(Delta, tl.trans(dS), allow_tf32=ALLOW_TF32)
            dDelta = tl.dot(Kscaled, dS, allow_tf32=ALLOW_TF32)
            dS0c = gamma_last * dS
            dK = kfacG[:, None] * dKscaled
            dkfacG += tl.sum(dKscaled * K, axis=1)
            dM0c = tl.zeros([BN, BV], dtype=tl.float32)

        # ---- (out) ----
        dgamma += tl.sum(dOut * QS0, axis=1)
        dQS0 = gamma[:, None] * dOut
        dAout = tl.where(lower_incl, tl.dot(dOut, tl.trans(Delta), allow_tf32=ALLOW_TF32), 0.0)
        dDelta += tl.dot(tl.trans(Aout), dOut, allow_tf32=ALLOW_TF32)
        if HAS_MOM:
            dpsiout = tl.sum(dOut * QM0, axis=1)
            dQM0 = psiout[:, None] * dOut

        # ---- (QS0,QM0) ----
        dQ = tl.dot(dQS0, tl.trans(S0c), allow_tf32=ALLOW_TF32)
        dS0c += tl.dot(tl.trans(Q), dQS0, allow_tf32=ALLOW_TF32)
        if HAS_MOM:
            dQ += tl.dot(dQM0, tl.trans(M0c), allow_tf32=ALLOW_TF32)
            dM0c += tl.dot(tl.trans(Q), dQM0, allow_tf32=ALLOW_TF32)

        # ---- (psiout = B1@gammam) ----
        dB1 = tl.zeros([BC, BC], dtype=tl.float32)
        if HAS_MOM:
            dB1 += dpsiout[:, None] * gammam[None, :]
            dgammam += tl.sum(B1 * dpsiout[:, None], axis=0)

        # ---- (Aout = where(incl, Phiout*QK)) ----
        dPhiout = tl.where(lower_incl, dAout * QK, 0.0)
        dQK = tl.where(lower_incl, dAout * Phiout, 0.0)

        # ---- (Delta = Tmat@RHS) ----
        dTmat = tl.dot(dDelta, tl.trans(RHS), allow_tf32=ALLOW_TF32)
        dRHS = tl.dot(tl.trans(Tmat), dDelta, allow_tf32=ALLOW_TF32)

        # ---- (RHS = P - decay_prev*US - psi*UM) ----
        dP = dRHS
        ddecay_prev = -tl.sum(dRHS * US, axis=1)
        dUS = -decay_prev[:, None] * dRHS
        dpsi = tl.zeros([BC], dtype=tl.float32)
        if HAS_MOM:
            dpsi = -tl.sum(dRHS * UM, axis=1)
            dUM = -psi[:, None] * dRHS

        # ---- (psi = A1@gammam) ----
        dA1 = tl.zeros([BC, BC], dtype=tl.float32)
        if HAS_MOM:
            dA1 += dpsi[:, None] * gammam[None, :]
            dgammam += tl.sum(A1 * dpsi[:, None], axis=0)

        # ---- (US=U@S0c, UM=U@M0c) ----
        dU = tl.dot(dUS, tl.trans(S0c), allow_tf32=ALLOW_TF32)
        dS0c += tl.dot(tl.trans(U), dUS, allow_tf32=ALLOW_TF32)
        if HAS_MOM:
            dU += tl.dot(dUM, tl.trans(M0c), allow_tf32=ALLOW_TF32)
            dM0c += tl.dot(tl.trans(U), dUM, allow_tf32=ALLOW_TF32)

        # ---- (Tmat = (I+Mscore)^-1) ----
        Tt = tl.trans(Tmat)
        dMscore = tl.where(lower_strict,
                           -tl.dot(tl.dot(Tt, dTmat, allow_tf32=ALLOW_TF32), Tt,
                                   allow_tf32=ALLOW_TF32), 0.0)

        # ---- (Mscore = where(strict, Phi*KU)) ----
        dPhi = tl.where(lower_strict, dMscore * KU, 0.0)
        dKU = tl.where(lower_strict, dMscore * Phi, 0.0)

        # ---- (QK = Q@K^T) ----
        dQ += tl.dot(dQK, K, allow_tf32=ALLOW_TF32)
        dK += tl.dot(tl.trans(dQK), Q, allow_tf32=ALLOW_TF32)

        # ---- (KU = U@K^T) ----
        dU += tl.dot(dKU, K, allow_tf32=ALLOW_TF32)
        dK += tl.dot(tl.trans(dKU), U, allow_tf32=ALLOW_TF32)

        # ---- (Phi/Phiout) ----
        if HAS_MOM:
            # Phi = A1@A2 ; Phiout = B1@A2
            dA2 = tl.dot(tl.trans(A1), dPhi, allow_tf32=ALLOW_TF32)
            dA2 += tl.dot(tl.trans(B1), dPhiout, allow_tf32=ALLOW_TF32)
            dA1 += tl.dot(dPhi, tl.trans(A2), allow_tf32=ALLOW_TF32)
            dB1 += tl.dot(dPhiout, tl.trans(A2), allow_tf32=ALLOW_TF32)
        else:
            dA1 += dPhi
            dB1 += dPhiout

        # ---- (A2 = where(incl, DAm)) ----
        dDA = tl.where(lower_incl, dB1, 0.0)
        # A1 = where(strict, DA*inv_decay)
        dDA += tl.where(lower_strict, dA1 * inv_decay[:, None], 0.0)
        dinv_decay = tl.sum(tl.where(lower_strict, dA1 * DA, 0.0), axis=1)

        # ---- exp-factor adjoints -> dG / dGm ----
        DD = dDA * DA
        dG += tl.sum(DD, axis=1) - tl.sum(DD, axis=0)
        dG += dgamma * gamma
        dG += ddecay_prev * decay_prev
        if HAS_MOM:
            dDAm = tl.where(lower_incl, dA2, 0.0)
            DDm = dDAm * DAm
            dGm += tl.sum(DDm, axis=1) - tl.sum(DDm, axis=0)
            # Scs = sum(kfacG*gammam)
            dkfacG += dScs * gammam
            dgammam += dScs * kfacG
            # gammam = exp(Gm): fold all gammam adjoints (psi/psiout/Scs) into dGm
            dGm += dgammam * gammam
            # kmfac = exp(min(Gm_last-Gm,0)): d/dGm_last=+kmfac, d/dGm=-kmfac
            dGm_last_s += tl.sum(dkmfac * kmfac)
            dGm += -dkmfac * kmfac
            # gammam_last folded into dGm_last_s by the M_new carry term above.
            # Wm / w (cross-chunk S momentum weighting)
            dEij = tl.where(upper_incl, dw[:, None] * Wm, 0.0)
            dG_last_s += tl.sum(dEij)
            dG += -tl.sum(dEij, axis=0)
            dGm += tl.sum(dEij, axis=0) - tl.sum(dEij, axis=1)
        # kfacG = exp(min(G_last-G,0))  (both paths)
        dG_last_s += tl.sum(dkfacG * kfacG)
        dG += -dkfacG * kfacG

        # ---- assemble dg / dgm (g via inclusive cumsum + G_last + direct terms) ----
        is_last = (cidx == (C - 1))
        dG += tl.where(is_last, dG_last_s, 0.0)
        dg_cum = tl.sum(tl.where(upper_incl, dG[None, :], 0.0), axis=1)
        dg = dg_cum - ddecay_prev * decay_prev - dinv_decay * inv_decay
        if HAS_MOM:
            dGm += tl.where(is_last, dGm_last_s, 0.0)
            dgm = tl.sum(tl.where(upper_incl, dGm[None, :], 0.0), axis=1)
        else:
            dgm = tl.zeros([BC], dtype=tl.float32)

        # ---- raw input grads ----
        dk = dK + dU * E
        de = dU * K
        dv = dP * Wg
        dwg = dP * Vv

        tl.store(dK_ptr + kn_off, dk.to(dK_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dQ_ptr + kn_off, dQ.to(dQ_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dE_ptr + kn_off, de.to(dE_ptr.dtype.element_ty), mask=kn_mask)
        tl.store(dV_ptr + pv_off, dv.to(dV_ptr.dtype.element_ty), mask=pv_mask)
        tl.store(dW_ptr + pv_off, dwg.to(dW_ptr.dtype.element_ty), mask=pv_mask)
        tl.store(dG_ptr + g_off, dg.to(dG_ptr.dtype.element_ty), mask=c_mask)
        if HAS_MOM:
            tl.store(dGM_ptr + g_off, dgm.to(dGM_ptr.dtype.element_ty), mask=c_mask)

        dS = dS0c
        dMs = dM0c


# ---------------------------------------------------------------------------
# Eager heavy-ball reference (REAL per-token recurrence; ground truth for parity).
# ---------------------------------------------------------------------------
def refit_eager_reference(k, v, q, decay, erase, write, mu,
                          log_gates=False, S0=None, M0=None):
    """Pure-PyTorch per-token `refit` recurrence. Inputs [B,T,H,*].

    decay/mu: [B,T,H] (decay g_t, momentum μ_t) in (0,1] if log_gates=False, else
    the LOG-gates (g_t=log decay, gm_t=log μ). erase: [B,T,H,N]; write: [B,T,H,V].
    Returns (out [B,T,H,V], S_final [B,H,N,V], M_final [B,H,N,V]).
    At μ≡0 this is e88_torch_reference(linear_state=True, raw_write=False).
    """
    B, T, H, N = k.shape
    Vd = v.shape[-1]
    dev, dt = k.device, torch.float32
    k = k.to(dt); v = v.to(dt); q = q.to(dt)
    erase = erase.to(dt); write = write.to(dt)
    g = decay.to(dt) if log_gates else decay.to(dt).clamp_min(1e-30).log()
    gm = mu.to(dt) if log_gates else mu.to(dt).clamp_min(1e-30).log()
    g = g.clamp_min(_GLOG_FLOOR); gm = gm.clamp_min(_GLOG_FLOOR)
    dec = g.exp(); mom = gm.exp()
    S = torch.zeros(B, H, N, Vd, device=dev, dtype=dt) if S0 is None else S0.to(dt).clone()
    M = torch.zeros(B, H, N, Vd, device=dev, dtype=dt) if M0 is None else M0.to(dt).clone()
    out = torch.empty(B, T, H, Vd, device=dev, dtype=dt)
    for t in range(T):
        u_t = (k[:, t] * erase[:, t])                       # [B,H,N]
        p_t = (v[:, t] * write[:, t])                       # [B,H,V]
        delta = p_t - torch.einsum('bhnv,bhn->bhv', S, u_t)  # [B,H,V]
        r = torch.einsum('bhn,bhv->bhnv', k[:, t], delta)    # [B,H,N,V]
        M = mom[:, t, :, None, None] * M + r
        S = dec[:, t, :, None, None] * S + M
        out[:, t] = torch.einsum('bhnv,bhn->bhv', S, q[:, t])
    return out, S, M


def _refit_forward_only(k, v, q, glog, gmlog, erase, write, chunk_size,
                        newton_steps=None, has_mom=True):
    """Direct forward-kernel call (no autograd). For forward-parity testing."""
    B, T, H, N = k.shape
    Vd = v.shape[-1]
    C = int(chunk_size)
    assert N <= 64 and Vd <= 64
    pad = (-T) % C
    if pad:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
        erase = torch.nn.functional.pad(erase, (0, 0, 0, 0, 0, pad))
        write = torch.nn.functional.pad(write, (0, 0, 0, 0, 0, pad))
        glog = torch.nn.functional.pad(glog, (0, 0, 0, pad), value=0.0)
        gmlog = torch.nn.functional.pad(gmlog, (0, 0, 0, pad), value=0.0)
    Tp = T + pad
    NC = Tp // C
    k = k.contiguous(); q = q.contiguous(); v = v.contiguous()
    U = (k * erase).contiguous()
    Pw = (v * write).contiguous()
    glog = glog.clamp_min(_GLOG_FLOOR).contiguous()
    gmlog = gmlog.clamp_min(_GLOG_FLOOR).contiguous()
    out = torch.empty((B, Tp, H, Vd), device=k.device, dtype=k.dtype)
    S_entry = torch.empty((B, H, NC, N, Vd), device=k.device, dtype=torch.float32)
    M_entry = torch.empty((B, H, NC, N, Vd), device=k.device, dtype=torch.float32)
    S_final = torch.empty((B, H, N, Vd), device=k.device, dtype=k.dtype)
    BN, BV, BC = _next_pow2(N), _next_pow2(Vd), _next_pow2(C)
    newton = max(1, (C - 1).bit_length()) if newton_steps is None else int(newton_steps)
    allow_tf32 = k.dtype in (torch.bfloat16, torch.float16)
    _refit_fwd_kernel[(B * H,)](
        k, q, U, Pw, glog, gmlog, out, S_entry, M_entry, S_final,
        B=B, T=Tp, H=H, N=N, V=Vd, C=C, NC=NC,
        BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
        HAS_MOM=has_mom, num_warps=4,
    )
    return out[:, :T].contiguous(), S_final


# ---------------------------------------------------------------------------
# autograd.Function — fused fwd+bwd. Inputs/outputs mirror e97_delta_chunked_triton
# with the added momentum gate `mu`. log_decay=True: `decay` and `mu` ARE the
# LOG-gates (g, gm); the backward returns grad wrt the log-gates directly (no
# d/decay division -> safe as gates -> 0). newton_steps=K exposes the inner-step
# count (default ceil(log2 C) = exact chunk refit).
# ---------------------------------------------------------------------------
class RefitChunkedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, q, decay, erase, write, mu, chunk_size,
                log_decay, has_mom, newton_steps):
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
            decay = torch.nn.functional.pad(decay, (0, 0, 0, pad),
                                            value=(0.0 if log_decay else 1.0))
            mu = torch.nn.functional.pad(mu, (0, 0, 0, pad),
                                         value=(0.0 if log_decay else 1.0))
        Tp = T + pad
        NC = Tp // C
        k = k.contiguous(); q = q.contiguous(); v = v.contiguous()
        erase = erase.contiguous(); write = write.contiguous()
        U = (k * erase).contiguous()
        Pw = (v * write).contiguous()
        glog = (decay if log_decay else decay.clamp_min(1e-9).log())
        gmlog = (mu if log_decay else mu.clamp_min(1e-9).log())
        clamp_g = glog < _GLOG_FLOOR
        clamp_gm = gmlog < _GLOG_FLOOR
        glog = glog.clamp_min(_GLOG_FLOOR).contiguous()
        gmlog = gmlog.clamp_min(_GLOG_FLOOR).contiguous()

        out = torch.empty((B, Tp, H, Vd), device=k.device, dtype=k.dtype)
        S_entry = torch.empty((B, H, NC, N, Vd), device=k.device, dtype=torch.float32)
        M_entry = torch.empty((B, H, NC, N, Vd), device=k.device, dtype=torch.float32)
        S_final = torch.empty((B, H, N, Vd), device=k.device, dtype=k.dtype)
        BN, BV, BC = _next_pow2(N), _next_pow2(Vd), _next_pow2(C)
        newton = (max(1, (C - 1).bit_length()) if newton_steps is None
                  else int(newton_steps))
        allow_tf32 = k.dtype in (torch.bfloat16, torch.float16)
        _refit_fwd_kernel[(B * H,)](
            k, q, U, Pw, glog, gmlog, out, S_entry, M_entry, S_final,
            B=B, T=Tp, H=H, N=N, V=Vd, C=C, NC=NC,
            BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            HAS_MOM=bool(has_mom), num_warps=4,
        )
        ctx.save_for_backward(k, q, v, erase, write, glog, gmlog,
                              S_entry, M_entry, clamp_g, clamp_gm)
        ctx.shape = (B, T, Tp, H, N, Vd, C, NC, BN, BV, BC, newton, allow_tf32)
        ctx.log_decay = bool(log_decay)
        ctx.has_mom = bool(has_mom)
        return out[:, :T].contiguous(), S_final

    @staticmethod
    def backward(ctx, dout, dSfinal):
        (k, q, v, erase, write, glog, gmlog,
         S_entry, M_entry, clamp_g, clamp_gm) = ctx.saved_tensors
        (B, T, Tp, H, N, Vd, C, NC, BN, BV, BC, newton, allow_tf32) = ctx.shape
        has_mom = ctx.has_mom
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
        dglog = torch.empty_like(glog)
        dgmlog = torch.empty_like(gmlog)
        _refit_bwd_kernel[(B * H,)](
            k, q, erase, v, write, glog, gmlog, S_entry, M_entry, dout, dSfinal,
            dk, dq, de, dv, dw, dglog, dgmlog,
            B=B, T=Tp, H=H, N=N, V=Vd, C=C, NC=NC,
            BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            HAS_MOM=bool(has_mom), num_warps=4, num_stages=1,
        )
        # floored-gate steps are constant wrt upstream params -> zero their grad.
        dglog = dglog.masked_fill(clamp_g, 0.0)
        ddecay = dglog if ctx.log_decay else dglog / glog.exp().clamp_min(1e-9)
        if has_mom:
            dgmlog = dgmlog.masked_fill(clamp_gm, 0.0)
            dmu = dgmlog if ctx.log_decay else dgmlog / gmlog.exp().clamp_min(1e-9)
        else:
            dmu = torch.zeros_like(gmlog)
        sl = slice(0, T)
        return (dk[:, sl], dv[:, sl], dq[:, sl], ddecay[:, sl],
                de[:, sl], dw[:, sl], dmu[:, sl], None, None, None, None)


_BWD_CHUNK_DEFAULT = 32


def refit_chunked_triton(k, v, q, decay, erase_gate, value_write_gate, mu,
                         chunk_size=_BWD_CHUNK_DEFAULT, log_decay=False,
                         has_mom=True, newton_steps=None):
    """Autograd-enabled fused chunked `refit` (momentum-delta TTT) cell.

    Returns (out [B,T,H,V], S_final [B,H,N,V]); grads flow to
    k, v, q, decay, erase_gate, value_write_gate, mu.

    has_mom=False routes the EXACT e97 gated-delta path (zero momentum overhead,
    mu ignored) — the delta-rule = one-inner-step special case. newton_steps=K
    exposes the inner-step count (None => exact chunk refit ceil(log2 C)).
    log_decay=True: `decay`/`mu` are the LOG-gates; grads returned wrt the logs.
    """
    return RefitChunkedFn.apply(k, v, q, decay, erase_gate, value_write_gate, mu,
                                chunk_size, log_decay, has_mom, newton_steps)
