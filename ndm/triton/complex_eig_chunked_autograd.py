"""Fused chunked-parallel COMPLEX-eigenvalue gated-delta scan — Triton fwd + bwd.

This is the REAL fused kernel for the complex-eigenvalue (rotation-scaling)
gated-delta head (``paper/review/COMPLEX_EIG_HEAD_SPEC.md``).  It replaces the
pure-``torch.complex`` reference (``complex_eig_chunked.py``) in the training hot
path — that reference is kept ONLY as the parity ground truth.

Triton has no native complex type, so every complex matrix is carried as a pair
of real tiles ``(x_r, x_i)`` and every complex matmul is the four real
``tl.dot`` products

    (A_r + iA_i)(B_r + iB_i) = (A_r B_r - A_i B_i) + i(A_r B_i + A_i B_r).

The eigenvalue ``lambda = r e^{i theta}`` is folded into per-channel cumulative
log-magnitude ``G`` and cumulative phase ``Phi`` (the S5/LRU diagonal scan), so
the intra-chunk delta system is the same strictly-lower-triangular nilpotent
``(I+M)`` solved by complex Newton-Schulz as in ``e97_chunked_autograd``, and the
cross-chunk carry is a per-channel complex diagonal.  Everything is real
``tl.dot`` on tensor cores; the ``[C,C]`` intermediates never touch HBM.

Two ``@triton.jit`` kernels:
  * ``_cplx_fwd_kernel`` — forward chunk scan; also writes the per-chunk ENTRY
    state (needed only by the backward).
  * ``_cplx_bwd_kernel`` — reverse chunk scan; recomputes the forward
    intermediates and applies the chunked VJP (the real-valued vector-Jacobian
    product of the real/imag-decomposed forward), threading the complex state
    gradient ``dS`` across chunks in registers.

The autograd.Function (``ComplexEigChunkedFn``) takes the L2-normed, query-scaled
real/imag key/query pairs (the cheap elementwise preprocessing — pairing, complex
L2-norm, ``1/sqrt(P)`` scale — is done in real arithmetic OUTSIDE the kernel so
autograd chains it), and returns ``(out_r, out_i, S_final_r, S_final_i)``.

Reductions (verified in tests): theta=0 -> real-positive decay (GDN regime);
theta=pi -> reflection / negative eigenvalue.  S0 is assumed zero (the head never
passes a nonzero initial state).
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


# Magnitude-exponent guard for the decay-absorbed inverse key KR = k / cumdecay.
# cumdecay = exp(Gprev) with Gprev <= 0, so 1/cumdecay = exp(-Gprev) overflows fp32
# for a run of strong-decay steps. The TRUE (I+M)/A entries are always bounded
# (they pair a <=1 factor with this >=1 one); clamping the intermediate exponent
# only affects channels whose within-chunk product already underflowed to ~0.
# Mirrors _INV_DECAY_GUARD in complex_eig_chunked.py and _GLOG_FLOOR in e97.
_INV_DECAY_GUARD = float(_os.environ.get('CPLX_INV_DECAY_GUARD', '80.0'))


# ---------------------------------------------------------------------------
# Forward kernel — one program per (batch, head). Carries the complex state
# (S_r, S_i) [P,V] in registers across the chunk loop; writes per-chunk entry
# state for the backward and the final state.
# ---------------------------------------------------------------------------
@triton.jit
def _cplx_fwd_kernel(
    Kr_ptr, Ki_ptr, Qr_ptr, Qi_ptr,   # [B,T,H,P] real/imag key & query (L2-normed)
    Vv_ptr,                            # [B,T,H,V] real value
    LR_ptr, TH_ptr,                    # [B,T,H,P] per-channel log-magnitude / phase
    Be_ptr,                            # [B,T,H]   scalar write strength
    Or_ptr, Oi_ptr,                    # [B,T,H,V] complex output
    Ser_ptr, Sei_ptr,                  # [B,H,NC,P,V] per-chunk entry state (r,i)
    Sfr_ptr, Sfi_ptr,                  # [B,H,P,V] final state (r,i)
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    P: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NC: tl.constexpr,
    BP: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NEWTON_STEPS: tl.constexpr, ALLOW_TF32: tl.constexpr, GUARD: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    cidx = tl.arange(0, BC)
    pidx = tl.arange(0, BP)
    vidx = tl.arange(0, BV)
    c_mask = cidx < C
    p_mask = pidx < P
    v_mask = vidx < V
    lower_incl = (cidx[:, None] >= cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    lower_strict = (cidx[:, None] > cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    eyeC = (cidx[:, None] == cidx[None, :]).to(tl.float32) * c_mask[:, None]

    Sr = tl.zeros([BP, BV], dtype=tl.float32)
    Si = tl.zeros([BP, BV], dtype=tl.float32)
    for c in range(NC):
        t0 = c * C
        cp_off = ((b * T + (t0 + cidx[:, None])) * H + h) * P + pidx[None, :]
        cp_mask = c_mask[:, None] & p_mask[None, :]
        Kr = tl.load(Kr_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        Ki = tl.load(Ki_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        Qr = tl.load(Qr_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        Qi = tl.load(Qi_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        lr = tl.load(LR_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        th = tl.load(TH_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        cv_off = ((b * T + (t0 + cidx[:, None])) * H + h) * V + vidx[None, :]
        cv_mask = c_mask[:, None] & v_mask[None, :]
        Vv = tl.load(Vv_ptr + cv_off, mask=cv_mask, other=0.0).to(tl.float32)
        be_off = (b * T + (t0 + cidx)) * H + h
        be = tl.load(Be_ptr + be_off, mask=c_mask, other=0.0).to(tl.float32)

        # save entry state for this chunk (before update)
        se_off = (((b * H + h) * NC + c) * P + pidx[:, None]) * V + vidx[None, :]
        se_mask = p_mask[:, None] & v_mask[None, :]
        tl.store(Ser_ptr + se_off, Sr.to(Ser_ptr.dtype.element_ty), mask=se_mask)
        tl.store(Sei_ptr + se_off, Si.to(Sei_ptr.dtype.element_ty), mask=se_mask)

        # cumulative log-magnitude G and phase Phi (inclusive within chunk).
        G = tl.cumsum(lr, axis=0)
        Phi = tl.cumsum(th, axis=0)
        Gprev = G - lr
        Phiprev = Phi - th
        # cp = exp(Gprev) cis(Phiprev) (|cp|<=1); c = exp(G) cis(Phi).
        eGp = tl.exp(Gprev)
        cphi = tl.cos(Phiprev)
        sphi = tl.sin(Phiprev)
        cp_r = eGp * cphi
        cp_i = eGp * sphi
        eG = tl.exp(G)
        cPhi = tl.cos(Phi)
        sPhi = tl.sin(Phi)
        c_r = eG * cPhi
        c_i = eG * sPhi
        # inverse cumulative (magnitude-guarded): inv_cp = exp(-Gprev) cis(-Phiprev)
        inv_mag = tl.exp(tl.minimum(-Gprev, GUARD))
        invcp_r = inv_mag * cphi
        invcp_i = -inv_mag * sphi

        # decay-absorbed keys:
        #   KL = conj(K)*cp ;  KR = K*inv_cp ;  QL = conj(Q)*c
        KL_r = Kr * cp_r + Ki * cp_i
        KL_i = Kr * cp_i - Ki * cp_r
        KR_r = Kr * invcp_r - Ki * invcp_i
        KR_i = Kr * invcp_i + Ki * invcp_r
        QL_r = Qr * c_r + Qi * c_i
        QL_i = Qr * c_i - Qi * c_r

        # pairwise complex dots [C,C]: KLKR[t,j]=KL_t.KR_j ; QLKR[t,j]=QL_t.KR_j
        KRrT = tl.trans(KR_r)
        KRiT = tl.trans(KR_i)
        KLKR_r = tl.dot(KL_r, KRrT, allow_tf32=ALLOW_TF32) - tl.dot(KL_i, KRiT, allow_tf32=ALLOW_TF32)
        KLKR_i = tl.dot(KL_r, KRiT, allow_tf32=ALLOW_TF32) + tl.dot(KL_i, KRrT, allow_tf32=ALLOW_TF32)
        QLKR_r = tl.dot(QL_r, KRrT, allow_tf32=ALLOW_TF32) - tl.dot(QL_i, KRiT, allow_tf32=ALLOW_TF32)
        QLKR_i = tl.dot(QL_r, KRiT, allow_tf32=ALLOW_TF32) + tl.dot(QL_i, KRrT, allow_tf32=ALLOW_TF32)

        be_col = be[:, None]
        M_r = tl.where(lower_strict, be_col * KLKR_r, 0.0)
        M_i = tl.where(lower_strict, be_col * KLKR_i, 0.0)
        A_r = tl.where(lower_incl, QLKR_r, 0.0)
        A_i = tl.where(lower_incl, QLKR_i, 0.0)

        # complex Newton-Schulz X <- X(2I - (I+M)X), exact for nilpotent M.
        ImM_r = eyeC + M_r
        ImM_i = M_i
        X_r = eyeC - M_r
        X_i = -M_i
        for _ in range(NEWTON_STEPS):
            LX_r = tl.dot(ImM_r, X_r, allow_tf32=ALLOW_TF32) - tl.dot(ImM_i, X_i, allow_tf32=ALLOW_TF32)
            LX_i = tl.dot(ImM_r, X_i, allow_tf32=ALLOW_TF32) + tl.dot(ImM_i, X_r, allow_tf32=ALLOW_TF32)
            T_r = 2.0 * eyeC - LX_r
            T_i = -LX_i
            nX_r = tl.dot(X_r, T_r, allow_tf32=ALLOW_TF32) - tl.dot(X_i, T_i, allow_tf32=ALLOW_TF32)
            nX_i = tl.dot(X_r, T_i, allow_tf32=ALLOW_TF32) + tl.dot(X_i, T_r, allow_tf32=ALLOW_TF32)
            X_r = nX_r
            X_i = nX_i

        # RHS_p = beta*V (imag 0) ; RHS_u = beta*KL
        RHSp_r = be_col * Vv
        RHSu_r = be_col * KL_r
        RHSu_i = be_col * KL_i
        # W_p = X@RHS_p ; W_u = X@RHS_u
        Wp_r = tl.dot(X_r, RHSp_r, allow_tf32=ALLOW_TF32)
        Wp_i = tl.dot(X_i, RHSp_r, allow_tf32=ALLOW_TF32)
        Wu_r = tl.dot(X_r, RHSu_r, allow_tf32=ALLOW_TF32) - tl.dot(X_i, RHSu_i, allow_tf32=ALLOW_TF32)
        Wu_i = tl.dot(X_r, RHSu_i, allow_tf32=ALLOW_TF32) + tl.dot(X_i, RHSu_r, allow_tf32=ALLOW_TF32)

        # --- within-chunk output using current entry state S (=S0c) ---
        # Delta = W_p - W_u@S ; O = QL@S + A@Delta ; out = O
        WuS_r = tl.dot(Wu_r, Sr, allow_tf32=ALLOW_TF32) - tl.dot(Wu_i, Si, allow_tf32=ALLOW_TF32)
        WuS_i = tl.dot(Wu_r, Si, allow_tf32=ALLOW_TF32) + tl.dot(Wu_i, Sr, allow_tf32=ALLOW_TF32)
        Delta_r = Wp_r - WuS_r
        Delta_i = Wp_i - WuS_i
        QLS_r = tl.dot(QL_r, Sr, allow_tf32=ALLOW_TF32) - tl.dot(QL_i, Si, allow_tf32=ALLOW_TF32)
        QLS_i = tl.dot(QL_r, Si, allow_tf32=ALLOW_TF32) + tl.dot(QL_i, Sr, allow_tf32=ALLOW_TF32)
        ADel_r = tl.dot(A_r, Delta_r, allow_tf32=ALLOW_TF32) - tl.dot(A_i, Delta_i, allow_tf32=ALLOW_TF32)
        ADel_i = tl.dot(A_r, Delta_i, allow_tf32=ALLOW_TF32) + tl.dot(A_i, Delta_r, allow_tf32=ALLOW_TF32)
        O_r = QLS_r + ADel_r
        O_i = QLS_i + ADel_i
        tl.store(Or_ptr + cv_off, O_r.to(Or_ptr.dtype.element_ty), mask=cv_mask)
        tl.store(Oi_ptr + cv_off, O_i.to(Oi_ptr.dtype.element_ty), mask=cv_mask)

        # --- cross-chunk state update: S_next = P_trans@S + P_const ---
        # gamma = chunk-total eigenvalue (per channel) = exp(Glast) cis(Philast)
        Glast = tl.sum(tl.where(c_mask[:, None], lr, 0.0), axis=0)    # [BP]
        Philast = tl.sum(tl.where(c_mask[:, None], th, 0.0), axis=0)  # [BP]
        egl = tl.exp(Glast)
        gamma_r = egl * tl.cos(Philast)
        gamma_i = egl * tl.sin(Philast)
        # KRtWu = KR^T@W_u [P,P] ; KRtWp = KR^T@W_p [P,V]
        KRtWu_r = tl.dot(KRrT, Wu_r, allow_tf32=ALLOW_TF32) - tl.dot(KRiT, Wu_i, allow_tf32=ALLOW_TF32)
        KRtWu_i = tl.dot(KRrT, Wu_i, allow_tf32=ALLOW_TF32) + tl.dot(KRiT, Wu_r, allow_tf32=ALLOW_TF32)
        KRtWp_r = tl.dot(KRrT, Wp_r, allow_tf32=ALLOW_TF32) - tl.dot(KRiT, Wp_i, allow_tf32=ALLOW_TF32)
        KRtWp_i = tl.dot(KRrT, Wp_i, allow_tf32=ALLOW_TF32) + tl.dot(KRiT, Wp_r, allow_tf32=ALLOW_TF32)
        eyeP = (pidx[:, None] == pidx[None, :]).to(tl.float32) * p_mask[:, None]
        # tmp = eyeP - KRtWu ; P_trans = diag(gamma)*tmp (gamma broadcast over rows)
        tmp_r = eyeP - KRtWu_r
        tmp_i = -KRtWu_i
        gr = gamma_r[:, None]
        gi = gamma_i[:, None]
        Pt_r = gr * tmp_r - gi * tmp_i
        Pt_i = gr * tmp_i + gi * tmp_r
        Pc_r = gr * KRtWp_r - gi * KRtWp_i
        Pc_i = gr * KRtWp_i + gi * KRtWp_r
        # S_next = P_trans@S + P_const
        PtS_r = tl.dot(Pt_r, Sr, allow_tf32=ALLOW_TF32) - tl.dot(Pt_i, Si, allow_tf32=ALLOW_TF32)
        PtS_i = tl.dot(Pt_r, Si, allow_tf32=ALLOW_TF32) + tl.dot(Pt_i, Sr, allow_tf32=ALLOW_TF32)
        Sr = PtS_r + Pc_r
        Si = PtS_i + Pc_i

    sf_off = ((b * H + h) * P + pidx[:, None]) * V + vidx[None, :]
    sf_mask = p_mask[:, None] & v_mask[None, :]
    tl.store(Sfr_ptr + sf_off, Sr.to(Sfr_ptr.dtype.element_ty), mask=sf_mask)
    tl.store(Sfi_ptr + sf_off, Si.to(Sfi_ptr.dtype.element_ty), mask=sf_mask)


# ---------------------------------------------------------------------------
# Backward kernel — reverse chunk scan. Recomputes the forward per-chunk
# quantities, then applies the chunked VJP. dS (complex state grad) is threaded
# in registers, initialised from dS_final at the last chunk.
# ---------------------------------------------------------------------------
@triton.jit
def _cplx_bwd_kernel(
    Kr_ptr, Ki_ptr, Qr_ptr, Qi_ptr,   # [B,T,H,P]
    Vv_ptr,                            # [B,T,H,V]
    LR_ptr, TH_ptr,                    # [B,T,H,P]
    Be_ptr,                            # [B,T,H]
    Ser_ptr, Sei_ptr,                  # [B,H,NC,P,V] entry states
    dOr_ptr, dOi_ptr,                  # [B,T,H,V] output cotangents
    dSfr_ptr, dSfi_ptr,                # [B,H,P,V] final-state cotangents
    dKr_ptr, dKi_ptr, dQr_ptr, dQi_ptr,   # [B,T,H,P] grads
    dVv_ptr,                              # [B,T,H,V]
    dLR_ptr, dTH_ptr,                     # [B,T,H,P]
    dBe_ptr,                              # [B,T,H]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    P: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NC: tl.constexpr,
    BP: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NEWTON_STEPS: tl.constexpr, ALLOW_TF32: tl.constexpr, GUARD: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    cidx = tl.arange(0, BC)
    pidx = tl.arange(0, BP)
    vidx = tl.arange(0, BV)
    c_mask = cidx < C
    p_mask = pidx < P
    v_mask = vidx < V
    lower_incl = (cidx[:, None] >= cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    lower_strict = (cidx[:, None] > cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    upper_incl = (cidx[:, None] <= cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    eyeC = (cidx[:, None] == cidx[None, :]).to(tl.float32) * c_mask[:, None]
    eyeP = (pidx[:, None] == pidx[None, :]).to(tl.float32) * p_mask[:, None]

    # dS = grad wrt state leaving the last chunk (= S_final)
    sf_off = ((b * H + h) * P + pidx[:, None]) * V + vidx[None, :]
    sf_mask = p_mask[:, None] & v_mask[None, :]
    dSr = tl.load(dSfr_ptr + sf_off, mask=sf_mask, other=0.0).to(tl.float32)
    dSi = tl.load(dSfi_ptr + sf_off, mask=sf_mask, other=0.0).to(tl.float32)

    for cc in range(NC):
        c = NC - 1 - cc
        t0 = c * C
        cp_off = ((b * T + (t0 + cidx[:, None])) * H + h) * P + pidx[None, :]
        cp_mask = c_mask[:, None] & p_mask[None, :]
        Kr = tl.load(Kr_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        Ki = tl.load(Ki_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        Qr = tl.load(Qr_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        Qi = tl.load(Qi_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        lr = tl.load(LR_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        th = tl.load(TH_ptr + cp_off, mask=cp_mask, other=0.0).to(tl.float32)
        cv_off = ((b * T + (t0 + cidx[:, None])) * H + h) * V + vidx[None, :]
        cv_mask = c_mask[:, None] & v_mask[None, :]
        Vv = tl.load(Vv_ptr + cv_off, mask=cv_mask, other=0.0).to(tl.float32)
        be_off = (b * T + (t0 + cidx)) * H + h
        be = tl.load(Be_ptr + be_off, mask=c_mask, other=0.0).to(tl.float32)
        dOr = tl.load(dOr_ptr + cv_off, mask=cv_mask, other=0.0).to(tl.float32)
        dOi = tl.load(dOi_ptr + cv_off, mask=cv_mask, other=0.0).to(tl.float32)
        se_off = (((b * H + h) * NC + c) * P + pidx[:, None]) * V + vidx[None, :]
        se_mask = p_mask[:, None] & v_mask[None, :]
        S0r = tl.load(Ser_ptr + se_off, mask=se_mask, other=0.0).to(tl.float32)
        S0i = tl.load(Sei_ptr + se_off, mask=se_mask, other=0.0).to(tl.float32)

        # ===== recompute forward intermediates =====
        G = tl.cumsum(lr, axis=0)
        Phi = tl.cumsum(th, axis=0)
        Gprev = G - lr
        Phiprev = Phi - th
        eGp = tl.exp(Gprev)
        cphi = tl.cos(Phiprev)
        sphi = tl.sin(Phiprev)
        cp_r = eGp * cphi
        cp_i = eGp * sphi
        eG = tl.exp(G)
        cPhi = tl.cos(Phi)
        sPhi = tl.sin(Phi)
        c_r = eG * cPhi
        c_i = eG * sPhi
        neg_clamped = (-Gprev) > GUARD
        inv_mag = tl.exp(tl.minimum(-Gprev, GUARD))
        invcp_r = inv_mag * cphi
        invcp_i = -inv_mag * sphi

        KL_r = Kr * cp_r + Ki * cp_i
        KL_i = Kr * cp_i - Ki * cp_r
        KR_r = Kr * invcp_r - Ki * invcp_i
        KR_i = Kr * invcp_i + Ki * invcp_r
        QL_r = Qr * c_r + Qi * c_i
        QL_i = Qr * c_i - Qi * c_r

        KRrT = tl.trans(KR_r)
        KRiT = tl.trans(KR_i)
        KLKR_r = tl.dot(KL_r, KRrT, allow_tf32=ALLOW_TF32) - tl.dot(KL_i, KRiT, allow_tf32=ALLOW_TF32)
        KLKR_i = tl.dot(KL_r, KRiT, allow_tf32=ALLOW_TF32) + tl.dot(KL_i, KRrT, allow_tf32=ALLOW_TF32)
        QLKR_r = tl.dot(QL_r, KRrT, allow_tf32=ALLOW_TF32) - tl.dot(QL_i, KRiT, allow_tf32=ALLOW_TF32)
        QLKR_i = tl.dot(QL_r, KRiT, allow_tf32=ALLOW_TF32) + tl.dot(QL_i, KRrT, allow_tf32=ALLOW_TF32)

        be_col = be[:, None]
        M_r = tl.where(lower_strict, be_col * KLKR_r, 0.0)
        M_i = tl.where(lower_strict, be_col * KLKR_i, 0.0)
        A_r = tl.where(lower_incl, QLKR_r, 0.0)
        A_i = tl.where(lower_incl, QLKR_i, 0.0)

        ImM_r = eyeC + M_r
        ImM_i = M_i
        X_r = eyeC - M_r
        X_i = -M_i
        for _ in range(NEWTON_STEPS):
            LX_r = tl.dot(ImM_r, X_r, allow_tf32=ALLOW_TF32) - tl.dot(ImM_i, X_i, allow_tf32=ALLOW_TF32)
            LX_i = tl.dot(ImM_r, X_i, allow_tf32=ALLOW_TF32) + tl.dot(ImM_i, X_r, allow_tf32=ALLOW_TF32)
            T_r = 2.0 * eyeC - LX_r
            T_i = -LX_i
            nX_r = tl.dot(X_r, T_r, allow_tf32=ALLOW_TF32) - tl.dot(X_i, T_i, allow_tf32=ALLOW_TF32)
            nX_i = tl.dot(X_r, T_i, allow_tf32=ALLOW_TF32) + tl.dot(X_i, T_r, allow_tf32=ALLOW_TF32)
            X_r = nX_r
            X_i = nX_i

        RHSp_r = be_col * Vv
        RHSu_r = be_col * KL_r
        RHSu_i = be_col * KL_i
        Wp_r = tl.dot(X_r, RHSp_r, allow_tf32=ALLOW_TF32)
        Wp_i = tl.dot(X_i, RHSp_r, allow_tf32=ALLOW_TF32)
        Wu_r = tl.dot(X_r, RHSu_r, allow_tf32=ALLOW_TF32) - tl.dot(X_i, RHSu_i, allow_tf32=ALLOW_TF32)
        Wu_i = tl.dot(X_r, RHSu_i, allow_tf32=ALLOW_TF32) + tl.dot(X_i, RHSu_r, allow_tf32=ALLOW_TF32)

        WuS_r = tl.dot(Wu_r, S0r, allow_tf32=ALLOW_TF32) - tl.dot(Wu_i, S0i, allow_tf32=ALLOW_TF32)
        WuS_i = tl.dot(Wu_r, S0i, allow_tf32=ALLOW_TF32) + tl.dot(Wu_i, S0r, allow_tf32=ALLOW_TF32)
        Delta_r = Wp_r - WuS_r
        Delta_i = Wp_i - WuS_i

        Glast = tl.sum(tl.where(c_mask[:, None], lr, 0.0), axis=0)    # [BP]
        Philast = tl.sum(tl.where(c_mask[:, None], th, 0.0), axis=0)  # [BP]
        egl = tl.exp(Glast)
        gamma_r = egl * tl.cos(Philast)
        gamma_i = egl * tl.sin(Philast)
        KRtWu_r = tl.dot(KRrT, Wu_r, allow_tf32=ALLOW_TF32) - tl.dot(KRiT, Wu_i, allow_tf32=ALLOW_TF32)
        KRtWu_i = tl.dot(KRrT, Wu_i, allow_tf32=ALLOW_TF32) + tl.dot(KRiT, Wu_r, allow_tf32=ALLOW_TF32)
        KRtWp_r = tl.dot(KRrT, Wp_r, allow_tf32=ALLOW_TF32) - tl.dot(KRiT, Wp_i, allow_tf32=ALLOW_TF32)
        KRtWp_i = tl.dot(KRrT, Wp_i, allow_tf32=ALLOW_TF32) + tl.dot(KRiT, Wp_r, allow_tf32=ALLOW_TF32)
        tmp_r = eyeP - KRtWu_r
        tmp_i = -KRtWu_i
        gr = gamma_r[:, None]
        gi = gamma_i[:, None]

        # ===== VJP =====
        dOr_t = dOr      # dO_r = dOut_r
        dOi_t = dOi      # dO_i = dOut_i

        # --- state update VJP: S_next = P_trans@S0 + P_const ; cotangent dS ---
        # P_const = diag(gamma)*KRtWp ; P_trans = diag(gamma)*tmp ; tmp=eyeP-KRtWu
        # dP_const = dS ; dP_trans = dS@S0^H ; dS0 += P_trans^H@dS
        dPc_r = dSr
        dPc_i = dSi
        # dP_trans = dS @ S0^H  (S0 [P,V], dS [P,V], dP_trans [P,P])
        S0rT = tl.trans(S0r)
        S0iT = tl.trans(S0i)
        dPt_r = tl.dot(dSr, S0rT, allow_tf32=ALLOW_TF32) + tl.dot(dSi, S0iT, allow_tf32=ALLOW_TF32)
        dPt_i = -tl.dot(dSr, S0iT, allow_tf32=ALLOW_TF32) + tl.dot(dSi, S0rT, allow_tf32=ALLOW_TF32)
        # dS0 from P_trans@S0: P_trans^H @ dS  (recompute P_trans)
        Pt_r = gr * tmp_r - gi * tmp_i
        Pt_i = gr * tmp_i + gi * tmp_r
        PtrT = tl.trans(Pt_r)
        PtiT = tl.trans(Pt_i)
        dS0r = tl.dot(PtrT, dSr, allow_tf32=ALLOW_TF32) + tl.dot(PtiT, dSi, allow_tf32=ALLOW_TF32)
        dS0i = -tl.dot(PtiT, dSr, allow_tf32=ALLOW_TF32) + tl.dot(PtrT, dSi, allow_tf32=ALLOW_TF32)

        # P_const = diag(gamma)*KRtWp -> dgamma, dKRtWp
        dgamma_r = tl.sum(KRtWp_r * dPc_r + KRtWp_i * dPc_i, axis=1)
        dgamma_i = tl.sum(-KRtWp_i * dPc_r + KRtWp_r * dPc_i, axis=1)
        dKRtWp_r = gr * dPc_r + gi * dPc_i
        dKRtWp_i = -gi * dPc_r + gr * dPc_i
        # P_trans = diag(gamma)*tmp -> dgamma, dtmp
        dgamma_r += tl.sum(tmp_r * dPt_r + tmp_i * dPt_i, axis=1)
        dgamma_i += tl.sum(-tmp_i * dPt_r + tmp_r * dPt_i, axis=1)
        dtmp_r = gr * dPt_r + gi * dPt_i
        dtmp_i = -gi * dPt_r + gr * dPt_i
        # tmp = eyeP - KRtWu -> dKRtWu = -dtmp
        dKRtWu_r = -dtmp_r
        dKRtWu_i = -dtmp_i

        # --- output VJP: O = QL@S0 + A@Delta ---
        # dQL += dO@S0^H ; dS0 += QL^H@dO ; dA += dO@Delta^H ; dDelta += A^H@dO
        dQL_r = tl.dot(dOr_t, S0rT, allow_tf32=ALLOW_TF32) + tl.dot(dOi_t, S0iT, allow_tf32=ALLOW_TF32)
        dQL_i = -tl.dot(dOr_t, S0iT, allow_tf32=ALLOW_TF32) + tl.dot(dOi_t, S0rT, allow_tf32=ALLOW_TF32)
        QLrT = tl.trans(QL_r)
        QLiT = tl.trans(QL_i)
        dS0r += tl.dot(QLrT, dOr_t, allow_tf32=ALLOW_TF32) + tl.dot(QLiT, dOi_t, allow_tf32=ALLOW_TF32)
        dS0i += -tl.dot(QLiT, dOr_t, allow_tf32=ALLOW_TF32) + tl.dot(QLrT, dOi_t, allow_tf32=ALLOW_TF32)
        DelrT = tl.trans(Delta_r)
        DeliT = tl.trans(Delta_i)
        dA_r = tl.dot(dOr_t, DelrT, allow_tf32=ALLOW_TF32) + tl.dot(dOi_t, DeliT, allow_tf32=ALLOW_TF32)
        dA_i = -tl.dot(dOr_t, DeliT, allow_tf32=ALLOW_TF32) + tl.dot(dOi_t, DelrT, allow_tf32=ALLOW_TF32)
        dA_r = tl.where(lower_incl, dA_r, 0.0)
        dA_i = tl.where(lower_incl, dA_i, 0.0)
        ArT = tl.trans(A_r)
        AiT = tl.trans(A_i)
        dDelta_r = tl.dot(ArT, dOr_t, allow_tf32=ALLOW_TF32) + tl.dot(AiT, dOi_t, allow_tf32=ALLOW_TF32)
        dDelta_i = -tl.dot(AiT, dOr_t, allow_tf32=ALLOW_TF32) + tl.dot(ArT, dOi_t, allow_tf32=ALLOW_TF32)

        # --- Delta = W_p - W_u@S0 ---
        dWp_r = dDelta_r
        dWp_i = dDelta_i
        # dW_u += -dDelta@S0^H
        dWu_r = -(tl.dot(dDelta_r, S0rT, allow_tf32=ALLOW_TF32) + tl.dot(dDelta_i, S0iT, allow_tf32=ALLOW_TF32))
        dWu_i = -(-tl.dot(dDelta_r, S0iT, allow_tf32=ALLOW_TF32) + tl.dot(dDelta_i, S0rT, allow_tf32=ALLOW_TF32))
        # dS0 += -W_u^H@dDelta
        WurT = tl.trans(Wu_r)
        WuiT = tl.trans(Wu_i)
        dS0r += -(tl.dot(WurT, dDelta_r, allow_tf32=ALLOW_TF32) + tl.dot(WuiT, dDelta_i, allow_tf32=ALLOW_TF32))
        dS0i += -(-tl.dot(WuiT, dDelta_r, allow_tf32=ALLOW_TF32) + tl.dot(WurT, dDelta_i, allow_tf32=ALLOW_TF32))

        # --- KRtWp = KR^T@W_p ; KRtWu = KR^T@W_u  (accumulate dKR, dW_p, dW_u) ---
        # C = KRt@W ; KRt=[P,C], W=[C,*]. dKRt += dC@W^H ; dW += KRt^H@dC ; dKR += dKRt^T
        # KRtWp -> dW_p, dKRt
        WprT = tl.trans(Wp_r)
        WpiT = tl.trans(Wp_i)
        dKRt_r = tl.dot(dKRtWp_r, WprT, allow_tf32=ALLOW_TF32) + tl.dot(dKRtWp_i, WpiT, allow_tf32=ALLOW_TF32)
        dKRt_i = -tl.dot(dKRtWp_r, WpiT, allow_tf32=ALLOW_TF32) + tl.dot(dKRtWp_i, WprT, allow_tf32=ALLOW_TF32)
        # dW_p += KRt^H @ dKRtWp ; KRt^H = conj(KR) as [C,P] (KRt=[P,C]).
        dWp_r += tl.dot(KR_r, dKRtWp_r, allow_tf32=ALLOW_TF32) + tl.dot(KR_i, dKRtWp_i, allow_tf32=ALLOW_TF32)
        dWp_i += -tl.dot(KR_i, dKRtWp_r, allow_tf32=ALLOW_TF32) + tl.dot(KR_r, dKRtWp_i, allow_tf32=ALLOW_TF32)
        # KRtWu -> dW_u, dKRt
        WurT2 = tl.trans(Wu_r)
        WuiT2 = tl.trans(Wu_i)
        dKRt_r += tl.dot(dKRtWu_r, WurT2, allow_tf32=ALLOW_TF32) + tl.dot(dKRtWu_i, WuiT2, allow_tf32=ALLOW_TF32)
        dKRt_i += -tl.dot(dKRtWu_r, WuiT2, allow_tf32=ALLOW_TF32) + tl.dot(dKRtWu_i, WurT2, allow_tf32=ALLOW_TF32)
        dWu_r += tl.dot(KR_r, dKRtWu_r, allow_tf32=ALLOW_TF32) + tl.dot(KR_i, dKRtWu_i, allow_tf32=ALLOW_TF32)
        dWu_i += -tl.dot(KR_i, dKRtWu_r, allow_tf32=ALLOW_TF32) + tl.dot(KR_r, dKRtWu_i, allow_tf32=ALLOW_TF32)
        # dKRt is [P,C]; dKR += dKRt^T  ([C,P])
        dKR_r = tl.trans(dKRt_r)
        dKR_i = tl.trans(dKRt_i)

        # --- W_p = X@RHS_p (RHS_p imag 0) ; W_u = X@RHS_u (complex) ---
        # dX += dW@RHS^H ; dRHS += X^H@dW
        RHSprT = tl.trans(RHSp_r)
        dX_r = tl.dot(dWp_r, RHSprT, allow_tf32=ALLOW_TF32)
        dX_i = tl.dot(dWp_i, RHSprT, allow_tf32=ALLOW_TF32)
        XrT = tl.trans(X_r)
        XiT = tl.trans(X_i)
        dRHSp_r = tl.dot(XrT, dWp_r, allow_tf32=ALLOW_TF32) + tl.dot(XiT, dWp_i, allow_tf32=ALLOW_TF32)
        # W_u = X@RHS_u
        RHSurT = tl.trans(RHSu_r)
        RHSuiT = tl.trans(RHSu_i)
        dX_r += tl.dot(dWu_r, RHSurT, allow_tf32=ALLOW_TF32) + tl.dot(dWu_i, RHSuiT, allow_tf32=ALLOW_TF32)
        dX_i += -tl.dot(dWu_r, RHSuiT, allow_tf32=ALLOW_TF32) + tl.dot(dWu_i, RHSurT, allow_tf32=ALLOW_TF32)
        dRHSu_r = tl.dot(XrT, dWu_r, allow_tf32=ALLOW_TF32) + tl.dot(XiT, dWu_i, allow_tf32=ALLOW_TF32)
        dRHSu_i = -tl.dot(XiT, dWu_r, allow_tf32=ALLOW_TF32) + tl.dot(XrT, dWu_i, allow_tf32=ALLOW_TF32)

        # --- RHS_p = beta*V ; RHS_u = beta*KL ---
        dVv = be_col * dRHSp_r
        dbeta = tl.sum(Vv * dRHSp_r, axis=1)
        dKL_r = be_col * dRHSu_r
        dKL_i = be_col * dRHSu_i
        dbeta += tl.sum(KL_r * dRHSu_r + KL_i * dRHSu_i, axis=1)

        # --- Newton inverse: X = (I+M)^{-1} ; dM = lower_strict( -X^H dX X^H ) ---
        XhdX_r = tl.dot(XrT, dX_r, allow_tf32=ALLOW_TF32) + tl.dot(XiT, dX_i, allow_tf32=ALLOW_TF32)
        XhdX_i = -tl.dot(XiT, dX_r, allow_tf32=ALLOW_TF32) + tl.dot(XrT, dX_i, allow_tf32=ALLOW_TF32)
        dY_r = -(tl.dot(XhdX_r, XrT, allow_tf32=ALLOW_TF32) - tl.dot(XhdX_i, (-XiT), allow_tf32=ALLOW_TF32))
        dY_i = -(tl.dot(XhdX_r, (-XiT), allow_tf32=ALLOW_TF32) + tl.dot(XhdX_i, XrT, allow_tf32=ALLOW_TF32))
        dM_r = tl.where(lower_strict, dY_r, 0.0)
        dM_i = tl.where(lower_strict, dY_i, 0.0)

        # --- M = lower_strict(beta*KLKR) ; A = lower_incl(QLKR) ---
        dbeta += tl.sum(KLKR_r * dM_r + KLKR_i * dM_i, axis=1)
        dKLKR_r = be_col * dM_r
        dKLKR_i = be_col * dM_i
        dQLKR_r = dA_r
        dQLKR_i = dA_i

        # --- KLKR = KL@KR^T ; QLKR = QL@KR^T  (direct component VJPs) ---
        dKL_r += tl.dot(dKLKR_r, KR_r, allow_tf32=ALLOW_TF32) + tl.dot(dKLKR_i, KR_i, allow_tf32=ALLOW_TF32)
        dKL_i += -tl.dot(dKLKR_r, KR_i, allow_tf32=ALLOW_TF32) + tl.dot(dKLKR_i, KR_r, allow_tf32=ALLOW_TF32)
        dKLKRrT = tl.trans(dKLKR_r)
        dKLKRiT = tl.trans(dKLKR_i)
        dKR_r += tl.dot(dKLKRrT, KL_r, allow_tf32=ALLOW_TF32) + tl.dot(dKLKRiT, KL_i, allow_tf32=ALLOW_TF32)
        dKR_i += -tl.dot(dKLKRrT, KL_i, allow_tf32=ALLOW_TF32) + tl.dot(dKLKRiT, KL_r, allow_tf32=ALLOW_TF32)
        dQL_r += tl.dot(dQLKR_r, KR_r, allow_tf32=ALLOW_TF32) + tl.dot(dQLKR_i, KR_i, allow_tf32=ALLOW_TF32)
        dQL_i += -tl.dot(dQLKR_r, KR_i, allow_tf32=ALLOW_TF32) + tl.dot(dQLKR_i, KR_r, allow_tf32=ALLOW_TF32)
        dQLKRrT = tl.trans(dQLKR_r)
        dQLKRiT = tl.trans(dQLKR_i)
        dKR_r += tl.dot(dQLKRrT, QL_r, allow_tf32=ALLOW_TF32) + tl.dot(dQLKRiT, QL_i, allow_tf32=ALLOW_TF32)
        dKR_i += -tl.dot(dQLKRrT, QL_i, allow_tf32=ALLOW_TF32) + tl.dot(dQLKRiT, QL_r, allow_tf32=ALLOW_TF32)

        # --- gamma feeds dc at last row: gamma = exp(Glast) cis(Philast) ---
        # dGlast/dPhilast (per channel) from dgamma; gamma=exp(Glast)cis(Philast)
        dGlast = dgamma_r * gamma_r + dgamma_i * gamma_i      # [BP]
        dPhilast = -dgamma_r * gamma_i + dgamma_i * gamma_r   # [BP]

        # --- decay-absorbed keys -> dK, dQ, dcp, dc, dinv_cp ---
        # KL = conj(K)*cp
        dKr = dKL_r * cp_r + dKL_i * cp_i
        dKi = dKL_r * cp_i - dKL_i * cp_r
        dcp_r = dKL_r * Kr - dKL_i * Ki
        dcp_i = dKL_r * Ki + dKL_i * Kr
        # KR = K*inv_cp
        dKr += dKR_r * invcp_r + dKR_i * invcp_i
        dKi += -dKR_r * invcp_i + dKR_i * invcp_r
        dinvcp_r = dKR_r * Kr + dKR_i * Ki
        dinvcp_i = -dKR_r * Ki + dKR_i * Kr
        # QL = conj(Q)*c
        dQr = dQL_r * c_r + dQL_i * c_i
        dQi = dQL_r * c_i - dQL_i * c_r
        dc_r = dQL_r * Qr - dQL_i * Qi
        dc_i = dQL_r * Qi + dQL_i * Qr
        # NOTE: gamma's grad enters lr/th directly via dGlast/dPhilast below
        # (gamma = exp(sum lr) cis(sum th)), NOT through the c[C-1] row.

        # ===== cp/c/inv_cp -> G, Phi, Gprev, Phiprev =====
        # c = exp(G) cis(Phi)
        dG_c = dc_r * c_r + dc_i * c_i
        dPhi_c = -dc_r * c_i + dc_i * c_r
        # cp = exp(Gprev) cis(Phiprev)
        dGprev = dcp_r * cp_r + dcp_i * cp_i
        dPhiprev = -dcp_r * cp_i + dcp_i * cp_r
        # inv_cp = inv_mag * cis(-Phiprev) ; inv_mag = exp(min(-Gprev,GUARD))
        dinv_mag = dinvcp_r * cphi - dinvcp_i * sphi
        dPhiprev += -inv_mag * (dinvcp_r * sphi + dinvcp_i * cphi)
        # inv_mag deriv wrt Gprev: d(exp(-Gprev))/dGprev = -inv_mag (unless clamped)
        dGprev += tl.where(neg_clamped, 0.0, -dinv_mag * inv_mag)

        # gamma -> last-row G/Phi (Glast=sum lr, Philast=sum th). Add as a direct
        # per-row contribution to dG_total / dPhi_total below via dGlast/dPhilast.

        # G inclusive cumsum of lr ; Gprev = G - lr.
        dG_total = dG_c + dGprev
        dlr_direct = -dGprev
        dPhi_total = dPhi_c + dPhiprev
        dth_direct = -dPhiprev
        # gamma's Glast/Philast = sum over rows of lr/th -> add to every row's direct grad
        dlr_direct += dGlast[None, :]
        dth_direct += dPhilast[None, :]
        # reverse cumsum along C: dlr += sum_{i>=t} dG_total[i,p]
        dlr = tl.dot(upper_incl.to(tl.float32), dG_total, allow_tf32=ALLOW_TF32) + dlr_direct
        dth = tl.dot(upper_incl.to(tl.float32), dPhi_total, allow_tf32=ALLOW_TF32) + dth_direct

        # ===== store grads =====
        tl.store(dKr_ptr + cp_off, dKr.to(dKr_ptr.dtype.element_ty), mask=cp_mask)
        tl.store(dKi_ptr + cp_off, dKi.to(dKi_ptr.dtype.element_ty), mask=cp_mask)
        tl.store(dQr_ptr + cp_off, dQr.to(dQr_ptr.dtype.element_ty), mask=cp_mask)
        tl.store(dQi_ptr + cp_off, dQi.to(dQi_ptr.dtype.element_ty), mask=cp_mask)
        tl.store(dVv_ptr + cv_off, dVv.to(dVv_ptr.dtype.element_ty), mask=cv_mask)
        tl.store(dLR_ptr + cp_off, dlr.to(dLR_ptr.dtype.element_ty), mask=cp_mask)
        tl.store(dTH_ptr + cp_off, dth.to(dTH_ptr.dtype.element_ty), mask=cp_mask)
        tl.store(dBe_ptr + be_off, dbeta.to(dBe_ptr.dtype.element_ty), mask=c_mask)

        # propagate state grad to previous chunk
        dSr = dS0r
        dSi = dS0i


class ComplexEigChunkedFn(torch.autograd.Function):
    """Fused chunked complex gated-delta scan (S0=0). fwd+bwd Triton.

    Inputs are the L2-normed, query-scaled real/imag key & query pairs. Returns
    ``(out_r, out_i, S_final_r, S_final_i)``.
    """

    @staticmethod
    def forward(ctx, kr, ki, qr, qi, v, log_r, theta, beta, chunk_size, allow_tf32):
        B, T, H, P = kr.shape
        Vd = v.shape[-1]
        C = int(chunk_size)
        assert P <= 64 and Vd <= 64, "kernel supports P,V<=64"
        pad = (-T) % C
        if pad:
            pz = (0, 0, 0, 0, 0, pad)
            kr = torch.nn.functional.pad(kr, pz)
            ki = torch.nn.functional.pad(ki, pz)
            qr = torch.nn.functional.pad(qr, pz)
            qi = torch.nn.functional.pad(qi, pz)
            v = torch.nn.functional.pad(v, pz)
            log_r = torch.nn.functional.pad(log_r, pz, value=0.0)
            theta = torch.nn.functional.pad(theta, pz, value=0.0)
            beta = torch.nn.functional.pad(beta, (0, 0, 0, pad), value=0.0)
        Tp = T + pad
        NC = Tp // C
        kr = kr.contiguous(); ki = ki.contiguous()
        qr = qr.contiguous(); qi = qi.contiguous()
        v = v.contiguous(); log_r = log_r.contiguous(); theta = theta.contiguous()
        beta = beta.contiguous()

        out_r = torch.empty((B, Tp, H, Vd), device=kr.device, dtype=torch.float32)
        out_i = torch.empty((B, Tp, H, Vd), device=kr.device, dtype=torch.float32)
        S_er = torch.empty((B, H, NC, P, Vd), device=kr.device, dtype=torch.float32)
        S_ei = torch.empty((B, H, NC, P, Vd), device=kr.device, dtype=torch.float32)
        S_fr = torch.empty((B, H, P, Vd), device=kr.device, dtype=torch.float32)
        S_fi = torch.empty((B, H, P, Vd), device=kr.device, dtype=torch.float32)
        BP, BV, BC = _next_pow2(P), _next_pow2(Vd), _next_pow2(C)
        newton = max(1, (C - 1).bit_length())
        allow_tf32 = bool(allow_tf32)
        _cplx_fwd_kernel[(B * H,)](
            kr, ki, qr, qi, v, log_r, theta, beta,
            out_r, out_i, S_er, S_ei, S_fr, S_fi,
            B=B, T=Tp, H=H, P=P, V=Vd, C=C, NC=NC,
            BP=BP, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            GUARD=_INV_DECAY_GUARD, num_warps=4,
        )
        ctx.save_for_backward(kr, ki, qr, qi, v, log_r, theta, beta, S_er, S_ei)
        ctx.shape = (B, T, Tp, H, P, Vd, C, NC, BP, BV, BC, newton, allow_tf32)
        return (out_r[:, :T].contiguous(), out_i[:, :T].contiguous(), S_fr, S_fi)

    @staticmethod
    def backward(ctx, dout_r, dout_i, dSfr, dSfi):
        kr, ki, qr, qi, v, log_r, theta, beta, S_er, S_ei = ctx.saved_tensors
        (B, T, Tp, H, P, Vd, C, NC, BP, BV, BC, newton, allow_tf32) = ctx.shape
        if dout_r.shape[1] != Tp:
            pz = (0, 0, 0, 0, 0, Tp - dout_r.shape[1])
            dout_r = torch.nn.functional.pad(dout_r, pz)
            dout_i = torch.nn.functional.pad(dout_i, pz)
        dout_r = dout_r.float().contiguous()
        dout_i = dout_i.float().contiguous()
        if dSfr is None:
            dSfr = torch.zeros((B, H, P, Vd), device=kr.device, dtype=torch.float32)
        else:
            dSfr = dSfr.float().contiguous()
        if dSfi is None:
            dSfi = torch.zeros((B, H, P, Vd), device=kr.device, dtype=torch.float32)
        else:
            dSfi = dSfi.float().contiguous()

        dkr = torch.empty_like(kr); dki = torch.empty_like(ki)
        dqr = torch.empty_like(qr); dqi = torch.empty_like(qi)
        dv = torch.empty_like(v)
        dlog_r = torch.empty_like(log_r); dtheta = torch.empty_like(theta)
        dbeta = torch.empty_like(beta)
        _cplx_bwd_kernel[(B * H,)](
            kr, ki, qr, qi, v, log_r, theta, beta, S_er, S_ei,
            dout_r, dout_i, dSfr, dSfi,
            dkr, dki, dqr, dqi, dv, dlog_r, dtheta, dbeta,
            B=B, T=Tp, H=H, P=P, V=Vd, C=C, NC=NC,
            BP=BP, BV=BV, BC=BC, NEWTON_STEPS=newton, ALLOW_TF32=allow_tf32,
            GUARD=_INV_DECAY_GUARD, num_warps=4, num_stages=1,
        )
        sl = slice(0, T)
        return (dkr[:, sl], dki[:, sl], dqr[:, sl], dqi[:, sl], dv[:, sl],
                dlog_r[:, sl], dtheta[:, sl], dbeta[:, sl], None, None)


# Default chunk: the fused complex backward holds many [C,C] complex (doubled)
# tiles live; C=32 keeps each fp32 tile at 4 KB and fits the 100 KB/SM limit on
# Ada/Ampere. The head passes chunk_size=32 by default.
_BWD_CHUNK_DEFAULT = 32


def _pair_l2_scale(x, l2norm, scale):
    """[..., N] real -> (xr, xi) [..., N/2], complex-L2-normed and optionally scaled.
    Pure real arithmetic (no torch.complex), differentiable; matches the eager
    reference (_pair_to_complex + _complex_l2norm + 1/sqrt(P) query scale)."""
    xr = x[..., 0::2].float()
    xi = x[..., 1::2].float()
    if l2norm:
        sq = (xr * xr + xi * xi).sum(dim=-1, keepdim=True)
        inv = torch.rsqrt(sq + 1e-6)
        xr = xr * inv
        xi = xi * inv
    if scale != 1.0:
        xr = xr * scale
        xi = xi * scale
    return xr.contiguous(), xi.contiguous()


def complex_gated_delta_chunked_triton(
    q, k, v, log_r, theta, beta,
    S0=None, chunk_size=_BWD_CHUNK_DEFAULT, read_mode="real", l2norm=True,
    allow_tf32=None,
):
    """Fused-Triton chunked complex gated-delta forward+backward.

    Drop-in for ``complex_gated_delta_chunked`` (same signature/semantics) but the
    hot path is the fused @triton.jit kernels — NO torch.complex. Returns
    ``(out, S_final)`` with ``S_final`` a complex64 tensor (assembled from the
    kernel's real/imag outputs only for parity reporting; not on the hot path).

    The phase/magnitude algebra (cos/sin/exp/cumsum, L2-norm) is always computed in
    fp32 for stability; ``allow_tf32`` only governs the heavy ``tl.dot`` matmuls.
    Default: TF32 tensor cores when ``q`` is bf16/fp16 (the production autocast
    path), exact fp32 when ``q`` is fp32 (the parity path) — mirrors the e97 kernel.

    S0 must be None/zero (the head never passes a nonzero initial state). For a
    nonzero S0 use the eager reference.
    """
    if S0 is not None and torch.is_tensor(S0) and S0.abs().sum().item() != 0.0:
        raise NotImplementedError("fused complex kernel assumes S0=0; use the eager reference")
    B, T, H, N = q.shape
    P = N // 2
    if allow_tf32 is None:
        allow_tf32 = q.dtype in (torch.bfloat16, torch.float16)
    # Env override: CPLX_ALLOW_TF32=0 forces full-fp32 tl.dot matmuls even on the
    # bf16 autocast path. The TF32 matmuls (~10-bit mantissa) in the Newton-Schulz
    # (I+M) inverse + decay-absorbed inverse-key products are too lossy for STABLE
    # LM training at lr 2e-3 (they diverge to NaN within ~10 optimizer steps, while
    # the full-fp32 path — matching the torch.complex reference's precision — is
    # stable). Inputs/states/accumulation are already fp32; this knob only governs
    # the heavy tl.dot precision. (task complex-eig-lm-2)
    _tf32_env = _os.environ.get('CPLX_ALLOW_TF32')
    if _tf32_env is not None:
        allow_tf32 = _tf32_env not in ('0', 'false', 'False', '')
    kr, ki = _pair_l2_scale(k, l2norm, 1.0)
    qr, qi = _pair_l2_scale(q, l2norm, P ** -0.5)
    out_r, out_i, S_fr, S_fi = ComplexEigChunkedFn.apply(
        kr, ki, qr, qi, v.float(), log_r.float(), theta.float(), beta.float(),
        int(chunk_size), bool(allow_tf32))
    if read_mode == "real":
        out = out_r
    elif read_mode == "reim":
        out = torch.cat([out_r, out_i], dim=-1)
    else:
        raise ValueError(f"unknown read_mode {read_mode!r}")
    S_final = torch.complex(S_fr, S_fi)
    return out.to(q.dtype), S_final


__all__ = [
    "complex_gated_delta_chunked_triton",
    "ComplexEigChunkedFn",
]
