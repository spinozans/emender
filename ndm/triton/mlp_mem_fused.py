"""Fused sequential-scan Triton kernels for the `mlp-mem` head (NONLIN_MEMORY_SPEC.md).

The recurrent STATE of this head is the parameter set of a tiny 1-hidden-layer MLP
``M_theta : R^N -> R^V`` carried per (batch, head):

    theta = (W1 [HID, N], W2 [V, HID])           # the fast weights
    M_theta(x) = W2 @ tanh(W1 @ x)

Per token the memory takes ONE gated inner gradient-descent step on the inner
reconstruction loss ``l_t(theta) = 1/2 ||M_theta(k_t) - v_t||^2`` with an
input-dependent inner learning-rate ``eta_t`` (write strength) and a multiplicative
forget gate ``gamma_t`` (spec sec 2-3):

    pre  = W1 @ k_t ; h = tanh(pre) ; yhat = W2 @ h ; e = yhat - v_t      # forward @ key
    g2   = W2^T @ e ; sp = 1 - h*h ; delta = g2 * sp                      # inner backprop
    W2   = gamma_t * W2 - eta_t * outer(e, h)                            # gated rank-1 write
    W1   = gamma_t * W1 - eta_t * outer(delta, k_t)                      # gated rank-1 write
    out_t = W2 @ tanh(W1 @ q_t)                                          # read @ query (post-write)

This transition is NON-affine in theta (delta couples through W2 and tanh'), so it is
NON-associative => there is NO chunked/parallel scan; the forward MUST be a sequential
per-token recurrence (spec sec 4). GDN's linear matrix memory is the degenerate
no-hidden-layer + linear-sigma corner of this cell (spec sec 2.3).

Two REAL fused @triton.jit kernels (no torch in the hot path), modelled on
``e88_triton_forward`` (sequential SRAM-resident state + sparse forward checkpoints)
and ``e88_triton_backward`` / ``e97_chunked_autograd`` (reverse-replay BPTT that
recomputes per-segment intermediates from sparse checkpoints and threads the
state-gradient ``dtheta`` across steps in registers):

  * ``_mlp_mem_fwd_kernel`` — one program per (b, head). Sequential time loop with
    W1/W2 resident in registers; saves theta every CKPT_INTERVAL steps to a sparse
    checkpoint buffer and writes theta_final.
  * ``_mlp_mem_bwd_kernel`` — one program per (b, head). Reverse-segment loop: reload
    the segment-entry checkpoint, forward-replay K steps caching theta into a
    per-program HBM scratch, then walk the K steps in reverse applying the exact
    per-step VJP (the closed-form second-order adjoint of the inner gradient step;
    for a 1-hidden tanh MLP every term is finite and bounded, no iterative inverse).

State is real-valued (no complex pairing -> no real/imag split). Scope: N, V, HID <= 64.
The PyTorch reference ``mlp_mem_torch_reference`` below defines the SAME recurrence and
is used ONLY for the parity gate in ``tests/test_mlp_mem_triton.py`` (it is not on any
experiment/training path — those run the fused kernels exclusively).
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


DEFAULT_CKPT_INTERVAL = 16


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(16, p)


# ---------------------------------------------------------------------------
# Forward kernel — sequential scan, sparse theta checkpoints.
# One program per (b, head). State W1 [HID,N], W2 [V,HID] resident in registers.
# ---------------------------------------------------------------------------
@triton.jit
def _mlp_mem_fwd_kernel(
    K_ptr, Q_ptr, Vv_ptr, Eta_ptr, Gam_ptr,   # [B,T,NH,*] inputs (eta,gam = [B,T,NH])
    Out_ptr,                                   # [B,T,NH,V]
    W1ck_ptr, W2ck_ptr,                        # [num_ck,B,NH,HID,N] / [num_ck,B,NH,V,HID]
    W1f_ptr, W2f_ptr,                          # [B,NH,HID,N] / [B,NH,V,HID] final state
    B: tl.constexpr, T: tl.constexpr, NH: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr, HID: tl.constexpr,
    BN: tl.constexpr, BV: tl.constexpr, BH: tl.constexpr,
    CKPT: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NH
    h = pid % NH

    nidx = tl.arange(0, BN)
    vidx = tl.arange(0, BV)
    hidx = tl.arange(0, BH)
    n_mask = nidx < N
    v_mask = vidx < V
    h_mask = hidx < HID

    m_w1 = h_mask[:, None] & n_mask[None, :]      # [HID, N]
    m_w2 = v_mask[:, None] & h_mask[None, :]       # [V, HID]

    W1 = tl.zeros([BH, BN], dtype=tl.float32)      # [HID, N]  (S0 = 0)
    W2 = tl.zeros([BV, BH], dtype=tl.float32)      # [V, HID]

    # checkpoint slot 0 = initial (zero) state
    w1ck0 = (((0 * B + b) * NH + h) * HID + hidx[:, None]) * N + nidx[None, :]
    w2ck0 = (((0 * B + b) * NH + h) * V + vidx[:, None]) * HID + hidx[None, :]
    tl.store(W1ck_ptr + w1ck0, W1.to(W1ck_ptr.dtype.element_ty), mask=m_w1)
    tl.store(W2ck_ptr + w2ck0, W2.to(W2ck_ptr.dtype.element_ty), mask=m_w2)

    for t in range(T):
        t_i64 = tl.full([1], t, dtype=tl.int64)
        kn_off = ((b * T + (t_i64)) * NH + h) * N + nidx
        v_off = ((b * T + (t_i64)) * NH + h) * V + vidx
        g_off = (b * T + t_i64) * NH + h
        k_t = tl.load(K_ptr + kn_off, mask=n_mask, other=0.0).to(tl.float32)   # [N]
        q_t = tl.load(Q_ptr + kn_off, mask=n_mask, other=0.0).to(tl.float32)   # [N]
        v_t = tl.load(Vv_ptr + v_off, mask=v_mask, other=0.0).to(tl.float32)   # [V]
        eta = tl.load(Eta_ptr + g_off).to(tl.float32)
        gam = tl.load(Gam_ptr + g_off).to(tl.float32)

        # forward at the key, at theta_{t-1}
        pre = tl.sum(W1 * k_t[None, :], axis=1)          # [HID]
        hh = tl.where(h_mask, 2.0 * tl.sigmoid(2.0 * pre) - 1.0, 0.0)
        yhat = tl.sum(W2 * hh[None, :], axis=1)          # [V]
        e = yhat - v_t                                   # [V]
        g2 = tl.sum(W2 * e[:, None], axis=0)             # [HID]  W2^T @ e
        sp = 1.0 - hh * hh                               # [HID]
        delta = g2 * sp                                  # [HID]

        # gated rank-1 writes -> theta_t
        W2 = gam * W2 - eta * (e[:, None] * hh[None, :])           # [V,HID]
        W1 = gam * W1 - eta * (delta[:, None] * k_t[None, :])      # [HID,N]

        # read at the query, at theta_t (post-write)
        preq = tl.sum(W1 * q_t[None, :], axis=1)         # [HID]
        hq = tl.where(h_mask, 2.0 * tl.sigmoid(2.0 * preq) - 1.0, 0.0)
        out = tl.sum(W2 * hq[None, :], axis=1)           # [V]
        tl.store(Out_ptr + v_off, out.to(Out_ptr.dtype.element_ty), mask=v_mask)

        is_ck = ((t + 1) % CKPT) == 0
        if is_ck:
            slot = tl.full([1], (t + 1) // CKPT, dtype=tl.int64)
            w1c = (((slot * B + b) * NH + h) * HID + hidx[:, None]) * N + nidx[None, :]
            w2c = (((slot * B + b) * NH + h) * V + vidx[:, None]) * HID + hidx[None, :]
            tl.store(W1ck_ptr + w1c, W1.to(W1ck_ptr.dtype.element_ty), mask=m_w1)
            tl.store(W2ck_ptr + w2c, W2.to(W2ck_ptr.dtype.element_ty), mask=m_w2)

    w1f = ((b * NH + h) * HID + hidx[:, None]) * N + nidx[None, :]
    w2f = ((b * NH + h) * V + vidx[:, None]) * HID + hidx[None, :]
    tl.store(W1f_ptr + w1f, W1.to(W1f_ptr.dtype.element_ty), mask=m_w1)
    tl.store(W2f_ptr + w2f, W2.to(W2f_ptr.dtype.element_ty), mask=m_w2)


# ---------------------------------------------------------------------------
# Backward kernel — reverse-replay BPTT, fully fused. One program per (b, head).
# Threads dtheta = (dW1, dW2) across steps in registers. Per segment: reload the
# entry checkpoint, replay K steps caching theta into per-program scratch, then
# walk the K steps in reverse applying the exact per-step VJP.
# ---------------------------------------------------------------------------
@triton.jit
def _mlp_mem_bwd_kernel(
    K_ptr, Q_ptr, Vv_ptr, Eta_ptr, Gam_ptr,    # [B,T,NH,*] forward inputs
    W1ck_ptr, W2ck_ptr,                          # sparse checkpoints
    W1scr_ptr, W2scr_ptr,                        # per-program replay scratch (fp32)
    dOut_ptr,                                    # [B,T,NH,V]
    dW1f_ptr, dW2f_ptr,                          # [B,NH,HID,N]/[B,NH,V,HID] grad of final state
    dK_ptr, dQ_ptr, dV_ptr, dEta_ptr, dGam_ptr,  # outputs
    B: tl.constexpr, T: tl.constexpr, NH: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr, HID: tl.constexpr,
    BN: tl.constexpr, BV: tl.constexpr, BH: tl.constexpr,
    CKPT: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NH
    h = pid % NH

    nidx = tl.arange(0, BN)
    vidx = tl.arange(0, BV)
    hidx = tl.arange(0, BH)
    n_mask = nidx < N
    v_mask = vidx < V
    h_mask = hidx < HID
    m_w1 = h_mask[:, None] & n_mask[None, :]      # [HID, N]
    m_w2 = v_mask[:, None] & h_mask[None, :]       # [V, HID]

    # scratch layout: [num_progs, (CKPT+1), tile]. W1 tile = BH*BN, W2 tile = BV*BH.
    w1_tile = BH * BN
    w2_tile = BV * BH
    w1_base = pid * (CKPT + 1) * w1_tile
    w2_base = pid * (CKPT + 1) * w2_tile
    w1_inner = hidx[:, None] * BN + nidx[None, :]
    w2_inner = vidx[:, None] * BH + hidx[None, :]

    # dtheta carry, init from grad of final state
    dw1f = ((b * NH + h) * HID + hidx[:, None]) * N + nidx[None, :]
    dw2f = ((b * NH + h) * V + vidx[:, None]) * HID + hidx[None, :]
    dW1 = tl.load(dW1f_ptr + dw1f, mask=m_w1, other=0.0).to(tl.float32)    # [HID,N]
    dW2 = tl.load(dW2f_ptr + dw2f, mask=m_w2, other=0.0).to(tl.float32)    # [V,HID]

    num_seg = T // CKPT
    for seg_rev in range(num_seg):
        seg = num_seg - 1 - seg_rev
        seg_i64 = tl.full([1], seg, dtype=tl.int64)

        # load segment-entry checkpoint (state before step seg*CKPT)
        w1c = (((seg_i64 * B + b) * NH + h) * HID + hidx[:, None]) * N + nidx[None, :]
        w2c = (((seg_i64 * B + b) * NH + h) * V + vidx[:, None]) * HID + hidx[None, :]
        W1 = tl.load(W1ck_ptr + w1c, mask=m_w1, other=0.0).to(tl.float32)
        W2 = tl.load(W2ck_ptr + w2c, mask=m_w2, other=0.0).to(tl.float32)

        # slot 0 = entry state (theta_{t-1} for the first step of the segment)
        tl.store(W1scr_ptr + w1_base + 0 * w1_tile + w1_inner, W1, mask=m_w1)
        tl.store(W2scr_ptr + w2_base + 0 * w2_tile + w2_inner, W2, mask=m_w2)

        # ---- forward-replay K steps, caching post-step theta into scratch ----
        for j in range(CKPT):
            t = seg * CKPT + j
            t_i64 = tl.full([1], t, dtype=tl.int64)
            kn_off = ((b * T + t_i64) * NH + h) * N + nidx
            v_off = ((b * T + t_i64) * NH + h) * V + vidx
            g_off = (b * T + t_i64) * NH + h
            k_t = tl.load(K_ptr + kn_off, mask=n_mask, other=0.0).to(tl.float32)
            v_t = tl.load(Vv_ptr + v_off, mask=v_mask, other=0.0).to(tl.float32)
            eta = tl.load(Eta_ptr + g_off).to(tl.float32)
            gam = tl.load(Gam_ptr + g_off).to(tl.float32)

            pre = tl.sum(W1 * k_t[None, :], axis=1)
            hh = tl.where(h_mask, 2.0 * tl.sigmoid(2.0 * pre) - 1.0, 0.0)
            yhat = tl.sum(W2 * hh[None, :], axis=1)
            e = yhat - v_t
            g2 = tl.sum(W2 * e[:, None], axis=0)
            sp = 1.0 - hh * hh
            delta = g2 * sp
            W2 = gam * W2 - eta * (e[:, None] * hh[None, :])
            W1 = gam * W1 - eta * (delta[:, None] * k_t[None, :])

            tl.store(W1scr_ptr + w1_base + (j + 1) * w1_tile + w1_inner, W1, mask=m_w1)
            tl.store(W2scr_ptr + w2_base + (j + 1) * w2_tile + w2_inner, W2, mask=m_w2)

        # ---- reverse walk over the K steps ----
        for j_rev in range(CKPT):
            j = CKPT - 1 - j_rev
            t = seg * CKPT + j
            t_i64 = tl.full([1], t, dtype=tl.int64)
            kn_off = ((b * T + t_i64) * NH + h) * N + nidx
            v_off = ((b * T + t_i64) * NH + h) * V + vidx
            g_off = (b * T + t_i64) * NH + h
            k_t = tl.load(K_ptr + kn_off, mask=n_mask, other=0.0).to(tl.float32)
            q_t = tl.load(Q_ptr + kn_off, mask=n_mask, other=0.0).to(tl.float32)
            v_t = tl.load(Vv_ptr + v_off, mask=v_mask, other=0.0).to(tl.float32)
            eta = tl.load(Eta_ptr + g_off).to(tl.float32)
            gam = tl.load(Gam_ptr + g_off).to(tl.float32)
            dout = tl.load(dOut_ptr + v_off, mask=v_mask, other=0.0).to(tl.float32)

            # theta_{t-1} (pre-state) from scratch slot j
            W1 = tl.load(W1scr_ptr + w1_base + j * w1_tile + w1_inner, mask=m_w1, other=0.0)
            W2 = tl.load(W2scr_ptr + w2_base + j * w2_tile + w2_inner, mask=m_w2, other=0.0)

            # recompute per-step forward intermediates at theta_{t-1}
            pre = tl.sum(W1 * k_t[None, :], axis=1)
            hh = tl.where(h_mask, 2.0 * tl.sigmoid(2.0 * pre) - 1.0, 0.0)
            yhat = tl.sum(W2 * hh[None, :], axis=1)
            e = yhat - v_t
            g2 = tl.sum(W2 * e[:, None], axis=0)
            sp = 1.0 - hh * hh
            delta = g2 * sp
            # post-write theta_t
            W2p = gam * W2 - eta * (e[:, None] * hh[None, :])
            W1p = gam * W1 - eta * (delta[:, None] * k_t[None, :])
            preq = tl.sum(W1p * q_t[None, :], axis=1)
            hq = tl.where(h_mask, 2.0 * tl.sigmoid(2.0 * preq) - 1.0, 0.0)

            # incoming adjoints of the post-write state
            dW1p = dW1
            dW2p = dW2

            # ---- read path: out = W2p @ hq ; hq = tanh(W1p @ q) ----
            dW2p = dW2p + dout[:, None] * hq[None, :]            # [V,HID]
            dhq = tl.sum(W2p * dout[:, None], axis=0)            # [HID]  W2p^T @ dout
            dpreq = dhq * (1.0 - hq * hq)                        # [HID]
            dW1p = dW1p + dpreq[:, None] * q_t[None, :]          # [HID,N]
            dq = tl.sum(W1p * dpreq[:, None], axis=0)            # [N]  W1p^T @ dpreq

            # ---- write path: W2p = gam W2 - eta e h^T ; W1p = gam W1 - eta delta k^T ----
            dgam = tl.sum(tl.where(m_w2, dW2p * W2, 0.0)) + tl.sum(tl.where(m_w1, dW1p * W1, 0.0))
            edh = tl.sum(dW2p * hh[None, :], axis=1)             # [V]  dW2p @ h
            de_w = -eta * edh                                   # [V]
            dh_w = -eta * tl.sum(dW2p * e[:, None], axis=0)      # [HID]  dW2p^T @ e
            dk1 = tl.sum(dW1p * k_t[None, :], axis=1)            # [HID]  dW1p @ k
            ddelta = -eta * dk1                                  # [HID]
            dk_w = -eta * tl.sum(dW1p * delta[:, None], axis=0)  # [N]  dW1p^T @ delta
            deta = -(tl.sum(e * edh) + tl.sum(delta * dk1))
            # decay path into the previous state
            dW1 = gam * dW1p                                     # [HID,N]
            dW2 = gam * dW2p                                     # [V,HID]

            # ---- inner forward path at the key (the second-order adjoint) ----
            # delta = g2 * sp
            dg2 = ddelta * sp
            dsp = ddelta * g2
            # g2 = W2^T @ e
            dW2 = dW2 + e[:, None] * dg2[None, :]                # [V,HID]
            de = de_w + tl.sum(W2 * dg2[None, :], axis=1)        # [V]  (W2 @ dg2)
            # sp = 1 - h*h
            dh = dh_w + dsp * (-2.0 * hh)                        # [HID]
            # e = yhat - v
            dyhat = de
            dv = -de
            # yhat = W2 @ h
            dW2 = dW2 + dyhat[:, None] * hh[None, :]             # [V,HID]
            dh = dh + tl.sum(W2 * dyhat[:, None], axis=0)        # [HID]  W2^T @ dyhat
            # h = tanh(pre)
            dpre = dh * (1.0 - hh * hh)                          # [HID]
            # pre = W1 @ k
            dW1 = dW1 + dpre[:, None] * k_t[None, :]             # [HID,N]
            dk = dk_w + tl.sum(W1 * dpre[:, None], axis=0)       # [N]  W1^T @ dpre

            tl.store(dK_ptr + kn_off, dk.to(dK_ptr.dtype.element_ty), mask=n_mask)
            tl.store(dQ_ptr + kn_off, dq.to(dQ_ptr.dtype.element_ty), mask=n_mask)
            tl.store(dV_ptr + v_off, dv.to(dV_ptr.dtype.element_ty), mask=v_mask)
            tl.store(dEta_ptr + g_off, deta.to(dEta_ptr.dtype.element_ty))
            tl.store(dGam_ptr + g_off, dgam.to(dGam_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# autograd.Function gluing the two fused kernels.
# ---------------------------------------------------------------------------
class MlpMemFn(torch.autograd.Function):
    """Fused sequential mlp-mem scan. fwd + bwd are both @triton.jit kernels."""

    @staticmethod
    def forward(ctx, k, q, v, eta, gamma, hid, ckpt_interval):
        assert k.is_cuda, "mlp-mem kernels require CUDA"
        B, T, NH, N = k.shape
        V = v.shape[-1]
        HID = int(hid)
        C = int(ckpt_interval)
        assert N <= 64 and V <= 64 and HID <= 64, "mlp-mem kernel supports N,V,HID<=64"
        pad = (-T) % C
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
            # identity step on the pad: eta=0 (no write), gamma=1 (no decay)
            eta = torch.nn.functional.pad(eta, (0, 0, 0, pad), value=0.0)
            gamma = torch.nn.functional.pad(gamma, (0, 0, 0, pad), value=1.0)
        Tp = T + pad
        num_ck = Tp // C + 1
        k = k.contiguous(); q = q.contiguous(); v = v.contiguous()
        eta = eta.contiguous(); gamma = gamma.contiguous()

        BN, BV, BH = _next_pow2(N), _next_pow2(V), _next_pow2(HID)
        out = torch.empty((B, Tp, NH, V), device=k.device, dtype=k.dtype)
        W1ck = torch.empty((num_ck, B, NH, HID, N), device=k.device, dtype=torch.float32)
        W2ck = torch.empty((num_ck, B, NH, V, HID), device=k.device, dtype=torch.float32)
        W1f = torch.empty((B, NH, HID, N), device=k.device, dtype=k.dtype)
        W2f = torch.empty((B, NH, V, HID), device=k.device, dtype=k.dtype)

        _mlp_mem_fwd_kernel[(B * NH,)](
            k, q, v, eta, gamma, out, W1ck, W2ck, W1f, W2f,
            B=B, T=Tp, NH=NH, N=N, V=V, HID=HID,
            BN=BN, BV=BV, BH=BH, CKPT=C, num_warps=4,
        )
        ctx.save_for_backward(k, q, v, eta, gamma, W1ck, W2ck)
        ctx.shape = (B, T, Tp, NH, N, V, HID, C, num_ck, BN, BV, BH)
        return out[:, :T].contiguous(), W1f, W2f

    @staticmethod
    def backward(ctx, dout, dW1f, dW2f):
        k, q, v, eta, gamma, W1ck, W2ck = ctx.saved_tensors
        (B, T, Tp, NH, N, V, HID, C, num_ck, BN, BV, BH) = ctx.shape
        if dout.shape[1] != Tp:
            dout = torch.nn.functional.pad(dout, (0, 0, 0, 0, 0, Tp - dout.shape[1]))
        dout = dout.contiguous()
        if dW1f is None:
            dW1f = torch.zeros((B, NH, HID, N), device=k.device, dtype=torch.float32)
        else:
            dW1f = dW1f.float().contiguous()
        if dW2f is None:
            dW2f = torch.zeros((B, NH, V, HID), device=k.device, dtype=torch.float32)
        else:
            dW2f = dW2f.float().contiguous()

        dk = torch.empty_like(k)
        dq = torch.empty_like(q)
        dv = torch.empty_like(v)
        deta = torch.empty_like(eta)
        dgam = torch.empty_like(gamma)
        n_prog = B * NH
        W1scr = torch.empty(n_prog * (C + 1) * BH * BN, device=k.device, dtype=torch.float32)
        W2scr = torch.empty(n_prog * (C + 1) * BV * BH, device=k.device, dtype=torch.float32)

        _mlp_mem_bwd_kernel[(n_prog,)](
            k, q, v, eta, gamma, W1ck, W2ck, W1scr, W2scr, dout, dW1f, dW2f,
            dk, dq, dv, deta, dgam,
            B=B, T=Tp, NH=NH, N=N, V=V, HID=HID,
            BN=BN, BV=BV, BH=BH, CKPT=C, num_warps=4, num_stages=1,
        )
        sl = slice(0, T)
        # grad order matches forward args: (k, q, v, eta, gamma, hid, ckpt_interval)
        return (dk[:, sl], dq[:, sl], dv[:, sl], deta[:, sl], dgam[:, sl], None, None)


def mlp_mem_triton(k, q, v, eta, gamma, hid: int,
                   ckpt_interval: int = DEFAULT_CKPT_INTERVAL):
    """Autograd-enabled fused sequential mlp-mem scan.

    Args (all [B,T,NH,*], S0 = 0):
        k, q : [B,T,NH,N] key / query.
        v    : [B,T,NH,V] target value.
        eta  : [B,T,NH]   inner learning-rate / write-strength gate (>= 0).
        gamma: [B,T,NH]   forget gate in (0,1].
        hid  : inner hidden width HID (the memory's capacity knob).

    Returns (out [B,T,NH,V], W1_final [B,NH,HID,N], W2_final [B,NH,V,HID]).
    Gradients flow to k, q, v, eta, gamma. The hot path is the two @triton.jit
    kernels — no torch fallback.
    """
    return MlpMemFn.apply(k, q, v, eta, gamma, int(hid), ckpt_interval)


# ---------------------------------------------------------------------------
# PyTorch reference (parity gate ONLY — not used in any experiment/training path).
# ---------------------------------------------------------------------------
def mlp_mem_torch_reference(k, q, v, eta, gamma, hid: int):
    """Eager reference for the mlp-mem recurrence. Mirrors the kernel math exactly.

    Inputs [B,T,NH,*] (S0 = 0). Returns (out, W1_final, W2_final). Differentiable in
    plain PyTorch so the test can finite-difference / autograd-check the fused grads.
    """
    B, T, NH, N = k.shape
    V = v.shape[-1]
    HID = int(hid)
    dtype = k.dtype
    dev = k.device
    W1 = torch.zeros(B, NH, HID, N, device=dev, dtype=torch.float32)
    W2 = torch.zeros(B, NH, V, HID, device=dev, dtype=torch.float32)
    outs = []
    for t in range(T):
        k_t = k[:, t].float()                    # [B,NH,N]
        q_t = q[:, t].float()
        v_t = v[:, t].float()                    # [B,NH,V]
        eta_t = eta[:, t].float().unsqueeze(-1)  # [B,NH,1]
        gam_t = gamma[:, t].float().unsqueeze(-1)

        pre = torch.einsum('bhin,bhn->bhi', W1, k_t)         # [B,NH,HID]
        hh = torch.tanh(pre)
        yhat = torch.einsum('bhvi,bhi->bhv', W2, hh)         # [B,NH,V]
        e = yhat - v_t                                       # [B,NH,V]
        g2 = torch.einsum('bhvi,bhv->bhi', W2, e)            # [B,NH,HID]  W2^T e
        sp = 1.0 - hh * hh
        delta = g2 * sp                                      # [B,NH,HID]

        W2 = gam_t.unsqueeze(-1) * W2 - eta_t.unsqueeze(-1) * (e.unsqueeze(-1) * hh.unsqueeze(-2))
        W1 = gam_t.unsqueeze(-1) * W1 - eta_t.unsqueeze(-1) * (delta.unsqueeze(-1) * k_t.unsqueeze(-2))

        preq = torch.einsum('bhin,bhn->bhi', W1, q_t)        # [B,NH,HID]
        hq = torch.tanh(preq)
        out = torch.einsum('bhvi,bhi->bhv', W2, hq)          # [B,NH,V]
        outs.append(out.to(dtype))
    out = torch.stack(outs, dim=1)                           # [B,T,NH,V]
    return out, W1.to(dtype), W2.to(dtype)


__all__ = ["mlp_mem_triton", "mlp_mem_torch_reference", "MlpMemFn",
           "DEFAULT_CKPT_INTERVAL"]
