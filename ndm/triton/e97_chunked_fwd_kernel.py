"""Fused Triton forward kernel for the chunked E97 split-edit delta recurrence.

One Triton program per (batch, head). The program loops over the T/C chunks
*in-kernel*, threading the recurrent state ``S`` ([N, V]) in registers/SRAM across
chunks (this is the legitimate cross-chunk recurrence — the only sequential axis).
Everything inside a chunk is a small ``tl.dot`` matmul on tensor cores, and the
``[C,C]`` intermediates never touch HBM. This is the fused analogue of the staged
PyTorch implementation in ``e97_chunked.py`` and is modeled on FLA's
chunk_gated_delta_rule (chunk + WY representation + in-kernel state scan).

The unit-lower-triangular inverse ``T = (I + M)^{-1}`` (the UT / WY transform) is
computed by Newton-Schulz iteration ``X <- X (2I - L X)``. Because ``M`` is
strictly-lower (nilpotent, ``M^C = 0``), the iteration residual is ``M^{2^k}`` and
becomes *exactly* zero after ``ceil(log2(C))`` steps — so 6 steps invert a 64x64
block exactly, using only ``tl.dot`` matmuls (no O(C) row-substitution loop, no
O(T) sequential scan).

Numerics: all matmul operands are kept in fp32 and the dots run on the TF32
tensor-core path (``allow_tf32=True``) for bf16/fp16 inputs — TF32 carries 10
mantissa bits (more than bf16's 8) so it meets the bf16-parity tolerance while
saturating the tensor cores, and it avoids Triton's "inconsistent return type"
limitation on dtype-switching jit helpers. For fp32 inputs we set
``allow_tf32=False`` for the exact-parity path.

Scope: N, V <= 64 (block-padded to powers of two), linear state (the GDN-2-class
cell — see e97_chunked.py for why per-step tanh is non-chunkable). Forward only;
training uses the autograd PyTorch-chunked path (parity-verified fwd+bwd). This
kernel is the throughput/inference fast-path and the benchmark subject.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e97_chunk_fwd_kernel(
    K_ptr, Q_ptr, U_ptr, P_ptr, G_ptr,   # inputs [B,T,H,*] (U=erase*k, P=wgate*v, G=log decay [B,T,H])
    Out_ptr,                              # [B,T,H,V]
    Sfinal_ptr,                           # [B,H,N,V]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, NC: tl.constexpr,
    BN: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NEWTON_STEPS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,   # TF32 tensor-core dots for bf16/fp16 inputs
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    cidx = tl.arange(0, BC)            # chunk-row index [0,C)
    nidx = tl.arange(0, BN)            # state key index [0,N)
    vidx = tl.arange(0, BV)            # value index [0,V)
    c_mask = cidx < C
    n_mask = nidx < N
    v_mask = vidx < V

    # causal masks over the chunk
    lower_incl = (cidx[:, None] >= cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    lower_strict = (cidx[:, None] > cidx[None, :]) & c_mask[:, None] & c_mask[None, :]
    eyeC = (cidx[:, None] == cidx[None, :]).to(tl.float32) * c_mask[:, None]

    # recurrent state S [N, V], starts at 0 (initial_state not supported here)
    S = tl.zeros([BN, BV], dtype=tl.float32)

    for c in range(NC):
        t0 = c * C
        # ---- load chunk tensors ----
        # K,Q,U: [C,N]  P: [C,V]   g: [C]
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

        # ---- within-chunk cumulative log-decay ----
        # G[t] = sum_{i<=t} g_i (inclusive). cumsum over the chunk axis.
        G = tl.cumsum(g, axis=0)                  # [C]
        gamma = tl.exp(G)                          # [C]
        decay_prev = tl.exp(G - g)                 # exp(G[t-1]) = exp(G[t]-g[t])
        inv_decay = tl.exp(-g)                     # exp(-g[t])
        G_last = tl.sum(tl.where(c_mask, g, 0.0))  # G[C-1] = total log decay in chunk

        # ---- pairwise score matrices (tensor cores) ----
        Kt = tl.trans(K)                           # [N,C]
        QK = tl.dot(Q, Kt, allow_tf32=ALLOW_TF32)          # [C,C] q_t . k_j
        KU = tl.dot(U, Kt, allow_tf32=ALLOW_TF32)          # [C,C] u_t . k_j

        # decay weighting. DA[t,j]=exp(G[t]-G[j]); A=tril_incl(DA*QK);
        # M=tril_strict(DA*inv_decay[t]*KU)
        DA = tl.exp(G[:, None] - G[None, :])
        A = tl.where(lower_incl, DA * QK, 0.0)     # [C,C]
        M = tl.where(lower_strict, DA * inv_decay[:, None] * KU, 0.0)

        # ---- Newton-Schulz inverse of L = I + M (exact for nilpotent M) ----
        L = eyeC + M
        X = eyeC - M                               # first-order seed (exact to order 2)
        for _ in range(NEWTON_STEPS):
            LX = tl.dot(L, X, allow_tf32=ALLOW_TF32)
            X = tl.dot(X, 2.0 * eyeC - LX, allow_tf32=ALLOW_TF32)
        Tmat = X                                   # (I+M)^{-1}, [C,C]

        # ---- delta vectors: (I+M) Delta = P - decay_prev * (U @ S) ----
        US = tl.dot(U, S, allow_tf32=ALLOW_TF32)                          # [C,V]
        RHS = P - decay_prev[:, None] * US
        Delta = tl.dot(Tmat, RHS, allow_tf32=ALLOW_TF32)                  # [C,V]

        # ---- chunk output: gamma * (Q @ S) + A @ Delta ----
        QS = tl.dot(Q, S, allow_tf32=ALLOW_TF32)                          # [C,V]
        out = gamma[:, None] * QS + tl.dot(A, Delta, allow_tf32=ALLOW_TF32)
        tl.store(Out_ptr + pv_off, out.to(Out_ptr.dtype.element_ty), mask=pv_mask)

        # ---- state update: S_new = gamma_last * S + Kscaled^T @ Delta ----
        kfac = tl.exp(G_last - G)                  # [C]
        Kscaled = K * kfac[:, None]                # [C,N]
        SdK = tl.dot(tl.trans(Kscaled), Delta, allow_tf32=ALLOW_TF32)     # [N,V]
        S = tl.exp(G_last) * S + SdK

    # write final state
    sf_off = ((b * H + h) * N + nidx[:, None]) * V + vidx[None, :]
    sf_mask = n_mask[:, None] & v_mask[None, :]
    tl.store(Sfinal_ptr + sf_off, S.to(Sfinal_ptr.dtype.element_ty), mask=sf_mask)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(16, p)


def e97_delta_chunked_fwd_triton(
    k: torch.Tensor,            # [B,T,H,N]
    v: torch.Tensor,            # [B,T,H,V]
    q: torch.Tensor,            # [B,T,H,N]
    decay: torch.Tensor,        # [B,T,H]
    erase_gate: torch.Tensor,   # [B,T,H,N]
    value_write_gate: torch.Tensor,  # [B,T,H,V]
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused-Triton forward for the chunked E97 split-edit delta recurrence.

    Returns (out [B,T,H,V], S_final [B,H,N,V]). Forward only. Linear state, S0=0.
    """
    B, T, H, N = k.shape
    Vd = v.shape[-1]
    C = int(chunk_size)
    assert N <= 64 and Vd <= 64, "kernel supports N,V<=64"

    pad = (-T) % C
    if pad:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
        erase_gate = torch.nn.functional.pad(erase_gate, (0, 0, 0, 0, 0, pad))
        value_write_gate = torch.nn.functional.pad(value_write_gate, (0, 0, 0, 0, 0, pad))
        decay = torch.nn.functional.pad(decay, (0, 0, 0, pad), value=1.0)
    Tp = T + pad
    NC = Tp // C

    U = (k * erase_gate).contiguous()
    Pw = (v * value_write_gate).contiguous()
    kc = k.contiguous()
    qc = q.contiguous()
    glog = decay.clamp_min(1e-9).log().contiguous()

    out = torch.empty((B, Tp, H, Vd), device=k.device, dtype=k.dtype)
    S_final = torch.empty((B, H, N, Vd), device=k.device, dtype=k.dtype)

    BN = _next_pow2(N)
    BV = _next_pow2(Vd)
    BC = _next_pow2(C)
    newton = max(1, (C - 1).bit_length())   # ceil(log2(C)) steps -> exact

    allow_tf32 = k.dtype in (torch.bfloat16, torch.float16)
    _e97_chunk_fwd_kernel[(B * H,)](
        kc, qc, U, Pw, glog, out, S_final,
        B=B, T=Tp, H=H, N=N, V=Vd, C=C, NC=NC,
        BN=BN, BV=BV, BC=BC, NEWTON_STEPS=newton,
        ALLOW_TF32=allow_tf32,
        num_warps=4,
    )
    return out[:, :T].contiguous(), S_final
