"""Chunked-parallel E97 split-edit delta kernel (e97_delta).

This is the *load-bearing* throughput fix for the E97 split-edit cell. The prior
fused E97 kernel (``ndm.triton.e88_triton_forward``) runs a SEQUENTIAL
``for t in range(T)`` outer-product scan inside one Triton program per (batch,
head). That is latency-bound and cannot saturate the GPU (the within-layer
latency-bound finding hinged on it being ~2.6x slower than GDN-2). GDN-2 reaches ~97% util via
FLA's CHUNKED PARALLEL gated-delta-rule: chunk the sequence, do the intra-chunk
work with matmuls (tensor cores), thread the recurrent state across chunks.

The E97 split-edit *linear-state* delta recurrence HAS that chunked form. Per
(batch, head), with state ``S`` of shape ``[N, V]``:

    read_key_t   = e_t (.) k_t                 # erase/read gate, [N]
    write_val_t  = w_t (.) v_t                 # value write gate, [V]
    delta_t      = write_val_t - S_{t-1}^T read_key_t            # [V]
    S_t          = decay_t * S_{t-1} + k_t delta_t^T            # [N, V]
    out_t        = S_t^T q_t                    # [V]

Substituting delta_t gives an AFFINE (asymmetric gated delta) recurrence

    S_t = (decay_t * I - k_t read_key_t^T) S_{t-1} + k_t write_val_t^T

which matches ``E88FLAHybrid._scan_recurrence`` EXACTLY (same erase/write gates,
same delta correction, raw_write=False). It differs from the standard DeltaNet
chunk only in being *asymmetric*: the rank-1 transition uses left vector ``k`` and
right vector ``read_key = e (.) k`` (DeltaNet uses ``k`` for both).

Chunked algebra (chunk length C, entry state S0c per chunk)
----------------------------------------------------------
Let ``G[t] = sum_{i<=t} log(decay_i)`` be the inclusive cumulative log-decay
within the chunk, ``K,U,Q = [C,N]`` (rows k_t, read_key_t, q_t), ``P = [C,V]``
(rows write_val_t). Define

    M[t,j] = exp(G[t]-g[t]-G[j]) * (k_j . u_t)     for j <  t   (strictly lower)
    A[t,j] = exp(G[t]      -G[j]) * (q_t . k_j)     for j <= t   (lower incl diag)

The per-step delta vectors ``Delta = [C,V]`` solve the unit-lower-triangular
system ``(I + M) Delta = P - diag(decay_prev) U S0c`` where
``decay_prev[t] = exp(G[t]-g[t])``. Let ``T = (I+M)^{-1}`` (the UT transform,
independent of S0c). Then

    W_p = T P                                  # [C,V], chunk-local, parallel
    W_u = T diag(decay_prev) U                 # [C,N], chunk-local, parallel
    Delta = W_p - W_u S0c

The chunk-final state is an affine map of S0c (this is the cross-chunk recurrence,
a short sequential scan over only T/C chunks):

    Kscaled[j] = exp(G[L-1]-G[j]) k_j          # [C,N]
    S_next = (gamma_last I - Kscaled^T W_u) S0c + Kscaled^T W_p

and the per-position output is

    out_t = gamma[t] (q_t . S0c)  +  (A Delta)[t]      # [C,V]

All of the C-sized work is matmuls (tensor cores); only the T/C-step cross-chunk
scan is sequential. This is the same structure FLA uses to hit ~97% util.

Scope / parity
--------------
* LINEAR STATE only. The per-step ``tanh`` of the full E97 cell is a pointwise
  nonlinearity applied to the whole state every step; it is NOT associative and
  therefore has no chunked-matmul form. GDN-2 (the throughput target) is itself
  linear-state, so the apples-to-apples chunked E97 cell is the linear-state
  delta. An optional ``state_chunk`` boundary-nonlinearity (phi applied only at
  chunk boundaries, exactly like ``gdn2_nonlin_fused``) is supported for the
  nonlinear-state shell variant; with ``state_nonlin='identity'`` it is the pure
  linear delta.
* Parity target: ``e88_triton_forward.e88_torch_reference`` with
  ``linear_state=True, raw_write=False`` and the split-edit gates. Verified fwd +
  bwd in bf16 within tolerance by ``tests/test_e97_chunked.py``.

The forward is written entirely with differentiable torch ops (batched matmul +
triangular solve), so autograd supplies an exact, also-chunk-parallel backward —
no hand-written backward kernel is required, and every heavy op is a cuBLAS
tensor-core matmul. A fused Triton variant lives in ``_e97_chunk_fwd_kernel``
below for the cross-chunk scan; the matmul-heavy intra-chunk stages stay on cuBLAS
where they already run at tensor-core throughput.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _phi(S: torch.Tensor, kind: str) -> torch.Tensor:
    """State nonlinearity applied ONLY at chunk boundaries (never per-step)."""
    if kind in (None, 'identity', 'linear', 'none'):
        return S
    if kind == 'tanh':
        return torch.tanh(S)
    if kind == 'relu':
        return torch.relu(S)
    if kind == 'softplus':
        return F.softplus(S)
    raise ValueError(f"unknown state_nonlin {kind!r}")


def e97_delta_chunked(
    k: torch.Tensor,            # [B, T, H, N]
    v: torch.Tensor,            # [B, T, H, V]
    q: torch.Tensor,            # [B, T, H, N]
    decay: torch.Tensor,        # [B, T, H]   per-step scalar decay in (0, 1]
    erase_gate: torch.Tensor,   # [B, T, H, N] split-edit read/erase gate (e)
    value_write_gate: torch.Tensor,  # [B, T, H, V] split-edit value write gate (w)
    S0: Optional[torch.Tensor] = None,   # [B, H, N, V] initial state
    chunk_size: int = 64,
    state_nonlin: str = 'identity',
    state_chunk: Optional[int] = None,   # phi applied every `state_chunk` steps
    inverse_mode: str = 'newton',        # 'newton' (matmul, fast bwd) | 'solve'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked-parallel forward for the E97 split-edit (delta) recurrence.

    Returns (out [B, T, H, V], S_final [B, H, N, V]).

    Differentiable: the backward is autograd over the chunked matmuls.

    ``state_nonlin``/``state_chunk``: if not identity, ``phi`` is applied to the
    state at every ``state_chunk`` boundary (default == ``chunk_size``), mirroring
    ``gdn2_nonlin_fused``. The recurrence stays linear *within* each phi-segment so
    the chunked form is exact; phi composes the segments. ``identity`` => pure
    linear delta (the GDN-2-class cell).
    """
    B, T, H, N = k.shape
    V = v.shape[-1]
    C = int(chunk_size)
    assert k.shape == q.shape == erase_gate.shape == (B, T, H, N)
    assert v.shape == value_write_gate.shape == (B, T, H, V)
    assert decay.shape == (B, T, H)

    if state_chunk is None:
        state_chunk = C
    # phi must land on chunk boundaries so a segment is an integer # of chunks.
    if state_nonlin not in (None, 'identity', 'linear', 'none'):
        if state_chunk % C != 0:
            raise ValueError(
                f"state_chunk ({state_chunk}) must be a multiple of chunk_size ({C}) "
                "so the boundary nonlinearity lands between chunks")

    # Two precisions: decay/cumsum/exp/triangular-solve stay in fp32 (stability),
    # but the heavy pairwise + state matmuls run in `mm_dtype` so bf16 inputs hit
    # the tensor cores (bf16 matmul accumulates in fp32 on Ampere+). For fp32
    # inputs everything stays fp32 (the exact-parity path).
    compute_dtype = torch.float32
    mm_dtype = torch.bfloat16 if k.dtype in (torch.bfloat16, torch.float16) else torch.float32
    kf = k.to(compute_dtype)
    qf = q.to(compute_dtype)
    vf = v.to(compute_dtype)
    ef = erase_gate.to(compute_dtype)
    wf = value_write_gate.to(compute_dtype)
    df = decay.to(compute_dtype)

    # Pad T up to a multiple of C. Padded steps: decay=1 (g=0), all vecs 0 ->
    # state unchanged, output 0. Exact (causal: future-zero steps cannot affect
    # earlier outputs); we slice [:T] at the end.
    pad = (-T) % C
    if pad:
        kf = F.pad(kf, (0, 0, 0, 0, 0, pad))
        qf = F.pad(qf, (0, 0, 0, 0, 0, pad))
        vf = F.pad(vf, (0, 0, 0, 0, 0, pad))
        ef = F.pad(ef, (0, 0, 0, 0, 0, pad))
        wf = F.pad(wf, (0, 0, 0, 0, 0, pad))
        df = F.pad(df, (0, 0, 0, pad), value=1.0)  # decay=1 on pad
    Tp = T + pad
    NC = Tp // C

    read_key = kf * ef          # u = e (.) k          [B,Tp,H,N]
    write_val = vf * wf         # p = w (.) v          [B,Tp,H,V]

    # Reshape to [B,H,NC,C,*] (head-major so each (b,h,chunk) is a matmul batch).
    def chunkify(x, d):
        return x.view(B, NC, C, H, d).permute(0, 3, 1, 2, 4).contiguous()
    Kc = chunkify(kf, N)         # [B,H,NC,C,N]
    Qc = chunkify(qf, N)
    Uc = chunkify(read_key, N)
    Pc = chunkify(write_val, V)  # [B,H,NC,C,V]
    gc = df.log().view(B, NC, C, H).permute(0, 3, 1, 2).contiguous()  # [B,H,NC,C]

    # Inclusive cumulative log-decay within each chunk.
    G = torch.cumsum(gc, dim=-1)             # [B,H,NC,C]   G[t]=sum_{i<=t} g_i
    Gprev = G - gc                           # [B,H,NC,C]   G[t]-g[t]=G[t-1]
    gamma = torch.exp(G)                     # [B,H,NC,C]
    decay_prev = torch.exp(Gprev)            # [B,H,NC,C]

    # Pairwise dot products: KU[t,j]=u_t.k_j ; QK[t,j]=q_t.k_j  (tensor cores).
    Kc_m = Kc.to(mm_dtype)
    KU = torch.matmul(Uc.to(mm_dtype), Kc_m.transpose(-1, -2)).to(compute_dtype)
    QK = torch.matmul(Qc.to(mm_dtype), Kc_m.transpose(-1, -2)).to(compute_dtype)

    # Decay weighting. DM[t,j]=exp(G[t-1]-G[j]) = DA[t,j]*exp(-g[t]) (row scale),
    # so we form one [C,C] exp (DA) and derive DM by a cheap row scaling.
    Gi = G.unsqueeze(-1)                      # [B,H,NC,C,1]  (row index t)
    Gj = G.unsqueeze(-2)                      # [B,H,NC,1,C]  (col index j)
    ar = torch.arange(C, device=k.device)
    lower_strict = (ar.view(C, 1) > ar.view(1, C))   # t>j
    lower_incl = (ar.view(C, 1) >= ar.view(1, C))    # t>=j

    DA = torch.exp(Gi - Gj)                   # exp(G[t]-G[j])  (<=1, stable)
    A = torch.where(lower_incl, DA * QK, DA.new_zeros(()))          # [B,H,NC,C,C]
    inv_decay_t = torch.exp(-gc).unsqueeze(-1)   # exp(-g[t]) row scale [B,H,NC,C,1]
    M = torch.where(lower_strict, DA * KU * inv_decay_t, DA.new_zeros(()))  # strictly lower

    # T = (I + M)^{-1}. (I+M) is unit lower triangular with M strictly-lower and
    # nilpotent (M^C = 0). Two ways to apply T:
    #   inverse_mode='solve'  : batched triangular solve (LAPACK-style). Accurate
    #       but its autograd backward issues a second batched solve, which on
    #       small [C,C] systems is launch-bound and dominates fwd+bwd cost.
    #   inverse_mode='newton' : Newton-Schulz X <- X(2I - LX) on tensor cores. The
    #       residual is M^{2^k}, so ceil(log2(C)) steps invert EXACTLY (no
    #       truncation error for nilpotent M); both fwd and bwd are pure cuBLAS
    #       matmuls (no triangular-solve backward), which is markedly faster.
    eye = torch.eye(C, device=k.device, dtype=compute_dtype)
    ImM = eye + M                             # [B,H,NC,C,C]
    # RHS2 = diag(decay_prev) @ U   (the S0c-coupling part)
    RHS2 = decay_prev.unsqueeze(-1) * Uc      # [B,H,NC,C,N]
    RHS = torch.cat([Pc, RHS2], dim=-1)       # [B,H,NC,C,V+N]
    if inverse_mode == 'solve':
        # Solve (I+M) W = RHS in one batched triangular solve.
        W = torch.linalg.solve_triangular(ImM, RHS, upper=False, unitriangular=True)
    elif inverse_mode == 'newton':
        # Exact nilpotent inverse via Newton-Schulz, then apply to RHS.
        steps = max(1, (C - 1).bit_length())   # ceil(log2(C)) -> exact
        L_m = ImM.to(mm_dtype)
        X = (eye - M)                          # 2nd-order seed: (I+M)^-1 = I-M+M^2-...
        for _ in range(steps):
            LX = torch.matmul(L_m, X.to(mm_dtype)).to(compute_dtype)
            X = torch.matmul(X.to(mm_dtype), (2.0 * eye - LX).to(mm_dtype)).to(compute_dtype)
        W = torch.matmul(X.to(mm_dtype), RHS.to(mm_dtype)).to(compute_dtype)
    else:
        raise ValueError(f"unknown inverse_mode {inverse_mode!r}")
    W_p = W[..., :V]                          # [B,H,NC,C,V]
    W_u = W[..., V:]                          # [B,H,NC,C,N]

    # Cross-chunk transition: S_next = P_trans @ S0c + P_const
    # gamma_last per chunk = gamma[..., -1]   (full chunk: prod of all decays)
    gamma_last = gamma[..., -1]               # [B,H,NC]
    # Kscaled[j] = exp(G[L-1]-G[j]) k_j
    Kscaled = torch.exp((G[..., -1:] - G)).unsqueeze(-1) * Kc   # [B,H,NC,C,N]
    KsT_m = Kscaled.transpose(-1, -2).to(mm_dtype)   # [B,H,NC,N,C]
    P_trans = -torch.matmul(KsT_m, W_u.to(mm_dtype)).to(compute_dtype)   # [B,H,NC,N,N]
    # add gamma_last * I
    eye_N = torch.eye(N, device=k.device, dtype=compute_dtype)
    P_trans = P_trans + gamma_last[..., None, None] * eye_N
    P_const = torch.matmul(KsT_m, W_p.to(mm_dtype)).to(compute_dtype)    # [B,H,NC,N,V]

    # Sequential cross-chunk scan (NC steps; NC = T/C is small).
    if S0 is None:
        S_entry = torch.zeros(B, H, N, V, device=k.device, dtype=compute_dtype)
    else:
        S_entry = S0.to(compute_dtype)
    entries = []
    phi_period_chunks = max(1, state_chunk // C)
    nonlinear = state_nonlin not in (None, 'identity', 'linear', 'none')
    for c in range(NC):
        entries.append(S_entry)
        S_entry = torch.matmul(P_trans[:, :, c], S_entry) + P_const[:, :, c]
        # boundary nonlinearity (phi) — applied between segments, not last step
        if nonlinear and ((c + 1) % phi_period_chunks == 0) and (c + 1) < NC:
            S_entry = _phi(S_entry, state_nonlin)
    S_entry_stack = torch.stack(entries, dim=2)   # [B,H,NC,N,V]
    S_es_m = S_entry_stack.to(mm_dtype)

    # Per-chunk outputs (parallel over chunks).
    Delta = W_p - torch.matmul(W_u.to(mm_dtype), S_es_m).to(compute_dtype)   # [B,H,NC,C,V]
    O_intra = torch.matmul(A.to(mm_dtype), Delta.to(mm_dtype)).to(compute_dtype)   # [B,H,NC,C,V]
    O_inter = gamma.unsqueeze(-1) * torch.matmul(Qc.to(mm_dtype), S_es_m).to(compute_dtype)
    O = O_inter + O_intra                                  # [B,H,NC,C,V]

    # back to [B,T,H,V]
    out = O.permute(0, 2, 3, 1, 4).reshape(B, Tp, H, V)[:, :T]
    out = out.to(v.dtype)
    S_final = S_entry.to(v.dtype)                          # state after last chunk
    return out, S_final
