"""Chunked-parallel COMPLEX-eigenvalue gated-delta scan (cplx_gdn).

Implements the complex-eigenvalue (rotation-scaling) gated-delta head of
``paper/review/COMPLEX_EIG_HEAD_SPEC.md``.  The real per-head scalar decay of
GDN-2 / e97 is replaced by a **per-key-channel complex eigenvalue**

    lambda_{t,p} = r_{t,p} * exp(i * theta_{t,p})         r in (0,1), theta in R

equivalently a 2x2 real rotation-scaling block acting on the (Re, Im) pair of
each complex channel.  Everything else in the gated-delta head is reused: the
delta correction (Hermitian ``k^H`` on the key axis), L2-norm, beta gate
(with ``allow_neg_eigval`` doubling), and the read ``o = Re(q^H S)``.

Two scans live here, both over REAL math (no mocks):

* ``complex_gated_delta_reference`` — the eager per-step recurrence

      retrieved_t = k_t^H S_{t-1}                         (Hermitian read)
      delta_t     = beta_t (v_t - retrieved_t)
      S_t         = diag(lambda_t) S_{t-1} + (lambda_t (.) k_t) delta_t^T
      o_t         = Re(q_t^H S_t)

  This is the spec recurrence (sec 1-2) and the ground-truth reference the
  chunked kernel is verified against.  An optional per-step bounded map
  ``phi`` (hardtanh) realizes the "nonlinear-subset" heads (sec 5): bounded
  per-step state is NOT associative, so those heads run this sequential scan.

* ``complex_gated_delta_chunked`` — the chunked-parallel form (sec 3).  The
  cumulative complex transition factors into TWO prefix sums per channel
  (cumulative log-magnitude ``G`` and cumulative phase ``Phi``), exactly the
  S5/LRU diagonal associative scan.  The per-channel cumulative eigenvalue is
  FOLDED into decay-absorbed Hermitian keys (KL, KR, QL) so the intra-chunk
  delta system is the same strictly-lower-triangular nilpotent ``(I+M)`` solved
  by complex Newton-Schulz in ceil(log2 C) steps, and the cross-chunk carry is
  a per-channel complex diagonal (the LRU carry).  All the C-sized work is
  complex matmuls (tensor cores); only the T/C cross-chunk scan is sequential.
  The forward is written with differentiable complex torch ops, so autograd
  supplies an exact, also-chunk-parallel backward (Wirtinger) — no hand-written
  backward needed, and the per-step ``for t in range(T)`` fallback is NEVER on
  the chunked path.

Reductions (verified in ``tests/test_complex_eig.py``):
  * theta == 0  -> lambda = r real-positive  -> the GDN positive-decay regime.
  * theta == pi -> lambda = -r real-negative -> the reflection / negative-
    eigenvalue (allow_neg_eigval) regime.  One disk, one knob (spec sec 4).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Magnitude-exponent guard for the decay-absorbed key KR = k / cumdecay.
# cumdecay = exp(G) with G <= 0, so 1/cumdecay = exp(-G) can overflow fp32 for a
# run of strong-decay steps inside one chunk.  The TRUE (I+M)/A entries are always
# bounded <= 1 (they pair a <=1 factor with this >=1 one), so clamping the
# intermediate magnitude exponent only affects channels whose within-chunk
# product already underflowed to ~0 — a standard chunked-kernel stability guard
# (mirrors _GLOG_FLOOR in e97_chunked_autograd).  exp(80) ~ 5.5e34 < fp32 max.
# It is kept as a final backstop, but the PRIMARY hardening is the adaptive
# sub-chunking below (_decay_safe_chunk_size): the clamp alone is LOSSY — once a
# run of strong-decay steps pushes -G past the guard, ALL positions beyond the
# saturation point collapse to the same magnitude, erasing the relative scale of
# the (banded) near-diagonal entries that actually carry the output.  That makes
# the chunked scan return garbage (rel err -> 1) for |lambda| << 1 even though it
# stays finite, which corrupts training (the failure that motivated this task).
# Env-overridable (mirrors CPLX_INV_DECAY_GUARD in complex_eig_chunked_autograd):
# a huge value disables the clamp so the raw 1/cp overflow -> inf -> NaN is
# observable (used to A/B the hardening).
_INV_DECAY_GUARD = float(os.environ.get('CPLX_INV_DECAY_GUARD', '80.0'))

# Largest intra-chunk cumulative-decay span (max over channels of -Gprev, i.e.
# sum of per-step -log_r within one chunk) we allow before SHRINKING the chunk.
# Keeping the span <= 30 holds 1/cp = exp(-Gprev) <= exp(30) ~ 1e13, far below
# both fp32 max and the _INV_DECAY_GUARD clamp, so the clamp never fires and the
# decay-absorbed key/inverse-key products stay numerically exact (the dropped
# precision lives only in cross-chunk carry terms that gamma=exp(G) then scales
# by exp(-span) ~ 0, so they are genuinely negligible).  The cross-chunk LRU
# carry is division-free and exact, so shrinking the chunk only trades throughput
# for stability — in the limit C=1 this IS the eager per-step scan.
# Env override CPLX_INTRA_CHUNK_SPAN_GUARD (mainly for A/B repro; a huge value
# disables the adaptive sub-chunking and restores the old overflow behaviour).
_INTRA_CHUNK_SPAN_GUARD = float(os.environ.get('CPLX_INTRA_CHUNK_SPAN_GUARD', '30.0'))


def _decay_safe_chunk_size(log_r: torch.Tensor, requested_C: int,
                           span_guard: float = _INTRA_CHUNK_SPAN_GUARD) -> int:
    """Largest chunk size <= requested_C whose worst-case intra-chunk decay span
    (-log_r accumulated over the chunk) stays <= span_guard.

    The chunked scan folds the cumulative decay into the keys as 1/cp = exp(-Gprev);
    when the model drives |lambda| small WITHIN a chunk this overflows / saturates
    the magnitude guard and the scan returns garbage.  Bounding the span by capping
    the chunk size keeps 1/cp in fp32's exact range.  We use the GLOBAL worst-case
    per-step decay so the bound holds for every (batch, head, chunk, channel); the
    only cost is more (still division-free, exact) cross-chunk steps.
    """
    if requested_C <= 1:
        return max(1, int(requested_C))
    # worst-case per-step magnitude decay, |log_r| at the strongest-decay step
    d = (-log_r.detach()).clamp_min(0.0).max()
    d_val = float(d)
    if not math.isfinite(d_val) or d_val <= 0.0:
        return int(requested_C)
    # (C-1) * d <= span_guard  ->  C <= span_guard/d + 1
    c_safe = int(span_guard / d_val) + 1
    return max(1, min(int(requested_C), c_safe))


def _pair_to_complex(x: torch.Tensor) -> torch.Tensor:
    """[..., N] real -> [..., N/2] complex, pairing (x[2p], x[2p+1]) -> Re+iIm."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"key/query dim must be even to pair into complex, got {x.shape[-1]}")
    xr = x[..., 0::2]
    xi = x[..., 1::2]
    return torch.complex(xr.float(), xi.float())


def _complex_l2norm(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-norm over the last (complex-channel) axis by complex magnitude.

    ||z|| = sqrt(sum_p |z_p|^2); reduces to the real head's L2-norm at theta=0.
    """
    sq = (z.real * z.real + z.imag * z.imag).sum(dim=-1, keepdim=True)
    return z * torch.rsqrt(sq + eps)


def phi_complex(S: torch.Tensor, kind: Optional[str]) -> torch.Tensor:
    """Per-step bounded state map on the complex state (Re, Im independently).

    The phi-explore result is that BOUNDEDNESS is the depth lever (tanh = hardtanh
    = softsign all perfect); hardtanh is the cheapest.  Applied to Re and Im
    independently so it is the natural 2x2-real (rotation pair) bounded map.
    """
    if kind in (None, "identity", "linear", "none"):
        return S
    if kind == "hardtanh":
        return torch.complex(torch.clamp(S.real, -1.0, 1.0), torch.clamp(S.imag, -1.0, 1.0))
    if kind == "tanh":
        return torch.complex(torch.tanh(S.real), torch.tanh(S.imag))
    if kind == "softsign":
        return torch.complex(S.real / (1.0 + S.real.abs()), S.imag / (1.0 + S.imag.abs()))
    raise ValueError(f"unknown phi {kind!r}")


# ---------------------------------------------------------------------------
# Eager per-step complex reference (ground truth) — REAL recurrence, no mocks.
# ---------------------------------------------------------------------------
def complex_gated_delta_reference(
    q: torch.Tensor,        # [B, T, H, N] real query projection
    k: torch.Tensor,        # [B, T, H, N] real key projection
    v: torch.Tensor,        # [B, T, H, V] real value
    log_r: torch.Tensor,    # [B, T, H, P] per-channel log-magnitude (<= 0), P = N/2
    theta: torch.Tensor,    # [B, T, H, P] per-channel phase
    beta: torch.Tensor,     # [B, T, H]    scalar write strength per head
    S0: Optional[torch.Tensor] = None,   # [B, H, P, V] complex initial state
    phi: Optional[str] = None,           # per-step bounded map for the nonlinear subset
    read_mode: str = "real",             # 'real' -> Re(q^H S); 'reim' -> concat[Re;Im]
    l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spec sec 1-2 recurrence, run step by step.  Returns (out, S_final).

    out: [B, T, H, V] (read_mode='real') or [B, T, H, 2V] ('reim'). S_final complex.
    """
    B, T, H, N = q.shape
    V = v.shape[-1]
    P = N // 2
    qc = _pair_to_complex(q)            # [B,T,H,P] complex
    kc = _pair_to_complex(k)
    if l2norm:
        qc = _complex_l2norm(qc)
        kc = _complex_l2norm(kc)
    qc = qc * (P ** -0.5)               # 1/sqrt(P) query scale (matches FLA head scale)
    vf = v.float()
    lam = torch.polar(torch.exp(log_r.float()), theta.float())   # [B,T,H,P] complex eigenvalue
    bf = beta.float()
    if S0 is None:
        S = torch.zeros(B, H, P, V, device=q.device, dtype=torch.complex64)
    else:
        S = S0.to(torch.complex64)
    outs = []
    for t in range(T):
        k_t = kc[:, t]                  # [B,H,P]
        lam_t = lam[:, t]               # [B,H,P]
        # Hermitian read k^H S_{t-1}: sum_p conj(k)[p] S[p,:]
        retrieved = torch.einsum("bhp,bhpv->bhv", k_t.conj(), S)            # [B,H,V] complex
        delta = bf[:, t].unsqueeze(-1) * (vf[:, t].to(torch.complex64) - retrieved)
        # S_t = diag(lam) S_{t-1} + (lam (.) k) delta^T
        S = lam_t.unsqueeze(-1) * S
        S = S + torch.einsum("bhp,bhv->bhpv", lam_t * k_t, delta)
        if phi is not None:
            S = phi_complex(S, phi)
        q_t = qc[:, t]
        o = torch.einsum("bhp,bhpv->bhv", q_t.conj(), S)                    # q^H S_t  [B,H,V] complex
        outs.append(o)
    O = torch.stack(outs, dim=1)        # [B,T,H,V] complex
    if read_mode == "real":
        out = O.real
    elif read_mode == "reim":
        out = torch.cat([O.real, O.imag], dim=-1)
    else:
        raise ValueError(f"unknown read_mode {read_mode!r}")
    return out.to(q.dtype), S


# ---------------------------------------------------------------------------
# Chunked-parallel complex scan (sec 3) — the production kernel.
# ---------------------------------------------------------------------------
def complex_gated_delta_chunked(
    q: torch.Tensor,        # [B, T, H, N] real
    k: torch.Tensor,        # [B, T, H, N] real
    v: torch.Tensor,        # [B, T, H, V] real
    log_r: torch.Tensor,    # [B, T, H, P] per-channel log-magnitude (<= 0)
    theta: torch.Tensor,    # [B, T, H, P] per-channel phase
    beta: torch.Tensor,     # [B, T, H]
    S0: Optional[torch.Tensor] = None,   # [B, H, P, V] complex
    chunk_size: int = 32,
    read_mode: str = "real",
    l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked-parallel forward for the complex gated-delta recurrence.

    Returns (out, S_final).  Differentiable via autograd over complex matmuls
    (the backward is the conjugate/Wirtinger VJP, supplied automatically).  The
    per-step T-loop is NEVER used here — only a T/C-step cross-chunk scan.
    """
    B, T, H, N = q.shape
    V = v.shape[-1]
    P = N // 2
    C = int(chunk_size)
    if C <= 0:
        raise ValueError(f"chunk_size must be positive, got {C}")
    # Harden against fp32 overflow when |lambda| << 1 within a chunk: shrink the
    # chunk so the intra-chunk decay span (-log_r accumulated over C steps) stays
    # in fp32's exact range.  Benign decay (the parity regime) is unaffected
    # (span small -> C unchanged); only the strong-decay regime that used to NaN
    # / return garbage falls back toward the exact, division-free eager limit.
    C = _decay_safe_chunk_size(log_r, C)

    qc = _pair_to_complex(q)            # [B,T,H,P] complex64
    kc = _pair_to_complex(k)
    if l2norm:
        qc = _complex_l2norm(qc)
        kc = _complex_l2norm(kc)
    qc = qc * (P ** -0.5)
    vf = v.float().to(torch.complex64)
    lr = log_r.float()
    th = theta.float()
    bf = beta.float()

    # Pad T up to a multiple of C: pad steps have log_r=0 (r=1), theta=0, beta=0,
    # zero q/k/v -> state unchanged, output 0.  Exact (causal); sliced at the end.
    pad = (-T) % C
    if pad:
        zc = (0, 0, 0, 0, 0, pad)       # pad the T axis (dim=1) of [B,T,H,*]
        qc = F.pad(qc, zc)
        kc = F.pad(kc, zc)
        vf = F.pad(vf, zc)
        lr = F.pad(lr, (0, 0, 0, 0, 0, pad), value=0.0)
        th = F.pad(th, (0, 0, 0, 0, 0, pad), value=0.0)
        bf = F.pad(bf, (0, 0, 0, pad), value=0.0)
    Tp = T + pad
    NC = Tp // C

    # Reshape to [B,H,NC,C,*] (head-major so each (b,h,chunk) is a matmul batch).
    def chunkify(x, last):
        return x.view(B, NC, C, H, last).permute(0, 3, 1, 2, 4).contiguous()
    Kc = chunkify(kc, P)                 # [B,H,NC,C,P] complex
    Qc = chunkify(qc, P)
    Vc = chunkify(vf, V)                 # [B,H,NC,C,V] complex
    LRc = chunkify(lr, P)                # [B,H,NC,C,P] real
    THc = chunkify(th, P)
    Bc = bf.view(B, NC, C, H).permute(0, 3, 1, 2).contiguous()   # [B,H,NC,C]

    # Two prefix sums per channel (the S5/LRU diagonal scan): cumulative
    # log-magnitude G and cumulative phase Phi (inclusive within the chunk).
    G = torch.cumsum(LRc, dim=-2)        # [B,H,NC,C,P]  G[t]=sum_{i<=t} log r_i
    Phi = torch.cumsum(THc, dim=-2)      # [B,H,NC,C,P]
    Gprev = G - LRc                      # G[t-1] (exclusive cumulative log-magnitude)
    Phiprev = Phi - THc                  # Phi[t-1]

    # cp = cumulative-exclusive eigenvalue product P_{t-1} = exp(Gprev) cis(Phiprev)
    # c  = cumulative-inclusive P_t = exp(G) cis(Phi)
    cp = torch.polar(torch.exp(Gprev), Phiprev)        # [B,H,NC,C,P]  |cp| <= 1
    c = torch.polar(torch.exp(G), Phi)                 # [B,H,NC,C,P]

    # Decay-absorbed Hermitian keys (sec 3.4 split-Re/Im twiddle, here as complex):
    #   KL[t] = conj(k_t) * cp_t           (folds cumulative decay into the read key)
    #   KR[j] = k_j / cp_j                  (inverse cumulative; magnitude-guarded)
    #   QL[t] = conj(q_t) * c_t            (folds inclusive decay into the query)
    # so M[t,j]=beta_t (KL_t . KR_j), A[t,j]=QL_t . KR_j are plain complex dots.
    KL = Kc.conj() * cp                                # [B,H,NC,C,P]
    inv_mag = torch.exp(torch.clamp(-Gprev, max=_INV_DECAY_GUARD))   # exp(-G[t-1]), guarded
    inv_cp = torch.polar(inv_mag, -Phiprev)            # 1/cp with magnitude guard
    KR = Kc * inv_cp                                   # [B,H,NC,C,P]
    QL = Qc.conj() * c                                 # [B,H,NC,C,P]

    ar = torch.arange(C, device=q.device)
    lower_strict = (ar.view(C, 1) > ar.view(1, C))     # t > j
    lower_incl = (ar.view(C, 1) >= ar.view(1, C))      # t >= j
    zero = torch.zeros((), dtype=torch.complex64, device=q.device)

    # Pairwise complex dots (tensor-core complex matmuls): [.,C,C].
    KLKR = torch.matmul(KL, KR.transpose(-1, -2))      # KLKR[t,j] = KL_t . KR_j
    QLKR = torch.matmul(QL, KR.transpose(-1, -2))      # QLKR[t,j] = QL_t . KR_j
    Bcol = Bc.unsqueeze(-1).to(torch.complex64)        # [B,H,NC,C,1] row scale beta_t
    M = torch.where(lower_strict, Bcol * KLKR, zero)   # strictly lower, nilpotent
    A = torch.where(lower_incl, QLKR, zero)            # lower incl diagonal

    # RHS of (I+M) Delta = diag(beta) v - diag(beta) (KL @ S0):
    #   W = (I+M)^{-1} [ diag(beta) v | diag(beta) KL ]  ->  W_p, W_u
    #   Delta = W_p - W_u @ S0
    RHS_p = Bcol * Vc                                  # [B,H,NC,C,V]
    RHS_u = Bcol * KL                                  # [B,H,NC,C,P]
    RHS = torch.cat([RHS_p, RHS_u], dim=-1)            # [B,H,NC,C,V+P]

    # Exact nilpotent inverse via complex Newton-Schulz X <- X(2I - LX).
    # Residual is M^{2^step}; M strictly-lower => M^C = 0 => ceil(log2 C) steps exact.
    eye = torch.eye(C, dtype=torch.complex64, device=q.device)
    ImM = eye + M
    steps = max(1, (C - 1).bit_length())
    X = eye - M                                        # 2nd-order seed (I+M)^-1 = I-M+M^2-...
    for _ in range(steps):
        LX = torch.matmul(ImM, X)
        X = torch.matmul(X, 2.0 * eye - LX)
    W = torch.matmul(X, RHS)                           # [B,H,NC,C,V+P]
    W_p = W[..., :V]                                   # [B,H,NC,C,V]
    W_u = W[..., V:]                                   # [B,H,NC,C,P]

    # Per-channel complex chunk-total eigenvalue gamma = P_{C-1} (the LRU carry).
    gamma = c[..., -1, :]                              # [B,H,NC,P]
    # Cross-chunk transition S_next = P_trans @ S0 + P_const (per-channel diag gamma).
    KRt = KR.transpose(-1, -2)                         # [B,H,NC,P,C]
    KRtWu = torch.matmul(KRt, W_u)                     # [B,H,NC,P,P]
    KRtWp = torch.matmul(KRt, W_p)                     # [B,H,NC,P,V]
    g_row = gamma.unsqueeze(-1)                        # [B,H,NC,P,1]
    eyeP = torch.eye(P, dtype=torch.complex64, device=q.device)
    P_trans = g_row * (eyeP - KRtWu)                   # diag(gamma) - gamma (.) (KR^T W_u)
    P_const = g_row * KRtWp                            # gamma (.) (KR^T W_p)

    # Sequential cross-chunk scan (NC = T/C steps; small).
    if S0 is None:
        S_entry = torch.zeros(B, H, P, V, dtype=torch.complex64, device=q.device)
    else:
        S_entry = S0.to(torch.complex64)
    entries = []
    for ci in range(NC):
        entries.append(S_entry)
        S_entry = torch.matmul(P_trans[:, :, ci], S_entry) + P_const[:, :, ci]
    S_stack = torch.stack(entries, dim=2)              # [B,H,NC,P,V] per-chunk entry states

    # Per-chunk outputs (parallel over chunks):
    #   Delta = W_p - W_u @ S0c ; O = A @ Delta + QL @ S0c ; out = Re(O)
    Delta = W_p - torch.matmul(W_u, S_stack)           # [B,H,NC,C,V]
    O_intra = torch.matmul(A, Delta)                   # [B,H,NC,C,V]
    O_inter = torch.matmul(QL, S_stack)                # [B,H,NC,C,V]  (QL_t . S0c)
    O = O_inter + O_intra                              # [B,H,NC,C,V] complex

    # back to [B,T,H,*]
    O = O.permute(0, 2, 3, 1, 4).reshape(B, Tp, H, V)[:, :T]
    if read_mode == "real":
        out = O.real
    elif read_mode == "reim":
        out = torch.cat([O.real, O.imag], dim=-1)
    else:
        raise ValueError(f"unknown read_mode {read_mode!r}")
    return out.to(q.dtype), S_entry


__all__ = [
    "complex_gated_delta_reference",
    "complex_gated_delta_chunked",
    "phi_complex",
]
