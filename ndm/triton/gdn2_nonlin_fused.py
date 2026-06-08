"""Triton fused nonlinear-state GatedDeltaNet scan.

This module implements the GDN-2 "nonlinear shell" recurrence without the old
Python loop over FLA ``chunk_gated_delta_rule`` chunks.  The recurrent state is
threaded through a single Triton launch and ``phi(S)`` is applied in-kernel after
each ``state_chunk`` boundary, after that boundary step's readout and before the
next step consumes the state.

Supported scope is the E99 shell geometry: equal-length batches, headwise beta,
one value-head per key-head, and ``K,V <= 64``.  That covers the production
``n_state=32, expand_v=1.0`` GDN shell used by the audit microbench.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


PHI_IDENTITY = 0
PHI_TANH = 1
PHI_RELU = 2
PHI_SOFTPLUS_C = 3
PHI_SOFTPLUS = 4

PHI_NAME_TO_CODE = {
    "identity": PHI_IDENTITY,
    "linear": PHI_IDENTITY,
    "none": PHI_IDENTITY,
    "tanh": PHI_TANH,
    "relu": PHI_RELU,
    "softplus_c": PHI_SOFTPLUS_C,
    "softplus": PHI_SOFTPLUS,
}


@triton.jit
def _apply_state_phi(x, phi_mode: tl.constexpr):
    if phi_mode == 0:
        return x
    elif phi_mode == 1:
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0
    elif phi_mode == 2:
        return tl.maximum(x, 0.0)
    elif phi_mode == 3:
        # 0.5 * softplus(2x) - 0.5 * log(2), implemented with a stable branch.
        y = 2.0 * x
        sp = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(tl.minimum(y, 20.0))))
        return 0.5 * sp - 0.34657359027997264
    else:
        return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))


@triton.jit
def _state_phi_grad(x, phi_mode: tl.constexpr):
    if phi_mode == 0:
        return tl.full(x.shape, 1.0, tl.float32)
    elif phi_mode == 1:
        y = 2.0 * tl.sigmoid(2.0 * x) - 1.0
        return 1.0 - y * y
    elif phi_mode == 2:
        return (x > 0.0).to(tl.float32)
    elif phi_mode == 3:
        return tl.sigmoid(2.0 * x)
    else:
        return tl.sigmoid(x)


@triton.jit
def _gdn2_nonlin_fwd_kernel(
    Q, K, V, G, BETA,
    O, STATES,
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr, K_DIM: tl.constexpr, V_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    STATE_CHUNK: tl.constexpr, PHI_MODE: tl.constexpr, SCALE: tl.constexpr,
    PRENORM: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)

    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)
    k_mask = k_idx < K_DIM
    v_mask = v_idx < V_DIM
    state_mask = k_mask[:, None] & v_mask[None, :]

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    state_base = ((0 * B + b) * H + h) * K_DIM * V_DIM
    state_off = state_base + k_idx[:, None] * V_DIM + v_idx[None, :]
    tl.store(STATES + state_off, S.to(STATES.dtype.element_ty), mask=state_mask)

    for t in range(T):
        q_base = ((b * T + t) * H + h) * K_DIM
        v_base = ((b * T + t) * H + h) * V_DIM
        gh_off = (b * T + t) * H + h

        q_raw = tl.load(Q + q_base + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        k_raw = tl.load(K + q_base + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        v_vec = tl.load(V + v_base + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        g = tl.load(G + gh_off).to(tl.float32)
        beta = tl.load(BETA + gh_off).to(tl.float32)

        # PRENORM: q,k arrive already L2-normalized and (for q) *SCALE, done in
        # autograd-tracked PyTorch off the sequential critical path. Skipping the
        # per-step rsqrt+normalize removes two reductions from each step's chain.
        if PRENORM:
            q_vec = q_raw
            k_vec = k_raw
        else:
            q_rstd = tl.rsqrt(tl.sum(q_raw * q_raw) + 1.0e-6)
            k_rstd = tl.rsqrt(tl.sum(k_raw * k_raw) + 1.0e-6)
            q_vec = q_raw * q_rstd * SCALE
            k_vec = k_raw * k_rstd

        S = S * tl.exp(g)
        retrieved = tl.sum(S * k_vec[:, None], axis=0)
        delta = beta * (v_vec - retrieved)
        S = S + k_vec[:, None] * delta[None, :]

        out_vec = tl.sum(S * q_vec[:, None], axis=0)
        tl.store(O + v_base + v_idx, out_vec.to(O.dtype.element_ty), mask=v_mask)

        if ((t + 1) % STATE_CHUNK) == 0 and (t + 1) < T:
            S = _apply_state_phi(S, PHI_MODE)
            S = tl.where(state_mask, S, 0.0)

        next_base = (((t + 1) * B + b) * H + h) * K_DIM * V_DIM
        next_off = next_base + k_idx[:, None] * V_DIM + v_idx[None, :]
        tl.store(STATES + next_off, S.to(STATES.dtype.element_ty), mask=state_mask)


@triton.jit
def _gdn2_nonlin_bwd_kernel(
    Q, K, V, G, BETA, STATES, DO,
    DQ, DK, DV, DG, DBETA,
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr, K_DIM: tl.constexpr, V_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    STATE_CHUNK: tl.constexpr, PHI_MODE: tl.constexpr, SCALE: tl.constexpr,
    PRENORM: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)

    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)
    k_mask = k_idx < K_DIM
    v_mask = v_idx < V_DIM
    state_mask = k_mask[:, None] & v_mask[None, :]

    carry = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for ti in range(T):
        t = T - 1 - ti
        q_base = ((b * T + t) * H + h) * K_DIM
        v_base = ((b * T + t) * H + h) * V_DIM
        gh_off = (b * T + t) * H + h

        state_base = ((t * B + b) * H + h) * K_DIM * V_DIM
        state_off = state_base + k_idx[:, None] * V_DIM + v_idx[None, :]
        S_prev = tl.load(STATES + state_off, mask=state_mask, other=0.0).to(tl.float32)

        q_raw = tl.load(Q + q_base + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        k_raw = tl.load(K + q_base + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        v_vec = tl.load(V + v_base + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        dout = tl.load(DO + v_base + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        g = tl.load(G + gh_off).to(tl.float32)
        beta = tl.load(BETA + gh_off).to(tl.float32)

        # PRENORM: q,k already L2-normalized (+SCALE on q) in autograd-tracked
        # PyTorch; the grads DQ/DK are then w.r.t. those pre-normalized inputs and
        # the rsqrt jacobian is handled outside the kernel. Skips 2 reductions/step.
        if PRENORM:
            q_norm = q_raw
            k_norm = k_raw
            q_scaled = q_raw
        else:
            q_rstd = tl.rsqrt(tl.sum(q_raw * q_raw) + 1.0e-6)
            k_rstd = tl.rsqrt(tl.sum(k_raw * k_raw) + 1.0e-6)
            q_norm = q_raw * q_rstd
            k_norm = k_raw * k_rstd
            q_scaled = q_norm * SCALE

        eg = tl.exp(g)
        S_dec = S_prev * eg
        retrieved = tl.sum(S_dec * k_norm[:, None], axis=0)
        residual = v_vec - retrieved
        delta = beta * residual
        S_new = S_dec + k_norm[:, None] * delta[None, :]

        if ((t + 1) % STATE_CHUNK) == 0 and (t + 1) < T:
            carry = carry * _state_phi_grad(S_new, PHI_MODE)

        dq_scaled = tl.sum(S_new * dout[None, :], axis=1)
        dS = carry + q_scaled[:, None] * dout[None, :]

        d_delta = tl.sum(dS * k_norm[:, None], axis=0)
        dk_norm = tl.sum(dS * delta[None, :], axis=1)
        dS_dec = dS

        dbeta = tl.sum(d_delta * residual)
        dv = beta * d_delta
        d_retrieved = -beta * d_delta
        dS_dec = dS_dec + k_norm[:, None] * d_retrieved[None, :]
        dk_norm = dk_norm + tl.sum(S_dec * d_retrieved[None, :], axis=1)

        dg = tl.sum(dS_dec * S_dec)
        carry = dS_dec * eg

        if PRENORM:
            # DQ/DK are grads w.r.t. the pre-normalized (+SCALE) inputs; the
            # normalization jacobian is differentiated by PyTorch autograd outside.
            dq_out = dq_scaled
            dk_out = dk_norm
        else:
            dq_norm = dq_scaled * SCALE
            dq_dot = tl.sum(dq_norm * q_norm)
            dq_out = (dq_norm - q_norm * dq_dot) * q_rstd

            dk_dot = tl.sum(dk_norm * k_norm)
            dk_out = (dk_norm - k_norm * dk_dot) * k_rstd

        tl.store(DQ + q_base + k_idx, dq_out, mask=k_mask)
        tl.store(DK + q_base + k_idx, dk_out, mask=k_mask)
        tl.store(DV + v_base + v_idx, dv, mask=v_mask)
        tl.store(DG + gh_off, dg)
        tl.store(DBETA + gh_off, dbeta)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(16, p)


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  g: torch.Tensor, beta: torch.Tensor, state_chunk: int) -> Tuple[int, int, int, int, int]:
    if not q.is_cuda:
        raise RuntimeError("fused nonlinear GDN scan requires CUDA tensors")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,T,H,D] tensors")
    B, T, H, K_dim = q.shape
    if v.shape[:3] != (B, T, H):
        raise ValueError(f"v must be [B,T,H,V] with same B/T/H, got {tuple(v.shape)}")
    V_dim = v.shape[-1]
    if g.shape != (B, T, H):
        raise ValueError(f"g must be [B,T,H], got {tuple(g.shape)}")
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must be headwise [B,T,H], got {tuple(beta.shape)}")
    if K_dim > 64 or V_dim > 64:
        raise NotImplementedError(
            f"fused nonlinear GDN scan supports K,V<=64 (got K={K_dim}, V={V_dim})"
        )
    if int(state_chunk) <= 0:
        raise ValueError(f"state_chunk must be positive, got {state_chunk}")
    return B, T, H, K_dim, V_dim


def gdn2_nonlinear_scan_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_chunk: int,
    phi_mode: int,
    prenorm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K_dim, V_dim = _check_inputs(q, k, v, g, beta, state_chunk)
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()
    out = torch.empty((B, T, H, V_dim), device=q.device, dtype=q.dtype)
    states = torch.empty((T + 1, B, H, K_dim, V_dim), device=q.device, dtype=q.dtype)
    block_k = _next_pow2(K_dim)
    block_v = _next_pow2(V_dim)
    _gdn2_nonlin_fwd_kernel[(B, H)](
        q_c, k_c, v_c, g_c, beta_c, out, states,
        T=T, B=B, H=H, K_DIM=K_dim, V_DIM=V_dim,
        BLOCK_K=block_k, BLOCK_V=block_v,
        STATE_CHUNK=int(state_chunk), PHI_MODE=int(phi_mode),
        SCALE=1.0 / math.sqrt(float(K_dim)),
        PRENORM=bool(prenorm),
        # num_warps=2 is the latency-optimal launch for the sequential forward scan
        # at the [K,V]<=64 tile (hetero-kernel micro-sweep: 1.26ms vs 1.31ms@4 /
        # 1.75ms@1, H=4 T=2048). The per-step [32,32] tile is small, so 2 warps
        # balance reduction-shuffle width against scheduling latency.
        num_warps=2,
    )
    return out, states


def gdn2_nonlinear_scan_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    states: torch.Tensor,
    d_out: torch.Tensor,
    state_chunk: int,
    phi_mode: int,
    prenorm: bool = False,
):
    B, T, H, K_dim, V_dim = _check_inputs(q, k, v, g, beta, state_chunk)
    block_k = _next_pow2(K_dim)
    block_v = _next_pow2(V_dim)
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()
    states_c = states.contiguous()
    d_out_c = d_out.contiguous()
    d_q = torch.empty((B, T, H, K_dim), device=q.device, dtype=torch.float32)
    d_k = torch.empty_like(d_q)
    d_v = torch.empty((B, T, H, V_dim), device=q.device, dtype=torch.float32)
    d_g = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    d_beta = torch.empty_like(d_g)
    _gdn2_nonlin_bwd_kernel[(B, H)](
        q_c, k_c, v_c, g_c, beta_c, states_c, d_out_c,
        d_q, d_k, d_v, d_g, d_beta,
        T=T, B=B, H=H, K_DIM=K_dim, V_DIM=V_dim,
        BLOCK_K=block_k, BLOCK_V=block_v,
        STATE_CHUNK=int(state_chunk), PHI_MODE=int(phi_mode),
        SCALE=1.0 / math.sqrt(float(K_dim)),
        PRENORM=bool(prenorm),
        # num_warps=1 is the latency-optimal launch for the sequential BACKWARD scan
        # (hetero-kernel micro-sweep: 2.43ms vs 2.93ms@4, H=4 T=2048). The backward's
        # per-step gradient algebra is a chain of 32-wide reductions; a single warp
        # does each reduction in one shuffle with no inter-warp sync, which dominates
        # the small loss of parallelism at this tile size.
        num_warps=1,
    )
    return d_q, d_k, d_v, d_g, d_beta


class _GDN2NonlinearScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, state_chunk: int, phi_mode: int, prenorm: bool = False):
        out, states = gdn2_nonlinear_scan_forward(q, k, v, g, beta, state_chunk, phi_mode, prenorm)
        ctx.save_for_backward(q, k, v, g, beta, states)
        ctx.state_chunk = int(state_chunk)
        ctx.phi_mode = int(phi_mode)
        ctx.prenorm = bool(prenorm)
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, g, beta, states = ctx.saved_tensors
        d_q, d_k, d_v, d_g, d_beta = gdn2_nonlinear_scan_backward(
            q, k, v, g, beta, states, d_out, ctx.state_chunk, ctx.phi_mode, ctx.prenorm
        )
        return (
            d_q.to(q.dtype),
            d_k.to(k.dtype),
            d_v.to(v.dtype),
            d_g.to(g.dtype),
            d_beta.to(beta.dtype),
            None,
            None,
            None,
        )


def _l2norm_scale(t: torch.Tensor, scale: float) -> torch.Tensor:
    """L2-normalize the last dim exactly as the in-kernel path (rsqrt(sum^2+1e-6)),
    then multiply by ``scale``. Runs in autograd-tracked PyTorch (parallel over all
    T), so its jacobian feeds the raw-q/k gradient and the sequential kernel can
    skip per-step normalization (PRENORM fast path)."""
    f = t.float()
    n = f * torch.rsqrt((f * f).sum(-1, keepdim=True) + 1e-6) * scale
    return n.to(t.dtype)


def fused_nonlinear_gated_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_chunk: int = 64,
    state_nonlin: str | int = "tanh",
    prenorm: bool = False,
) -> torch.Tensor:
    """Differentiable fused nonlinear-state gated-delta scan.

    Args:
        q, k: ``[B,T,H,K]`` raw query/key tensors. They are L2-normalized inside
            the Triton kernel to match FLA ``use_qk_l2norm_in_kernel=True``.
        v: ``[B,T,H,V]`` values.
        g: ``[B,T,H]`` log-space decay gate.
        beta: ``[B,T,H]`` headwise delta-rule beta.
        state_chunk: apply ``phi`` after every ``state_chunk`` steps except the
            final step.
        state_nonlin: ``identity``, ``tanh``, ``relu``, ``softplus_c`` or
            ``softplus``.
        prenorm: if True, L2-normalize q,k (and scale q) in autograd-tracked
            PyTorch BEFORE the kernel, so the sequential scan skips the per-step
            rsqrt/normalize on its critical path (hetero-kernel fast path). The
            normalization jacobian is differentiated by PyTorch. Numerically
            identical to the in-kernel normalization (both use rsqrt(sum^2+1e-6)).
    """
    phi_mode = PHI_IDENTITY if state_nonlin is None else (
        int(state_nonlin) if isinstance(state_nonlin, int) else PHI_NAME_TO_CODE[state_nonlin]
    )
    if prenorm:
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        q = _l2norm_scale(q, scale)
        k = _l2norm_scale(k, 1.0)
    return _GDN2NonlinearScanFunction.apply(q, k, v, g, beta, int(state_chunk), phi_mode, bool(prenorm))


def _phi_torch(S: torch.Tensor, kind: str | int) -> torch.Tensor:
    phi_mode = PHI_IDENTITY if kind is None else (
        int(kind) if isinstance(kind, int) else PHI_NAME_TO_CODE[kind]
    )
    if phi_mode == PHI_IDENTITY:
        return S
    if phi_mode == PHI_TANH:
        return torch.tanh(S)
    if phi_mode == PHI_RELU:
        return torch.relu(S)
    if phi_mode == PHI_SOFTPLUS_C:
        return torch.nn.functional.softplus(2.0 * S) * 0.5 - 0.34657359027997264
    if phi_mode == PHI_SOFTPLUS:
        return torch.nn.functional.softplus(S)
    raise ValueError(f"unknown phi mode {kind}")


def nonlinear_gated_delta_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_chunk: int = 64,
    state_nonlin: str | int = "tanh",
) -> torch.Tensor:
    """Autograd reference for tests. Matches FLA q/k L2 norm and scale."""
    B, T, H, K_dim = q.shape
    scale = K_dim ** -0.5
    S = torch.zeros(B, H, K_dim, v.shape[-1], device=q.device, dtype=torch.float32)
    outs = []
    for t in range(T):
        q_raw = q[:, t].float()
        k_raw = k[:, t].float()
        q_t = q_raw * torch.rsqrt((q_raw * q_raw).sum(dim=-1, keepdim=True) + 1e-6) * scale
        k_t = k_raw * torch.rsqrt((k_raw * k_raw).sum(dim=-1, keepdim=True) + 1e-6)
        v_t = v[:, t].float()
        S = S * torch.exp(g[:, t].float()).unsqueeze(-1).unsqueeze(-1)
        retrieved = torch.einsum("bhkv,bhk->bhv", S, k_t)
        delta = beta[:, t].float().unsqueeze(-1) * (v_t - retrieved)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q_t))
        if (t + 1) % int(state_chunk) == 0 and (t + 1) < T:
            S = _phi_torch(S, state_nonlin)
    return torch.stack(outs, dim=1).to(q.dtype)
