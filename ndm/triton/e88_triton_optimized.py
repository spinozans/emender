"""Drop-in Triton replacement for E88OptimizedCUDAFunction.

Mirrors the call signature of
``ndm.models.e88_fla_hybrid.E88OptimizedCUDAFunction.apply`` but uses
the Triton forward + backward kernels under the hood. Gate (silu) and L2
normalization are applied as differentiable PyTorch ops outside the
kernel — the recurrence math is unchanged.

Layout: production E88 uses ``[B, T, H, *]``; the Triton kernels use
``[T, B, H, *]``, so we transpose at the boundary.

This wrapper is meant for parity / portability work — it should produce
the same loss as the CUDA path (within numerical tolerance) so you can
swap backends on the fly with ``use_triton=True``.
"""
from __future__ import absolute_import

from typing import Tuple

import torch

from .e88_triton_backward import e88_triton


def e88_triton_optimized_apply(
    training: bool,
    k: torch.Tensor,        # [B, T, H, n_state]
    v: torch.Tensor,        # [B, T, H, head_v_dim]
    q: torch.Tensor,        # [B, T, H, n_state]
    decay: torch.Tensor,    # [B, T, H]
    g: torch.Tensor = None, # [B, T, H, head_v_dim] gate (None or empty if no gate)
    S0: torch.Tensor = None,# [B, H, n_state, head_v_dim]
    n_heads: int = None,
    apply_gate: bool = True,
    normalize_kq: bool = False,
    checkpoint_interval: int = 16,  # ignored — Triton stores all checkpoints
    apply_silu_qkv: bool = False,
    raw_write: bool = False,
    erase_gate: torch.Tensor = None,
    value_write_gate: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-backed E88 recurrence with optional pre-norm and post-gate.

    Returns:
        S_final: [B, H, n_state, head_v_dim]
        output:  [B, T, H, head_v_dim]   (after gate if apply_gate=True)
    """
    assert k.dim() == 4 and v.dim() == 4 and q.dim() == 4, \
        "k/v/q must be [B, T, H, *]"
    B, T, H, N = k.shape
    Vsz = v.shape[-1]
    assert q.shape == (B, T, H, N)
    assert v.shape == (B, T, H, Vsz)
    assert decay.shape == (B, T, H)

    # L2 normalization is fused inside the Triton kernel when normalize_kq=True
    # — this saves a Python `linalg_vector_norm` + `aten::div` per layer call,
    # which adds up to ~50-60 ms/step at 1.27B production. The kernel handles
    # the L2-norm gradient via the standard chain rule.

    # Triton kernels expect [T, B, H, *]. We use .transpose() WITHOUT
    # .contiguous() — the kernel reads via explicit strides, and last-dim
    # contiguity is preserved for these axes (transposing dims 0 and 1
    # doesn't touch stride[-1]==1). Skipping the copy saves ~100 MB per
    # tensor at production scale; 14 layers * 3 grad_ckpt invocations *
    # 4 tensors = many GB of bandwidth per training step.
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    q_t = q.transpose(0, 1)
    decay_t = decay.transpose(0, 1)
    erase_t = erase_gate.transpose(0, 1) if erase_gate is not None else None
    value_write_t = value_write_gate.transpose(0, 1) if value_write_gate is not None else None
    if S0 is None:
        S0 = torch.zeros(
            (B, H, N, Vsz), dtype=k.dtype, device=k.device,
        )

    # Forward gate is fused INSIDE the kernel when apply_gate=True and
    # g is non-empty. This saves two extra kernel launches per layer
    # (silu, multiply) and matches CUDA's register-owned forward.
    use_fused_gate = (
        apply_gate
        and g is not None
        and getattr(g, "numel", lambda: 0)() > 0
    )
    if use_fused_gate:
        g_t = g.transpose(0, 1)  # [T, B, H, V] view (last dim contiguous)
        out_t, S_final = e88_triton(
            S0, k_t, v_t, q_t, decay_t, g_t, normalize_kq=normalize_kq,
            apply_silu_qkv=apply_silu_qkv,
            raw_write=raw_write,
            erase_gate=erase_t,
            value_write_gate=value_write_t,
        )
        output = out_t.transpose(0, 1)
    else:
        out_t, S_final = e88_triton(
            S0, k_t, v_t, q_t, decay_t, None, normalize_kq=normalize_kq,
            apply_silu_qkv=apply_silu_qkv,
            raw_write=raw_write,
            erase_gate=erase_t,
            value_write_gate=value_write_t,
        )
        output = out_t.transpose(0, 1)

    return S_final, output
