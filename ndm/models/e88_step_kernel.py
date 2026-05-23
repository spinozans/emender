"""E88 fused single-token step kernel (Triton).

One kernel launch per (b, h) pair. Each block:
  - loads k, q, v, decay, g (and S_in) for its head
  - optionally L2-normalizes k, q
  - runs the recurrence update  S = tanh(decay * S + outer(v - S@k, k))
  - computes output = gated(S @ q)
  - writes S_out and output

For the CMA-ES-optimal E88 config (n_state=32, head_v_dim=32) this fuses
what was ~20 separate kernel launches in the vectorized-PyTorch path into 1.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e88_step_kernel(
    k_ptr, q_ptr, v_ptr, decay_ptr, g_ptr,
    S_in_ptr, S_out_ptr, output_ptr,
    H,                       # num_heads (runtime)
    N: tl.constexpr,         # n_state (compile-time)
    V: tl.constexpr,         # head_v_dim (compile-time)
    USE_GATE: tl.constexpr,
    GATE_ACT: tl.constexpr,  # 0=sigmoid, 1=silu
    USE_L2: tl.constexpr,
    LINEAR_STATE: tl.constexpr,
):
    # One program per (b, h) — grid is (B*H,)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    n_off = tl.arange(0, N)
    v_off = tl.arange(0, V)

    # Offsets into flat [B, H, *] tensors
    bh_n_base = (b * H + h) * N
    bh_v_base = (b * H + h) * V
    bh_S_base = (b * H + h) * N * V
    bh_scalar = b * H + h

    # Load in fp32 (Triton requires fp32 for sqrt; all accumulation stays fp32)
    k = tl.load(k_ptr + bh_n_base + n_off).to(tl.float32)
    q = tl.load(q_ptr + bh_n_base + n_off).to(tl.float32)
    v = tl.load(v_ptr + bh_v_base + v_off).to(tl.float32)
    decay = tl.load(decay_ptr + bh_scalar).to(tl.float32)
    if USE_GATE:
        g = tl.load(g_ptr + bh_v_base + v_off).to(tl.float32)

    if USE_L2:
        k_norm = tl.sqrt(tl.sum(k * k)) + 1e-6
        k = k / k_norm
        q_norm = tl.sqrt(tl.sum(q * q)) + 1e-6
        q = q / q_norm

    # Load S_in: shape [N, V] flattened row-major
    S_off = n_off[:, None] * V + v_off[None, :]
    S = tl.load(S_in_ptr + bh_S_base + S_off).to(tl.float32)

    # retrieved[v] = sum_i S[i, v] * k[i]
    retrieved = tl.sum(S * k[:, None], axis=0)       # [V]
    delta = v - retrieved                             # [V]

    # outer[i, v] = delta[v] * k[i]
    outer = delta[None, :] * k[:, None]               # [N, V]

    # State update in fp32
    pre = decay * S + outer
    if LINEAR_STATE:
        S_new = pre
    else:
        e2x = tl.exp(2.0 * pre)
        S_new = (e2x - 1.0) / (e2x + 1.0)

    # Store as bf16
    tl.store(S_out_ptr + bh_S_base + S_off, S_new.to(tl.bfloat16))

    # Readout in fp32 for precision
    Sq = tl.sum(S_new * q[:, None], axis=0)          # [V], fp32

    # Output gating
    if USE_GATE:
        if GATE_ACT == 0:
            out = Sq * tl.sigmoid(g)
        else:
            # silu(g) = g * sigmoid(g)
            out = Sq * (g * tl.sigmoid(g))
    else:
        out = Sq

    tl.store(output_ptr + bh_v_base + v_off, out.to(tl.bfloat16))


def e88_step(
    k, q, v, decay, S_in,
    g=None,
    use_l2_norm=True,
    linear_state=False,
    gate_activation='silu',
):
    """Single-step E88 recurrence.

    Args:
        k, q: [B, H, N] L2-normalizable keys/queries (bf16)
        v: [B, H, V] values (bf16)
        decay: [B, H] per-head decay (bf16)
        S_in: [B, H, N, V] previous state (bf16)
        g: [B, H, V] or None - gate activations (pre-activation). When None,
           output is raw Sq without gating.
        use_l2_norm: apply L2 norm to k, q inside the kernel
        linear_state: skip tanh (ablation)
        gate_activation: 'sigmoid' or 'silu'

    Returns:
        output: [B, H, V] (bf16) - gated readout
        S_out: [B, H, N, V] (bf16) - new state
    """
    assert k.dtype == torch.bfloat16, f"expected bfloat16, got {k.dtype}"
    B, H, N = k.shape
    V = v.shape[-1]
    assert S_in.shape == (B, H, N, V), f"S_in shape mismatch: {S_in.shape}"

    # Make contiguous for flat indexing
    k = k.contiguous()
    q = q.contiguous()
    v = v.contiguous()
    decay = decay.contiguous()
    S_in = S_in.contiguous()

    S_out = torch.empty_like(S_in)
    output = torch.empty_like(v)

    use_gate = g is not None
    if use_gate:
        g = g.contiguous()
        g_ptr = g
    else:
        # Pass a dummy tensor; kernel won't read it when USE_GATE=False
        g_ptr = torch.empty(1, dtype=torch.bfloat16, device=k.device)

    gate_act = 0 if gate_activation == 'sigmoid' else 1

    grid = (B * H,)
    _e88_step_kernel[grid](
        k, q, v, decay, g_ptr,
        S_in, S_out, output,
        H, N, V,
        USE_GATE=use_gate,
        GATE_ACT=gate_act,
        USE_L2=use_l2_norm,
        LINEAR_STATE=linear_state,
    )
    return output, S_out
