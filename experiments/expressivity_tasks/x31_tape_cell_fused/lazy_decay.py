"""Device port of X30's exact rounded-multiplication interval jumps.

The scan calls these jumps for timestamped references and heap scheduling.
Using pow for a nonunit reference is not equivalent to repeated fp32 decay.
No host tensor reads or per-token host loops are used here. Compile callers
with enable_fp_fusion=False to preserve multiplication boundaries.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _jump(s, decay: tl.constexpr):
    """Return a constant-mantissa-decrement interval, or one native step.

    For d>=1/2, its exact ratio is p/2**24; every integer intermediate
    fits in signed int64. Smaller d always crosses an exponent boundary,
    so no interval optimization is needed. Ties are handled by native
    multiplication at interval boundaries, just as in X30 _interval.
    """
    bits = s.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = ((bits & 0x7fffff) + 0x800000).to(tl.int64)
    count = tl.full(s.shape, 1, tl.int64)
    decrement = tl.full(s.shape, 0, tl.int64)
    eligible = tl.full(s.shape, False, tl.int1)
    if decay >= 0.5 and decay < 1.0:
        p = tl.full((), int(decay * 16777216), tl.int64)
        denominator = tl.full((), 16777216, tl.int64)
        product = mantissa * p
        quotient = product // denominator
        remainder = product % denominator
        rounded = quotient + ((2*remainder > denominator) |
                              ((2*remainder == denominator) & ((quotient & 1) != 0)))
        decrement = mantissa - rounded
        lower = ((2*decrement-1)*denominator) // (2*(denominator-p)) + 1
        normalization = (8388608*denominator+p-1)//p
        lower = tl.maximum(lower, normalization)
        eligible = ((exponent > 0) & (product >= 8388608*denominator) &
                    (decrement > 0) & (mantissa >= lower))
        count = tl.where(eligible, (mantissa-lower)//tl.maximum(decrement, 1)+1, 1)
    return exponent, mantissa, decrement, count, eligible


@triton.jit
def advance_exact(s, steps, decay: tl.constexpr):
    remaining = steps.to(tl.int64)
    if decay != 1.0:
        while tl.sum((remaining > 0).to(tl.int32), 0) > 0:
            exponent, mantissa, decrement, count, eligible = _jump(s, decay)
            count = tl.minimum(count, remaining)
            bits = ((exponent.to(tl.int64) << 23) |
                    (mantissa-count*decrement-8388608)).to(tl.int32)
            native = s * decay
            updated = tl.where(eligible, bits.to(tl.float32, bitcast=True), native)
            active = remaining > 0
            # Subnormal rounding can reach a fixed point above zero.
            fixed = ~eligible & (native == s)
            s = tl.where(active, updated, s)
            remaining = tl.where(active, tl.where(fixed, 0, remaining-count), 0)
    return s


@triton.jit
def death_exact(s, threshold, decay: tl.constexpr):
    """Positive exact horizon and final strength; -1 means no finite death."""
    elapsed = tl.full(s.shape, 0, tl.int64)
    active = s > threshold
    if decay == 1.0:
        elapsed = tl.full(s.shape, -1, tl.int64)
    else:
        target_bits = threshold.to(tl.int32, bitcast=True)
        target_exponent = (target_bits >> 23) & 255
        target_mantissa = ((target_bits & 0x7fffff) + 8388608).to(tl.int64)
        while tl.sum(active.to(tl.int32), 0) > 0:
            exponent, mantissa, decrement, count, eligible = _jump(s, decay)
            needed = (mantissa-target_mantissa+tl.maximum(decrement, 1)-1)//tl.maximum(decrement, 1)
            count = tl.where(eligible & (exponent == target_exponent) & (needed <= count),
                             needed, count)
            bits = ((exponent.to(tl.int64) << 23) |
                    (mantissa-count*decrement-8388608)).to(tl.int32)
            native = s*decay
            updated = tl.where(eligible, bits.to(tl.float32, bitcast=True), native)
            fixed = ~eligible & (native == s)
            elapsed = tl.where(active, tl.where(fixed, -1, elapsed+count), elapsed)
            s = tl.where(active, updated, s)
            active = active & ~fixed & (s > threshold)
    return elapsed, s


@triton.jit
def _evaluate(Strength, Steps, Out, Horizons, Terminal, N: tl.constexpr,
              Decay: tl.constexpr, Threshold: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    s = tl.load(Strength+i, i<N, 1.)
    steps = tl.load(Steps+i, i<N, 0)
    result = advance_exact(s, steps, Decay)
    horizon, terminal = death_exact(s, tl.full((BLOCK,), Threshold, tl.float32), Decay)
    tl.store(Out+i, result, i<N)
    tl.store(Horizons+i, horizon, i<N)
    tl.store(Terminal+i, terminal, i<N)


def evaluate_decay(strength, steps, decay, threshold):
    """Evaluate exact jumps and schedules for live fp32 references on CUDA.

    Caller guarantees finite nonnegative strengths above threshold, and
    nonnegative int64 steps. Validation deliberately never reads CUDA data.
    """
    # Round the compile-time constant exactly once, matching X30 f32(decay).
    import struct
    decay = struct.unpack('<f', struct.pack('<f', decay))[0]
    threshold = struct.unpack('<f', struct.pack('<f', threshold))[0]
    if not (math.isfinite(decay) and 0 < decay <= 1 and
            math.isfinite(threshold) and threshold >= 0):
        raise ValueError('invalid decay/threshold')
    if (strength.dtype != torch.float32 or steps.dtype != torch.int64 or
        strength.shape != steps.shape or strength.device != steps.device or
        not strength.is_cuda or not strength.is_contiguous() or not steps.is_contiguous()):
        raise ValueError('contiguous CUDA fp32 strengths/int64 steps of equal shape required')
    out = torch.empty_like(strength)
    horizon = torch.empty_like(steps)
    terminal = torch.empty_like(strength)
    _evaluate[(triton.cdiv(strength.numel(), 128),)](
        strength, steps, out, horizon, terminal, strength.numel(), decay, threshold, 128,
        enable_fp_fusion=False)
    return out, horizon, terminal
