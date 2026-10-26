"""Exact fp32 repeated-decay jumps without a slot or token scan.

Within a normal-float exponent bin, rounded multiplication by d<1 subtracts
an integer mantissa decrement. That decrement stays constant over an integer
interval. Jump across the interval, then handle its boundary. IEEE precision
bounds the number of intervals independently of the tape's slot budget.

Used for architectural death scheduling, pop touches, and nonunit references.
Unit references use a shared exact discrete-exponential lookup curve. Rounding
must NEVER move an architectural death to another token.
"""
import functools
import math
import struct


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def _bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def _value(exponent, mantissa):
    return struct.unpack('<f', struct.pack('<I', (exponent << 23) | (mantissa-(1 << 23))))[0]


def _round_ratio(numerator, denominator):
    q, r = divmod(numerator, denominator)
    return q + (2*r > denominator or (2*r == denominator and q % 2 == 1))


def _interval(s, d):
    bits = _bits(s)
    exponent = (bits >> 23) & 255
    mantissa = (bits & ((1 << 23)-1)) + (1 << 23)
    p, denominator = d.as_integer_ratio()
    if exponent == 0 or mantissa*p < (1 << 23)*denominator:
        return None  # a normalization/subnormal boundary: one exact multiply
    decrement = mantissa - _round_ratio(mantissa*p, denominator)
    if decrement <= 0:
        return None
    # Strict lower bound excludes ties; their parity is handled individually.
    lower = ((2*decrement-1)*denominator)//(2*(denominator-p)) + 1
    lower = max(lower, ((1 << 23)*denominator+p-1)//p)
    if mantissa < lower:
        return None
    steps = (mantissa-lower)//decrement + 1
    return exponent, mantissa, decrement, steps


@functools.lru_cache(maxsize=65536)
def advance(strength, decay, steps):
    """Exactly the scalar fp32 recurrence for `steps` multiplies."""
    s, d = f32(strength), f32(decay)
    if steps < 0 or not 0 < d <= 1 or not math.isfinite(s) or s < 0:
        raise ValueError('nonnegative finite strength/steps and 0<decay<=1 required')
    if d == 1 or s == 0:
        return s
    while steps:
        interval = _interval(s, d)
        if interval is None:
            next_s = f32(s*d)
            if next_s == s:  # possible subnormal fixed point
                return s
            s, steps = next_s, steps-1
        else:
            exponent, mantissa, decrement, count = interval
            count = min(count, steps)
            s = _value(exponent, mantissa-count*decrement)
            steps -= count
    return s


@functools.lru_cache(maxsize=65536)
def death(strength, decay, threshold):
    """Return (exact positive horizon, death strength), or (None, strength)."""
    s, d, threshold = f32(strength), f32(decay), f32(threshold)
    if not 0 < d <= 1 or s <= threshold or threshold < 0:
        raise ValueError('schedule only live strengths above threshold')
    if d == 1:
        return None, s
    elapsed = 0
    threshold_bits = _bits(threshold)
    target_exponent = (threshold_bits >> 23) & 255
    target_mantissa = (threshold_bits & ((1 << 23)-1)) + (1 << 23)
    while s > threshold:
        interval = _interval(s, d)
        if interval is None:
            next_s = f32(s*d)
            if next_s == s:
                return None, s
            s, elapsed = next_s, elapsed+1
        else:
            exponent, mantissa, decrement, count = interval
            if exponent == target_exponent:
                needed = (mantissa-target_mantissa+decrement-1)//decrement
                if needed <= count:
                    return elapsed+needed, _value(exponent, mantissa-needed*decrement)
            s = _value(exponent, mantissa-count*decrement)
            elapsed += count
    return elapsed, s


# A single shared curve per decay/device, not one curve per slot. It is native
# rounding's representation of exp(dt*log(decay)); only accesses gather it.
# Its length is bounded by the maximum live reference age requested, not S.
_unit_curves = {}
_device_curves = {}


def unit_factors(decay, horizon, device):
    import torch
    decay = f32(decay)
    curve = _unit_curves.setdefault(decay, [1.])
    if horizon >= len(curve):
        for _ in range(horizon+1-len(curve)):
            curve.append(f32(curve[-1]*decay))
    key = (decay, str(device))
    tensor = _device_curves.get(key)
    if tensor is None or tensor.numel() <= horizon:
        tensor = torch.tensor(curve, device=device, dtype=torch.float32)
        _device_curves[key] = tensor
    return tensor
