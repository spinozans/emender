"""Experimental rounding primitive for diagnostics only; not wired into training.

FP32 temporary arithmetic + stochastic BF16 writeback is not an FP32 master
weight. Production use still requires qualified CUDA rounding, Schedule-Free
basis semantics, bucket-independent RNG identity, DDP and exact-resume tests.
"""
from __future__ import annotations
import torch


def stochastic_round_bf16(value: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Unbiased stochastic rounding of bounded finite FP32 values to BF16.

The discarded low 16 bits determine the probability of advancing to the next
BF16 magnitude. The caller owns RNG state; global RNG is not consumed. This
prototype consumes one draw per coordinate, including exactly representable
values. Deliberately reject overflow/nonfinite inputs rather than map to inf.
"""
    if value.dtype != torch.float32:
        raise ValueError('FP32 temporary input required')
    if not bool(torch.isfinite(value).all()) or bool((value.abs() > torch.finfo(torch.bfloat16).max).any()):
        raise ValueError('input must be finite and inside BF16 finite range')
    bits = value.contiguous().view(torch.int32)
    noise = torch.randint(0, 65536, bits.shape, dtype=torch.int32,
                          device=value.device, generator=generator)
    return ((bits + noise) >> 16).to(torch.int16).view(torch.bfloat16)
