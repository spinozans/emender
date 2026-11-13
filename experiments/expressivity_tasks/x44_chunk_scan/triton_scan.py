"""One Triton kernel for the intra-chunk affine scan.

``affine.scan_with_entry`` runs a Python loop over the chunk steps, which is
one elementwise launch per step per pass: at chunk length 64 that is 1536 of the
2624 CUDA-graph nodes of a chunk, and a CUDA-graph replay costs about 2.4 us per
node.  The kernel below performs the same recurrence with one thread per lane
and the chunk loop in registers, so a scan is one node.

The arithmetic is the same operation sequence as the PyTorch pass
(``s = m * s; s = s + b``, no reassociation), which keeps the strength channel
-- and therefore the derived death boundaries and routing -- identical.  Use
``same_as_torch`` to verify that on the shapes in play before relying on it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn(a, b):
    """``a*b`` with one round-to-nearest step and no FMA contraction."""
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _affine_scan_kernel(multiplier, injection, entry, states, lanes,
                        steps: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < lanes
    state = tl.load(entry + offset, mask=mask, other=0.)
    for step in range(steps):
        block = step * lanes + offset
        mult = tl.load(multiplier + block, mask=mask, other=1.)
        inject = tl.load(injection + block, mask=mask, other=0.)
        # Explicit round-per-operation: Triton would otherwise contract this
        # into an FMA and move the strength trajectory (and death boundaries)
        # away from the certified sequential path.
        state = _add_rn(_mul_rn(state, mult), inject)
        tl.store(states + block, state, mask=mask)


def scan_with_entry(multiplier: torch.Tensor, injection: torch.Tensor,
                    entry: torch.Tensor, block: int = 256):
    """``[C, L]`` multiplier/injection plus ``[L]`` entry -> ``([C, L], [L])``."""
    steps, lanes = multiplier.shape
    if injection.shape != multiplier.shape or entry.shape != (lanes,):
        raise ValueError('affine scan shape mismatch')
    multiplier = multiplier.contiguous()
    injection = injection.contiguous()
    entry = entry.contiguous()
    states = torch.empty_like(multiplier)
    _affine_scan_kernel[(triton.cdiv(lanes, block),)](
        multiplier, injection, entry, states, lanes,
        steps=steps, BLOCK=block, num_warps=4)
    return states, states[-1]


def same_as_torch(multiplier: torch.Tensor, injection: torch.Tensor,
                  entry: torch.Tensor) -> tuple[bool, float]:
    """Compare against the reference implementation; returns (exact, max deviation)."""
    from . import affine

    reference, reference_final = affine.scan_with_entry(multiplier, injection, entry)
    states, final = scan_with_entry(multiplier, injection, entry)
    deviation = max((states - reference).abs().max().item(),
                    (final - reference_final).abs().max().item())
    return bool(torch.equal(states, reference)), deviation


__all__ = ['scan_with_entry', 'same_as_torch']
