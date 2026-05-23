"""Triton kernel implementations for Elman models.

This package mirrors selected CUDA kernels in Triton, primarily as a
research vehicle for studying alternative implementations alongside the
hand-written CUDA kernels in ``elman.cuda``.
"""

# IMPORTANT: avoid shadowing the global ``triton`` package. We use absolute
# imports inside this subpackage's modules (``import triton`` and
# ``import triton.language as tl``); Python's import machinery resolves
# those against site-packages, not against this package, because there is
# no relative-import ambiguity in those statements.

from __future__ import absolute_import

from .e88_triton_forward import (
    e88_triton_forward,
    e88_torch_reference,
)
from .e88_triton_backward import (
    e88_triton_backward,
    e88_triton,
    E88TritonFunction,
)
from .e88_triton_optimized import (
    e88_triton_optimized_apply,
)

__all__ = [
    "e88_triton_forward",
    "e88_torch_reference",
    "e88_triton_backward",
    "e88_triton",
    "E88TritonFunction",
    "e88_triton_optimized_apply",
]
