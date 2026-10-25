"""X28E phase-2 fused Triton kernels for the slot-store tape head (pre-M2)."""
from .tape_kernels import (
    DEFAULT_CKPT_INTERVAL, DEFAULT_MAX_RESIDENT_SLOTS, DEFAULT_TILE_S,
    FusedBuffers, TapeFusedFunction, buffers_to_state, buffers_to_trace,
    fused_fwd_raw, fused_replay_raw, fused_tape_scan, tape_bwd_kernel,
    tape_fwd_kernel,
)
from .fused_layer import FusedTapeHead, FusedTapeLayer

__all__ = [
    "tape_fwd_kernel", "tape_bwd_kernel", "TapeFusedFunction", "fused_tape_scan",
    "fused_fwd_raw", "fused_replay_raw", "buffers_to_trace", "buffers_to_state",
    "FusedBuffers", "FusedTapeHead", "FusedTapeLayer",
    "DEFAULT_CKPT_INTERVAL", "DEFAULT_TILE_S", "DEFAULT_MAX_RESIDENT_SLOTS",
]
