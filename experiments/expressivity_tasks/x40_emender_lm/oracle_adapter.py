"""Read-only upstream fixtures with the X40 exact-decay reverse implementation."""
from ..x31_tape_cell_fused import cell_fused as x31
from ..x36_tape_perf import cell_fused as x36
from ..x36_tape_perf import backward as original
from .backward import autograd_scan
x31.tape_scan=x36.tape_scan
x31.TapeCell=x36.TapeCell
original.autograd_scan=autograd_scan
