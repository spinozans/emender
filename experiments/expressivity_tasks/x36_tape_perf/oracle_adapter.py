"""Pytest process-local adapter: run unchanged X31 tests against X36."""
from ..x31_tape_cell_fused import cell_fused as x31
from .cell_fused import tape_scan, TapeCell
x31.tape_scan=tape_scan
x31.TapeCell=TapeCell
