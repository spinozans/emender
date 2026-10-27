"""Apply unchanged X31 tests to X35 by redirecting only the public cell module."""
import os
import sys
import pytest
from . import cell_fused

if __name__ == '__main__':
    assert os.environ.get('GPU_LEASE_HELD') in list('0123456')
    sys.modules['experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused'] = cell_fused
    raise SystemExit(pytest.main([
        'experiments/expressivity_tasks/x31_tape_cell_fused/tests',
        'experiments/expressivity_tasks/x35_tape_cell_v3/tests',
        '-q', '--junitxml=experiments/expressivity_tasks/x35_tape_cell_v3/validation_part3.xml',
        *sys.argv[1:],
    ]))
