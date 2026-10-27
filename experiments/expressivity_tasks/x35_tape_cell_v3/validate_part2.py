"""Run unchanged X31 oracle suite plus X35 component tests on an X35 lease."""
import os
import pytest

if __name__ == '__main__':
    assert os.environ.get('GPU_LEASE_HELD') in list('0123456')
    raise SystemExit(pytest.main([
        'experiments/expressivity_tasks/x31_tape_cell_fused/tests',
        'experiments/expressivity_tasks/x35_tape_cell_v3/tests',
        '-q', '--junitxml=experiments/expressivity_tasks/x35_tape_cell_v3/validation_part2.xml',
    ]))
