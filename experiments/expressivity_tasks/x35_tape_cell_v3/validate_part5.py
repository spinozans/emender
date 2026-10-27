"""Unchanged X31/X35 suites plus backing tests; bind parity to source hashes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
from . import cell_fused

ROOT = Path(__file__).resolve().parent
SOURCES = ('cell_fused.py','backing.py','backward.py','kernels.py','causal_read.py',
           'all_field_journal.py','tests/test_backing.py', 'structural_cost_part5.py',
           'gates_part5.py','validate_part5.py','tests/test_structural_cost.py')


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


if __name__ == '__main__':
    assert os.environ.get('GPU_LEASE_HELD') in list('0123456')
    receipt = ROOT/'parity_part5.json'
    receipt.unlink(missing_ok=True)
    hashes = source_hashes()
    sys.modules['experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused'] = cell_fused
    status = pytest.main([
        'experiments/expressivity_tasks/x31_tape_cell_fused/tests',
        'experiments/expressivity_tasks/x35_tape_cell_v3/tests',
        '-q', f'--junitxml={ROOT}/validation_part5.xml', *sys.argv[1:]])
    if status == 0:
        assert source_hashes() == hashes, 'source changed during validation'
        receipt.write_text(json.dumps(dict(source_sha256=hashes,physical_gpu=os.environ['GPU_LEASE_HELD']),indent=2)+'\n')
    raise SystemExit(status)
