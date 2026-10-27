"""Run both unchanged suites against v3; retain independent part-6 receipts."""
import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
from . import cell_fused

ROOT=Path(__file__).resolve().parent

def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}

if __name__=='__main__':
    assert os.environ.get('GPU_LEASE_HELD') in list('0123456')
    before=hashes()
    sys.modules['experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused']=cell_fused
    status=pytest.main(['experiments/expressivity_tasks/x31_tape_cell_fused/tests',
                       'experiments/expressivity_tasks/x35_tape_cell_v3/tests',
                       '-q',f'--junitxml={ROOT}/validation_part6.xml'])
    assert before==hashes()
    (ROOT/'parity_part6.json').write_text(json.dumps(dict(exit_code=int(status),
        source_sha256=before,physical_gpu=os.environ['GPU_LEASE_HELD']),indent=2)+'\n')
    raise SystemExit(status)
