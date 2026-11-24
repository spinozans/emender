"""Exercise actual entry points without acquiring a lease or exposing a device."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', ['k15_coupled_training', 'k15_relation_training',
                                    'k15_coupled_diagnostic', 'k15_coupled_launch'])
def test_optimized_entry_points_fail_closed(module, tmp_path):
    out = tmp_path / 'must-not-exist'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', PYTHONOPTIMIZE='1')
    proc = subprocess.run([sys.executable, '-m', f'experiments.expressivity_tasks.x45_matrix_emender.{module}', str(out)],
                          cwd=Path(__file__).resolve().parents[3], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0 and 'K15 I/J requires Python assertions' in proc.stderr
    assert not out.exists()
