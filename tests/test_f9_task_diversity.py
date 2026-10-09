"""Exercise the staged-only F9 closure in an isolated interpreter/checkout."""
import os
from pathlib import Path
import subprocess
import sys


def test_f9_staged_task_diversity(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'scripts/rl-loop-v1/F9-staged'
    root = tmp_path / 'checkout'
    prepared = subprocess.run([sys.executable, str(source / 'prepare.py'), '--root', str(root)], capture_output=True, text=True, timeout=90)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    env = {**os.environ, 'F9_STAGE_ROOT': str(root), 'F9_AUTHOR_WORK': str(root / 'author-work'),
           'PYTHONPATH': f'{root}:{root / "loop"}:{repo}:/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts'}
    checked = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(source / 'unit_tests.py')], cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert checked.returncode == 0, checked.stdout + checked.stderr
