"""Exercise the staged-only F9 closure in an isolated interpreter/checkout."""
import os
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def _preparer():
    path = Path(__file__).resolve().parents[1] / 'scripts/rl-loop-v1/F9-staged/prepare.py'
    spec = importlib.util.spec_from_file_location('f9_prepare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f9_staged_validator_matches_reviewed_pin():
    prepare = _preparer()
    validator = prepare.SOURCE / 'scripts/e97_first_party_validator_diversity.py'
    assert hashlib.sha256(validator.read_bytes()).hexdigest() == prepare.EXPECTED_VALIDATOR_SHA256


def test_f9_preparation_rejects_validator_drift_before_creating_stage(tmp_path, monkeypatch):
    prepare = _preparer()
    source = tmp_path / 'modified-source'
    (source / 'scripts').mkdir(parents=True)
    (source / 'scripts/e97_first_party_validator_diversity.py').write_text('# changed\n')
    monkeypatch.setattr(prepare, 'SOURCE', source)
    output = tmp_path / 'stage'
    with pytest.raises(ValueError, match='sealed validator bytes differ'):
        prepare.prepare(output)
    assert not output.exists()


def test_f9_staged_task_diversity(tmp_path):
    prepare = _preparer()
    required = [prepare.REPO / 'configs/pi/e97-firstparty-generator-manifest-v1.json',
                prepare.REPO / '.venv/bin/python',
                prepare.WORK / 'scripts/lake_expand.py',
                prepare.WORK / 'scripts/teacher_author.py']
    if not all(path.is_file() for path in required):
        pytest.skip('requires the private F9 deployment checkout and reviewed authoring sources')
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'scripts/rl-loop-v1/F9-staged'
    root = tmp_path / 'checkout'
    prepared = subprocess.run([sys.executable, str(source / 'prepare.py'), '--root', str(root)], capture_output=True, text=True, timeout=90)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    env = {**os.environ, 'F9_STAGE_ROOT': str(root), 'F9_AUTHOR_WORK': str(root / 'author-work'),
           'PYTHONPATH': f'{root}:{root / "loop"}:{repo}:/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts'}
    checked = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(source / 'unit_tests.py')], cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert checked.returncode == 0, checked.stdout + checked.stderr
