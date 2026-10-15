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


def test_f9_portable_functional_and_security_cases(tmp_path):
    """Run the original portable cases even when deployment sources are absent."""
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'scripts/rl-loop-v1/F9-staged'
    cases = (
        'test_end_state_exact_regex_absent_command_and_no_token_echo',
        'test_end_state_rejects_symlink_escape',
        'test_user_script_determinism_and_before_acting_gate',
        'test_versioned_script_continuation_and_legacy_finish_close',
        'test_full_exchange_cannot_skip_turn_or_finish_early',
        'test_private_script_is_prescreened_without_publishing_it',
        'test_fresh_scripted_correction_rule_is_shared_by_pool_and_sync_paths',
        'test_new_families_reject_token_echo_artifacts',
        'test_raw_quote_boundary_preserves_authorized_literals',
        'test_rollback_boundary_is_read_only_and_refuses_inflight_work',
        'test_seed_intake_checks_manifest_corpus_and_authority_before_use',
        'test_seed_intake_explicitly_disables_absent_corrupt_or_unauthorized',
    )
    boundary = tmp_path / 'test_offline_boundary.py'
    boundary.write_text('''import pytest
import f9_original
from scripts.e97_offline_shell import shell_allowed, assertion_argv_allowed

def commands(name):
    return next(mark.args[1] for mark in getattr(f9_original, name).pytestmark
                if mark.name == "parametrize")

@pytest.mark.parametrize("command", commands("test_offline_shell_reference_gate_rejects")
                         + commands("test_f9r_expansion_rejected_by_reference_and_policy"))
def test_staged_offline_shell_rejects_unsafe_commands(command):
    with pytest.raises(ValueError):
        shell_allowed(command)

def test_staged_assertion_commands_cannot_mutate_workspace():
    with pytest.raises(ValueError):
        assertion_argv_allowed(["rm", "reports/summary.txt"])
''')
    bootstrap = '''import importlib.util, sys, types
from pathlib import Path
import scripts
import pytest
source = Path(sys.argv[1])
scripts.__path__ = [str(source / "scripts"), *scripts.__path__]
# These deployment-only imports are unavailable, not permissive mocks:
# accidental use by a supposedly portable test raises AttributeError.
for name in ("teacher_author", "standing_supply"):
    sys.modules[name] = types.ModuleType(name)
spec = importlib.util.spec_from_file_location("f9_original", source / "unit_tests.py")
original = importlib.util.module_from_spec(spec)
sys.modules["f9_original"] = original
spec.loader.exec_module(original)
raise SystemExit(pytest.main(["-q", *sys.argv[2:]]))
'''
    env = {**os.environ, 'F9_STAGE_ROOT': str(tmp_path),
           'F9_AUTHOR_WORK': str(tmp_path / 'author-work'),
           'PYTHONPATH': f'{source / "loop"}:{repo / "scripts/rl-loop-v1"}:{repo}'}
    checked = subprocess.run(
        [sys.executable, '-c', bootstrap, str(source),
         *(f'{source / "unit_tests.py"}::{name}' for name in cases), str(boundary)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert checked.returncode == 0, checked.stdout + checked.stderr


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
