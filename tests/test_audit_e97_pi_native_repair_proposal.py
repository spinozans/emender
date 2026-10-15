import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_e97_pi_native_repair_proposal as auditor

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / 'configs/pi/e97-pi-native-repair-training-proposal-v1.json'


def run_audit(proposal_path, tmp):
    return subprocess.run(
        [sys.executable, 'scripts/audit_e97_pi_native_repair_proposal.py',
         '--proposal', str(proposal_path), '--output', str(tmp / 'audit.json')],
        cwd=ROOT, capture_output=True, text=True)


@pytest.fixture
def prepared_proposal(tmp_path, monkeypatch):
    """Keep semantic and digest checks real without a private training checkpoint."""
    proposal = json.loads(PROPOSAL.read_text())
    preparation = tmp_path / 'preparation'
    openhands = tmp_path / 'openhands'
    bridge = tmp_path / 'bridge'
    for directory in (preparation / 'packs', openhands, bridge):
        directory.mkdir(parents=True)

    def document(path, value):
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    proposal['preparation_root'] = str(preparation)
    proposal['openhands_rehearsal_source'] = str(openhands)
    proposal['rehearsal_source'] = str(bridge)
    proposal['openhands_rehearsal_source_sha256'] = document(
        openhands / 'manifest.json', {
            'schema': 'emender-open-swe-source-native-candidate-v1',
            'training_eligible': False,
        })
    proposal['rehearsal_source_sha256'] = document(bridge / 'manifest.json', {
        'schema': 'emender-e97-tulu3-masked-sft-v1', 'training_eligible': True,
    })
    proposal['authority_manifest_sha256'] = document(preparation / 'manifest.json', {
        'training_eligible': False, 'packing_authorized': False,
        'optimizer_updates_authorized': 0,
        'source_target_totals': proposal['scheduled_source_targets'],
        'source_record_counts': {
            'openhands-execution-rehearsal': proposal['openhands_rehearsal_records'],
        },
        'openhands_rehearsal': {
            'authority_sha256': proposal['openhands_rehearsal_source_sha256'],
            'seed': proposal['openhands_rehearsal_seed'],
            'target_token_budget': proposal['openhands_rehearsal_target_budget'],
        },
    })
    proposal['pack_manifest_sha256'] = document(preparation / 'packs/manifest.json', {
        'training_eligible': False, 'diagnostic_system_gate': 'cpu-system-gate',
    })
    proposal['schedule_sha256'] = document(preparation / 'schedule-proposal.json', {
        'training_eligible': False, 'optimizer_updates_authorized': 0,
        'sampler_key': proposal['sampler_key'],
        'world_size': proposal['data_world_size'],
        'context_size': proposal['context_size'],
        'steps': [
            {'update': update, 'source_targets': proposal['scheduled_source_targets']}
            for update in range(proposal['proposed_updates'])
        ],
        'source_target_totals': proposal['scheduled_source_targets'],
        'source_unique_records': proposal['scheduled_source_unique_records'],
        'source_token_totals': {'fixture': proposal['scheduled_input_tokens']},
        'unique_packs': proposal['scheduled_unique_packs'],
        'unique_records': proposal['scheduled_unique_records'],
    })
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'bounded checkpoint identity fixture')
    proposal['parent_checkpoint'] = str(checkpoint)
    original_sha = auditor.sha

    def fixture_sha(path):
        # The production checker hard-pins a multi-GB checkpoint digest. Mock
        # only that exact fixture input, retaining rejection of changed bytes
        # and real SHA-256 checks for every manifest and proposal.
        if Path(path) == checkpoint and checkpoint.read_bytes() == b'bounded checkpoint identity fixture':
            return proposal['parent_checkpoint_sha256']
        return original_sha(path)

    monkeypatch.setattr(auditor, 'sha', fixture_sha)
    path = tmp_path / 'proposal.json'
    path.write_text(json.dumps(proposal))
    return path, proposal


def audit_fixture(path, tmp_path):
    auditor.audit(argparse.Namespace(proposal=path, output=tmp_path / 'audit.json'))
    return json.loads((tmp_path / 'audit.json').read_text())


def test_audit_accepts_bound_proposal(prepared_proposal, tmp_path):
    path, _ = prepared_proposal
    receipt = audit_fixture(path, tmp_path)
    assert receipt['status'] == 'qualified-proposal-not-authorized'
    assert receipt['optimizer_updates_authorized'] == 0
    assert receipt['checkpoint_promotion'] is False
    assert receipt['per_update_cohort_stratification'].startswith('verified')


@pytest.mark.parametrize('field,value', [
    ('optimizer_updates_authorized', 3), ('automatic_retry', True),
    ('packing_authorized', True), ('checkpoint_promotion', True),
])
def test_audit_rejects_authorization_tampering(prepared_proposal, tmp_path, field, value):
    path, proposal = prepared_proposal
    proposal[field] = value
    path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match='proposal authorization'):
        audit_fixture(path, tmp_path)


@pytest.mark.parametrize('field', ['parent_checkpoint', 'preparation_root'])
def test_audit_rejects_artifact_tampering(prepared_proposal, tmp_path, field):
    path, proposal = prepared_proposal
    artifact = Path(proposal[field])
    if field == 'preparation_root':
        artifact /= 'manifest.json'
    artifact.write_bytes(artifact.read_bytes() + b' ')
    with pytest.raises(ValueError, match='proposal input identity'):
        audit_fixture(path, tmp_path)


def test_audit_rejects_unstratified_schedule(prepared_proposal, tmp_path):
    path, proposal = prepared_proposal
    schedule_path = Path(proposal['preparation_root']) / 'schedule-proposal.json'
    schedule = json.loads(schedule_path.read_text())
    schedule['steps'][0]['source_targets'].pop('pi-native-curriculum')
    schedule_path.write_text(json.dumps(schedule))
    proposal['schedule_sha256'] = auditor.sha(schedule_path)
    path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match='updates missing cohorts'):
        audit_fixture(path, tmp_path)


def test_retained_private_proposal_artifacts_integration(tmp_path):
    proposal = json.loads(PROPOSAL.read_text())
    required = [Path(proposal['parent_checkpoint'])]
    required += [Path(proposal[key]) / 'manifest.json' for key in (
        'preparation_root', 'openhands_rehearsal_source', 'rehearsal_source')]
    required += [Path(proposal['preparation_root']) / relative for relative in (
        'packs/manifest.json', 'schedule-proposal.json')]
    if not all(path.is_file() for path in required):
        pytest.skip('requires the retained private repair-proposal artifacts and checkpoint')
    result = run_audit(PROPOSAL, tmp_path)
    assert result.returncode == 0, (result.stdout, result.stderr)
    receipt = json.loads((tmp_path / 'audit.json').read_text())
    assert receipt['status'] == 'qualified-proposal-not-authorized'
    assert receipt['optimizer_updates_authorized'] == 0
    assert receipt['checkpoint_promotion'] is False
