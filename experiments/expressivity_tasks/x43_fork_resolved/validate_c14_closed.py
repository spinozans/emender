"""Validate closure receipts, without claiming consultation training acceptance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/c14_emender_closed'


def main():
    cache = json.loads((OUT / 'cache_validation.json').read_text())
    audit = json.loads((ROOT / 'receipts/c14_registered_scale/stream_audit.json').read_text())
    projection = json.loads((OUT / 'eval_projection.json').read_text())
    assert cache['status'] == 'passed'
    assert cache['windows'] == audit['arms'][0]['windows']
    assert cache['positions'] == audit['arms'][0]['targets']
    assert cache['last_window_index'] == cache['windows'] - 1
    assert cache['journal_sha256'] == cache['boundary']['journal_sha256']
    assert cache['cache_bytes'] == cache['boundary']['cache_bytes']
    assert cache['parity_attestations_green'] is True
    assert cache['request_cardinality_matches'] is False
    assert cache['training_status'] == 'blocked_scope_conflict'
    assert projection['frozen_windows_sha256'] == audit['eval']['windows_sha256']
    assert projection['committed_targets'] < projection['expected_targets']
    for name in ['memory_before.json', 'memory_after.json']:
        sample = json.loads((OUT / name).read_text())
        assert 'ActiveState=active' in sample['service']
        assert 'NRestarts=2' in sample['service']
    journal = (OUT / 'systemd_journal.txt').read_text()
    assert '15:17:07' in journal and '15:57:46' in journal
    result = dict(
        status='passed', scope='cache closure and diagnosis receipt consistency only',
        full_task_acceptance='not-satisfied', training_run=False, evaluation_run=False,
        evidence_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'validation.json'})
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
