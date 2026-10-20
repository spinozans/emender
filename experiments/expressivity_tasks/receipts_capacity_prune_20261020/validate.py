"""Read-only independent acceptance checks against durable prune receipts."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from prune import BASE, OUT, REPO, MATCHED, TARGETS, digest, entry_hash, read_json, write_json


def main():
    checks = {}
    total = count = 0
    prior = read_json(BASE / 'results_invertible_stage2b/receipts/postcompletion_capacity_prune.json')
    for label in TARGETS:
        m = read_json(OUT / (label + '.json'))
        events = [json.loads(s) for s in (OUT / (label + '.execution.jsonl')).read_text().splitlines()]
        assert len(events) == len(m['removed'])
        event_by_path = {e['path']: e for e in events}
        original = {e['path']: e for e in m['removed'] + m['retained']}
        retained = {e['path']: e for e in m['retained']}
        root = REPO / m['root']
        assert root.stat().st_dev == REPO.stat().st_dev
        for entry in m['removed']:
            e = dict(entry)
            checksum = e.pop('manifest_entry_sha256')
            assert entry_hash(e) == checksum
            assert not (REPO / e['path']).exists()
            event = event_by_path[e['path']]
            assert event['manifest_entry_sha256'] == checksum and event['removed_bytes'] == e['bytes']
            ledger = read_json(REPO / e['checkpoint_sha256_source'])
            cp = next(c for c in ledger.values() if Path(c['path']).name == Path(e['path']).name)
            assert cp['sha256'] == e['checkpoint_sha256']
            assert e['device'] == REPO.stat().st_dev
        if not m['active_processes']:
            for e in retained.values():
                p = REPO / e['path']
                assert p.exists() and p.stat().st_size == e['bytes']
            for p, sha in m['non_checkpoint_evidence_sha256'].items():
                assert digest(REPO / p) == sha, p
        if label == '2b':
            for run in sorted(root.glob('*/validation.json')):
                chain = set()
                incumbent = None
                for row in read_json(run):
                    key = (-row['accuracy'], row['loss'], row['step'])
                    if incumbent is None or key < incumbent:
                        incumbent = key
                        chain.add(row['step'])
                for entry in m['removed']:
                    p = REPO / entry['path']
                    if p.parent == run.parent:
                        step = int(p.stem.split('_')[1])
                        assert step in chain, entry['path']
                        assert step not in MATCHED | set(prior['broadcast_x2b_stop_steps'])
        freshly_rehashed = {}
        if label in {'2b', '3_8m'}:
            for result_path in sorted(root.glob('*/result.json')):
                r = read_json(result_path)
                required = {r['step'], r['stop_step']}
                if label == '2b':
                    required |= MATCHED | set(prior['broadcast_x2b_stop_steps'])
                for step in required:
                    p = result_path.parent / f'step_{step:06d}.pt'
                    key = str(p.relative_to(REPO))
                    if key in original:
                        assert key in retained, key
                        if step in {r['step'], r['stop_step']}:
                            sha = digest(p)
                            assert sha == retained[key]['checkpoint_sha256'], key
                            freshly_rehashed[key] = sha
                if label == '3_8m' and result_path.parent.name.startswith('one_step_'):
                    assert all(e['path'] in retained for e in original.values()
                               if Path(e['path']).parent == result_path.parent.relative_to(REPO))
        if label == '5c_growth_8m':
            for key, e in retained.items():
                sha = digest(REPO / key)
                assert sha == e['checkpoint_sha256'], key
                freshly_rehashed[key] = sha
        freed = sum(e['bytes'] for e in m['removed'])
        assert freed == sum(e['removed_bytes'] for e in events)
        total += freed
        count += len(events)
        checks[label] = {'removed_entries_and_execution_verified': len(events),
                         'freed_bytes': freed, 'retained_count': len(retained),
                         'freshly_rehashed': freshly_rehashed,
                         'non_checkpoint_evidence': 'all byte-identical' if not m['active_processes'] else 'active root retained in full; training evidence naturally advances'}
    logs = read_json(OUT / 'tmp_logs.json')
    events = [json.loads(s) for s in (OUT / 'tmp_logs.execution.jsonl').read_text().splitlines()]
    assert len(events) == logs['compressed_count']
    for e, event in zip(logs['compress'], events):
        entry = dict(e)
        assert entry_hash({k: v for k, v in entry.items() if k != 'manifest_entry_sha256'}) == entry['manifest_entry_sha256']
        assert not Path(e['path']).exists() and Path(e['gzip_path']).exists()
        assert event['manifest_entry_sha256'] == e['manifest_entry_sha256']
        h = hashlib.sha256()
        with gzip.open(e['gzip_path'], 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        assert h.hexdigest() == e['sha256']
    # Verify the misleading large bioinformatics example was not modified.
    bio = next(e for e in logs['retained'] if e['path'] == '/tmp/pi-bash-cab7545f335f075e.log')
    p = Path(bio['path'])
    assert p.stat().st_size == bio['bytes'] and p.stat().st_mtime_ns == bio['mtime_ns']
    df = subprocess.check_output(['df', '-h', '/'], text=True).strip()
    free = __import__('shutil').disk_usage(REPO).free
    assert free >= 500 * 1024 ** 3
    result = {'status': 'passed', 'targets': checks, 'checkpoint_files_removed': count,
              'checkpoint_bytes_freed': total, 'log_bytes_freed': logs['freed_bytes'],
              'total_bytes_freed': total + logs['freed_bytes'], 'free_bytes': free,
              'df': df, 'large_bioinformatics_log_unchanged': True,
              'execution_source_sha256': digest(OUT / 'prune_at_execution.py'),
              'ledger_provenance': 'Removed checkpoint SHA256s come from original checkpoint ledgers, not a new full-payload hash; every entry and execution receipt independently checked. Retained best/stop weights and every retained X5c weight freshly rehashed.'}
    write_json(OUT / 'validation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'targets'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
