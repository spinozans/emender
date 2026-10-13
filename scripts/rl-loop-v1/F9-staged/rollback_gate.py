"""Read-only, fail-closed old-only rollback gate; never changes bank ownership."""
import argparse
import hashlib
import json
from pathlib import Path


def rollback_inventory(bank):
    bank = Path(bank)
    if not (bank / 'pool/pending').is_dir() or not (bank / 'pool/claims').is_dir() or not (bank / 'lanes').is_dir():
        raise ValueError('bank layout missing; cannot certify rollback boundary')
    new_work, active_jobs = [], []
    for directory, pattern in [('pool/pending', '*.json'), ('pool/claims', '*.claim')]:
        for path in sorted((bank / directory).glob(pattern)):
            task = json.loads(path.read_bytes())
            body = task['body']
            binding = body.get('task_lake')
            if binding is not None:
                pin = binding['validator']
                payload = Path(pin['spec_path']).read_bytes()
                if hashlib.sha256(payload).hexdigest() != pin['spec_sha256']:
                    raise ValueError('sealed spec drift; rollback boundary unknown')
                spec = json.loads(payload)
                if 'user_script' in spec or 'end_state_assertions' in spec:
                    new_work.append(str(path))
    for job in sorted((bank / 'lanes').glob('lane-*/corrections/job-*')):
        if not (job / 'spec.json').is_file():
            raise ValueError('durable job missing spec; rollback boundary unknown')
        spec = json.loads((job / 'spec.json').read_bytes())
        task_id = spec['task']['task_id']
        if not (job / 'published.json').exists():
            active_jobs.append(str(job))
        else:
            published = json.loads((job / 'published.json').read_bytes())
            if published['outcome']['task_id'] != task_id or not isinstance(published['published_unix'], (int, float)):
                raise ValueError('publication identity drift; rollback boundary unknown')
    return {'new_family_pending_or_claimed': new_work, 'unpublished_correction_jobs': active_jobs,
            'old_only_restore_allowed': not new_work and not active_jobs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    args = parser.parse_args()
    result = rollback_inventory(args.bank)
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result['old_only_restore_allowed'] else 1)
