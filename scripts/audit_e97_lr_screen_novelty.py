#!/usr/bin/env python3
"""Aggregate-only exact marker overlap audit; no semantic/family claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tiktoken
from ndm.e97_atomic import publish_bytes_no_replace


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    expected = '08c16e0fa08b960ada87e197ba94f1a842dd0062691aaf28911f1df6bc552247'
    mp = args.authority / 'manifest.json'
    assert file_hash(mp) == expected
    manifest = json.loads(mp.read_text())
    panel = json.loads(args.panel.read_text())
    markers = {t['marker'] for t in panel['tasks']}
    assert len(markers) == len(panel['tasks']) == 24
    sources = []
    collisions = set()
    descriptor = manifest['outputs']['tokens']
    path = args.authority / descriptor['path']
    assert path.name == descriptor['path'] and file_hash(path) == descriptor['sha256']
    data = np.memmap(path, dtype='<u4', mode='r')
    encoding = tiktoken.get_encoding('p50k_base')
    for offset in range(0, len(data), 1000000):
        text = encoding.decode(data[max(0, offset-128):offset+1000000].tolist())
        collisions.update(m for m in markers if m in text)
    sources.append({'path': str(path), 'sha256': descriptor['sha256']})
    # Only aggregate collision evidence is emitted from consumed protected panels.
    for name in ('pi-native-core-v1', 'pi-core-eval-v2-template-heldout',
                 'pi-core-eval-v3-blind-family-heldout', 'pi-core-eval-v4-post-broad-heldout'):
        path = Path('/mnt/nvme1n1/erikg/sft') / name / 'records.jsonl'
        with path.open() as f:
            for line in f:
                collisions.update(m for m in markers if m in line)
        sources.append({'path': str(path), 'sha256': file_hash(path)})
    result = {'schema': 'emender-e97-lr-screen-exact-marker-audit-v1',
              'status': 'passed' if not collisions else 'failed',
              'panel_sha256': file_hash(args.panel), 'authority_sha256': expected,
              'markers': len(markers), 'colliding_markers': len(collisions),
              'sources': sources, 'scope': 'exact unique task markers only; familiar family/template overlap remains',
              'production_admission': False, 'training_eligible': False,
              'auditor_sha256': file_hash(Path(__file__))}
    publish_bytes_no_replace(args.output, (json.dumps(result, indent=2, sort_keys=True)+'\n').encode(), mode=0o600)
    print(json.dumps(result))
    if collisions:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
