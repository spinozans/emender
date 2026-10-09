#!/usr/bin/env python3
"""Local first-party seed intake; no requests/downloads and no admission."""
import hashlib
import json
import os
from pathlib import Path

from scripts.e97_diversity_author import harvest_solved, digest
import standing_supply

WORK = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1')
seeds = harvest_solved(WORK, standing_supply.TASK_LAKE, standing_supply.verify_admitted, count=8)
seeds += [
    {'kind': 'code', 'payload': 'def mean(xs):\n    return sum(xs) // len(xs)\n', 'provenance': {'origin': 'invented'}},
    {'kind': 'config', 'payload': 'archive_enabled=false\nretention_days=seven\n', 'provenance': {'origin': 'invented'}},
    {'kind': 'tree', 'payload': {'inbox/': ['records.csv', 'notes.md'], 'archive/': []}, 'provenance': {'origin': 'invented'}},
    {'kind': 'error', 'payload': 'ValueError: invalid literal for int(): seven', 'provenance': {'origin': 'invented'}},
]
output = Path(os.environ['F9_AUTHOR_WORK']) / 'seed-corpus.json'
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x') as handle:
    json.dump(seeds, handle, sort_keys=True, indent=2)
os.chmod(output, 0o400)
manifest = {'schema': 'emender-era8-local-seed-corpus-v1', 'seeds': len(seeds),
            'corpus_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'items': [{'kind': s['kind'], 'seed_sha256': digest(s), 'provenance': s['provenance']} for s in seeds]}
(output.parent / 'seed-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'path': str(output), 'seeds': len(seeds), 'corpus_sha256': manifest['corpus_sha256']}))
