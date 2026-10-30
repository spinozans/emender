"""Validate immutable completed anchor receipts before skipping training."""
import hashlib
import json
import math
from pathlib import Path


def complete(arm, root):
    manifest = json.loads((Path(root)/'completed_rows.json').read_text())
    if arm not in manifest:
        if (Path(root)/arm/'result.json').exists():
            raise ValueError('unregistered completed row requires review')
        return False
    entry = manifest[arm]
    path = Path(entry['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('completed row receipt changed: '+arm)
    row = json.loads(path.read_text())
    if not (row['status'] == 'completed' and row['score_eligible'] is True
            and row['train_seconds'] >= 2700 and row['train_budget_seconds'] == 2700
            and row['batch'] == 64 and row['context'] == 2048
            and row['eval_windows'] == 24 and math.isfinite(row['score'])
            and row.get('arm', 'pure') == arm):
        raise ValueError('invalid completed anchor: '+arm)
    return True


if __name__ == '__main__':
    import sys
    if complete(sys.argv[1], sys.argv[2]):
        print('preserving completed anchor: '+sys.argv[1])
        sys.exit(0)
    sys.exit(10)
