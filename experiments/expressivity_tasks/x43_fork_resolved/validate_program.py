"""Fail-closed audit of PROGRAM.md's committed receipt bindings (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)


def validate(program, repos):
    text = program.read_text()
    blocks = re.findall(r'^```json claim-bindings\n(.*?)\n```$', text, re.M | re.S)
    if len(blocks) != 1:
        raise ValueError('exactly one claim-bindings block required')
    ledger = json.loads(blocks[0])
    if ledger['schema'] != 1:
        raise ValueError('unsupported schema')
    claims = ledger['claims']
    expected = [f'C{i}' for i in range(1, 11)]
    if [c['id'] for c in claims] != expected:
        raise ValueError('require C1..C10 exactly once in order')
    headings = re.findall(r'^### (C\d+) — (registered|measured|refuted):', text, re.M)
    if headings != [(c['id'], c['status']) for c in claims]:
        raise ValueError('claim headings and binding statuses disagree')
    count = 0
    for claim in claims:
        status, receipts = claim['status'], claim['receipts']
        if status not in {'registered', 'measured', 'refuted'}:
            raise ValueError(f"{claim['id']}: invalid status")
        if (status == 'registered') != (len(receipts) == 0):
            raise ValueError(f"{claim['id']}: registered requires no receipts; outcomes require receipts")
        seen = set()
        for receipt in receipts:
            repo = repos[receipt['repo']]
            path, commit, digest = (receipt[k] for k in ('path', 'commit', 'sha256'))
            if Path(path).is_absolute() or '..' in Path(path).parts or Path(path).suffix == '.pt':
                raise ValueError(f'unsafe or checkpoint receipt path: {path}')
            if not re.fullmatch(r'[0-9a-f]{40}', commit) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError(f'{path}: full commit and SHA-256 required')
            key = (receipt['repo'], path, commit)
            if key in seen:
                raise ValueError(f'duplicate binding: {key}')
            seen.add(key)
            if git(repo, 'rev-parse', commit + '^{commit}').decode().strip() != commit:
                raise ValueError(f'{path}: not an exact commit')
            committed = git(repo, 'show', commit + ':' + path)
            current = (repo / path).read_bytes()
            if hashlib.sha256(committed).hexdigest() != digest:
                raise ValueError(f'{path}: committed hash mismatch')
            if hashlib.sha256(current).hexdigest() != digest:
                raise ValueError(f'{path}: working receipt differs from pinned commit')
            count += 1
    return {'status': 'green', 'claims': len(claims), 'bindings': count,
            'measured': sum(c['status'] == 'measured' for c in claims),
            'refuted': sum(c['status'] == 'refuted' for c in claims),
            'registered': sum(c['status'] == 'registered' for c in claims)}


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', type=Path, default=Path(__file__).with_name('PROGRAM.md'))
    parser.add_argument('--sxgc-repo', type=Path, default=root.parent / 'sxgc')
    args = parser.parse_args()
    try:
        result = validate(args.program, {'emender': root, 'sxgc': args.sxgc_repo})
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({'status': 'red', 'error': str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
