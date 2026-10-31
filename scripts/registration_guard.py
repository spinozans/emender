"""Diagnose source bindings without importing training or touching a GPU.

History explains committed drift; it never authorizes a protocol amendment.
Rebind writes an explicit receipt and refuses dirty or unproven source hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PYTHON = '/home/erikg/emender/.venv/bin/python'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bindings(raw):
    if raw.lstrip().startswith(b'{'):
        return json.loads(raw)['sources']
    return {line.split(maxsplit=1)[1].strip(): line.split()[0]
            for line in raw.decode().splitlines() if line.strip()}


def recipe(path):
    return (f'{PYTHON} scripts/registration_guard.py rebind {shlex.quote(str(path))} '
            f"--receipt {shlex.quote(str(path)+'.rebind-')}$(git rev-parse --short HEAD).json --cause HEAD")


def diagnose(path, root='.'):
    root = Path(root).resolve()
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(root)
    rows = []
    for name, expected in bindings((root/path).read_bytes()).items():
        history_root, history_name = root, name
        if Path(name).is_absolute():
            try:
                history_root = Path(git(Path(name).parent, 'rev-parse', '--show-toplevel').decode().strip())
                history_name = str(Path(name).relative_to(history_root))
            except (subprocess.CalledProcessError, ValueError):
                rows.append(dict(source=name, status='TAMPER SUSPECT', registered=expected,
                                 head=None, working=digest(Path(name).read_bytes()) if Path(name).is_file() else None,
                                 commits=[], reason='external repository history unavailable'))
                continue
        try:
            head = digest(git(history_root, 'show', f'HEAD:{history_name}'))
        except subprocess.CalledProcessError:
            head = None
        file = root/name
        work = digest(file.read_bytes()) if file.is_file() else None
        commits = []
        known = expected == head
        if not known:
            for commit in git(history_root, 'rev-list', 'HEAD', '--', history_name).decode().splitlines():
                try:
                    historical = digest(git(history_root, 'show', f'{commit}:{history_name}'))
                except subprocess.CalledProcessError:
                    historical = None
                if historical == expected:
                    known = True
                    break
                commits.append(git(history_root, 'show', '-s', '--format=%h %s', commit).decode().strip())
        if not known or (work != head and expected != head and work != expected):
            status = 'TAMPER SUSPECT'
        elif work != head:
            status = 'WORKING TREE DIRTY vs HEAD'
        elif expected != head:
            status = 'REGISTRATION STALE vs HEAD'
        else:
            status = 'MATCH'
        rows.append(dict(source=name, status=status, registered=expected,
                         head=head, working=work, commits=commits))
    return dict(registration=str(path), matches=all(r['status']=='MATCH' for r in rows),
                sources=rows, rebind_command=recipe(path))


def precommit(root='.'):
    """Compare the proposed index, including partially staged rebindings."""
    changed = set(git(root, 'diff', '--cached', '--name-only', '-z').decode().split('\0'))
    warnings = []
    for name in git(root, 'ls-files', '-z').decode().split('\0'):
        if not (name.endswith('registration.json') or name.endswith('/sources.sha256')
                or name.endswith('/velocity_regime.json')):
            continue
        try:
            sources = bindings(git(root, 'show', f':{name}'))
        except (subprocess.CalledProcessError, ValueError, KeyError):
            continue
        for source in changed & sources.keys():
            try:
                actual = digest(git(root, 'show', f':{source}'))
            except subprocess.CalledProcessError:
                actual = None
            if actual != sources[source]:
                warnings.append(f'WARNING: {source}: committing without registration re-bind: {name}\n{recipe(name)}')
    return warnings


def rebind(path, receipt, cause, root='.'):
    report = diagnose(path, root)
    if any(r['status'] in ('TAMPER SUSPECT', 'WORKING TREE DIRTY vs HEAD') for r in report['sources']):
        raise ValueError('refusing rebind: dirty or unproven source identity')
    target = Path(root)/path
    before = target.read_bytes()
    sources = {r['source']: r['head'] for r in report['sources']}
    if before.lstrip().startswith(b'{'):
        data = json.loads(before)
        data['sources'] = sources
        after = (json.dumps(data, indent=2)+'\n').encode()
    else:
        after = ''.join(f'{value}  {name}\n' for name, value in sources.items()).encode()
    evidence = dict(base_head=git(root, 'rev-parse', 'HEAD').decode().strip(),
                    cause=cause, old_sha256=digest(before), new_sha256=digest(after), **report)
    with Path(receipt).open('x') as output:
        output.write(json.dumps(evidence, indent=2)+'\n')
    target.write_bytes(after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check','pre-commit','rebind'))
    parser.add_argument('registration', nargs='?')
    parser.add_argument('--receipt')
    parser.add_argument('--cause')
    args = parser.parse_args()
    root = Path(git('.', 'rev-parse', '--show-toplevel').decode().strip())
    if args.action == 'pre-commit':
        for warning in precommit(root): print(warning)
        return 0  # Loud advisory: do not bypass other hooks.
    if not args.registration: parser.error('registration required')
    if args.action == 'rebind':
        if not args.receipt or not args.cause: parser.error('--receipt and --cause required')
        rebind(args.registration, args.receipt, args.cause, root)
        return 0
    report = diagnose(args.registration, root)
    print(json.dumps(report, indent=2))
    return 0 if report['matches'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
