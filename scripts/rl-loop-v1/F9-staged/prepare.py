#!/usr/bin/env python3
"""Create a private era8 checkout/authority overlay. Never deploy or admit.

Run with the project interpreter on lambda01. Operator must stop/restart supply
and affected collectors before deploying this coordinated versioned closure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path('/home/erikg/emender')
WORK = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1')
LOOP = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1')
SOURCE = Path(__file__).resolve().parent


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((REPO / 'configs/pi/e97-firstparty-generator-manifest-v1.json').read_bytes())
    members = [c['path'] for c in manifest['components']]
    for name in members:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    for name in ('e97-onpolicy-source-registry-v1.json', 'e97-onpolicy-source-policy-v1.json', 'e97-firstparty-collection-authorizations-v1.json'):
        source = REPO / 'configs/pi' / name
        if source.is_file():
            shutil.copyfile(source, root / 'configs/pi' / name)
    # Extend import paths to unchanged project packages, but load staged changed
    # roots first. Nothing in the live checkout/lake is overwritten.
    for package in ('ndm', 'scripts'):
        (root / package / '__init__.py').write_text('from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n')
    for directory in ('scripts', 'ndm', 'configs', 'loop'):
        for path in (SOURCE / directory).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            target = root / path.relative_to(SOURCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copyfile(WORK / 'scripts/lake_expand.py', root / 'loop/lake_expand.py')
    original = WORK / 'scripts/teacher_author.py'
    teacher = root / 'loop/teacher_author.py'
    original_bytes = original.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != '6881b6330cc7949bc6d21bc597c44f2a37cab6fa7ca91ec5681830d36d0449fe':
        raise ValueError('live teacher source changed; explicitly rebase/review F9 patch before preparing')
    teacher.write_bytes(original_bytes)
    subprocess.run(['patch', '--batch', str(teacher), str(SOURCE / 'teacher_author.patch')], check=True)
    members.append('scripts/e97_first_party_validator_diversity.py')
    manifest['components'] = [{'path': name, 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in sorted(members)]
    (root / 'configs/pi/e97-firstparty-generator-manifest-v1.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, F9_STAGE_ROOT=str(root), F9_AUTHOR_WORK=str(root / 'author-work'),
               PYTHONPATH=f'{root}:{root / "loop"}:{REPO}:{LOOP / "scripts"}')
    # Run closure verification in a NEW interpreter so it cannot accidentally
    # mix live and staged ndm modules already imported by this preparer.
    subprocess.run([str(REPO / '.venv/bin/python'), '-m', 'scripts.build_e97_first_party_source_archive',
                    '--manifest', str(root / 'configs/pi/e97-firstparty-generator-manifest-v1.json'),
                    '--archive', str(root / 'configs/pi/e97-firstparty-source-v1.tar'),
                    '--checkout-root', str(root)], cwd=root, env=env, check=True)
    registry_path = root / 'configs/pi/e97-onpolicy-source-registry-v1.json'
    registry = json.loads(registry_path.read_bytes())
    archive_sha = hashlib.sha256((root / 'configs/pi/e97-firstparty-source-v1.tar').read_bytes()).hexdigest()
    for registered in registry['sources']:
        if registered['kind'] == 'first-party' and registered['status'] == 'admitted':
            registered['receipts']['source_archive_sha256'] = archive_sha
    registry_path.write_text(json.dumps(registry, sort_keys=True, indent=2) + '\n')
    (root / 'activate.sh').write_text(f'export F9_STAGE_ROOT={root}\nexport F9_AUTHOR_WORK={root}/author-work\nexport PYTHONPATH={env["PYTHONPATH"]}\n')
    print(f'Staged only: source {root}/activate.sh; use project interpreter. No deployment or admission performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root)
