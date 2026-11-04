"""Fail-closed C14-B registration audit; never a certificate to train B1."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REGISTRATION = ROOT/'receipts/c14_b/registration.json'
PROTOCOL = ROOT/'PREREGISTRATION_C14_B.md'
KEY_AMENDMENT_COMMIT = '060445bc'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bindings(registration, repo=REPO):
    for name, binding in registration['files'].items():
        path = repo/name
        if sha(path) != binding['sha256']:
            raise ValueError(f'frozen dependency changed: {name}')
        blob = subprocess.check_output(
            ['git', 'rev-parse', registration['source_commit']+':'+name],
            cwd=repo, text=True).strip()
        if blob != binding['git_blob']:
            raise ValueError(f'frozen Git object changed: {name}')


def audit():
    if sys.executable != '/home/erikg/emender/.venv/bin/python':
        raise RuntimeError('project interpreter required')
    registration = json.loads(REGISTRATION.read_text())
    check_bindings(registration)
    # Locate the first commit containing the registration, rather than relying
    # on a mutable receipt's assertion about whether it was committed.
    relative = str(PROTOCOL.relative_to(REPO))
    commits = subprocess.check_output(
        ['git', 'log', '--diff-filter=A', '--format=%H', '--', relative],
        cwd=REPO, text=True).splitlines()
    if len(commits) != 1:
        raise ValueError('missing or ambiguous registration commit')
    commit = commits[0]
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
                   cwd=REPO, check=True)
    paths = subprocess.check_output(
        ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit],
        cwd=REPO, text=True).splitlines()
    expected = {relative, str(REGISTRATION.relative_to(REPO))}
    if set(paths) != expected:
        raise ValueError('registration was not committed alone')
    for path in (PROTOCOL, REGISTRATION):
        committed = subprocess.check_output(
            ['git', 'show', commit+':'+str(path.relative_to(REPO))], cwd=REPO)
        if path == PROTOCOL:
            if hashlib.sha256(committed).hexdigest() != registration['preregistration_sha256']:
                raise ValueError('original preregistration hash mismatch')
            subprocess.run(['git', 'merge-base', '--is-ancestor',
                            KEY_AMENDMENT_COMMIT, 'HEAD'], cwd=REPO, check=True)
            amended = subprocess.check_output(
                ['git', 'show', KEY_AMENDMENT_COMMIT+':'+relative], cwd=REPO)
            amendment_paths = subprocess.check_output(
                ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r',
                 KEY_AMENDMENT_COMMIT], cwd=REPO, text=True).splitlines()
            if amendment_paths != [relative] or not amended.startswith(committed):
                raise ValueError('key amendment must append to original registration alone')
            if amended != path.read_bytes():
                raise ValueError('preregistration changed after key amendment')
        elif committed != path.read_bytes():
            raise ValueError(f'registration commit mismatch: {path.name}')
    import torch
    from .c11 import Boosted
    from .c6 import count
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    model = Boosted('emender', 2000000)
    shapes = json.loads((ROOT/'receipts/c14_byte/model_shapes.json').read_text())
    expected_shapes = shapes['models'][0]['shapes']
    actual_shapes = {k: list(v.shape) for k, v in model.named_parameters()}
    if actual_shapes != expected_shapes or count(model) != 2133700:
        raise ValueError('standing model geometry changed')
    policies = [[{'slots': c.slots, 'decay_rate': c.decay_rate,
                  'free_threshold': c.free_threshold}
                 for c in b.cell.configs] for b in model.head.base.blocks]
    if [p['slots'] for p in policies[0]] != [32, 64] or len(policies) != 2:
        raise ValueError('standing physical budget changed')
    return dict(status='passed_registration_audit', registration_commit=commit,
                key_amendment_commit=KEY_AMENDMENT_COMMIT,
                preregistration_sha256=sha(PROTOCOL), frozen_bindings=len(registration['files']),
                trainable_parameters=count(model), standing_shape_parity=True,
                layer_policies=policies, gpu_used=False, training_started=False,
                launch_ready=False, remaining_gates=[
                    'Enabled shared-store consultation implementation',
                    'Native forward and checkpoint/replay backward parity and gradients',
                    'Cache oracle bit parity, causality, shared-budget and lifecycle tests',
                    'Full relevant test battery before training',
                    '2700 active-second broker-leased B1 training',
                    'Frozen-window paired evaluation and conditional ablations'])


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
