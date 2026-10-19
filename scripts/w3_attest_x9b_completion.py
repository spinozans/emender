#!/usr/bin/env python3
"""Completion-only X9b terminal attestation; never edits live/historical receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from scripts.w3_matcheddose_preflight import REPO, file_sha, digest

NAME = 'e97_bf16_seed0'
X9B = REPO / 'experiments/expressivity_tasks/results_invertible_stage9b_sf_anneal_8m'
ATTESTATION = REPO / 'receipts/w3-matcheddose/x9b-completed-trajectory.json'


def completed_snapshot(root, expected_bindings=None):
    """Snapshot full terminal/trajectory bytes once, then check the terminal chain."""
    paths = dict(marker=root / 'markers' / f'{NAME}.DONE',
                 terminal=root / 'receipts' / NAME / 'terminal.json',
                 result=root / NAME / 'result.json', validation=root / NAME / 'validation.json',
                 config=root / NAME / 'config.json', data_receipt=root / NAME / 'data_receipt.json',
                 frozen_validation=root / NAME / 'frozen_validation.npz')
    if (root / 'markers' / f'{NAME}.FAIL').exists() or not paths['marker'].exists():
        raise ValueError('WAIT: X9b has no successful final receipt; partial trajectory is not branch evidence')
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    bindings = {name: dict(path=str(paths[name].resolve()), sha256=digest(payload))
                for name, payload in payloads.items()}
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError('X9b completed-trajectory terminal digest mismatch; no intermediate counts read')
    values = {name: json.loads(payload) for name, payload in payloads.items() if name != 'frozen_validation'}
    marker, terminal, result = (values[key] for key in ('marker', 'terminal', 'result'))
    if (marker != terminal or terminal.get('name') != NAME or terminal.get('exit_code') != 0
            or terminal.get('result_sha256') != digest(payloads['result'])
            or result.get('status') != 'complete' or result.get('total_steps') != 500000
            or result.get('stop_reason') != 'safety_ceiling' or result.get('name') != NAME):
        raise ValueError('X9b terminal binding invalid')
    trajectory = values['validation']
    if (not isinstance(trajectory, list) or len(trajectory) != 1001
            or [row['step'] for row in trajectory] != list(range(0, 500001, 500))
            or result['final'] != trajectory[-1]):
        raise ValueError('X9b completed frozen trajectory coverage/binding invalid')
    config = values['config']
    if (config.get('name') != NAME or config.get('safety_ceiling') != 500000
            or config.get('optimizer') != 'AdamWScheduleFree' or config.get('internal_warmup') != 0
            or config.get('eval_examples') != 512 or config.get('eval_every') != 500):
        raise ValueError('X9b configuration binding invalid')
    if (values['data_receipt'].get('sha256') != digest(payloads['frozen_validation'])
            or values['data_receipt'].get('examples') != 512):
        raise ValueError('X9b frozen data binding invalid')
    return bindings, values


def attest(root):
    bindings, _values = completed_snapshot(root)
    return dict(schema='emender-w3-x9b-completed-trajectory-terminal-v1', status='complete',
                name=NAME, total_steps=500000, validation_rows=1001, bindings=bindings,
                validation_sha256=bindings['validation']['sha256'],
                method='completion-only immutable snapshot of all validation bytes/order; original DONE/terminal/result/config/data retained untouched',
                publication='create-once supplemental terminal evidence; selector must verify before reading intermediate counts')


def require_committed(path):
    relative = str(path.resolve().relative_to(REPO))
    committed = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=REPO)
    if digest(committed) != file_sha(path):
        raise ValueError(f'attestor/test evidence is not committed: {relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x9b-root', type=Path, default=X9B)
    parser.add_argument('--output', type=Path, default=ATTESTATION)
    args = parser.parse_args()
    evidence = [Path(__file__), REPO / 'tests/test_w3_regime_and_gate.py',
                REPO / 'receipts/w3-matcheddose/tests-attestor.txt',
                REPO / 'receipts/w3-matcheddose/tests-attestor-exit.txt']
    for path in evidence:
        require_committed(path)
    if evidence[-1].read_text().strip() != '0':
        raise ValueError('attestor tests did not pass')
    report = attest(args.x9b_root)
    report['attestor_source_sha256'] = file_sha(Path(__file__))
    report['test_evidence'] = [dict(path=str(p), sha256=file_sha(p)) for p in evidence[1:]]
    report['attestor_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print('W3_X9B_COMPLETED_TRAJECTORY_ATTESTED', report['validation_sha256'])


if __name__ == '__main__':
    main()
