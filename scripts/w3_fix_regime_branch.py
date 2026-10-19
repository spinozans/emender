#!/usr/bin/env python3
"""Fix W3 arm B only from X9b's verified successful 500K terminal receipt."""
import argparse
import json
from pathlib import Path

from scripts.w3_matcheddose_preflight import REPO, file_sha
from scripts.w3_attest_x9b_completion import ATTESTATION, completed_snapshot

X9B = REPO / 'experiments/expressivity_tasks/results_invertible_stage9b_sf_anneal_8m'
NAME = 'e97_bf16_seed0'


def fix_branch(root, attestation=ATTESTATION):
    if not attestation.exists():
        raise ValueError('WAIT: missing completed-trajectory terminal attestation; no intermediate counts read')
    try:
        terminal_evidence = json.loads(attestation.read_text())
        if (terminal_evidence['schema'] != 'emender-w3-x9b-completed-trajectory-terminal-v1'
                or terminal_evidence['status'] != 'complete' or terminal_evidence['name'] != NAME
                or terminal_evidence['total_steps'] != 500000 or terminal_evidence['validation_rows'] != 1001
                or terminal_evidence['validation_sha256'] != terminal_evidence['bindings']['validation']['sha256']):
            raise ValueError('malformed completed-trajectory terminal attestation')
        # Verify every snapshotted byte before parsing/reading any intermediate
        # counts; the digest includes every row in its original order.
        bindings, values = completed_snapshot(root, expected_bindings=terminal_evidence['bindings'])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('malformed completed-trajectory terminal attestation') from exc
    trajectory = values['validation']
    maxima = [0] * 16
    for row in trajectory:
        positions = row['positions']
        if ([p['driver_position'] for p in positions] != list(range(1, 17))
                or any(p['count'] != 512 or not 0 <= p['correct'] <= 512 for p in positions)):
            raise ValueError('X9b frozen position coverage invalid')
        maxima = [max(old, p['correct']) for old, p in zip(maxima, positions)]
    material = max(maxima[1:]) >= 50
    final = [p['correct'] for p in trajectory[-1]['positions']]
    return dict(schema='emender-w3-fixed-regime-v1', regime='sf-cosine' if material else 'adamw-cosine',
                criterion='max-anywhere P2+ >=50/512 over completed frozen trajectory; supervisor ruling before W3 results',
                threshold=50, max_p2_plus=max(maxima[1:]), maxima_per_position=maxima,
                fixed_final_per_position=final, escape_without_consolidation=material and max(final[1:]) < 50,
                evidence=[dict(path=str(attestation), sha256=file_sha(attestation)), *bindings.values()],
                optimizer=('schedulefree.AdamWScheduleFree' if material else 'torch.optim.AdamW'),
                betas=[.9, .95], eps=1e-8, weight_decay=.01, internal_warmup=0,
                schedule='lr(t)=1e-7 + .5*(1e-5-1e-7)*(1+cos(pi*(t-1)/511)), t=1..512',
                mechanism_claim='schedule-under-retained-SF local contrast' if material else 'regime bundle, not unique mechanism attribution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x9b-root', type=Path, default=X9B)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attestation', type=Path, default=ATTESTATION)
    args = parser.parse_args()
    result = fix_branch(args.x9b_root, args.attestation)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(result['regime'])


if __name__ == '__main__':
    main()
