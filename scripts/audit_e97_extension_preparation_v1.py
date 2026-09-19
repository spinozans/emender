#!/usr/bin/env python3
"""Fail-closed audit for the staged extension-preparation-v1 authority.

Mirrors the v2 repair-prep coverage pattern: the prep authority itself is
machine-verified end to end (manifest identity, payload identities, index
accounting, per-record mask sums, cohort table reconstruction from the
metadata + index, and every recorded cohort-authority binding re-hashed from
disk), and the receipt states plainly that this preparation is not admitted.
The generic proposal auditor (scripts/audit_e97_pi_native_repair_generic.py)
remains the staged mechanism for the extension's own segment proposals after
v10's gates, with operator sign-off.
"""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import struct

INDEX = struct.Struct('<QQQB7x')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(args):
    root = args.preparation
    manifest = json.loads((root / 'manifest.json').read_text())
    if sha(root / 'manifest.json') != args.manifest_sha256:
        raise ValueError('preparation manifest identity')
    if (manifest.get('schema') != 'emender-e97-tulu3-masked-sft-v1'
            or manifest.get('status') != 'complete'
            or manifest.get('tokenizer') != 'p50k_base'):
        raise ValueError('preparation schema')
    if (manifest.get('training_eligible') or manifest.get('packing_authorized')
            or manifest.get('optimizer_updates_authorized')):
        raise ValueError('preparation authorization')
    outputs = manifest['outputs']
    if set(outputs) != {'tokens', 'mask', 'index', 'metadata'}:
        raise ValueError('preparation outputs')
    paths = {}
    for name, spec in outputs.items():
        path = root / spec['path']
        if path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
            raise ValueError(f'preparation payload identity: {name}')
        paths[name] = path
    metadata = [json.loads(line) for line in paths['metadata'].open()]
    index = paths['index'].read_bytes()
    if len(index) != INDEX.size * len(metadata):
        raise ValueError('preparation index shape')
    tokens = paths['tokens']
    mask = paths['mask']
    if tokens.stat().st_size != 4 * mask.stat().st_size:
        raise ValueError('preparation payload shape')
    offset = 0
    records = targets = token_total = 0
    cohort_records, cohort_targets, cohort_tokens = Counter(), Counter(), Counter()
    for i, row in enumerate(metadata):
        start, n, want, split = INDEX.unpack_from(index, i * INDEX.size)
        if start != offset or split:
            raise ValueError('preparation index accounting')
        with mask.open('rb') as mf:
            mf.seek(start)
            actual = mf.read(n)
        if len(actual) != n or sum(actual) != want:
            raise ValueError('preparation mask reconstruction')
        with tokens.open('rb') as tf:
            tf.seek(4 * start)
            if len(tf.read(4 * n)) != 4 * n:
                raise ValueError('preparation token extent')
        source = row.get('source')
        if not isinstance(source, str) or not source:
            raise ValueError('preparation cohort label')
        offset += n
        records += 1
        targets += want
        token_total += n
        cohort_records[source] += 1
        cohort_targets[source] += want
        cohort_tokens[source] += n
    counts = manifest['counts']
    if (records != counts['records'] or token_total != counts['tokens']
            or targets != counts['assistant_target_tokens']
            or counts['train_records'] != records or counts['validation_records']):
        raise ValueError('preparation aggregate counts')
    if (dict(cohort_records) != manifest['source_record_counts']
            or dict(cohort_targets) != manifest['source_target_totals']):
        raise ValueError('preparation cohort table')
    bindings = []
    entries = [manifest[k] for k in ('openhands_rehearsal', 'conversation_rehearsal',
                                     'loopbreak_rehearsal', 'extra_rehearsal',
                                     'extra2_rehearsal', 'extra3_rehearsal',
                                     'extra4_rehearsal', 'authored_rehearsal',
                                     'correction_rehearsal')]
    entries.extend(manifest.get('spec_cohorts') or [])
    for entry in entries:
        if entry is None:
            continue
        if sha(Path(entry['authority']) / 'manifest.json') != entry['authority_sha256']:
            raise ValueError('cohort authority binding: %s' % entry['authority'])
        bindings.append(entry['authority_sha256'])
    selected_root = args.selected_root
    if (sha(selected_root / 'candidate-authority/manifest.json') != manifest['selected_authority_sha256']
            or sha(selected_root / 'selection-audit.json') != manifest['selected_selection_audit_sha256']
            or sha(selected_root / 'overlap-audit.json') != manifest['selected_overlap_audit_sha256']):
        raise ValueError('selected-curriculum provenance triple')
    parent = Path(manifest['parent_checkpoint'])
    if sha(parent) != manifest['parent_checkpoint_sha256']:
        raise ValueError('parent checkpoint identity')
    receipt = {
        'schema': 'emender-e97-extension-preparation-audit-v1',
        'status': 'qualified-preparation-not-admitted',
        'preparation_root': str(root.resolve()),
        'preparation_manifest_sha256': args.manifest_sha256,
        'records': records,
        'tokens': token_total,
        'assistant_target_tokens': targets,
        'cohort_table': {c: {'records': cohort_records[c],
                              'tokens': cohort_tokens[c],
                              'assistant_target_tokens': cohort_targets[c]}
                         for c in sorted(cohort_records)},
        'cohort_authority_sha256s': sorted(bindings),
        'selected_provenance_triple': {
            'authority': manifest['selected_authority_sha256'],
            'selection_audit': manifest['selected_selection_audit_sha256'],
            'overlap_audit': manifest['selected_overlap_audit_sha256']},
        'parent_checkpoint_sha256': manifest['parent_checkpoint_sha256'],
        'training_eligible': False,
        'packing_authorized': False,
        'optimizer_updates_authorized': 0,
        'checkpoint_promotion': False,
        'automatic_retry': False,
        'checker_sha256': sha(Path(__file__)),
        'note': ('segment proposals for the extension arc are frozen, audited with '
                 'scripts/audit_e97_pi_native_repair_generic.py and admitted only after '
                 "v10's gates, with operator sign-off"),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('EXTENSION_PREPARATION_AUDIT', records, token_total, targets, sha(args.output))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--selected-root', type=Path,
                  default=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-curriculum-2000-selected-v1'),
                  help='root containing candidate-authority/, selection-audit.json, overlap-audit.json')
    p.add_argument('--output', type=Path, required=True)
    audit(p.parse_args())


if __name__ == '__main__':
    main()
