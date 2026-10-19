#!/usr/bin/env python3
"""Read-only W3 mask recount and wash-2 instrument binding; never launches arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1')
PARENT_SHA = 'cb50c6586c91cb8f01e3c614b504ecc53dbad087a38af225b9e62ef762a24737'
AUTHORITY_SHA = '55aca12a4952338e576909d2e52541b5b5807ad14a970410309f740eff510184'
PACK_SHA = 'f9b3a5e103c5d94f692fcd736dda95cfea75c78bc9614d247c37019f1aacc17e'
REPO = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    path = Path(path)
    actual = file_sha(path)
    if actual != expected:
        raise ValueError(f'identity mismatch: {path}: {actual} != {expected}')
    return dict(path=str(path), sha256=actual, bytes=path.stat().st_size)


def independent_pack(pack_id, pack_index, pack_records, records, masks, tokens, context):
    """Decode little-endian indexes directly; exclude every record's first target.

    This deliberately does not call the training Dataset or trust target totals.
    Returned masks are prediction-aligned uint8 bytes, including padded positions.
    """
    first, count, expected_tokens, expected_targets = struct.unpack_from('<QQQQ', pack_index, pack_id * 32)
    record_ids = list(struct.unpack_from(f'<{count}I', pack_records, first * 4))
    loss_mask = bytearray(context)
    valid = bytearray(context + 1)
    reset = bytearray(context + 1)
    token_bytes = bytearray((context + 1) * 4)
    cursor = 0
    for record_id in record_ids:
        offset, length, _targets, split = struct.unpack_from('<QQQB7x', records, record_id * 32)
        if split != 0 or length <= 0 or cursor + length > context + 1:
            raise ValueError('invalid record membership/length')
        record_mask = masks[offset:offset + length]
        if len(record_mask) != length or any(v not in (0, 1) for v in record_mask):
            raise ValueError('invalid authority mask')
        token_bytes[cursor * 4:(cursor + length) * 4] = tokens[offset * 4:(offset + length) * 4]
        valid[cursor:cursor + length] = b'\x01' * length
        reset[cursor] = 1
        for local in range(1, length):
            loss_mask[cursor + local - 1] = record_mask[local]
        cursor += length
    targets = sum(loss_mask)
    if (cursor, targets) != (expected_tokens, expected_targets):
        raise ValueError('independent pack count differs from pack index')
    return dict(pack_id=f'pack-{pack_id:08d}', record_ids=record_ids,
                inputs=cursor, targets=targets, loss_mask_sha256=digest(loss_mask),
                valid_mask_sha256=digest(valid), reset_mask_sha256=digest(reset),
                tokens_sha256=digest(token_bytes))


def recount(root):
    authority, packs = root / 'wash-v2/authority', root / 'wash-v2/packs'
    bindings = [checked(authority / 'manifest.json', AUTHORITY_SHA),
                checked(packs / 'manifest.json', PACK_SHA)]
    for directory in (authority, packs):
        manifest = json.loads((directory / 'manifest.json').read_text())
        for descriptor in manifest['outputs'].values():
            binding = checked(directory / descriptor['path'], descriptor['sha256'])
            if binding['bytes'] != descriptor['bytes']:
                raise ValueError('payload length mismatch')
            bindings.append(binding)
    records = (authority / 'records.idx').read_bytes()
    masks = (authority / 'assistant_mask.uint8.bin').read_bytes()
    tokens = (authority / 'tokens.uint32.bin').read_bytes()
    pack_records = (packs / 'pack_records.uint32.bin').read_bytes()
    pack_index = (packs / 'train_packs.idx').read_bytes()
    log_path = root / 'wash-v2/logs/training.jsonl'
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    steps = [r for r in events if r['event'] == 'step']
    if [r['update'] for r in steps] != list(range(1, 513)):
        raise ValueError('historical step coverage')
    rows = []
    for step in steps:
        sample = step['sampled_pack']
        pack_id = int(sample['pack_id'].removeprefix('pack-'))
        row = independent_pack(pack_id, pack_index, pack_records, records, masks, tokens, 12288)
        if (row['record_ids'] != sample['record_ids'] or row['inputs'] != step['global_tokens']
                or row['targets'] != step['global_targets']):
            raise ValueError(f'historical mask/exposure mismatch at {step["update"]}')
        rows.append(dict(update=step['update'], sample_ids=step['rank_sample_ids'],
                         loss_weight=1.0 / row['targets'], **row))
    inputs, targets = sum(r['inputs'] for r in rows), sum(r['targets'] for r in rows)
    if (inputs, targets) != (3571182, 154386):
        raise ValueError(f'HARD STOP: exposure mismatch: {inputs}/{targets}')
    return dict(payloads=bindings, historical_log=dict(path=str(log_path), sha256=file_sha(log_path)),
                updates=512, inputs=inputs, targets=targets, sequence=rows,
                recount_method='direct struct index decoding + authority mask[1:] per record; prediction-aligned padded masks',
                loss_reduction='sum CE divided by each update target count; every target weighted 1/step_targets')


def instruments(root):
    path = root / 'bank-heldout-reads/wash2-u512-verdict/verdict-receipt.json'
    verdict = json.loads(path.read_text())
    pins = verdict['instrument_pins']
    bindings = [dict(path=str(path), sha256=file_sha(path))]
    for relative, expected in pins['sources'].items():
        bindings.append(checked(REPO / relative, expected))
    stage = pins['stage_b']
    for key in ('panel', 'args', 'cli_image', 'pi_inventory', 'stage_b_tools', 'tool_manifest'):
        path_key = 'args_json' if key == 'args' else key
        value = Path(stage[path_key])
        bindings.append(checked(value if value.is_absolute() else REPO / value, stage[key + '_sha256']))
    execution = pins['execution']
    bindings.append(checked(execution['image_manifest'], execution['image_manifest_sha256']))
    for relative, expected in execution['bundle_files'].items():
        bindings.append(checked(REPO / relative, expected))
    panel_path = root / 'bank-heldout-reads/wash2-u512-read-v4/panel.json'
    bindings.append(checked(panel_path, verdict['execution']['panel_sha256']))
    panel = json.loads(panel_path.read_text())
    if len(panel['cases']) != 96 or len(stage['case_ids']) != 14:
        raise ValueError('instrument case coverage')
    return dict(bindings=bindings, stage_b=stage, execution_panel=str(panel_path),
                case_ids=[c['id'] for c in panel['cases']],
                primary_execution_replica=0, primary_execution_ranks=[0, 1],
                stage_b_repetitions=3, execution_replicas=4,
                primary_weight_policy='train-y for parent/A/B; saved-x diagnostic only',
                retention_status='UNRESOLVED: no retention limits in wash-2 verdict; authority-chain investigation required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = args.root / 'wash-v1/checkpoints/wash_u256_y_seed_fresh_state.pt'
    receipt = dict(schema='emender-w3-preflight-v1', source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        preflight_script_sha256=file_sha(Path(__file__)),
        parent=checked(parent, PARENT_SHA), exposure=recount(args.root), instruments=instruments(args.root),
        launch_ready=False, branch='WAIT_X9b_FINAL',
        outstanding=['fresh-state conversion verification', 'ordinary AdamW production-precision viability',
                     'retention authority/bounds', 'X9b final receipt', 'preregistration commit',
                     'arm runner validation', 'pin-at-read and frozen gate runner validation'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(dict(updates=receipt['exposure']['updates'], inputs=receipt['exposure']['inputs'],
                          targets=receipt['exposure']['targets'], launch_ready=False)))


if __name__ == '__main__':
    main()
