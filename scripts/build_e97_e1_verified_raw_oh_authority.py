#!/usr/bin/env python3
"""Build the E1 verified-raw-OH cohort authority (non-authorizing data authority).

Selects, from the qualified OpenHands source-native fulltraj pool
(e97-4b-open-swe-source-native-fulltraj-v1), exactly the RAW records whose
trajectories the translated+replay-verified collection
(e97-oh-pi-native-translation-v1/candidate-authority, 5,087 records) verified:
each translated record carries (instance_id, trajectory_id); each raw record
carries problem_key = [repo, instance_id] and trajectory_identity =
"open-swe:<uuid>". The mapping is trajectory_id -> raw trajectory_identity uuid,
cross-checked against instance_id == problem_key[1].

The records are copied BYTE-EXACT in their NATIVE OH format (tokens.bin +
loss_mask.bin slices): this cohort is the execution dialect's training data,
NOT a translation. The authority uses the tulu3 masked-sft payload layout so
the standard preparer spec-cohort reader can budget-select from it; the
manifest omits training_eligible (a plain data authority; eligibility flows
only from the E1 preparation's admission chain — the prep itself remains
training_eligible:false).

Outputs: tokens.uint32.bin, assistant_mask.uint8.bin, records.idx,
records.jsonl, id-mapping.jsonl (the documented verified-id -> raw-record
mapping), manifest.json.
"""
import argparse, hashlib, json, struct
from pathlib import Path

INDEX = struct.Struct('<QQQB7x')
RAW_INDEX = struct.Struct('<QQQB7x')  # same layout for records.idx


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--translated-root', type=Path, required=True,
                   help='e97-oh-pi-native-translation-v1/candidate-authority')
    p.add_argument('--translated-sha256', required=True)
    p.add_argument('--raw-root', type=Path, required=True,
                   help='e97-4b-open-swe-source-native-fulltraj-v1')
    p.add_argument('--raw-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    translated_manifest_path = args.translated_root / 'manifest.json'
    if sha(translated_manifest_path) != args.translated_sha256:
        raise ValueError('translated collection manifest identity')
    translated_manifest = json.loads(translated_manifest_path.read_text())
    if (translated_manifest.get('schema') != 'emender-e97-tulu3-masked-sft-v1'
            or translated_manifest.get('status') != 'complete'):
        raise ValueError('translated collection schema')
    translated_rows = [json.loads(x) for x in
                       (args.translated_root / 'records.jsonl').read_text().splitlines()]
    translated_index = (args.translated_root / 'records.idx').read_bytes()
    if len(translated_index) != INDEX.size * len(translated_rows):
        raise ValueError('translated index shape')

    raw_manifest_path = args.raw_root / 'manifest.json'
    if sha(raw_manifest_path) != args.raw_sha256:
        raise ValueError('raw pool manifest identity')
    raw_manifest = json.loads(raw_manifest_path.read_text())
    if raw_manifest.get('profile') != 'e97-open-swe-source-native-v1':
        raise ValueError('raw pool profile')
    raw_rows = [json.loads(x) for x in (args.raw_root / 'records.jsonl').read_text().splitlines()]
    raw_index = (args.raw_root / 'records.idx').read_bytes()
    if len(raw_index) != RAW_INDEX.size * len(raw_rows):
        raise ValueError('raw pool index shape')
    raw_outputs = raw_manifest['outputs']
    for name in ('tokens.bin', 'loss_mask.bin'):
        spec = raw_outputs[name]
        path = args.raw_root / name
        if path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
            raise ValueError(f'raw pool payload identity: {name}')

    # verified id set: (instance_id, trajectory_id) from the translated collection
    verified = {}
    for row in translated_rows:
        key = (row['instance_id'], row['trajectory_id'])
        if key in verified:
            raise ValueError(f'duplicate verified id pair: {key}')
        verified[key] = row
    if len(verified) != translated_manifest['counts']['records']:
        raise ValueError('verified id count')

    raw_by_tid = {}
    for i, row in enumerate(raw_rows):
        ident = row.get('trajectory_identity')
        if not isinstance(ident, str) or not ident.startswith('open-swe:'):
            raise ValueError(f'raw record {i} trajectory identity')
        tid = ident.split(':', 1)[1]
        if tid in raw_by_tid:
            raise ValueError(f'duplicate raw trajectory id: {tid}')
        raw_by_tid[tid] = i

    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    out = args.output
    paths = {'tokens': out / 'tokens.uint32.bin', 'mask': out / 'assistant_mask.uint8.bin',
             'index': out / 'records.idx', 'metadata': out / 'records.jsonl',
             'idmap': out / 'id-mapping.jsonl'}
    mapping_records = []
    matched_keys = []
    with paths['tokens'].open('xb') as tf, paths['mask'].open('xb') as mf, \
            paths['index'].open('xb') as ix, paths['metadata'].open('x') as meta, \
            paths['idmap'].open('x') as idmap, \
            open(args.raw_root / 'tokens.bin', 'rb') as rtf, \
            open(args.raw_root / 'loss_mask.bin', 'rb') as rmf:
        offset = records = targets = total_tokens = train_records = train_targets = 0
        for (instance_id, trajectory_id), translated_row in verified.items():
            i = raw_by_tid.get(trajectory_id)
            if i is None:
                raise ValueError(f'verified trajectory absent from raw pool: {trajectory_id}')
            raw_row = raw_rows[i]
            if raw_row['problem_key'][1] != instance_id:
                raise ValueError(f'id mapping mismatch: {instance_id} vs {raw_row["problem_key"][1]}')
            roff, rn, rwant, rsplit = RAW_INDEX.unpack_from(raw_index, i * RAW_INDEX.size)
            if rn != raw_row['tokens'] or rwant != raw_row['targets'] or rsplit != raw_row['split']:
                raise ValueError(f'raw index accounting: record {i}')
            # NOTE: the raw pool carries its own train/validation split; the
            # verified set includes 383 validation-split trajectories. The
            # authority keeps the split flag so the preparer's readers (and
            # every consumer) skip validation rows exactly as they do for the
            # parent pool; the split census is recorded in the manifest.
            rtf.seek(4 * roff)
            tokens = rtf.read(4 * rn)
            rmf.seek(roff)
            mask = rmf.read(rn)
            if len(tokens) != 4 * rn or len(mask) != rn or sum(mask) != rwant:
                raise ValueError(f'raw record slice: record {i}')
            row = dict(raw_row)
            row['id'] = f"verified-raw-oh-{records:05d}"
            row['source'] = 'verified-raw-oh'
            row['verified_provenance'] = {
                'translated_instance_id': instance_id,
                'translated_trajectory_id': trajectory_id,
                'translated_record_sha256': translated_row.get('sha256'),
                'raw_record_index': i,
            }
            tf.write(tokens)
            mf.write(mask)
            ix.write(INDEX.pack(offset, rn, rwant, rsplit))
            meta.write(json.dumps(row, sort_keys=True) + '\n')
            idmap.write(json.dumps({
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'translated_record_index': translated_row.get('record_index'),
                'raw_record_index': i,
                'raw_problem_key': raw_row['problem_key'],
                'raw_source_file': raw_row.get('source_file'),
                'raw_source_row': raw_row.get('source_row'),
                'tokens': rn, 'targets': rwant}, sort_keys=True) + '\n')
            mapping_records.append(i)
            offset += rn
            records += 1
            targets += rwant
            total_tokens += rn
            train_records += 1 if rsplit == 0 else 0
            train_targets += rwant if rsplit == 0 else 0

    unmatched_raw = len(raw_rows) - records
    manifest = {
        'schema': 'emender-e97-tulu3-masked-sft-v1',
        'status': 'complete',
        'purpose': ('E1 verified-raw-OH cohort authority: the 5,087 replay-verified '
                    'OpenHands trajectories in their NATIVE OH format, selected from '
                    'the qualified source-native fulltraj pool by the verified '
                    '(instance_id, trajectory_id) pairs recorded by the translated '
                    'collection. Byte-exact payload copies; NOT translated; the '
                    'execution dialect\'s training data per the v10 scrub-reversal '
                    'finding and the v6 passing recipe (raw-OH family demonstrably '
                    'fed execution).'),
        'tokenizer': 'p50k_base',
        'packing_authorized': False,
        'optimizer_updates_authorized': 0,
        'admission_note': ('plain data authority; training eligibility flows only from '
                           'the E1 preparation admission chain (the prep manifest stays '
                           'training_eligible:false); the raw parent pool declares '
                           'admission blockers that the E1 operator directive (verified '
                           'subset at 15-25% of targets) stages for this prep'),
        'counts': {'records': records, 'train_records': train_records,
                   'validation_records': records - train_records,
                   'tokens': total_tokens, 'assistant_target_tokens': targets},
        'split_census': {
            'note': ('the replay-verified set includes 383 raw-pool validation-split '
                     'trajectories; consumers skip them via the split flag as for the '
                     'parent pool — every prior prep cohort is train-split only'),
            'train_records': train_records,
            'validation_records': records - train_records,
            'train_targets': train_targets,
        },
        'id_mapping': {
            'method': ('translated record (instance_id, trajectory_id) -> raw record '
                       'trajectory_identity "open-swe:<trajectory_id>"; cross-checked '
                       'instance_id == raw problem_key[1]; 1:1, no orphans on either side'),
            'verified_records': len(verified),
            'matched_raw_records': records,
            'raw_records_not_verified_excluded': len(raw_rows) - records,
            'artifact': 'id-mapping.jsonl',
        },
        'sources': {
            'translated_collection': {
                'authority': str(args.translated_root.resolve()),
                'manifest_sha256': args.translated_sha256,
                'records': translated_manifest['counts']['records'],
                'replay_passed': translated_manifest['provenance']['replay_passed'],
                'replay_dropped': translated_manifest['provenance']['replay_dropped'],
            },
            'raw_pool': {
                'authority': str(args.raw_root.resolve()),
                'manifest_sha256': args.raw_sha256,
                'records': raw_manifest['counts']['records'],
                'targets': raw_manifest['counts']['targets'],
                'tokens': raw_manifest['counts']['tokens'],
                'dataset_id': raw_manifest['dataset_id'],
            },
        },
        'outputs': {k: {'path': v.name, 'bytes': v.stat().st_size, 'sha256': sha(v)}
                    for k, v in paths.items()},
    }
    tmp = out / 'manifest.json.partial'
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    tmp.replace(out / 'manifest.json')
    print('E1_VERIFIED_RAW_OH_AUTHORITY', records, total_tokens, targets,
          sha(out / 'manifest.json'))


if __name__ == '__main__':
    main()
