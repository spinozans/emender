#!/usr/bin/env python3
"""Assemble replay-verified OH->Pi-native records into a tulu3-schema authority.

Final phase of T1: packs the verified collection (produced by
scripts/verify_e97_oh_translation_replay.py) into the emender-e97-tulu3-masked-sft-v1
payload layout the repair preparer consumes: tokens.uint32.bin (p50k),
assistant_mask.uint8.bin (1 on supervised five-line assistant turns only),
records.idx ('<QQQB7x' offset/tokens/targets/split), records.jsonl metadata, and
a manifest with training_eligible=false (admission flips it later).

Records exceeding the 65536-token context bound are excluded and counted.
No oracle metadata is copied into the payload.
"""
import argparse, hashlib, json, struct
from pathlib import Path

RS = '\x1e'
INDEX = struct.Struct('<QQQB7x')
REPO_ROOT = Path(__file__).resolve().parent.parent
sys_path = str(REPO_ROOT)
if sys_path not in __import__('sys').path:
    __import__('sys').path.insert(0, sys_path)
import tiktoken  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def piecewise_encode(text, enc):
    """Linear piecewise tokenization: encode (segment, is_target) pieces.

    The record is a sequence of '\n\nLabel:\n<payload>' blocks; assistant
    turn payloads are the supervised targets. Encoding piecewise matches the
    open-swe authority convention (render() pieces) and is O(text); BPE
    boundary effects are confined to block separators, as in every prior
    authority built this way.
    """
    marker = '\n\nAssistant:\n'
    labels = ('\n\nToolResult:', '\n\nAssistant:', '\n\nUser:', '\n\nSystem:')
    pieces = []
    pos = 0
    while True:
        i = text.find(marker, pos)
        if i < 0:
            pieces.append((text[pos:], False))
            break
        pieces.append((text[pos:i + len(marker)], False))
        start = i + len(marker)
        end = len(text)
        for label in labels:
            j = text.find(label, start)
            if 0 <= j < end:
                end = j
        if text.endswith(RS) and end == len(text):
            end = len(text) - 1
        pieces.append((text[start:end], True))
        pos = end
    ids = []
    mask = []
    for seg, is_target in pieces:
        seg_ids = enc.encode_ordinary(seg)
        ids.extend(seg_ids)
        mask.extend([1] * len(seg_ids) if is_target else [0] * len(seg_ids))
    return ids, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verified-dir', type=Path, required=True)
    p.add_argument('--replay-report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--context-tokens', type=int, default=65536)
    args = p.parse_args()

    enc = tiktoken.get_encoding('p50k_base')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    records = sorted(args.verified_dir.glob('*.json'))
    tokens_f = open(out / 'tokens.uint32.bin', 'wb')
    mask_f = open(out / 'assistant_mask.uint8.bin', 'wb')
    idx_f = open(out / 'records.idx', 'wb')
    meta_f = open(out / 'records.jsonl', 'w')

    offset = 0
    total_targets = 0
    counts = {'records': 0, 'kept': 0, 'over_context': 0, 'empty_targets': 0}
    for rp in records:
        d = json.loads(rp.read_text())
        text = d['record_text']
        ids, mask = piecewise_encode(text, enc)
        if len(ids) > args.context_tokens:
            counts['over_context'] += 1
            continue
        want = sum(mask)
        if want == 0:
            counts['empty_targets'] += 1
            continue
        payload = struct.pack(f'<{len(ids)}I', *ids)
        tokens_f.write(payload)
        mask_f.write(bytes(mask))
        idx_f.write(INDEX.pack(offset, len(ids), want, 0))
        row = {'trajectory_id': d['trajectory_id'], 'instance_id': d['instance_id'],
               'problem_key': d['problem_key'], 'source': 'oh-pi-native-replay-verified',
               'sha256': hashlib.sha256(text.encode()).hexdigest(),
               'offset': offset, 'tokens': len(ids), 'targets': want, 'split': 0,
               'record_index': counts['kept']}
        meta_f.write(json.dumps(row, sort_keys=True) + '\n')
        offset += len(ids)
        total_targets += want
        counts['kept'] += 1
        counts['records'] += 1
    tokens_f.close()
    mask_f.close()
    idx_f.close()
    meta_f.close()

    replay_report = json.loads(args.replay_report.read_text())
    manifest = {
        'schema': 'emender-e97-tulu3-masked-sft-v1',
        'status': 'complete',
        'purpose': ('OpenHands execution trajectories re-rendered as canonical '
                    'e97-pi-native-v1 records under the eleven-tool surface; every '
                    'record replay-verified against the recorded repository state '
                    '(file ops + deterministic bash + model-patch final-state oracle) '
                    'with fail-closed drops.'),
        'tokenizer': 'p50k_base',
        'training_eligible': False,
        'packing_authorized': False,
        'optimizer_updates_authorized': 0,
        'provenance': {
            'source_pool': 'nvidia/Open-SWE-Traces derivative (e97-4b-open-swe-source-native-fulltraj-v1 source_messages)',
            'translation_script': 'scripts/build_e97_oh_pi_native_translation.py',
            'replay_script': 'scripts/verify_e97_oh_translation_replay.py',
            'replay_passed': replay_report.get('passed'),
            'replay_dropped': replay_report.get('dropped'),
            'oracle_metadata_copied': False,
        },
        'counts': {'records': counts['kept'], 'train_records': counts['kept'],
                   'validation_records': 0, 'tokens': offset,
                   'assistant_target_tokens': total_targets,
                   'excluded_over_context': counts['over_context'],
                   'excluded_empty_targets': counts['empty_targets'],
                   'replay_candidates': replay_report.get('candidates')},
        'outputs': {},
    }
    for name in ('tokens.uint32.bin', 'assistant_mask.uint8.bin', 'records.idx',
                 'records.jsonl'):
        manifest['outputs'][name] = {'path': name, 'bytes': (out / name).stat().st_size,
                                     'sha256': sha(out / name)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('OH_VERIFIED_AUTHORITY', json.dumps(
        {'kept': counts['kept'], 'over_context': counts['over_context'],
         'tokens': offset, 'targets': total_targets}, sort_keys=True))


if __name__ == '__main__':
    main()
