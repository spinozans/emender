#!/usr/bin/env python3
"""Build the E1 tool-talk full-intake masked-SFT authority (non-authorizing).

Consumes the full-intake render produced by scripts/render_e97_public_tooltalk.py
(per-source records.jsonl carrying `sections`), applies the spike's mechanical
eligibility lint, tokenizes with assistant-frame supervision (the five-line
Pi-native frames are the supervised targets; Protocol/System/User/ToolResult
sections are context), and emits one combined tulu3-masked-sft data authority
for the E1 prep's tool-talk cohort.

Eligibility lint per record (machine-checked, fail-closed):
  - not when2call-tagged (never enters);
  - at least one assistant frame, zero invalid frames (malformed / undeclared
    action / bad finish);
  - zero mid-conversation orphan calls and zero unmatched results (terminal
    dangling call of a parallel group is allowed per the spike's
    no-terminal-orphan validity);
  - total tokens <= 65536 (whole-record 64K packing);
  - tokenization must not cross a supervision boundary (records that fail are
    dropped and counted).

Selection: deterministic seeded shuffle of eligible records per source, filled
to the source's assistant-target budget. The manifest omits training_eligible
(plain data authority; the E1 prep stays training_eligible:false and admission
flows only through the prep's proposal/admission chain).
"""
import argparse, hashlib, json, random, struct, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tiktoken  # noqa: E402
from scripts.build_e97_tulu3_sft import _encode_pieces, _worker_init, TOKENIZER  # noqa: E402

INDEX = struct.Struct('<QQQB7x')
RS = "\x1e"
CONTEXT_LIMIT = 65536


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def eligible(record):
    flags = record.get('flags') or {}
    if flags.get('when2call_excluded'):
        return 'when2call'
    m = record['metrics']
    if not m['frames']:
        return 'no_frames'
    if m['invalid_frames']:
        return 'invalid_frame'
    if m['unmatched_results']:
        return 'unmatched_result'
    if m['orphan_mid_calls']:
        return 'mid_orphan'
    if m['tokens'] > CONTEXT_LIMIT:
        return 'oversized'
    return None


def pieces_for(record):
    sections = record.get('sections')
    if not sections:
        raise ValueError('record lacks sections (re-render with the sections-aware renderer)')
    pieces = []
    for index, section in enumerate(sections):
        if index:
            pieces.append(("\n\n", False))
        text = section['text']
        if section['kind'] == 'assistant':
            if not text.startswith('Assistant:\n'):
                raise ValueError('assistant section header')
            pieces.append((text[:len('Assistant:\n')], False))
            pieces.append((text[len('Assistant:\n'):], True))
        else:
            pieces.append((text, False))
    pieces.append((RS, True))
    return pieces


def expected_targets(record, encoding):
    """Supervised target count for a record: the in-situ five-line frame
    bodies plus the record-separator token. The renderer's assistant_tokens
    metric joins frames with \n\n for its own accounting, which merges tokens
    at frame boundaries; this walks the sections instead."""
    total = 0
    for section in record['sections']:
        if section['kind'] == 'assistant':
            total += len(encoding.encode_ordinary(section['text'][len('Assistant:\n'):]))
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rendered-root', type=Path, required=True,
                   help='full-intake render output root (per-source dirs from render_e97_public_tooltalk.py)')
    p.add_argument('--source', action='append', required=True,
                   help='JSON: {"source": "<key>", "seed": N, "budget_targets": N}')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    _worker_init()
    encoding = tiktoken.get_encoding(TOKENIZER)

    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    paths = {'tokens': args.output / 'tokens.uint32.bin',
             'mask': args.output / 'assistant_mask.uint8.bin',
             'index': args.output / 'records.idx',
             'metadata': args.output / 'records.jsonl'}
    per_source = []
    drop_counts = Counter()
    _t, _m, _c = _encode_pieces([("x", False), (RS, True)])
    rs_tokens = sum(_m)
    with paths['tokens'].open('xb') as tf, paths['mask'].open('xb') as mf, \
            paths['index'].open('xb') as ix, paths['metadata'].open('x') as meta:
        offset = records = targets = total_tokens = 0
        for spec_json in args.source:
            spec = json.loads(spec_json)
            key = spec['source']
            source_dir = args.rendered_root / key
            rendered_manifest = json.loads((source_dir / 'manifest.json').read_text())
            rows = [json.loads(x) for x in (source_dir / 'records.jsonl').read_text().splitlines()]
            candidates = []
            for record in rows:
                reason = eligible(record)
                if reason:
                    drop_counts[f'{key}:{reason}'] += 1
                    continue
                candidates.append(record)
            rng = random.Random(spec['seed'])
            order = list(range(len(candidates)))
            rng.shuffle(order)
            chosen = []
            consumed = 0
            for j in order:
                record = candidates[j]
                want = expected_targets(record, encoding) + rs_tokens
                if consumed + want > spec['budget_targets']:
                    continue
                pieces = pieces_for(record)
                try:
                    tokens, masks, complete = _encode_pieces(pieces)
                except ValueError as error:
                    drop_counts[f'{key}:token_boundary'] += 1
                    continue
                if sum(masks) != want:
                    drop_counts[f'{key}:mask_sum'] += 1
                    continue
                row = {'id': record['id'], 'source': 'tooltalk-rehearsal',
                       'tooltalk_source': key, 'tooltalk_row_index': record['row_index'],
                       'tooltalk_metrics': {k: v for k, v in record['metrics'].items()},
                       'serialization_sha256': hashlib.sha256(complete.encode()).hexdigest()}
                tf.write(struct.pack(f'<{len(tokens)}I', *tokens))
                mf.write(bytes(masks))
                ix.write(INDEX.pack(offset, len(tokens), sum(masks), 0))
                meta.write(json.dumps(row, sort_keys=True) + '\n')
                offset += len(tokens)
                records += 1
                consumed += sum(masks)
                targets += sum(masks)
                total_tokens += len(tokens)
                chosen.append(record['id'])
            per_source.append({
                'source': key, 'seed': spec['seed'],
                'rendered_manifest_sha256': sha(source_dir / 'manifest.json'),
                'rendered_records': len(rows),
                'eligible_records': len(candidates),
                'selected_records': len(chosen),
                'target_token_budget': spec['budget_targets'],
                'consumed_target_tokens': consumed,
            })
    manifest = {
        'schema': 'emender-e97-tulu3-masked-sft-v1',
        'status': 'complete',
        'purpose': ('E1 tool-talk full-intake authority: public intermingled '
                    'chat+tool-call conversations rendered Pi-native by '
                    'scripts/render_e97_public_tooltalk.py (five-line frames as '
                    'supervised targets; Protocol/System/User/ToolResult as '
                    'context), mechanically linted per the codec-spike '
                    'admission recommendations, admitted into the E1 prep by '
                    'the operator\'s E1 directive (Toucan SFT + Nemotron-Agentic '
                    'interactive_agent + Tool-Reasoning-31K; When2Call excluded).'),
        'tokenizer': 'p50k_base',
        'packing_authorized': False,
        'optimizer_updates_authorized': 0,
        'admission_note': ('plain data authority; the E1 prep manifest stays '
                           'training_eligible:false and eligibility flows only '
                           'through the prep proposal/admission chain'),
        'counts': {'records': records, 'train_records': records, 'validation_records': 0,
                   'tokens': total_tokens, 'assistant_target_tokens': targets},
        'sources': per_source,
        'dropped_records': dict(drop_counts),
        'outputs': {k: {'path': v.name, 'bytes': v.stat().st_size, 'sha256': sha(v)}
                    for k, v in paths.items()},
    }
    tmp = args.output / 'manifest.json.partial'
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    tmp.replace(args.output / 'manifest.json')
    print('E1_TOOLTALK_AUTHORITY', records, total_tokens, targets,
          sha(args.output / 'manifest.json'))


if __name__ == '__main__':
    main()
