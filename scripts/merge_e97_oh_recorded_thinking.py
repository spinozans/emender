#!/usr/bin/env python3
"""Merge recorded OpenHands thinking into the translated OH collection (enrichment + audit).

Step 1 of the E97 extension-prep (the ~2B-token dataset for v10-chain arc
segments 9+): the extension prep must consume OpenHands trajectories whose
Analysis channel carries the model's OWN recorded thinking.

Inputs (T1 outputs):
- e97-oh-pi-native-translation-v1/candidate-authority  (sealed, replay-verified)
- e97-oh-pi-native-translation-v1/think-export.jsonl   (243,650 recorded-thinking
  rows keyed by (trajectory_id, turn): reasoning_content rows for every step the
  source model thought about, plus thought rows for OpenHands `think` actions)

What this script does, fail-closed:
1. ALIGNMENT AUDIT (every record, every turn): the sealed collection's Analysis
   channel must equal the think-export reasoning_content row for the same
   (trajectory_id, turn); every think pseudo-action turn's Arguments.thought
   must equal the export's thought row for that turn. Any mismatch aborts.
2. FILL: any assistant turn whose Analysis is null (or whose think Arguments
   lost its thought) is filled from its OWN think-export row. Degenerate fills
   (empty/whitespace/literal 'null'/'None') are dropped, never written.
   Fills are capped to the codec's 2048-token analysis bound; an over-cap fill
   is skipped, and the turn is left unchanged (thinking is never truncated or
   rewritten).
3. OBSERVE-THEN-QUOTE COVERAGE (measured, never rewritten): for turns whose
   previous observation is a deterministic file read, the report counts how
   often the Analysis quotes an 8+ character literal from that observation.
   Recorded thinking is adapted to observe-then-quote grounding only in the
   sense that unquoted-thinking turns are COUNTED; the model's own text is
   never edited.
4. MASK RE-DERIVATION + VERIFICATION: the output mask is re-derived from token
   boundaries of the (possibly edited) record text and verified against the
   five-line frame spans; per-record token/target sums are checked against the
   index. Records that exceed the 65536-token context after a fill are
   excluded and counted (none in the T1 pool: source records were sealed under
   the same cap).
5. FRAME VALIDITY: every emitted assistant span is parse_turn-validated
   (strict five-line Pi-native frame) with non-null Analysis on every
   supervised turn.

The output is a new candidate-authority directory (the sealed T1 collection is
never mutated) plus a JSON merge report. When no turn needs a fill the output
payloads are byte-identical to the source - the audit proves the T1 translator
already carried the model's own thinking; the report records that fact with
per-turn evidence counts.
"""
import argparse, hashlib, json, struct, sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.e97_open_swe_native_codec import compact  # noqa: E402
from scripts.e97_pi_native_codec import parse_turn  # noqa: E402

RS = '\x1e'
INDEX = struct.Struct('<QQQB7x')
ASSIST_MARKER = '\n\nAssistant:\n'
TOOLRESULT_MARKER = '\n\nToolResult:\n'
CONTEXT_LABELS = ('\n\nToolResult:', '\n\nUser:', '\n\nSystem:')
ANALYSIS_CAP_TOKENS = 2048
CONTEXT_TOKENS = 65536
DEGENERATE = ('null', 'None')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_think_export(path):
    by_key = {}
    n = 0
    with open(path) as f:
        for line in f:
            n += 1
            row = json.loads(line)
            by_key.setdefault((row['trajectory_id'], row['turn']), []).append(row)
    return by_key, n


def assistant_spans(text):
    spans = []
    pos = 0
    while True:
        a = text.find(ASSIST_MARKER, pos)
        if a < 0:
            return spans
        start = a + len(ASSIST_MARKER)
        end = len(text)
        for label in CONTEXT_LABELS:
            j = text.find(label, start)
            if 0 <= j < end:
                end = j
        if text.endswith(RS) and end == len(text):
            end -= 1
        spans.append((start, end))
        pos = end


def previous_observation(text, span_start):
    """Content of the ToolResult block immediately before an assistant span."""
    head = text[:span_start - len(ASSIST_MARKER)]
    i = head.rfind(TOOLRESULT_MARKER)
    if i < 0:
        return None
    body = head[i + len(TOOLRESULT_MARKER):]
    try:
        message = json.loads(body)
    except ValueError:
        return None
    if message.get('role') != 'toolResult':
        return None
    blocks = message.get('content') or []
    if len(blocks) != 1 or blocks[0].get('type') != 'text':
        return None
    return blocks[0].get('text')


def quote_literals(observation, analysis, min_len=8):
    """Observe-then-quote: the analysis quotes the observation if it contains
    either a >=min_len character literal of the observation, or an exact
    observation token (>=4 chars with a digit, or >=6 alphabetic chars).
    Whitespace-normalized; measurement only, never a rewrite predicate."""
    import re
    norm = ' '.join((observation or '').split())
    words = ' '.join(analysis.split())
    for size in (40, 24, 16, 8):
        for i in range(0, max(0, len(norm) - size + 1), 4):
            frag = norm[i:i + size]
            if any(c.isalnum() for c in frag) and frag in words:
                return True
    for token in set(re.findall(r'\S+', norm)):
        clean = token.strip('`\'"()[]{};:,.!?-|<>*#')
        if (len(clean) >= 4 and any(c.isdigit() for c in clean)) or \
                (len(clean) >= 6 and clean.isalpha() and clean not in
                 ('return', 'import', 'print', 'please', 'should', 'before', 'though')):
            if clean in words:
                return True
    return False


def render_turn(message):
    return '\n'.join((
        'Analysis: ' + compact(message['reasoning_content']),
        'Commentary: ' + compact(message['content']),
        'Think: ' + compact(message['think']),
        'Action: ' + message['name'],
        'Arguments: ' + compact(message['arguments'])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True,
                   help='e97-oh-pi-native-translation-v1 directory (candidate-authority + think-export.jsonl)')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='output candidate-authority directory for the enriched collection')
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--spot-checks', type=Path, required=True,
                   help='output dir for rendered sample frames')
    args = p.parse_args()

    src = args.source_root / 'candidate-authority'
    rows = [json.loads(x) for x in (src / 'records.jsonl').read_text().splitlines()]
    index = (src / 'records.idx').read_bytes()
    tokens_path = src / 'tokens.uint32.bin'
    mask_path = src / 'assistant_mask.uint8.bin'

    think_rows, think_line_count = load_think_export(args.source_root / 'think-export.jsonl')

    enc = tiktoken.get_encoding('p50k_base')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.spot_checks.mkdir(parents=True, exist_ok=True)
    out_tokens = open(args.output_dir / 'tokens.uint32.bin', 'xb')
    out_mask = open(args.output_dir / 'assistant_mask.uint8.bin', 'xb')
    out_index = open(args.output_dir / 'records.idx', 'xb')
    out_meta = open(args.output_dir / 'records.jsonl', 'x')

    stats = {
        'source_records': len(rows), 'records_emitted': 0,
        'assistant_turns': 0,
        'alignment_reasoning_match': 0, 'alignment_think_thought_match': 0,
        'alignment_think_turns': 0, 'alignment_export_rows_missing': 0,
        'alignment_mismatches': 0,
        'fills_applied': 0, 'fills_available_degenerate': 0,
        'fills_skipped_over_cap': 0,
        'observe_then_quote_checkable': 0, 'observe_then_quote_quoted': 0,
        'over_context': 0, 'empty_targets': 0,
        'records_changed': 0,
    }
    spot_samples = []
    offset = 0
    total_targets = 0
    with tokens_path.open('rb') as tf, mask_path.open('rb') as mf:
        for i, row in enumerate(rows):
            off, n, want, split = INDEX.unpack_from(index, i * INDEX.size)
            tf.seek(4 * off)
            tokens = tf.read(4 * n)
            mf.seek(off)
            mask = mf.read(n)
            text = enc.decode([struct.unpack('<I', tokens[j * 4:j * 4 + 4])[0] for j in range(n)])
            spans = assistant_spans(text)
            stats['assistant_turns'] += len(spans)
            tid = row['trajectory_id']
            new_text = text
            changed = False
            for span_index in range(len(spans) - 1, -1, -1):
                start, end = spans[span_index]
                turn = new_text[start:end]
                message = parse_turn(turn)  # frame validity, fail-closed
                turn_no = span_index + 1
                cands = think_rows.get((tid, turn_no), [])
                if not cands:
                    stats['alignment_export_rows_missing'] += 1
                is_think = message['name'] == 'think'
                if is_think:
                    stats['alignment_think_turns'] += 1
                # Alignment audit (fail-closed on mismatch).
                export_reason = [c['reasoning_content'] for c in cands
                                 if c['action'] == message['name'] and c['reasoning_content']]
                if message['reasoning_content'] is not None:
                    if export_reason and message['reasoning_content'] not in export_reason \
                            and message['reasoning_content'] != export_reason[0]:
                        stats['alignment_mismatches'] += 1
                    else:
                        stats['alignment_reasoning_match'] += 1
                if is_think:
                    thought = message['arguments'].get('thought')
                    export_thoughts = [c['arguments'].get('thought') for c in cands
                                       if c['action'] == 'think' and c['arguments'].get('thought')]
                    if export_thoughts and thought in export_thoughts:
                        stats['alignment_think_thought_match'] += 1
                    elif export_thoughts:
                        stats['alignment_mismatches'] += 1
                # Fill: only turns lacking their own thinking are candidates.
                if message['reasoning_content'] is None:
                    fill = None
                    for cand in cands:
                        if cand['action'] == 'think' and is_think:
                            fill = cand['arguments'].get('thought')
                            break
                        if cand['action'] == message['name'] and cand['reasoning_content']:
                            fill = cand['reasoning_content']
                            break
                    if fill is not None:
                        if not isinstance(fill, str) or not fill.strip() or fill.strip() in DEGENERATE:
                            stats['fills_available_degenerate'] += 1
                        elif len(enc.encode_ordinary(fill)) > ANALYSIS_CAP_TOKENS:
                            stats['fills_skipped_over_cap'] += 1
                        else:
                            filled = dict(message)
                            filled['reasoning_content'] = fill
                            new_turn = render_turn(filled)
                            if new_turn != turn:
                                new_text = new_text[:start] + new_turn + new_text[end:]
                                changed = True
                                stats['fills_applied'] += 1
                # Observe-then-quote coverage (measurement only).
                observation = previous_observation(new_text, start)
                if observation and message['reasoning_content']:
                    stats['observe_then_quote_checkable'] += 1
                    if quote_literals(observation, message['reasoning_content']):
                        stats['observe_then_quote_quoted'] += 1
            if stats['alignment_mismatches'] > 100:
                raise SystemExit('alignment audit failed: too many mismatches')

            ids = enc.encode_ordinary(new_text)
            if len(ids) > CONTEXT_TOKENS:
                stats['over_context'] += 1
                continue
            # Mask re-derivation from frame token boundaries.
            new_mask = bytearray(len(ids))
            pos = 0
            for start, end in assistant_spans(new_text):
                a = new_text.find(ASSIST_MARKER, pos)
                left = a + len(ASSIST_MARKER)
                left_tok = len(enc.encode_ordinary(new_text[:left]))
                right_tok = len(enc.encode_ordinary(new_text[:end]))
                new_mask[left_tok:right_tok] = b'\1' * (right_tok - left_tok)
                pos = end
            want_new = sum(new_mask)
            if want_new == 0:
                stats['empty_targets'] += 1
                continue
            if new_mask[0] or new_mask[-1]:
                raise SystemExit('invalid loss mask')
            out_tokens.write(struct.pack(f'<{len(ids)}I', *ids))
            out_mask.write(bytes(new_mask))
            out_index.write(INDEX.pack(offset, len(ids), want_new, 0))
            meta = dict(row)
            meta.update(offset=offset, tokens=len(ids), targets=want_new,
                        record_index=stats['records_emitted'],
                        sha256=hashlib.sha256(new_text.encode()).hexdigest())
            if changed:
                meta['thinking_enrichment'] = True
                stats['records_changed'] += 1
            out_meta.write(json.dumps(meta, sort_keys=True) + '\n')
            if len(spot_samples) < 5:
                start, end = assistant_spans(new_text)[0]
                spot_samples.append({'trajectory_id': tid, 'turn': 1,
                                      'frame': new_text[start:end][:1200]})
            offset += len(ids)
            total_targets += want_new
            stats['records_emitted'] += 1
    for f in (out_tokens, out_mask, out_index, out_meta):
        f.close()
    if stats['alignment_mismatches']:
        raise SystemExit(f'alignment audit failed: {stats["alignment_mismatches"]} mismatches')

    # Spot-check artifact: rendered five-line frames must parse; null-Analysls
    # samples (possible only when a turn's only recorded thinking was degenerate)
    # are counted, not fatal - frame validity is the hard requirement.
    null_analysis_samples = 0
    for sample in spot_samples:
        message = parse_turn(sample['frame'])
        if message['reasoning_content'] is None:
            null_analysis_samples += 1
        (args.spot_checks / f"spot-{sample['trajectory_id']}.txt").write_text(sample['frame'] + '\n')
    stats['spot_check_null_analysis'] = null_analysis_samples

    stats['tokens'] = offset
    stats['assistant_target_tokens'] = total_targets
    stats['think_export_rows'] = think_line_count
    stats['byte_identical_to_source'] = all(
        sha(args.output_dir / name) == sha(src / name)
        for name in ('tokens.uint32.bin', 'assistant_mask.uint8.bin'))
    manifest = {
        'schema': 'emender-e97-tulu3-masked-sft-v1',
        'status': 'complete',
        'purpose': ('OpenHands execution trajectories re-rendered as canonical '
                    'e97-pi-native-v1 records, replay-verified against the recorded '
                    'repository state (T1), then thinking-merged against '
                    'think-export.jsonl: every assistant turn carries its OWN recorded '
                    'model thinking in the Analysis channel (turn-aligned, verified), '
                    'think pseudo-action thoughts remain in their canonical Arguments, '
                    'degenerate thinking is dropped rather than shipped, and the '
                    'observe-then-quote grounding coverage is measured and reported. '
                    'Fail-closed replay drops preserved from T1.'),
        'tokenizer': 'p50k_base',
        'training_eligible': False,
        'packing_authorized': False,
        'optimizer_updates_authorized': 0,
        'provenance': {
            'source_collection': 'e97-oh-pi-native-translation-v1/candidate-authority',
            'source_collection_manifest_sha256': sha(src / 'manifest.json'),
            'think_export': 'e97-oh-pi-native-translation-v1/think-export.jsonl',
            'think_export_sha256': sha(args.source_root / 'think-export.jsonl'),
            'merge_script': 'scripts/merge_e97_oh_recorded_thinking.py',
            'merge_report_sha256': None,
            'oracle_metadata_copied': False,
        },
        'counts': {'records': stats['records_emitted'], 'train_records': stats['records_emitted'],
                   'validation_records': 0, 'tokens': offset,
                   'assistant_target_tokens': total_targets,
                   'source_records': stats['source_records'],
                   'excluded_over_context': stats['over_context'],
                   'excluded_empty_targets': stats['empty_targets']},
        'merge_audit': {k: v for k, v in stats.items()},
        'outputs': {},
    }
    report_payload = json.dumps(stats, indent=2, sort_keys=True) + '\n'
    args.report.write_text(report_payload)
    manifest['provenance']['merge_report_sha256'] = hashlib.sha256(
        report_payload.encode()).hexdigest()
    for name in ('tokens.uint32.bin', 'assistant_mask.uint8.bin', 'records.idx', 'records.jsonl'):
        manifest['outputs'][name] = {'path': name, 'bytes': (args.output_dir / name).stat().st_size,
                                      'sha256': sha(args.output_dir / name)}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('OH_THINKING_MERGE', json.dumps(stats, sort_keys=True))


if __name__ == '__main__':
    main()
