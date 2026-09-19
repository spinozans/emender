"""Tests for the OH recorded-thinking merge (extension-prep step 1)."""
import json, struct, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tiktoken
from scripts.e97_open_swe_native_codec import compact
from scripts.merge_e97_oh_recorded_thinking import (
    assistant_spans, quote_literals, render_turn)

RS = '\x1e'
INDEX = struct.Struct('<QQQB7x')
ENC = tiktoken.get_encoding('p50k_base')


def turn(analysis, action, arguments, commentary=None):
    return '\n'.join((
        'Analysis: ' + compact(analysis),
        'Commentary: ' + compact(commentary),
        'Think: ' + compact(None),
        'Action: ' + action,
        'Arguments: ' + compact(arguments)))


def record_text(turns, observations):
    """turns: list of (turn_text, tool_name or None); observations per tool turn."""
    parts = ['Protocol:\n{"profile":"e97-pi-native-v1"}']
    parts.append('\n\nSystem:\n' + compact({'role': 'system', 'content': 'sys'}))
    parts.append('\n\nUser:\n' + compact({'role': 'user', 'content': 'do it'}))
    obs = list(observations)
    for t, tool in turns:
        parts.append('\n\nAssistant:\n' + t)
        if tool is not None:
            parts.append('\n\nToolResult:\n' + compact({
                'role': 'toolResult', 'toolCallId': 'c1', 'toolName': tool,
                'content': [{'type': 'text', 'text': obs.pop(0)}], 'isError': False}))
    return ''.join(parts) + RS


def build_source(root, texts):
    """Materialize a minimal candidate-authority + think-export from record texts."""
    auth = root / 'candidate-authority'
    auth.mkdir(parents=True)
    tf = open(auth / 'tokens.uint32.bin', 'wb')
    mf = open(auth / 'assistant_mask.uint8.bin', 'wb')
    ix = open(auth / 'records.idx', 'wb')
    meta = open(auth / 'records.jsonl', 'w')
    offset = 0
    for i, text in enumerate(texts):
        ids = ENC.encode_ordinary(text)
        mask = bytearray(len(ids))
        pos = 0
        for start, end in assistant_spans(text):
            a = text.find('\n\nAssistant:\n', pos)
            left_tok = len(ENC.encode_ordinary(text[:a + len('\n\nAssistant:\n')]))
            right_tok = len(ENC.encode_ordinary(text[:end]))
            mask[left_tok:right_tok] = b'\1' * (right_tok - left_tok)
            pos = end
        tf.write(struct.pack(f'<{len(ids)}I', *ids))
        mf.write(bytes(mask))
        ix.write(INDEX.pack(offset, len(ids), sum(mask), 0))
        meta.write(json.dumps({'trajectory_id': f'traj-{i}', 'instance_id': f'inst-{i}',
                               'record_index': i, 'offset': offset, 'tokens': len(ids),
                               'targets': sum(mask), 'split': 0}, sort_keys=True) + '\n')
        offset += len(ids)
    for f in (tf, mf, ix, meta):
        f.close()
    (auth / 'manifest.json').write_text('{}\n')
    return auth


def run_merge_argv(tmp_path, texts, think_rows):
    """Run the merge script via its argparse main with sys.argv swapped."""
    root = tmp_path / 'src'
    root.mkdir()
    build_source(root, texts)
    with open(root / 'think-export.jsonl', 'w') as f:
        for row in think_rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    out = tmp_path / 'out'
    report = tmp_path / 'report.json'
    spots = tmp_path / 'spots'
    argv = ['merge', '--source-root', str(root), '--output-dir', str(out),
            '--report', str(report), '--spot-checks', str(spots)]
    old = sys.argv
    sys.argv = argv
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'merge_e97_oh_run', 'scripts/merge_e97_oh_recorded_thinking.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.main()
    finally:
        sys.argv = old
    stats = json.loads(report.read_text())
    rows = [json.loads(x) for x in (out / 'records.jsonl').read_text().splitlines()]
    return out, stats, rows


def test_merge_is_identity_when_thinking_already_present(tmp_path):
    t1 = turn('Plan: read first.', 'read', {'path': '/w/f.py'})
    t2 = turn('The file shows "file body"; done.', 'finish', {'message': 'done'})
    text = record_text([(t1, 'read'), (t2, None)], ['file body\n'])
    rows = [
        {'trajectory_id': 'traj-0', 'turn': 1, 'reasoning_content': 'Plan: read first.',
         'action': 'read', 'arguments': {'path': '/w/f.py'}},
        {'trajectory_id': 'traj-0', 'turn': 2, 'reasoning_content': 'The file shows "file body"; done.',
         'action': 'finish', 'arguments': {'message': 'done'}},
    ]
    out, stats, meta = run_merge_argv(tmp_path, [text], rows)
    assert stats['alignment_mismatches'] == 0
    assert stats['fills_applied'] == 0
    assert stats['byte_identical_to_source'] is True
    assert stats['records_emitted'] == 1
    assert stats['observe_then_quote_checkable'] == 1
    # the post-observation turn quotes the observed body
    assert stats['observe_then_quote_quoted'] == 1


def test_merge_fills_null_analysis_from_own_recorded_thinking(tmp_path):
    t1 = turn('Plan: read first.', 'read', {'path': '/w/f.py'})
    t2 = turn(None, 'finish', {'message': 'done'})
    text = record_text([(t1, 'read'), (t2, None)], ['file body\n'])
    rows = [
        {'trajectory_id': 'traj-0', 'turn': 1, 'reasoning_content': 'Plan: read first.',
         'action': 'read', 'arguments': {'path': '/w/f.py'}},
        {'trajectory_id': 'traj-0', 'turn': 2, 'reasoning_content': 'Observed body, finishing.',
         'action': 'finish', 'arguments': {'message': 'done'}},
    ]
    out, stats, meta = run_merge_argv(tmp_path, [text], rows)
    assert stats['fills_applied'] == 1
    assert stats['byte_identical_to_source'] is False
    from scripts.e97_pi_native_codec import parse_turn
    tokens = (out / 'tokens.uint32.bin').read_bytes()
    decoded = ENC.decode([struct.unpack('<I', tokens[j:j + 4])[0]
                          for j in range(0, len(tokens), 4)])
    start = decoded.find('\n\nAssistant:\n', decoded.find('ToolResult'))
    turn2 = decoded[start + len('\n\nAssistant:\n'):].split('\x1e')[0]
    assert parse_turn(turn2)['reasoning_content'] == 'Observed body, finishing.'
    # mask re-verified: targets grew by exactly the filled analysis tokens
    source_row = json.loads((tmp_path / 'src' / 'candidate-authority' / 'records.jsonl')
                            .read_text().splitlines()[0])
    mask = (out / 'assistant_mask.uint8.bin').read_bytes()
    expected = source_row['targets'] + len(ENC.encode_ordinary('"Observed body, finishing."')) \
        - len(ENC.encode_ordinary('null'))
    assert sum(mask) == meta[0]['targets'] == expected


def test_merge_drops_degenerate_and_skips_over_cap(tmp_path):
    t1 = turn(None, 'read', {'path': '/w/f.py'})
    long_thinking = 'word ' * 3000  # > 2048 tokens
    t2 = turn(None, 'finish', {'message': 'done'})
    text = record_text([(t1, 'read'), (t2, None)], ['file body\n'])
    rows = [
        {'trajectory_id': 'traj-0', 'turn': 1, 'reasoning_content': 'null',
         'action': 'read', 'arguments': {'path': '/w/f.py'}},
        {'trajectory_id': 'traj-0', 'turn': 2, 'reasoning_content': long_thinking,
         'action': 'finish', 'arguments': {'message': 'done'}},
    ]
    out, stats, meta = run_merge_argv(tmp_path, [text], rows)
    assert stats['fills_applied'] == 0
    assert stats['fills_available_degenerate'] == 1
    assert stats['fills_skipped_over_cap'] == 1
    assert stats['byte_identical_to_source'] is True


def test_merge_fills_think_turn_from_own_thought(tmp_path):
    t1 = turn(None, 'think', {'thought': 'I should check the config first.'})
    t2 = turn('Config checked.', 'finish', {'message': 'done'})
    text = record_text([(t1, 'think'), (t2, None)], ['Your thought has been logged.'])
    rows = [
        {'trajectory_id': 'traj-0', 'turn': 1, 'reasoning_content': None,
         'action': 'think', 'arguments': {'thought': 'I should check the config first.'}},
        {'trajectory_id': 'traj-0', 'turn': 2, 'reasoning_content': 'Config checked.',
         'action': 'finish', 'arguments': {'message': 'done'}},
    ]
    out, stats, meta = run_merge_argv(tmp_path, [text], rows)
    assert stats['fills_applied'] == 1
    assert stats['alignment_think_thought_match'] == 1


def test_merge_aborts_on_alignment_mismatch(tmp_path):
    t1 = turn('Wrong thinking text.', 'read', {'path': '/w/f.py'})
    t2 = turn('Done thinking.', 'finish', {'message': 'done'})
    text = record_text([(t1, 'read'), (t2, None)], ['file body\n'])
    rows = [
        {'trajectory_id': 'traj-0', 'turn': 1, 'reasoning_content': 'Plan: read first.',
         'action': 'read', 'arguments': {'path': '/w/f.py'}},
        {'trajectory_id': 'traj-0', 'turn': 2, 'reasoning_content': 'Done thinking.',
         'action': 'finish', 'arguments': {'message': 'done'}},
    ]
    with pytest.raises(SystemExit):
        run_merge_argv(tmp_path, [text], rows)


def test_quote_literals_detects_observed_values():
    assert quote_literals('port = 8443\nmode = alpha\n', 'The observed port 8443 is what I need.')
    assert not quote_literals('port = 8443\n', 'I will guess a port value.')
