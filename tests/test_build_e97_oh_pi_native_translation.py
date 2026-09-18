"""Tests for the OH -> Pi-native translation and replay verification pipeline."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_e97_oh_pi_native_translation import (
    Translator, assemble, map_action, parse_bash_observation, parse_catn_body,
    render_read_output)
from scripts.e97_pi_native_codec import parse_turn

SURFACE = {
    'profile': 'e97-pi-native-v1',
    'instructions': 'Use only the declared Pi tools and their exact argument schemas.',
    'tools': [
        {'name': 'read', 'label': 'read', 'description': 'Read a file.',
         'parameters': {'type': 'object', 'properties': {
             'path': {'type': 'string'}, 'offset': {'type': 'number'}, 'limit': {'type': 'number'}},
             'required': ['path'], 'additionalProperties': False}},
        {'name': 'bash', 'label': 'bash', 'description': 'Run a command.',
         'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}},
                        'required': ['command'], 'additionalProperties': False}},
        {'name': 'edit', 'label': 'edit', 'description': 'Edit a file.',
         'parameters': {'type': 'object', 'properties': {
             'path': {'type': 'string'},
             'edits': {'type': 'array', 'items': {'type': 'object', 'properties': {
                 'oldText': {'type': 'string'}, 'newText': {'type': 'string'}},
                 'required': ['oldText', 'newText'], 'additionalProperties': False}}},
             'required': ['path', 'edits'], 'additionalProperties': False}},
        {'name': 'write', 'label': 'write', 'description': 'Write a file.',
         'parameters': {'type': 'object', 'properties': {
             'path': {'type': 'string'}, 'content': {'type': 'string'}},
             'required': ['path', 'content'], 'additionalProperties': False}},
    ],
    'pseudo_actions': [
        {'name': 'think', 'description': 'Private thought.', 'parameters': {
            'type': 'object', 'properties': {'thought': {'type': 'string'}},
            'required': ['thought'], 'additionalProperties': False}},
        {'name': 'finish', 'description': 'Final answer.', 'parameters': {
            'type': 'object', 'properties': {'message': {'type': 'string'}},
            'required': ['message'], 'additionalProperties': False}},
    ],
}


def test_map_action_vocabulary():
    assert map_action('execute_bash', {'command': 'ls'}) == (
        'bash', {'command': 'ls'}, {'kind': 'bash', 'recorded_core': '', 'recorded_exit': 0})
    name, args, _ = map_action('str_replace_editor', {
        'command': 'str_replace', 'path': '/w/f.py', 'old_str': 'a', 'new_str': 'b'})
    assert (name, args) == ('edit', {'path': '/w/f.py', 'edits': [{'oldText': 'a', 'newText': 'b'}]})
    name, args, _ = map_action('str_replace_editor', {
        'command': 'create', 'path': '/w/n.py', 'file_text': 'x'})
    assert (name, args) == ('write', {'path': '/w/n.py', 'content': 'x'})
    name, args, _ = map_action('str_replace_editor', {
        'command': 'view', 'path': '/w/f.py', 'view_range': [10, -1]})
    assert args == {'path': '/w/f.py', 'offset': 10}
    # end < start is the OH malformed-range error case: rejected, never reinterpreted
    import pytest
    for bad_range in ([88, 65], [5, 2]):
        with pytest.raises(ValueError):
            map_action('str_replace_editor', {'command': 'view', 'path': '/w/f.py',
                                              'view_range': bad_range})


def test_bash_observation_parsing():
    obs = 'out line\n\n[The command completed with exit code 0.]\n[Current working directory: /workspace/x]\n'
    core, code = parse_bash_observation(obs)
    assert code == 0 and core == 'out line\n'
    obs2 = 'boom\n[The command completed with exit code 3.]\n'
    core2, code2 = parse_bash_observation(obs2)
    assert code2 == 3 and core2 == 'boom\n'


def test_catn_parsing():
    body = '   1\tfirst\n  10\tsecond\n'
    assert parse_catn_body(body) == {1: 'first', 10: 'second'}


def test_render_read_output_truncation_and_limits():
    out, err = render_read_output('a\nb\nc\nd', 2, 2)
    assert err is None and out == 'b\nc\n\n[1 more lines in file. Use offset=4 to continue.]'
    out, err = render_read_output('\n'.join(f'L{i}' for i in range(3000)), None, None)
    assert err is None and 'Use offset=2001 to continue.]' in out


def test_full_episode_translation_and_codec_parity():
    messages = [
        {'role': 'system', 'content': 'You are OpenHands agent.'},
        {'role': 'user', 'content': '<uploaded_files>\n/workspace/x__y__1.0\n</uploaded_files>\nFix it.'},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'Plan: read then edit.',
         'think': None, 'tool_calls': [{'type': 'function', 'function': {
             'name': 'str_replace_editor',
             'arguments': '{"command": "view", "path": "/workspace/x__y__1.0/f.py"}'}}]},
        {'role': 'tool', 'content': "Here's the result of running `cat -n` on /workspace/x__y__1.0/f.py:\n   1\thello\n"},
        {'role': 'assistant', 'content': '', 'reasoning_content': None,
         'think': None, 'tool_calls': [{'type': 'function', 'function': {
             'name': 'str_replace_editor',
             'arguments': '{"command": "str_replace", "path": "/workspace/x__y__1.0/f.py", '
                          '"old_str": "hello", "new_str": "world"}'}}]},
        {'role': 'tool', 'content': "The file /workspace/x__y__1.0/f.py has been edited.\n"},
        {'role': 'assistant', 'content': '', 'reasoning_content': None, 'think': None,
         'tool_calls': [{'type': 'function', 'function': {
             'name': 'finish', 'arguments': '{"message": "done"}'}}]},
    ]
    t = Translator(SURFACE)
    record, reason = t.translate({'trajectory_id': 'test-traj', 'messages': messages})
    assert record is not None, reason
    assert len(record['actions']) == 3
    assert [a['tool'] for a in record['actions']] == ['read', 'edit', 'finish']
    assert record['think_rows'] == [
        {'trajectory_id': 'test-traj', 'turn': 1, 'reasoning_content': 'Plan: read then edit.',
         'action': 'read', 'arguments': {'path': '/workspace/x__y__1.0/f.py'}}]
    # every assistant turn parses under the canonical codec
    turns = [b['turn'] for b in record['blocks'] if b['kind'] == 'assistant']
    parsed = [parse_turn(turn) for turn in turns]
    assert [p['name'] for p in parsed] == ['read', 'edit', 'finish']
    # assembled record structure
    texts = {a['index']: a.get('recorded_observation', '') for a in record['actions']}
    text = assemble(record['blocks'], texts)
    assert text.startswith('Protocol:\n{"instructions"')
    assert '\n\nSystem:\n' in text and '\n\nUser:\n' in text
    assert text.endswith('\x1e')
    assert 'Analysis: "Plan: read then edit."' in text


def test_translate_rejects_malformed():
    t = Translator(SURFACE)
    bad = {'trajectory_id': 'x', 'messages': [
        {'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'content': '', 'reasoning_content': None, 'think': None,
         'tool_calls': [{'type': 'function', 'function': {
             'name': 'str_replace_editor',
             'arguments': '{"command": "insert", "path": "/w/f", "insert_line": 2}'}}]},
        {'role': 'tool', 'content': 'ok'}]}
    record, reason = t.translate(bad)
    assert record is None and 'untranslatable' in reason
