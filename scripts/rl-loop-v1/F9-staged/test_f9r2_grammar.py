"""Permanent F9R2 117-case battery; negatives are never executed.

Matrix construction below is retained from F9R2-grammar-qa.py. The original
reviewer results are retained in F9R2-grammar-results.json as historical evidence.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess

import pytest
import tiktoken
import teacher_author as author
import rl_loop_driver as driver
from scripts.e97_pi_native_codec import native_turn
from scripts.e97_pi_native_tool_bridge import MODEL, PROVIDER, BridgeStopped
from scripts.e97_pi_scripted_v8 import SafeWorkspaceBridgeV8
from scripts.e97_diversity import canonical
from scripts.e97_offline_shell import shell_allowed, parse_shell

S = Path(__file__).resolve().parent
fixture = runpy.run_path(str(S / 'unit_tests.py'))['terminal_fixture']

original = ['printf %s {1..999999999}', 'printf %s ~root', 'printf %s *']
negative = []
for payload in ['{1..999999999}', '~root', '*', 'file?', '[ab]', '{a,b}']:
    for prefix in ['printf %s ', "printf '%s' ", 'pr"int"f %s ', "printf '%s\\n' ", 'printf\t%s\t']:
        negative.append(('raw-expansion', prefix + payload))
    for surround in [('x""', ''), ("'prefix'", "'suffix'"), ('"nested\'quote"', ''), ('', '\u00a0'), ('\u2003', '')]:
        negative.append(('fragmented-or-unicode-expansion', 'printf %s ' + surround[0] + payload + surround[1]))
for command in ['printf %s "$(pwd)"', 'printf %s "`pwd`"', 'printf %s "$HOME"',
                'printf %s "\\*"', 'printf %s \\*', 'printf %s $\'x\'',
                'printf %s "a\'$(pwd)\'b"', "printf %s 'a'\"$(pwd)\"'b'",
                'printf %s "unterminated', "printf %s 'unterminated",
                'printf %s *\n', 'printf %s ~root\r', 'printf %s {1..9}\x00',
                'printf %b literal', 'printf %n literal', "printf '%s%s' literal",
                "printf '%s\\x41' literal", "printf -- '%s' literal",
                'printf\u00a0%s *', 'printf\u2003%s ~root']:
    negative.append(('quoting-format-whitespace', command))
awk = "awk -F, '/^[a-z]+,[0-9]+$/{print $1 \":\" $2+2}' metrics.csv"
for bad in [awk.replace("'", '"'), awk.replace('+2', '+1000000'), awk.replace('+2', '/2'),
            awk.replace('$2+2', '$3+2'), awk.replace('print ', 'system '),
            awk.replace('}', ';system("id")}'), awk.replace('-F,', '-F:'),
            awk.replace('metrics.csv', '../metrics.csv'), awk.replace('metrics.csv', '~root'),
            awk.replace('metrics.csv', '*.csv'), awk.replace('+2', '+٢'), awk.replace('print ', 'print\t')]:
    negative.append(('awk-boundary', bad))
for command in ['printf %s x || printf %b literal', 'printf %s x || python evil.py',
                "printf %s 'x' || 'sh' 'evil.sh'", 'printf %s x ;; printf %s y',
                'printf %s x | | printf %s y', 'printf %s x && printf %s y',
                'printf %s x & printf %s y', 'printf %s x > ../outside']:
    negative.append(('operator-boundary', command))
positive = [awk, awk.replace('+2', '*999999'), awk.replace('+2', '-0'),
            "printf '%s\\n' '{1..999999999} ~root * [ab] ?' > report.txt",
            'printf %s "{1..999999999} ~root * [ab] ?"', "printf %s 'a\"*\"b'",
            'printf %s "a\'*\'b"', "printf %s '*'\"*\"", 'printf\t%s\tliteral',
            'printf %s \uff0a', 'printf %s \uff5eroot', 'printf %s \uff5b1..9\uff5d',
            "pr'int'f '%s' literal", 'printf %s x || printf %s y']

B1_R2_NAMES = {
    'printf %s x || printf %b literal': 'b1_r2_forbidden_printf_format',
    'printf %s x || python evil.py': 'b1_r2_python_loader',
    "printf %s 'x' || 'sh' 'evil.sh'": 'b1_r2_quoted_sh_loader',
    'printf %s x ;; printf %s y': 'b1_r2_double_semicolon',
    'printf %s x || printf %s y': 'b1_r2_plain_double_pipe',
}
CASES = ([('original', command, 'REJECT') for command in original]
         + [(category, command, 'REJECT') for category, command in negative]
         + [('authorized-literal-control', command, 'ACCEPT') for command in positive[:-1]]
         + [('operator-boundary', positive[-1], 'REJECT')])


def test_f9r2_matrix_matches_retained_reviewer_results():
    retained = json.loads((S / 'F9R2-grammar-results.json').read_text())['cases']
    assert CASES == [(r['category'], r['command'], r['expected']) for r in retained]
    assert len(CASES) == 117
    assert sum(expected == 'REJECT' for _, _, expected in CASES) == 104
    assert sum(expected == 'ACCEPT' for _, _, expected in CASES) == 13
    assert len(B1_R2_NAMES) == 5
    assert set(B1_R2_NAMES) <= {c for _, c, e in CASES if e == 'REJECT'}


@pytest.fixture(scope='module')
def gates():
    generated = runpy.run_path(str(S / 'sync_validator.py'))['validator_source']()
    assert generated.encode() == (S / 'scripts/e97_first_party_validator_diversity.py').read_bytes()
    assert generated.encode() == (Path(os.environ['F9_STAGE_ROOT']) / 'scripts/e97_first_party_validator_diversity.py').read_bytes()
    sealed = {}
    exec(compile(generated, '<sealed-validator>', 'exec'), sealed)
    enc = tiktoken.get_encoding('p50k_base')
    curriculum = driver.load_curriculum()
    tools = driver.tool_manifest(curriculum)['model_visible_tools']
    return sealed, enc, curriculum, tools


@pytest.mark.parametrize('category,command,expected', CASES,
    ids=[B1_R2_NAMES.get(command, f'{category}-{i:03}') for i, (category, command, _) in enumerate(CASES)])
def test_f9r2_four_gate_matrix(category, command, expected, gates):
    sealed, enc, curriculum, tools = gates
    spec = fixture()
    spec['solution']['actions'][1]['arguments']['command'] = command
    for check in (lambda: shell_allowed(command), lambda: sealed['shell_allowed'](command),
                  lambda: author.guards(spec, set(), set())):
        if expected == 'REJECT':
            with pytest.raises(ValueError):
                check()
        else:
            check()
    frame = native_turn({'role': 'assistant', 'content': None,
        'reasoning_content': 'Check admission only; do not execute.',
        'tool_calls': [{'id': 'probe', 'type': 'function', 'function': {
            'name': 'bash', 'arguments': canonical({'command': command})}}]}, tools)
    bridge = SafeWorkspaceBridgeV8(driver.make_panel(curriculum.SYSTEM, tools, driver.POLICY_PANEL),
        'opening', enc, lambda *a: (frame, enc.encode_ordinary(frame), 'valid'))
    request = {'systemPrompt': curriculum.SYSTEM, 'model': MODEL, 'provider': PROVIDER,
        'tools': tools, 'messages': deepcopy(bridge.history)}
    if expected == 'REJECT':
        with pytest.raises(BridgeStopped):
            bridge.next(request)
        assert bridge.failed and bridge.pending is None and len(bridge.history) == 1
        assert bridge.episode.text().endswith(frame)
        from scripts.build_e97_pi_native_curriculum import encode_candidate
        _, mask, _ = encode_candidate(bridge.episode.text(), bridge.generations, 0, enc)
        assert sum(mask) > 0
    else:
        result = bridge.next(request)
        canonical_command = shell_allowed(command)
        assert result['stop_reason'] == 'toolUse' and not bridge.failed
        assert canonical_command != command
        assert bridge.pending['arguments']['command'] == canonical_command
        for message in (result['message'], bridge.history[-1]):
            call = next(b for b in message['content'] if b['type'] == 'toolCall')
            assert call['arguments']['command'] == canonical_command
        assert bridge.episode.text().endswith(frame)  # raw PG source is retained
        assert sealed['shell_allowed'](command) == canonical_command
        assert parse_shell(command) == parse_shell(canonical_command)
        # The actual next-call ingest must agree with canonical public history.
        bridge._ingest(deepcopy(bridge.history) + [{'role': 'toolResult',
            'toolCallId': bridge.pending['id'], 'toolName': 'bash',
            'content': [{'type': 'text', 'text': 'fixture only'}], 'isError': False}])
        assert bridge.pending is None


@pytest.mark.parametrize('command', positive[:-1], ids=[f'authorized-{i:02}' for i in range(13)])
def test_f9r2_authorized_canonical_execution_equivalence(command, tmp_path):
    # Only the 13 explicitly approved controls execute, in isolated CPU fixtures.
    # No negative candidate or loader command ever reaches subprocess.
    outputs = []
    for index, form in enumerate((command, shell_allowed(command))):
        cwd = tmp_path / str(index)
        cwd.mkdir()
        (cwd / 'metrics.csv').write_text('north,7\nsouth,11\n')
        result = subprocess.run(['/bin/bash', '--noprofile', '--norc', '-c', form],
            cwd=cwd, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, capture_output=True, timeout=5)
        outputs.append((result.returncode, result.stdout, result.stderr,
                        {p.name: p.read_bytes() for p in cwd.iterdir()}))
    assert outputs[0] == outputs[1]
    assert outputs[0][0] == 0


@pytest.mark.parametrize('command', [
    'printf %s x >', 'printf %s x > report.txt extra',
    'printf %s x > report.txt | cat report.txt',
    'printf %s x ;', 'printf %s x |', '; printf %s x',
    'printf %s x >>> report.txt', 'printf %s x ;| cat report.txt',
    'printf %s x > report.txt ;; cat report.txt',
    'printf %s x \\'+ '\n' + 'python evil.py',
    'printf %s x\v', 'printf %s x\f',
])
def test_f9e_full_consumption_rejects_unsupported_suffix(command, gates):
    for check in (shell_allowed, gates[0]['shell_allowed']):
        with pytest.raises(ValueError):
            check(command)


@pytest.mark.parametrize('command', [
    "printf %s '|| ;; && > <'", "printf %s 'a'\"'\"'b'",
    "cat report.txt | sort report.txt; printf %s y > report.txt; cat report.txt",
])
def test_f9e_quoted_operators_and_supported_composition_are_canonical(command):
    canonical_command = shell_allowed(command)
    assert parse_shell(canonical_command) == parse_shell(command)
    assert shell_allowed(canonical_command) == canonical_command
    # Independent POSIX word decoding, not just a parser/serializer roundtrip.
    if command.startswith('printf'):
        assert shlex.split(canonical_command) == shlex.split(command)


@pytest.mark.parametrize('canonical_input', [False, True])
def test_f9e_projection_describes_executed_not_generated_action(canonical_input):
    raw = "pr'int'f %s literal"
    executed = shell_allowed(raw)
    generated = executed if canonical_input else raw
    source = [{'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'probe', 'type': 'function', 'function': {'name': 'bash',
         'arguments': canonical({'command': generated})}}]},
        {'role': 'toolResult', 'toolName': 'bash', 'toolCallId': 'probe',
         'content': [{'type': 'text', 'text': 'literal'}], 'isError': False}]
    record = {'source_messages': source, 'status': 'incomplete', 'end_state_task': True}
    projected = driver.gym_terminal_projection(record, 'system', 'opening')
    assert projected['actions'][0]['arguments']['command'] == executed
    call = projected['messages'][2]['tool_calls'][0]['function']
    assert json.loads(call['arguments'])['command'] == executed
    assert json.loads(source[0]['tool_calls'][0]['function']['arguments'])['command'] == generated
    if not canonical_input:
        assert executed != generated
    assert 'executed canonical' in projected['projection_note']
    ordinary = driver.gym_terminal_projection({**record, 'end_state_task': False}, 'system', 'opening')
    assert ordinary['actions'][0]['arguments']['command'] == generated
    assert ordinary['messages'][2]['tool_calls'][0]['function']['arguments'] == source[0]['tool_calls'][0]['function']['arguments']
    assert 'format-only' in ordinary['projection_note']
