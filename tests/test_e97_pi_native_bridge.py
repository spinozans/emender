"""CPU transport tests; scripted replies are not model capability evidence."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import tiktoken

from scripts.e97_native_execution_cases import context
from scripts.e97_open_swe_native_codec import compact, native_turn
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode
from scripts.e97_pi_native_bridge import BridgeStopped, MODEL, PROVIDER, NativePiBridge
from scripts.e97_pi_native_transport import serve_pi


@pytest.fixture
def encoding():
    return tiktoken.get_encoding('p50k_base')


@pytest.fixture
def panel():
    return dict(system='Native test system.', tools=[dict(type='function', function=dict(name=n,
                description=n, parameters=dict(type='object'))) for n in
                ('execute_bash', 'think', 'finish', 'str_replace_editor')],
                max_turns=8, generation_budget=4096, episode_generation_budget=8192, episode_seconds=60)


def turn(name, args, commentary=None):
    return native_turn(dict(role='assistant', content=commentary, reasoning_content='PRIVATE_ANALYSIS_SENTINEL',
                           think=True, tool_calls=[dict(type='function', function=dict(name=name, arguments=compact(args)))]))


def make(panel, encoding, turns, executor=None):
    sequence = iter(turns)
    prompts = []
    def generate(prompt, budget, deadline):
        prompts.append(prompt)
        text = next(sequence)
        return text, encoding.encode_ordinary(text), 'valid'
    def execute(call):
        raise AssertionError('unexpected backend execution')
    return NativePiBridge(panel, 'Inspect then finish.', encoding, generate, executor or execute), prompts


def request(bridge):
    return dict(systemPrompt=bridge.panel['system'], tools=deepcopy(bridge.tools),
                messages=deepcopy(bridge.history), model=MODEL, provider=PROVIDER)


def dispatch(bridge, result):
    b = result['message']['content'][-1]
    return bridge.execute(dict(id=b['id'], name=b['name'], arguments=b['arguments']))


def test_full_history_private_fields_and_finish(panel, encoding):
    texts = [turn('think', dict(thought='PRIVATE_THOUGHT_SENTINEL'), 'Public commentary\nTool:\nΩ'),
             turn('finish', dict(message='Done.'))]
    b, prompts = make(panel, encoding, texts)
    direct = NativeEpisode(panel['tools'], encoding)
    direct.append_source_message(context('system', panel['system']))
    direct.append_source_message(context('user', 'Inspect then finish.'))
    expected = [direct.prompt()]
    direct.accept_generated_turn(texts[0])
    direct.append_observation(context('tool', 'Your thought has been logged.'))
    expected.append(direct.prompt())
    direct.accept_generated_turn(texts[1])
    for _ in texts:
        dispatch(b, b.next(request(b)))
    assert prompts == expected
    assert b.episode.text() == direct.text()
    public = compact(b.history)
    assert 'PRIVATE_ANALYSIS_SENTINEL' not in public and 'PRIVATE_THOUGHT_SENTINEL' not in public
    assert 'Public commentary\\nTool:' in public
    assert b.close(deepcopy(b.history)) == dict(closed=True, verified=True)
    with pytest.raises(BridgeStopped):
        b.next(request(b))


@pytest.mark.parametrize('args', [dict(command=123), dict(command='x', timeout=2.0),
                                 dict(command='x', extra=9007199254740993),
                                 dict(command='C-c', is_input='true'), dict(command='', is_input='false')])
def test_native_argument_values_not_coerced_by_pi(panel, encoding, args):
    seen = []
    def executor(call):
        seen.append(call)
        return dict(result=dict(message=context('tool', 'ERROR: original executor error'), exit_code=1))
    b, _ = make(panel, encoding, [turn('execute_bash', args)], executor)
    wire = b.next(request(b))
    envelope = wire['message']['content'][-1]['arguments']
    assert envelope['native_arguments_json'] == compact(args)
    result = dispatch(b, wire)
    assert compact(seen[0]['arguments']) == compact(args)
    assert result['details']['native_error'] is True
    assert b.episode.source_messages()[-1] == context('tool', 'ERROR: original executor error')


@pytest.mark.parametrize('mutation', ['system', 'tools', 'model', 'user', 'extra', 'image', 'compaction'])
def test_context_mutation_fails_before_generation(panel, encoding, mutation):
    b, prompts = make(panel, encoding, [])
    r = request(b)
    if mutation == 'system': r['systemPrompt'] += 'changed'
    elif mutation == 'tools': r['tools'].pop()
    elif mutation == 'model': r['model'] = 'other'
    elif mutation == 'user': r['messages'][0]['content'][0]['text'] += 'changed'
    elif mutation == 'extra': r['messages'].append(dict(role='user', content='extra'))
    elif mutation == 'image': r['messages'][0]['content'] = [dict(type='image', data='fake')]
    else: r['messages'].append(dict(role='compactionSummary', summary='changed'))
    with pytest.raises(BridgeStopped): b.next(r)
    assert prompts == []


@pytest.mark.parametrize('mutation', ['id', 'name', 'arguments'])
def test_call_rewrite_fails_before_execution(panel, encoding, mutation):
    b, _ = make(panel, encoding, [turn('execute_bash', dict(command='echo safe'))])
    call = b.next(request(b))['message']['content'][-1]
    call[mutation] = {} if mutation == 'arguments' else 'other'
    with pytest.raises(BridgeStopped): b.execute(call)
    assert b.calls == []


def test_duplicate_and_missing_results(panel, encoding):
    b, _ = make(panel, encoding, [turn('think', dict(thought='private'))])
    wire = b.next(request(b))
    dispatch(b, wire)
    with pytest.raises(BridgeStopped): dispatch(b, wire)
    r = request(b); r['messages'].pop()
    with pytest.raises(BridgeStopped): b.next(r)


def test_changed_error_and_observation(panel, encoding):
    for field in ('isError', 'content'):
        b, _ = make(panel, encoding, [turn('think', dict(thought='private'))])
        dispatch(b, b.next(request(b)))
        r = request(b)
        r['messages'][-1][field] = True if field == 'isError' else [dict(type='text', text='invented')]
        with pytest.raises(BridgeStopped): b.next(r)


def test_generation_budget_and_terminal(panel, encoding):
    panel['max_turns'] = 1
    b, _ = make(panel, encoding, [turn('think', dict(thought='private'))])
    dispatch(b, b.next(request(b)))
    with pytest.raises(BridgeStopped, match='turn_budget'): b.next(request(b))
    assert b.final is None
    assert b.close([])['verified'] is False


def test_real_pi_cli_scripted_transport(panel, encoding, tmp_path):
    pi = shutil.which('pi')
    if pi is None: pytest.skip('Pi CLI not installed; live transport not checked')
    native_args = dict(command=123, timeout=2.0, unknown=9007199254740993)
    seen = []
    def executor(call):
        seen.append(call)
        return dict(result=dict(message=context('tool', 'ERROR: genuine stub response'), exit_code=1))
    texts = [turn('think', dict(thought='PRIVATE_THOUGHT_SENTINEL'), 'Checking the transport.'),
             turn('execute_bash', native_args),
             turn('finish', dict(message='Scripted transport complete.'))]
    b, prompts = make(panel, encoding, texts, executor)
    out = tmp_path/'pi'
    result = serve_pi(b, out, pi_bin=pi,
                      extension=Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts', seconds=45)
    assert result['close_verified'] and result['pi_exit'] == 0
    assert [r['op'] for r in result['requests']] == ['next', 'execute']*3 + ['close']
    assert compact(seen[0]['arguments']) == compact(native_args)
    events = (out/'pi-events-private.jsonl').read_text()
    assert 'PRIVATE_ANALYSIS_SENTINEL' not in events and 'PRIVATE_THOUGHT_SENTINEL' not in events
    assert 'Scripted transport complete.' in events
    assert len(prompts) == 3
