import io
import json
import tarfile
from copy import deepcopy
import pytest
from scripts.e97_native_execution_cases import cases, grade, aggregate_results
from scripts.e97_native_execution_sandbox import archive_file


def calls_for(c):
    observed = dict(request=dict(name='execute_bash', arguments={'command': 'cat file'}),
                    result={'message': {'content': c['answer']}})
    if c['family'] == 'recovery':
        return [dict(request=dict(name='str_replace_editor', arguments=dict(command='view', path='/testbed/missing.json')),
                     result={'message': {'content': 'ERROR: missing file'}}), observed]
    return [observed]


def snapshot(c):
    return {**c['files'], **({'result.json': json.dumps(c['expected_output'])} if c['expected_output'] else {})}


def test_counterfactual_pairs():
    a = cases('test-seed')
    assert a == cases('test-seed') and a != cases('other-seed')
    assert len({c['id'] for c in a}) == 8
    for first, second in zip(a[::2], a[1::2]):
        assert first['prompt'] == second['prompt'] and first['files'] != second['files']
        assert first['expected_output'] != second['expected_output'] or first['answer'] != second['answer']
        assert all('/testbed' in c['prompt'] for c in (first, second))
        if first['family'] != 'edit':
            assert first['answer'] not in first['prompt'] and second['answer'] not in second['prompt']


@pytest.mark.parametrize('c', cases('test-seed'), ids=lambda c:c['id'])
def test_grades_require_finish_execution_and_unchanged_inputs(c):
    assert grade(c, c['answer'], calls_for(c), snapshot(c))['success']
    assert not grade(c, 'wrong', calls_for(c), snapshot(c))['success']
    assert not grade(c, None, calls_for(c), snapshot(c))['success']
    assert not grade(c, c['answer'], [], snapshot(c))['success']
    corrupted = snapshot(c); corrupted[next(iter(c['files']))] = 'changed'
    assert not grade(c, c['answer'], calls_for(c), corrupted)['success']


def test_output_cannot_be_claimed_without_correct_file():
    c = cases('seed')[4]
    for value in ('', '{}', '{"count": 7}', 'not json', None):
        assert not grade(c, c['answer'], calls_for(c), {**c['files'], 'result.json': value})['success']


def test_recovery_requires_error_before_later_observation():
    c = cases('seed')[6]
    for calls in (calls_for(c)[1:], calls_for(c)[::-1], calls_for(c)[:1]):
        assert not grade(c, c['answer'], calls, snapshot(c))['success']


def archive(kind=tarfile.REGTYPE, content=b'hello'):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as f:
        info = tarfile.TarInfo('result.json'); info.type = kind
        if kind == tarfile.REGTYPE:
            info.size = len(content); f.addfile(info, io.BytesIO(content))
        else:
            info.linkname = '/etc/passwd'; f.addfile(info)
    return out.getvalue()


def test_regular_bounded_archive_only():
    assert archive_file(archive()) == 'hello'
    for data in (archive(tarfile.SYMTYPE), archive(tarfile.LNKTYPE), archive(tarfile.DIRTYPE), archive(content=b'x'*65537)):
        with pytest.raises(ValueError):
            archive_file(data)


def test_aggregate_coverage_and_pair_success():
    panel = {'models': [{'name': str(i)} for i in range(4)], 'cases': cases('seed')}
    reports = {}
    for rank in range(8):
        rows = [dict(id=c['id'], family=c['family'], reason='finished', grade={'success': True})
                for c in panel['cases'] if c['pair_index'] % 2 == rank % 2]
        reports[rank] = dict(rank=rank, model=panel['models'][rank//2], results=rows)
    result = aggregate_results(panel, reports)
    assert result['models']['0']['successes'] == 8
    reports[0]['results'][0]['grade']['success'] = False
    assert not aggregate_results(panel, reports)['models']['0']['families']['lookup']['paired_success']
    broken = deepcopy(reports); broken[0]['results'].pop()
    with pytest.raises(ValueError, match='coverage'):
        aggregate_results(panel, broken)


@pytest.mark.parametrize('finish', [True, False])
def test_episode_feeds_observation_before_finish(monkeypatch, tmp_path, finish):
    import tiktoken
    import scripts.eval_e97_native_execution as evaluator
    from scripts.e97_open_swe_native_codec import native_turn
    c = cases('causal-test')[0]
    tools = [{'type': 'function', 'function': {'name': n}} for n in
             ('execute_bash', 'str_replace_editor', 'think', 'finish')]
    panel = dict(tools=tools, system='Use tools.', episode_seconds=60, max_turns=8,
                 generation_budget=4096, episode_generation_budget=8192)
    prompts = []
    def generate(loaded, prompt, encoding, budget, deadline):
        prompts.append(prompt)
        name, args = ('str_replace_editor', {'command': 'view', 'path': '/testbed/config.json'}) if len(prompts) == 1 or not finish else ('finish', {'message': c['answer']})
        text = native_turn(dict(role='assistant', content=None, reasoning_content=None, think=None,
                                tool_calls=[{'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]))
        return text, encoding.encode_ordinary(text), 'valid'
    class Sandbox:
        def __init__(self, panel, output):
            output.mkdir()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def request(self, op, **kwargs):
            return {'result': {'message': evaluator.context('tool', c['files']['config.json'])}}
        def snapshot(self, names): return c['files']
    monkeypatch.setattr(evaluator, 'generate_turn', generate)
    monkeypatch.setattr(evaluator, 'NativeSandbox', Sandbox)
    result = evaluator.episode(None, c, panel, tiktoken.get_encoding('p50k_base'), tmp_path/'episode')
    assert result['grade']['success'] is finish and len(prompts) == (2 if finish else 8)
    assert c['answer'] not in prompts[0] and c['answer'] in prompts[1]
    assert 'Tool:' in prompts[1] and len(result['calls']) == (1 if finish else 8)
    assert result['reason'] == ('finished' if finish else 'turn_budget')


def test_paused_reader_no_follow_and_bounded_files(tmp_path):
    import os
    from scripts.e97_native_snapshot_reader import read_regular_files
    root = tmp_path/'testbed'; root.mkdir()
    (root/'good').write_text('verified')
    (root/'large').write_bytes(b'x'*65537)
    (root/'binary').write_bytes(b'\xff')
    (root/'link').symlink_to(root/'good')
    (root/'dirlink').symlink_to(root, target_is_directory=True)
    os.mkfifo(root/'fifo')
    names = ['good', 'large', 'binary', 'link', 'dirlink/good', 'fifo', 'missing']
    values = read_regular_files(tmp_path, names)
    assert values == {name: 'verified' if name == 'good' else None for name in names}
    for names in (['../escape'], ['/absolute'], [], ['good', 'good']):
        with pytest.raises(ValueError):
            read_regular_files(tmp_path, names)
