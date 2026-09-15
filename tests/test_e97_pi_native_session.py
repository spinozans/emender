"""CPU session control tests, not live Pi, native executor or model evidence."""
from copy import deepcopy
import pytest
import tiktoken

from scripts.e97_native_execution_cases import context
from scripts.e97_open_swe_native_codec import compact, native_turn
from scripts.e97_pi_native_bridge import MODEL, PROVIDER, BridgeStopped
from scripts.e97_pi_native_session import NativeTaskSession, history_sha


def text(name, args):
    return native_turn(dict(role='assistant', content=None, reasoning_content='retained private note', think=None,
                           tool_calls=[dict(type='function', function=dict(name=name, arguments=compact(args)))]))


def fixture(**limits):
    panel = dict(system='native system', tools=[dict(type='function', function=dict(name=n)) for n in
                 ('execute_bash', 'str_replace_editor', 'think', 'finish')],
                 max_turns=8, generation_budget=4096, episode_generation_budget=8192, episode_seconds=60)
    encoding = tiktoken.get_encoding('p50k_base')
    queue = []; seen = []; state = {}
    def generate(prompt, budget, deadline):
        seen.append(prompt)
        value = queue.pop(0)
        return value, encoding.encode_ordinary(value), 'valid'
    def execute(call):
        if call['arguments']['command'] == 'set': state['value'] = 23
        return dict(result=dict(message=context('tool', str(state['value'])), exit_code=0))
    return NativeTaskSession(panel, encoding, generate, execute, **limits), queue, seen, state


def begin(s, name):
    return s.begin_task(name, 'Task '+name, expected_history_sha=history_sha(s.completed_history),
                        acknowledge_fresh_record=True)


def run_turn(s, queue, name, args):
    queue.append(text(name, args))
    b = s.current
    response = s.next(dict(systemPrompt=b.panel['system'], tools=b.tools, model=MODEL, provider=PROVIDER,
                           messages=deepcopy(s.completed_history+b.history)))
    call = response['message']['content'][-1]
    return s.execute(dict(id=call['id'], name=call['name'], arguments=call['arguments']))


def settle(s):
    return s.settle_task(deepcopy(s.completed_history+s.current.history))


def test_two_tasks_retain_private_history_and_executor_but_explicitly_reset_record():
    s, q, seen, state = fixture()
    first = begin(s, 'one')
    assert first['previous_tasks_in_model_context'] is False
    run_turn(s, q, 'execute_bash', dict(command='set'))
    run_turn(s, q, 'finish', dict(message='done one'))
    first_text = s.current.episode.text()
    settle(s)
    prefix = deepcopy(s.completed_history)
    second = begin(s, 'two')
    assert second['prior_tasks_retained'] == 1 and second['executor_reused']
    assert 'Task one' not in s.current.episode.prompt()
    run_turn(s, q, 'execute_bash', dict(command='get'))
    assert state == dict(value=23)
    assert s.current.episode.source_messages()[-1]['content'] == '23'
    assert s.tasks[0].episode.text() == first_text and 'retained private note' in first_text
    run_turn(s, q, 'finish', dict(message='done two')); settle(s)
    assert s.completed_history[:len(prefix)] == prefix
    receipt = s.close(expected_history_sha=history_sha(s.completed_history))
    assert receipt['private_native_records_retained'] == 2 and not receipt['conversational_memory_claim']
    assert len(seen) == 4


def test_followup_not_implicitly_new_task():
    s, q, seen, _ = fixture(); begin(s, 'one')
    run_turn(s, q, 'finish', dict(message='done')); settle(s)
    with pytest.raises(BridgeStopped, match='explicit_task_boundary'):
        s.next(dict(messages=s.completed_history+[dict(role='user', content='Do another thing')]))
    assert len(seen) == 1


def test_boundary_transactional_until_verified_settlement():
    s, q, _, _ = fixture(); begin(s, 'one')
    with pytest.raises(BridgeStopped, match='previous_task_not_settled'): begin(s, 'two')
    run_turn(s, q, 'finish', dict(message='done'))
    with pytest.raises(BridgeStopped, match='previous_task_not_settled'): begin(s, 'two')
    settle(s); begin(s, 'two')
    assert len(s.tasks) == 2


@pytest.mark.parametrize('case', ['missing_ack', 'bad_hash', 'duplicate', 'exhausted'])
def test_bad_boundaries_rejected_without_creating_task(case):
    s, q, _, _ = fixture(max_tasks=1 if case == 'exhausted' else 2)
    begin(s, 'one'); run_turn(s, q, 'finish', dict(message='done')); settle(s)
    kwargs = dict(expected_history_sha=history_sha(s.completed_history), acknowledge_fresh_record=True)
    if case == 'missing_ack': kwargs['acknowledge_fresh_record'] = False
    if case == 'bad_hash': kwargs['expected_history_sha'] = 'bad'
    with pytest.raises(BridgeStopped): s.begin_task('one' if case == 'duplicate' else 'two', 'next', **kwargs)
    assert len(s.tasks) == 1 and s.current is None


def test_prior_history_mutation_rejected_before_next_model_call():
    s, q, seen, _ = fixture(); begin(s, 'one')
    run_turn(s, q, 'finish', dict(message='done')); settle(s); begin(s, 'two')
    messages = deepcopy(s.completed_history+s.current.history)
    messages[0]['content'][0]['text'] += 'changed'
    with pytest.raises(BridgeStopped, match='completed_session_history_changed'): s.next(dict(messages=messages))
    assert len(seen) == 1


def test_session_token_and_deadline_budgets():
    s, q, _, _ = fixture(session_tokens=1000); begin(s, 'one')
    run_turn(s, q, 'finish', dict(message='done')); settle(s)
    used = s.tasks[0].tokens; begin(s, 'two')
    assert s.current.panel['episode_generation_budget'] == 1000-used
    assert s.current.deadline <= s.deadline
    t, _, _, _ = fixture(); t.deadline = 0
    with pytest.raises(BridgeStopped, match='session_not_active'): begin(t, 'late')


@pytest.mark.parametrize('kwargs', [dict(max_tasks=True), dict(max_tasks=5), dict(session_tokens=0), dict(session_seconds=2401)])
def test_session_limits(kwargs):
    with pytest.raises(ValueError): fixture(**kwargs)
