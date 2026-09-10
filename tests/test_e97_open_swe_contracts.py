import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

from scripts.audit_e97_open_swe_contracts import field_relation, inspect_trajectory, run
from test_e97_open_swe_source_fidelity import create_fixture


def assistant(name, arguments, reasoning='Plan.', content=None):
    return {'role': 'assistant', 'reasoning_content': reasoning, 'content': content,
            'tool_calls': [{'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def test_containment_is_explicitly_lexical():
    assert field_relation('  ', 'x') == 'empty'
    assert field_relation('a\n b', 'a b') == 'same_as_reasoning'
    assert field_relation('b', 'a b c') == 'contained_in_reasoning'
    assert field_relation('a b c', 'b') == 'contains_reasoning_plus_text'
    assert field_relation('new fact', 'another fact') == 'not_contained_in_reasoning'


def test_think_payload_omission_and_hypothetical_cap():
    encoding = tiktoken.get_encoding('p50k_base')
    row = {'messages': [assistant('think', {'thought': 'Important fact. '*100}),
                        assistant('str_replace_editor', {'command': 'view', 'path': '/a'}),
                        assistant('finish', {'message': 'Done.'})]}
    c, samples = inspect_trajectory(row, encoding, token_cap=20)
    assert c['think_calls'] == 1
    assert c['think_thought_not_contained_in_reasoning'] == 1
    assert c['hypothetical_thought_fold_overcap_units'] == 1
    assert c['analysis_units'] == 2
    assert c['initial_executable_call/str_replace_editor'] == 1
    assert c['hypothetical_analysis_with_thought_tokens'] > c['baseline_analysis_tokens']


def test_shell_arguments_never_execute(tmp_path):
    destination = tmp_path/'must-not-exist'
    row = {'messages': [assistant('execute_bash', {'command': f'touch {destination}; export X=1',
                                                   'is_input': 'true', 'timeout': 300}, content='Different commentary.'),
                        assistant('finish', {'message': 'Done.\nTwo paragraphs.'})]}
    c, _ = inspect_trajectory(row, tiktoken.get_encoding('p50k_base'))
    assert c['bash_is_input_true'] == 1 and c['bash_explicit_timeout'] == 1
    assert c['bash_timeout_over_120'] == 1
    assert c['bash_session_mutation_lexical_candidates'] == 1
    assert c['finish_message_whitespace_changed'] == 1
    assert not destination.exists()


def test_full_contract_census(tmp_path):
    authority, raw, old_digest, row = create_fixture(tmp_path)
    row['messages'][1]['reasoning_content'] = 'Check.'
    row['messages'][3]['reasoning_content'] = 'Done.'
    row['tools'] = [json.dumps({'type': 'function', 'function': {'name': 'execute_bash',
                   'description': 'Persistent shell', 'parameters': {'type': 'object'}}})]
    p = raw/'fake.parquet';pq.write_table(pa.Table.from_pylist([row]), p)
    manifest = json.loads((authority/'manifest.json').read_text())
    manifest['input_files'][0].update(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (authority/'manifest.json').write_text(json.dumps(manifest))
    native = tmp_path/'native.ts';native.write_text('trusted fixture source')
    digest = hashlib.sha256((authority/'manifest.json').read_bytes()).hexdigest()
    run(authority, raw, digest, native, hashlib.sha256(native.read_bytes()).hexdigest(), tmp_path/'results')
    result = json.loads((tmp_path/'results/summary.json').read_text())
    assert result['included_trajectories'] == 1
    assert result['counts']['analysis_units'] == 2
    assert result['tool_specification_variants'] == 1
