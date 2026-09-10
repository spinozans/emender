import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tiktoken

from scripts.audit_e97_open_swe_semantics import run as census
from scripts.audit_e97_open_swe_source_fidelity import (
    load_converter, raw_diagnostics, run, source_read_category, unit_hash,
)
from test_e97_open_swe_semantics import make_authority


def test_categories_and_unit_identity():
    assert source_read_category({}) == 'no_view_range'
    assert source_read_category({'view_range': [1, -1]}) == 'open_ended_view_range'
    assert source_read_category({'view_range': [1, 200]}) == 'finite_view_range'
    assert source_read_category({'view_range': ['bad', 'bad']}) == 'other_view_range'
    assert unit_hash('a', 'bc') != unit_hash('ab', 'c')
    assert unit_hash('a', None) != unit_hash('a', '')


def create_fixture(tmp_path):
    authority = tmp_path/'authority'
    make_authority(authority)
    manifest = json.loads((authority/'manifest.json').read_text())
    metadata = authority/manifest['outputs']['metadata']['path']
    rows = [json.loads(x) for x in metadata.read_text().splitlines()]
    for row in rows:
        row['identity'] = 'open-swe:'+row['identity']
        row['trajectory_identity'] = 'open-swe:t'
    metadata.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def descriptor(path):
        return {'path': path.name, 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest['outputs']['metadata'] = descriptor(metadata)
    normalizer = '''import re
raise RuntimeError("TOP LEVEL MUST NEVER EXECUTE")
_WORKSPACE = re.compile(r"/workspace/repo/")
def normalize_text(text):
    return _WORKSPACE.sub("", text).strip()
def normalize_path(path):
    return normalize_text(path)
def normalize_command(command):
    return normalize_text(command)
def action_from_call(call, next_tool):
    v=json.loads(call['function']['arguments'])
    a={'path':normalize_path(v['path']), 'offset':1, 'limit':2}
    return 'action', 'Action: read\\nArguments: '+json.dumps(a,separators=(',', ':'))
'''
    builder = '''raise RuntimeError("TOP LEVEL MUST NEVER EXECUTE")
def _finish_from_call(call):
    return 'Final: Done.'
def _reasoning_prefix(reasoning, encoding=None):
    return reasoning, 2
def normalize_private_messages(row, encoding=None):
    return [(role, text) for role, text in row['normalized']], [2,2], 0
'''
    manifest['source_snapshots'] = {}
    for key, text in [('action_normalizer_source', normalizer), ('builder_source', builder)]:
        p = authority/(key+'.py'); p.write_text(text)
        manifest['source_snapshots'][key] = descriptor(p)
    manifest['analysis_policy']['admission_byte_cap'] = 65536
    manifest['system_prompt'] = 'Use tools.'
    raw = tmp_path/'raw';raw.mkdir()
    (raw/'README.md').write_text('fixture card')
    manifest['dataset_card_sha256'] = hashlib.sha256((raw/'README.md').read_bytes()).hexdigest()
    read = 'Analysis: "Check."\nAction: read\nArguments: {"path":"a.py","offset":1,"limit":2}'
    obs = "Here's the result of running `cat -n` on a.py:\n     1\ta\n     2\tb\n     3\tc"
    row = {'trajectory_id': 't', 'repo': 'test/repo', 'normalized': [
        ['system', 'Use tools.'], ['user', 'Check a.py.'], ['assistant', read],
        ['tool', obs], ['assistant', 'Analysis: "Done."\nFinal: Done.']],
        'messages': [{'role':'user','content':'Check a.py.', 'tool_calls':None},
                     {'role':'assistant','content':None, 'tool_calls':[{'function':{
                         'name':'str_replace_editor','arguments':json.dumps({'command':'view','path':'/workspace/repo/a.py','view_range':[1,2]})}}]},
                     {'role':'tool','content':obs, 'tool_calls':None},
                     {'role':'assistant','content':None, 'tool_calls':[{'function':{
                         'name':'finish','arguments':'{"message":"Done."}'}}]}]}
    p = raw/'fake.parquet';pq.write_table(pa.Table.from_pylist([row]),p)
    d = descriptor(p);d['name']=d.pop('path');manifest['input_files']=[d]
    (authority/'manifest.json').write_text(json.dumps(manifest))
    digest = hashlib.sha256((authority/'manifest.json').read_bytes()).hexdigest()
    return authority, raw, digest, row


def test_archived_function_isolation_and_diagnostics(tmp_path):
    authority, raw, digest, row = create_fixture(tmp_path)
    manifest = json.loads((authority/'manifest.json').read_text())
    converter = load_converter(authority, manifest, tiktoken.get_encoding('p50k_base'))
    counts, ranges, roots, examples = raw_diagnostics(row, converter)
    assert counts['out_of_bounds_reads'] == 1
    assert ranges['finite_view_range'] == 1
    assert counts['user_messages_before_finish'] == 1
    assert converter['normalize_path']('/workspace/repo/a.py') == 'a.py'
    assert len(examples) == 1


def test_full_raw_sequence_reconciliation_and_mutation_rejection(tmp_path):
    authority, raw, digest, row = create_fixture(tmp_path)
    census(authority, digest, tmp_path/'census')
    receipt = tmp_path/'census/summary.json'
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    run(authority, raw, digest, receipt, receipt_hash, tmp_path/'source-results')
    result = json.loads((tmp_path/'source-results/summary.json').read_text())
    assert result['counts']['verified_assistant_units'] == 2
    assert result['counts']['verified_trajectory_sequences'] == 1
    assert result['counts']['out_of_bounds/finite_view_range'] == 1
    assert result['affected_trajectory_counts']['bounds_trajectories'] == 1
    (raw/'fake.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='raw source identity mismatch'):
        run(authority, raw, digest, receipt, receipt_hash, tmp_path/'invalid-results')
    assert not (tmp_path/'invalid-results/summary.json').exists()
