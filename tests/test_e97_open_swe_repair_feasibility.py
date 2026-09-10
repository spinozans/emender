import json

import pytest
import tiktoken

from scripts.audit_e97_open_swe_repair_feasibility import project


def assistant(name, args, reason='Plan.', content=None):
    return {'role':'assistant','reasoning_content':reason,'content':content,
            'tool_calls':[{'function':{'name':name,'arguments':json.dumps(args)}}]}


def fixture():
    return {'tools':[], 'messages':[
        {'role':'system','content':'Original system.'},
        {'role':'user','content':'Work in /workspace/repo.'},
        assistant('str_replace_editor', {'command':'view','path':'/workspace/repo/a.py'},
                  reason='  Inspect.\n',content='Public source commentary.\n'),
        {'role':'tool','content':"Here's the result of running `cat -n` on /workspace/repo/a.py:\n     1\tx\n"},
        assistant('finish',{'message':'First paragraph.\n\nSecond paragraph.'})]}


def test_source_fields_remain_exact_with_declared_commentary_placement():
    row=fixture(); result,pieces=project(row,tiktoken.get_encoding('p50k_base'))
    assert result['within_proposed_envelope']
    assert result['provenance_piece_counts']['content_pieces']==1
    targets=[text for text,target in pieces if target and text!='\x1e']
    reasoning=json.loads(targets[0].split('\n')[0].removeprefix('Analysis: '))
    assert reasoning=='  Inspect.\n\n\nPublic source commentary.\n'
    args=json.loads(targets[0].partition('\nArguments: ')[2])
    assert args=={'command':'view','path':'/workspace/repo/a.py'}
    assert any(text=='\n\nTool:\n'+row['messages'][3]['content'] for text,_ in pieces)
    assert targets[-1].endswith('Final: First paragraph.\n\nSecond paragraph.')


def test_whole_context_and_analysis_caps_exclude_without_truncation():
    row=fixture();encoding=tiktoken.get_encoding('p50k_base')
    result,pieces=project(row,encoding,max_record=10)
    assert not result['within_proposed_envelope']
    assert 'whole_trajectory_context_cap' in result['exclusion_reasons']
    assert any('Second paragraph.' in text for text,_ in pieces)
    row['messages'][2]['reasoning_content']='word '*100
    result,_=project(row,encoding,analysis_cap=20)
    assert 'analysis_cap' in result['exclusion_reasons']


def test_think_argument_preserved_and_acknowledgement_checked():
    row=fixture()
    row['messages'][2:2]=[assistant('think',{'thought':'Distinct thought.'},reason='Think rationale.'),
                          {'role':'tool','content':'Your thought has been logged.'}]
    result,pieces=project(row,tiktoken.get_encoding('p50k_base'))
    assert result['provenance_piece_counts']['think_thought_pieces']==1
    assert any('Distinct thought.' in text for text,target in pieces if target)
    row['messages'][3]['content']='Unexpected information.'
    with pytest.raises(ValueError,match='noncanonical_think'):
        project(row,tiktoken.get_encoding('p50k_base'))


def test_literal_separator_fails_closed():
    row=fixture();row['messages'][3]['content']='raw\x1eseparator'
    with pytest.raises(ValueError,match='literal_record_separator'):
        project(row,tiktoken.get_encoding('p50k_base'))
