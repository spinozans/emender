from collections import Counter
import pytest
from scripts.build_e97_grounding_correction import fixtures
from scripts.e97_native_execution_cases import cases


def test_correction_fixtures_are_balanced_and_disjoint():
    rows=fixtures('grounding-correction-train-20260912-v1',1024)
    assert rows==fixtures('grounding-correction-train-20260912-v1',1024)
    assert Counter(r['family'] for r in rows)==dict(lookup=256,sum=256,edit=256,recovery=256)
    assert len({r['id'] for r in rows})==1024
    assert len({r['base'] for r in rows})==1024
    for family in ('lookup','sum','edit','recovery'):
        assert sum(r['source_style'] for r in rows if r['family']==family)==128
    forbidden={r['answer'] for r in cases('grounding-correction-validation-20260912-v1') if r['family']!='edit'}
    assert not any(r['answer'] in forbidden for r in rows if r['family']!='edit')
    assert all(all(p.startswith('/testbed/train/') for p in r['files']) for r in rows)
    assert all(r['path'] in r['prompt'] for r in rows)


def test_answer_not_leaked_into_task_prompt():
    for row in fixtures('test',128):
        if row['family']!='edit':
            assert row['answer'] not in row['prompt']
