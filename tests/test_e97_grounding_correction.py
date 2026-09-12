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


@pytest.mark.parametrize('successes,tool_accuracy,expected', [(4,1.,True),(3,1.,False),(8,.97,False)])
def test_correction_gate_never_auto_expands(tmp_path,successes,tool_accuracy,expected):
    import json
    from types import SimpleNamespace
    from scripts.prepare_e97_grounding_correction import gate,sha
    out=tmp_path/'evaluation';(out/'execution').mkdir(parents=True);(out/'learning').mkdir()
    (tmp_path/'summary.json').write_text('{}')
    for kind in ('execution','learning'):(out/kind/'panel.json').write_text('{}')
    def metrics(accuracy):
        return {c:{'metrics':{'assistant':{'token_accuracy':accuracy,'record_macro_nll':1.}}}
                for c in ('tool-retention','conversation-retention','native-development')}
    learning=dict(panel_sha256=sha(out/'learning/panel.json'),models={'pre-y':metrics(1.),'correction-y':metrics(tool_accuracy),'correction-x':metrics(1.)})
    execution=dict(panel_sha256=sha(out/'execution/panel.json'),models={'correction-y':{'outcomes':[
        {'id':f'fresh-{i}','grade':{'success':i<successes}} for i in range(8)]}})
    (out/'learning/summary.json').write_text(json.dumps(learning));(out/'execution/summary.json').write_text(json.dumps(execution))
    gate(SimpleNamespace(phase=tmp_path))
    result=json.loads((out/'gate.json').read_text())
    assert result['positive_correction_evidence'] is expected
    assert not result['automatic_expansion'] and not result['checkpoint_promotion']
