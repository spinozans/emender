import json
import math
import time
from collections import Counter
from types import SimpleNamespace

import pytest
import tiktoken

from scripts.e97_native_execution_cases import context, grade
from scripts.e97_native_onpolicy_canary import candidate, tasks
from scripts.e97_open_swe_native_codec import native_turn, render

TOOLS=[{'type':'function','function':{'name':n}} for n in ('execute_bash','str_replace_editor','think','finish')]


def assistant(name,args):
    return dict(role='assistant',content=None,reasoning_content=None,think=None,
                tool_calls=[dict(type='function',function=dict(name=name,arguments=json.dumps(args)))])


def test_training_tasks_vary_selectors_and_deltas_without_answer_leakage():
    rows=tasks('canary-unit',16)
    assert rows==tasks('canary-unit',16)
    assert Counter(r['family'] for r in rows)==dict(lookup=4,sum=4,edit=4,recovery=4)
    assert {r['delta'] for r in rows if r['family']=='edit'}=={1,3,7,11}
    for c in rows:
        assert all(p.startswith('onpolicy/') for p in c['files'])
        if c['family']!='edit':assert c['answer'] not in c['prompt']
        if c['family']=='lookup':
            d=json.loads(c['files'][c['path'].removeprefix('/testbed/')])
            assert d['values'][d['active']]==c['answer']
            assert len(d['values'])==3 and d['active']!='selected'


def test_teacher_masks_preserve_complete_unsupervised_failed_prefix():
    enc=tiktoken.get_encoding('p50k_base')
    messages=[context('system','Use tools'),context('user','Read the requested file.'),
              assistant('str_replace_editor',dict(command='view',path='/testbed/missing.json')),
              context('tool','ERROR: file absent'),assistant('finish',dict(message='verified'))]
    row=candidate(messages,TOOLS,1,enc)
    pieces,_=render(dict(messages=messages,tools=TOOLS),enc)
    assert enc.decode(row['token_ids'])==''.join(t for t,_ in pieces)
    selected=[i for i,m in zip(row['token_ids'],row['assistant_mask']) if m]
    assert enc.decode(selected)==native_turn(messages[-1])
    assert row['messages']==messages and not row['training_eligible']
    with pytest.raises(ValueError):candidate(messages,TOOLS,2,enc)


def test_generalized_oracle_requires_requested_output_and_missing_path():
    c=next(r for r in tasks('oracle') if r['family']=='edit')
    snapshot={**c['files'],c['output_path']:json.dumps(c['expected_output'])}
    calls=[{'request':{'name':'str_replace_editor','arguments':{'command':'view','path':c['path']}}}]
    assert grade(c,'done',calls,snapshot)['success']
    snapshot['result.json']=snapshot.pop(c['output_path'])
    assert not grade(c,'done',calls,snapshot)['success']


def test_sampling_trace_records_untruncated_policy_probabilities(monkeypatch):
    import torch
    import ndm.e97 as model
    from scripts.eval_e97_native_execution import generate_turn
    enc=tiktoken.get_encoding('p50k_base')
    text=native_turn(assistant('finish',dict(message='done')));ids=enc.encode_ordinary(text)
    cache=SimpleNamespace(next_logits=torch.zeros(enc.n_vocab));stream=iter(ids)
    monkeypatch.setattr(model,'advance_e97_cache_segment',lambda *a:cache)
    def generate(*a,**kw):
        assert kw['temperature']==1. and kw['top_k']==0 and kw['top_p']==0.
        return [next(stream)],cache
    monkeypatch.setattr(model,'generate_e97_from_cache',generate)
    trace={};actual,tokens,reason=generate_turn(None,'prompt',enc,4096,time.monotonic()+60,sampling_trace=trace)
    assert actual==text and tokens==ids and reason=='valid'
    assert trace['prompt_token_ids']==enc.encode_ordinary('prompt')
    assert trace['selected_logprobs']==pytest.approx([-math.log(enc.n_vocab)]*len(ids))


@pytest.mark.parametrize('owned,paused,running',[(False,True,True),(True,False,True),(True,True,False)])
def test_resume_rejects_unowned_or_wrong_state(monkeypatch,owned,paused,running):
    import scripts.e97_native_execution_sandbox as module
    sandbox=module.NativeSandbox.__new__(module.NativeSandbox)
    sandbox.identity='container';sandbox.nonce='owner'
    spec={'Config':{'Labels':{'emender.native-qualification':'owner' if owned else 'other'}},
          'State':{'Paused':paused,'Running':running}}
    monkeypatch.setattr(module,'inspect_container',lambda _:spec)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('must reject before unpause'))
    with pytest.raises(ValueError):sandbox.resume_for_continuation()


def test_teacher_result_does_not_change_autonomous_grade(monkeypatch,tmp_path):
    import scripts.eval_e97_native_execution as evaluator
    from scripts.e97_native_execution_cases import cases
    c=cases('unit')[0]
    panel=dict(tools=TOOLS,system='Use tools',max_turns=1,generation_budget=4096,episode_generation_budget=8192,episode_seconds=60)
    text=native_turn(assistant('finish',dict(message='wrong')))
    monkeypatch.setattr(evaluator,'generate_turn',lambda *a:(text,[], 'valid'))
    class Sandbox:
        def __init__(self,p,out):out.mkdir()
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def request(self,*a,**kw):return {}
        def snapshot(self,names):return c['files']
    monkeypatch.setattr(evaluator,'NativeSandbox',Sandbox)
    def continuation(sandbox,messages,original):
        assert not original['grade']['success']
        return dict(verified=True,kind='teacher-repair')
    row=evaluator.episode(None,c,panel,tiktoken.get_encoding('p50k_base'),tmp_path/'episode',continuation=continuation)
    assert row['continuation']['verified']
    assert not row['autonomous_success'] and not row['grade']['success'] and row['final']=='wrong'
