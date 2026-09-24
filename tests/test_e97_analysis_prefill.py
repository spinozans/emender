from collections import deque
import json
from types import SimpleNamespace
import pytest
import tiktoken
import torch
from scripts import eval_e97_analysis_prefill as probe
from scripts.e97_lr_screen_panel import make_panel
from scripts.eval_e97_lr_screen import evaluate_task


@pytest.mark.parametrize('tail',['Action: read\nArguments: {"path":"a","offset":1,"limit":2}', 'Final: 42\n'])
def test_failfast_preserves_every_prefix_of_valid_turn(tail):
    text='Analysis: '+json.dumps('Observe λ and a newline\ncarefully.',ensure_ascii=False)+'\n'+tail
    for i in range(len(text)+1): assert not probe.impossible_prefix(text[:i])


@pytest.mark.parametrize('text',['Action: read','Analysis:\n','Analysis: nope','Analysis: ""\n','Analysis: "ok"\nHere is'])
def test_irrecoverably_invalid_prefix(text):
    assert probe.impossible_prefix(text)


class Inner:
    def __init__(self):
        self.encoding=tiktoken.get_encoding('p50k_base'); self.loaded=object(); self.histories=[]
    def encode(self,text): return self.encoding.encode_ordinary(text)
    def decode(self,ids): return self.encoding.decode(list(ids))
    def advance(self,ids,cache=None):
        tokens=(() if cache is None else cache.token_ids)+tuple(ids)
        logits=torch.zeros(self.encoding.n_vocab); logits[self.encode('Action')[0]]=1
        self.histories.append(self.decode(tokens))
        return SimpleNamespace(token_ids=tokens,has_complete_token_history=True,next_logits=logits)


def test_prefill_only_supplies_header_and_preserves_real_observation_history(monkeypatch):
    panel=make_panel(); task=next(t for t in panel['tasks'] if t['kind']=='two_reads')
    outputs=[]
    for path in task['required_reads']:
        outputs.append('Analysis: "Read actual fixture. λ"\nAction: read\nArguments: '+json.dumps({'path':path,'offset':1,'limit':10}))
    outputs.append('Analysis: "Use the observed values."\nFinal: '+task['answer']+'\n')
    inner=Inner(); prefix=inner.encode(probe.PREFILL)
    stream=deque(token for text in outputs for token in inner.encode(text)[len(prefix):])
    def generate(loaded,cache,**kwargs):
        assert kwargs['max_new_tokens']==1 and kwargs['temperature']==0
        token=stream.popleft(); return [token],inner.advance([token],cache)
    monkeypatch.setattr(probe,'generate_e97_from_cache',generate)
    engine=probe.PrefillEngine(inner)
    result=evaluate_task(engine,inner.encoding,panel,task)
    assert result['success'] and result['reads']==task['required_reads']
    assert len(engine.events)==3 and not stream
    assert all(e['prefill']=='Analysis:' and e['stop_reason']=='complete_turn' for e in engine.events)
    assert all(e['unforced_first_token_text']=='Action' for e in engine.events)
    assert any('Tool:\n' in h and h.endswith('Assistant:\nAnalysis:') for h in inner.histories)
    assert not any('Analysis:Analysis:' in h for h in inner.histories)


def test_bad_opening_stops_before_consuming_budget(monkeypatch):
    inner=Inner(); engine=probe.PrefillEngine(inner)
    def generate(loaded,cache,**kwargs):
        token=inner.encode('\n')[0]; return [token],inner.advance([token],cache)
    monkeypatch.setattr(probe,'generate_e97_from_cache',generate)
    ids,_=engine.generate(inner.advance(inner.encode('Assistant:\n')),max_new_tokens=4096,temperature=0,top_p=1)
    assert len(ids)==3
    assert engine.events[0]['stop_reason']=='provably_invalid_prefix'
