import math
import pytest
import torch
from experiments.expressivity_tasks.x42_pile_scale.velocity_length_sweep import sweep, LENGTHS

class Fake:
    def eval(self): pass
    def loss(self,x,y):
        assert not torch.is_grad_enabled()
        return math.log(2)+len(x)/4096

def test_windows_and_ratios():
    calls=[]
    def window(stream,start,length):
        calls.append((start,length))
        return range(length),None
    r=sweep(Fake(),range(8193),window_fn=window)
    assert calls==[(i*t,t) for t in LENGTHS for i in range(2)]
    assert [x['context'] for x in r['rows']]==list(LENGTHS)
    assert r['rows'][0]['ppl_ratio_vs_T512']==1
    assert r['rows'][-1]['ppl_ratio_vs_T512']==pytest.approx(math.exp(1-.125))
    assert not r['score_eligible']

def test_fail_closed():
    with pytest.raises(ValueError):sweep(Fake(),range(8192))
    class Bad(Fake):
        def loss(self,x,y):return float('nan')
    with pytest.raises(ValueError):sweep(Bad(),range(8193),window_fn=lambda *a:(None,None))

def test_pure_occupancy(monkeypatch):
    from experiments.expressivity_tasks.x42_pile_scale import velocity_length_sweep as mod
    row=dict(occ_frac=.5,gate_mean=.7,event_density=.8,ever_touched_fraction=.9,peak_live_per_head=[16])
    monkeypatch.setattr(mod,'telemetry',lambda model:[row])
    r=sweep(Fake(),range(8193),pure=True,window_fn=lambda s,i,t:(range(t),None))
    assert r['rows'][0]['telemetry']==[[row],[row]]

def test_table_does_not_substitute_historical_c06(tmp_path):
    import json
    from experiments.expressivity_tasks.x42_pile_scale.velocity_anchor_report import report
    (tmp_path/'c06_result.json').write_text('{}')
    report(tmp_path)
    r=json.loads((tmp_path/'length_table.json').read_text())
    assert r['status']=='incomplete'
    assert all(x['measurements']==[] for x in r['rows'])
