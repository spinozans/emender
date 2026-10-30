"""Resume a partially scored second generation without re-scoring survivors."""
import json
import pytest
from experiments.expressivity_tasks.x42_pile_scale import velocity_parent as parent


def test_mid_generation_restart_keeps_population_and_completed(tmp_path,monkeypatch):
    registration=tmp_path/'registration.json';registration.write_text('{}')
    out=tmp_path/'run';launched=[];calls=0
    class Child:
        pid=-1
        def poll(self):return 0
    def launch(policy,path,registration):
        path.mkdir(parents=True,exist_ok=True);launched.append((path.name,policy))
        return Child(),path
    def result(child,path):
        nonlocal calls
        calls+=1
        if calls==16:raise RuntimeError('simulated child failure')
        return dict(score=2.,status='completed',score_eligible=True,train_seconds=2700)
    monkeypatch.setattr(parent,'verify_registration',lambda _: {})
    monkeypatch.setattr(parent,'validate_smoke',lambda _:None)
    monkeypatch.setattr(parent,'launch',launch)
    monkeypatch.setattr(parent,'wait_result',result)
    monkeypatch.setattr(parent,'process_receipt',lambda c,p:dict(pid=-1,output=str(p),start_ticks='0'))
    monkeypatch.setattr(parent.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='child failure'):
        parent.run(registration,out,generations=2)
    state=json.loads((out/'generation.json').read_text())
    assert state['cma']['generation']==1
    assert state['cma']['phase']=='asked'
    assert set(state['completed'])=={'0','1'}
    population=state['cma']['x']
    before=len(launched)
    parent.run(registration,out,generations=2)
    resumed=launched[before:]
    assert len(resumed)==10
    assert {name for name,_ in resumed}=={f'g2_c{i:02}' for i in range(2,12)}
    assert json.loads((out/'generation.json').read_text())['cma']['generation']==2
    assert json.loads((out/'generation.json').read_text())['cma']['x']==population
