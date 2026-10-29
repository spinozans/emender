import json
from pathlib import Path
from experiments.expressivity_tasks.x42_pile_scale import velocity_parent as parent


def test_parent_owns_one_population_and_resumes_completed_generation(tmp_path,monkeypatch):
    registration=tmp_path/'registration.json';registration.write_text('{}')
    output=tmp_path/'run';launched=[]
    class Child:
        pid=-1
        def poll(self):return 0
    def launch(policy,path,registration):
        path.mkdir(parents=True,exist_ok=True);launched.append(policy)
        return Child(),path
    monkeypatch.setattr(parent,'verify_registration',lambda _: {})
    monkeypatch.setattr(parent,'launch',launch)
    monkeypatch.setattr(parent,'wait_result',lambda *args:dict(score=2.,status='completed',score_eligible=True,train_seconds=2700))
    monkeypatch.setattr(parent,'process_receipt',lambda c,p:dict(pid=-1,output=str(p),start_ticks='0'))
    monkeypatch.setattr(parent.time,'sleep',lambda _:None)
    parent.run(registration,output)
    saved=json.loads((output/'generation.json').read_text())
    assert saved['cma']['generation']==1
    assert len(saved['completed'])==12
    assert len(launched)==13
    parent.run(registration,output)
    assert len(launched)==13


def test_killed_candidate_is_not_a_valid_endpoint(tmp_path):
    import pytest
    class Killed:
        pid=-1
        def wait(self):return -9
    with pytest.raises(RuntimeError,match='exit=-9'):parent.wait_result(Killed(),tmp_path)
