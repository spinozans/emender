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
    monkeypatch.setattr(parent,'wait_result',lambda *args:dict(score=2.,status='completed',score_eligible=True,train_seconds=2700,batch=64,context=2048,tokens_per_second=15800,regime_sha256=parent.sha(parent.ROOT/'velocity_regime.json')))
    monkeypatch.setattr(parent,'process_receipt',lambda c,p:dict(pid=-1,output=str(p),start_ticks='0'))
    monkeypatch.setattr(parent.time,'sleep',lambda _:None)
    parent.run(registration,output)
    saved=json.loads((output/'generation.json').read_text())
    assert saved['cma']['generation']==1
    assert len(saved['completed'])==12
    assert len(launched)==13
    parent.run(registration,output)
    assert len(launched)==13
    parent.run(registration,output,generations=3)
    saved=json.loads((output/'generation.json').read_text())
    assert saved['cma']['generation']==3
    assert saved['cma']['phase']=='told'
    assert len(launched)==37
    assert launched[1]!=launched[13]
    parent.run(registration,output,generations=3)
    assert len(launched)==37


def test_killed_candidate_is_not_a_valid_endpoint(tmp_path):
    import pytest
    class Killed:
        pid=-1
        def wait(self):return -9
    with pytest.raises(RuntimeError,match='exit=-9'):parent.wait_result(Killed(),tmp_path)


def test_slow_smoke_requires_bound_committed_decomposition(tmp_path,monkeypatch):
    import pytest
    from experiments.expressivity_tasks.x42_pile_scale import velocity_train
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(parent,'ROOT',tmp_path)
    regime=tmp_path/'velocity_regime.json';regime.write_text('{}')
    out=tmp_path/'run';out.mkdir()
    smoke=out/'smoke.json'
    smoke.write_text(json.dumps(dict(batch=64,context=2048,regime_sha256=parent.sha(regime),tokens_per_second=8000)))
    with pytest.raises(RuntimeError,match='decompose before generation'):parent.validate_smoke(out)
    path=tmp_path/'receipts/velocity_production_3b/throughput_decision.json';path.parent.mkdir(parents=True)
    decision=dict(status='decomposed_proceed',smoke_sha256=parent.sha(smoke),regime_sha256=parent.sha(regime),analysis=['Measured backward cost'],residual_risks=['Below target'])
    path.write_text(json.dumps(decision))
    monkeypatch.setattr(velocity_train,'subprocess_bytes',lambda c,p:Path(p).read_bytes())
    parent.validate_smoke(out)
    decision['smoke_sha256']='wrong';path.write_text(json.dumps(decision))
    with pytest.raises(RuntimeError,match='does not bind'):parent.validate_smoke(out)
    monkeypatch.setattr(velocity_train,'subprocess_bytes',lambda c,p:b'{}')
    with pytest.raises(RuntimeError,match='must be committed'):parent.validate_smoke(out)


def test_smoke_reuse_allows_only_orchestration_changes(tmp_path,monkeypatch):
    import hashlib
    import pytest
    from experiments.expressivity_tasks.x42_pile_scale import velocity_train
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(parent,'ROOT',tmp_path)
    old=dict(batch=64,context=2048,train_seconds=2700,flags={'test':'1'},allocator='expandable_segments:True',registration_sha256='registration',sources={'velocity_parent.py':'old','velocity_train.py':'train'})
    raw=json.dumps(old).encode();new=json.loads(raw);new['compute_compatible_previous_commit']='old_commit';new['sources']['velocity_parent.py']='new'
    regime=tmp_path/'velocity_regime.json';regime.write_text(json.dumps(new))
    out=tmp_path/'run';out.mkdir();(out/'smoke.json').write_text(json.dumps(dict(batch=64,context=2048,regime_sha256=hashlib.sha256(raw).hexdigest(),tokens_per_second=15000)))
    monkeypatch.setattr(velocity_train,'subprocess_bytes',lambda c,p:raw)
    parent.validate_smoke(out)
    assert (out/'smoke_compatibility.json').exists()
    new['sources']['velocity_train.py']='changed';regime.write_text(json.dumps(new))
    with pytest.raises(RuntimeError,match='compute source changed'):parent.validate_smoke(out)
