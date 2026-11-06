"""Fresh paired seed, frozen source and matched exposure; independent artifacts."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT=Path('/home/erikg/emender')
REL=Path('experiments/expressivity_tasks/x43_fork_resolved')
BOUND=Path('/home/erikg/emender_c14_matched_bound')
OUT=ROOT/REL/'receipts/c14_b_seed3402'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def replace(source,before,after,count=1):
    assert source.count(before)==count,(before,source.count(before))
    return source.replace(before,after)

def main():
    assert sys.executable==str(ROOT/'.venv/bin/python')
    arm,phase=sys.argv[1:3];sys.argv=sys.argv[:1]+sys.argv[3:]
    assert arm in ('budgeted','unbudgeted') and phase in ('preflight','train','eval')
    reg=json.loads((OUT/'registration.json').read_text())
    assert sha(Path(__file__))==reg['runner_sha256']
    for n,h in reg['code_sha256'].items():assert sha(BOUND/REL/n)==h,n
    sys.path.insert(0,str(BOUND))
    from experiments.expressivity_tasks.x43_fork_resolved import c14_b_train as train
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_v2 import configure
    configure();train.OUT=OUT/arm;train.CODE=tuple(reg['code_sha256'])
    train.CHECKPOINT=Path('/home/erikg/c14_dense_bytes')/f'emender_c14_b_seed3402_{arm}.pt'
    if arm=='unbudgeted':
        from experiments.expressivity_tasks.x43_fork_resolved.c14_b_control_model import StoreIntegrated
        train.StoreIntegrated=lambda:StoreIntegrated('unbudgeted')
        # Freeze lazy backward/fast-path dependencies before exposing installers.
        # Otherwise the current optimized replay could be imported accidentally.
        import importlib
        for name in reg['code_sha256']:
            if name.endswith('.py'):
                mod=importlib.import_module('experiments.expressivity_tasks.x43_fork_resolved.'+name[:-3])
                assert Path(mod.__file__).resolve().is_relative_to(BOUND),name
        import experiments.expressivity_tasks.x43_fork_resolved as package
        package.__path__=[str(ROOT/REL),*package.__path__]
        binding=json.loads((ROOT/REL/'receipts/scalability_phase2/registration.json').read_text())
        for n,h in binding['sources'].items():assert sha(ROOT/n)==h,n
        from experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_staged import install_rowkeys
        install_rowkeys()
    if phase=='preflight':
        train.preflight();print('Frozen preflight passed',flush=True);return
    if phase=='train':
        source=inspect.getsource(train.main)
        source=replace(source,'3401','3402',3)
        source=replace(source,'budget_active_seconds=2700','budget_active_seconds=None, matched_steps=35419, matched_targets=71677042')
        source=replace(source,'if stopped or active()>=2700:','if stopped or steps>=35419:')
        source=replace(source,"status='interrupted' if stopped else 'completed_active_budget'","status='interrupted' if stopped else 'completed_matched_exposure'")
        source=replace(source,'        save(status)\n    except',"        if status == 'completed_matched_exposure':\n            assert steps == 35419 and targets == 71677042\n        save(status)\n    except")
        (train.OUT/'effective_main.txt').write_text(source)
        exec(compile(source,str(Path(__file__)),'exec'),train.__dict__);train.main();return
    training=json.loads((train.OUT/'training.json').read_text())
    assert training['status']=='completed_matched_exposure'
    assert (training['steps'],training['actual_bytes_targets'])==(35419,71677042)
    if arm=='unbudgeted':
        for v in training['telemetry'].values():assert v['peak_occupancy']<257 and v['counts']['capacity_merges']==0
    if arm=='budgeted':
        from experiments.expressivity_tasks.x43_fork_resolved import c14_b_eval as ev
    else:
        from experiments.expressivity_tasks.x43_fork_resolved import c14_b_control_eval as ev
    ev.OUT=train.OUT;ev.StoreIntegrated=train.StoreIntegrated
    if arm=='unbudgeted':
        ev.synthetic_decay=lambda model:[]
        src=inspect.getsource(ev.fast.forward)
        src=replace(src,'BS=triton.next_power_of_2(cfg.slots)','BS=triton.next_power_of_2(min(cfg.slots,max(1,2*t)))')
        exec(compile(src,'<seed3402-fast>','exec'),ev.fast.__dict__)
    source=replace(inspect.getsource(ev.main),"assert training['status']=='completed_active_budget'","assert training['status']=='completed_matched_exposure'")
    (train.OUT/'effective_eval_main.txt').write_text(source)
    exec(compile(source,str(Path(__file__)),'exec'),ev.__dict__);ev.main()

if __name__=='__main__':main()
