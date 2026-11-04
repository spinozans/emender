"""Fresh registered controls, enabled only by the complete B1-v2 PB1 result."""
import json,sys
from pathlib import Path
from . import c14_b_train as training
from .c14_b_v2 import configure
from .c14_b_control_model import StoreIntegrated
from .c14_documents import sha

EXTRA=('c14_b_control_replay.py','c14_b_control_graph.py','c14_b_control_native.py',
    'c14_b_control_model.py','c14_b_control_fast.py','c14_b_control_eval.py','c14_b_controls.py',
    'c14_b_ablation_native.py','c14_b_ablation_store.py','c14_b_ablation_model.py')

def main():
    control,phase=sys.argv[1:3];sys.argv=sys.argv[:1]+sys.argv[3:]
    assert control in ('append_only','unbudgeted') and phase in ('gates','train','eval')
    configure();base=training.OUT
    verdict=json.loads((base/'verdicts.json').read_text())
    assert verdict['status']=='complete' and verdict['conditional_ablations_required'] and verdict['PB1']['delta']!=0
    names=(*training.CODE,*EXTRA)
    if phase=='gates':
        log=base/'control_battery.txt';text=log.read_text()
        assert 'passed' in text and 'failed' not in text.lower() and 'skipped' not in text.lower()
        tests=sorted(p.name for p in training.ROOT.glob('test_c14*.py') if p.name!='test_c14_stacking.py')
        assert all(n in text for n in tests)
        gate=dict(status='passed',code_sha256={n:sha(training.ROOT/n) for n in names},
            tests_sha256={n:sha(training.ROOT/n) for n in tests},test_log_sha256=sha(log),
            B1_verdict_sha256=sha(base/'verdicts.json'),B1_checkpoint_sha256=verdict['training']['checkpoint_sha256'],
            controls=['append_only','unbudgeted'],unbudgeted_finite_cap=257,
            requirement='Peak occupancy strictly below 257 and capacity merges zero; full runtime telemetry verifies this after training and eval.',
            registration='unchanged active budget, seed, optimizer, dropout, document boundaries and frozen windows; existing control kernels unchanged')
        training.write('control_gates.json',gate);print(json.dumps(gate,indent=2));return
    out=training.ROOT/('receipts/c14_b_v2_'+control);out.mkdir(exist_ok=True)
    passed=json.loads((base/'control_gates.json').read_text());assert passed['status']=='passed'
    for n,h in passed['code_sha256'].items():assert sha(training.ROOT/n)==h,n
    training.OUT=out;training.CODE=names
    training.CHECKPOINT=Path('/home/erikg/c14_dense_bytes')/('emender_c14_b1_v2_'+control+'_resume.pt')
    training.StoreIntegrated=lambda:StoreIntegrated(control)
    gate=json.loads((base/'launch_gates.json').read_text())
    gate.update(code_sha256=passed['code_sha256'],tests_sha256=passed['tests_sha256'],test_log_sha256=passed['test_log_sha256'],
        conditional_ablation=control,condition_B1_verdict_sha256=passed['B1_verdict_sha256'],control_gates_sha256=sha(base/'control_gates.json'))
    target=out/'launch_gates.json'
    if target.exists():assert json.loads(target.read_text())==gate
    else:training.write('launch_gates.json',gate)
    if phase=='train':training.main()
    else:
        from . import c14_b_control_eval as evaluation
        evaluation.OUT=out;evaluation.StoreIntegrated=lambda:StoreIntegrated(control)
        evaluation.synthetic_decay=lambda model:[]
        evaluation.main()

if __name__=='__main__':main()
