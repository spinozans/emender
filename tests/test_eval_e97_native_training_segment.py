from copy import deepcopy
import math
import pytest
from scripts.eval_e97_native_training_segment import continuation_checks,gate
from ndm.data.masked_sft_dataset import sha256
import json
from types import SimpleNamespace

POLICY=dict(minimum_tool_retention_token_accuracy=.98,maximum_conversation_nll_increase_over_parent=.15,
            maximum_native_development_nll_increase_over_parent=0.)


def models():
    parent={cohort:dict(metrics=dict(assistant=dict(record_macro_nll=nll,token_accuracy=accuracy)))
            for cohort,nll,accuracy in [('native-development',1.55,.6),('conversation-retention',1.68,.55),('tool-retention',.002,1.)]}
    return {name:deepcopy(parent) for name in ('parent-y','current-x','current-y')}


def test_both_representations_must_pass_frozen_boundaries():
    values=models()
    values['current-y']['native-development']['metrics']['assistant']['record_macro_nll']=1.0
    values['current-x']['conversation-retention']['metrics']['assistant']['record_macro_nll']=1.68+.15
    values['current-y']['tool-retention']['metrics']['assistant']['token_accuracy']=.98
    checks=continuation_checks(values,POLICY)
    assert len(checks)==6 and all(c['passed'] for c in checks)
    values['current-x']['tool-retention']['metrics']['assistant']['token_accuracy']=.979
    checks=continuation_checks(values,POLICY)
    assert sum(not c['passed'] for c in checks)==1


@pytest.mark.parametrize('accepted',[True,False])
def test_gate_receipt_binds_training_panel_program_and_quality(tmp_path,accepted):
    run=tmp_path;phase=run/'segment';root=phase/'evaluation';root.mkdir(parents=True)
    def save(path,value):path.write_text(json.dumps(value))
    save(run/'program.json',{'evaluation_policy':POLICY})
    save(run/'base-panel.json',{'examples':[{'id':'fixture'}]})
    save(phase/'recipe.json',dict(run_root=str(run),program_sha256=sha256(run/'program.json'),base_panel_sha256=sha256(run/'base-panel.json')))
    checkpoint=dict(path='/fixture/checkpoint.pt',sha256='a'*64)
    save(phase/'summary.json',dict(status='passed',recipe_sha256=sha256(phase/'recipe.json'),checkpoint=checkpoint))
    model_rows=[dict(name=name,checkpoint=checkpoint['path'],sha256=checkpoint['sha256'],mode='saved' if name=='current-x' else 'train')
                for name in ('parent-y','pilot-1e-5-y','current-y','current-x')]
    save(root/'panel.json',dict(examples=[{'id':'fixture'}],models=model_rows,
         training_summary_sha256=sha256(phase/'summary.json'),program_sha256=sha256(run/'program.json')))
    measured=models();measured['pilot-1e-5-y']=deepcopy(measured['parent-y'])
    if not accepted:measured['current-x']['tool-retention']['metrics']['assistant']['token_accuracy']=.97
    save(root/'summary.json',dict(status='passed',panel_sha256=sha256(root/'panel.json'),models=measured))
    if accepted:gate(SimpleNamespace(phase=phase))
    else:
        with pytest.raises(SystemExit) as failure:gate(SimpleNamespace(phase=phase))
        assert failure.value.code==2
    receipt=json.loads((root/'gate.json').read_text())
    assert receipt['continue_training'] is accepted and receipt['checkpoint_promotion'] is False
    assert receipt['evaluation_summary_sha256']==sha256(root/'summary.json')
    assert receipt['training_summary_sha256']==sha256(phase/'summary.json')


@pytest.mark.parametrize('model,cohort,metric,value',[
    ('current-y','native-development','record_macro_nll',math.nan),
    ('parent-y','conversation-retention','record_macro_nll',math.inf),
    ('current-x','tool-retention','token_accuracy',1.1),
    ('current-y','native-development','record_macro_nll',-1.)])
def test_nonfinite_or_invalid_metrics_are_not_accepted(model,cohort,metric,value):
    values=models();values[model][cohort]['metrics']['assistant'][metric]=value
    with pytest.raises(ValueError):continuation_checks(values,POLICY)
