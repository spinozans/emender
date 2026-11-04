"""Conditional matched controls. Never run before a nonzero full PB1 verdict."""
import json
import os
import sys
from pathlib import Path
from . import c14_b_train as training
from . import c14_b_eval as evaluation
from .c14_b_ablation_model import StoreIntegrated
from .c14_documents import sha


def main():
    control,phase=sys.argv[1:3];sys.argv=sys.argv[:1]+sys.argv[3:]
    assert control in ('append_only','unbudgeted') and phase in ('train','eval')
    base=training.OUT
    verdict=json.loads((base/'verdicts.json').read_text())
    assert verdict['status']=='complete' and verdict['conditional_ablations_required']
    out=training.ROOT/('receipts/c14_b_'+control);out.mkdir(exist_ok=True)
    extra=('c14_b_ablation_store.py','c14_b_ablation_native.py','c14_b_ablation_model.py','c14_b_ablation.py','c14_b_ablation_fast.py')
    if phase=='train':
        passed=json.loads((base/'ablation_gates.json').read_text())
        assert passed['status']=='passed'
        for n,h in passed['code_sha256'].items():assert sha(training.ROOT/n)==h
        original_preflight=training.preflight
        def checked():
            training.OUT=base
            try:return original_preflight()
            finally:training.OUT=out
        training.preflight=checked;training.OUT=out
        training.StoreIntegrated=lambda:StoreIntegrated(control)
        training.CHECKPOINT=Path('/home/erikg/c14_dense_bytes')/('emender_c14_b_'+control+'_resume.pt')
        training.CODE=training.CODE+extra
        gate=json.loads((base/'launch_gates.json').read_text());gate['conditional_ablation']=control
        (out/'launch_gates.json').write_text(json.dumps(gate,indent=2)+'\n')
        training.main()
    else:
        evaluation.OUT=out;evaluation.StoreIntegrated=lambda:StoreIntegrated(control)
        # These are PB1 operation controls; B1 alone owns the PB4 decay test.
        evaluation.synthetic_decay=lambda model:[]
        # The control uses its own native scan; retain full native telemetry.
        from . import c14_b_ablation_fast
        evaluation.fast=c14_b_ablation_fast
        evaluation.main()


if __name__=='__main__':main()
