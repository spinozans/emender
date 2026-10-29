"""Fresh bare 22M pilot gate using certified launch fusion and sparse pooling.

No change to reverse arithmetic, no value-consuming event is discarded.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .part8_surface import ROOT,OUT


def main():
    from .part8_repair import main as repair_preflight
    repair_preflight()
    env=os.environ.copy()
    env.update({k:'1' for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_STABLE_MAPPING','X40_FUSED_INIT')})
    env.update(PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    lease=str(ROOT/'run_leased.sh');module='experiments.expressivity_tasks.x40_emender_lm.'
    label='part8_pilot_B128_bare_fused_sparse'
    commands=[('fusion_parity',[lease,'-m',module+'part8_pilot','parity']),
              ('throughput',[lease,'-m',module+'part8_pilot','throughput','--batch','128','--no-checkpoint','--fused-pool','--label',label]),
              ('curve',[lease,'-m',module+'part8_pilot','curve','--receipt',label+'.json'])]
    outcomes=[]
    for name,cmd in commands:
        with (OUT/f'gate_{name}.log').open('w') as log:
            r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        outcomes.append(dict(name=name,command=cmd,exit_code=r.returncode))
        if r.returncode:break
    rate_path=OUT/f'{label}.json'
    curve_path=OUT/f'matched_{label}'/'summary.json'
    rate=json.loads(rate_path.read_text()) if rate_path.exists() else {}
    curve=json.loads(curve_path.read_text()) if curve_path.exists() else {}
    qualified=all(o['exit_code']==0 for o in outcomes) and len(outcomes)==3 and rate.get('status')=='passed' and curve.get('status')=='passed'
    result=dict(status='passed' if qualified and rate['tokens_per_second']>=100000 else 'failed',
        qualified_measurement=qualified,threshold=100000,tokens_per_second=rate.get('tokens_per_second'),
        envelope='bare',commands=outcomes,curve=curve,
        throughput_receipt=str(rate_path.relative_to(ROOT)),
        levers='certified integer pool launch fusion; exact sparse entry/allocation union backward; detached-key adjoint elision; existing no-event suppression',
        limitations='No new arithmetic sparsification was accepted. All pilot scans consume values; suppressing those events without a proof would violate parity. This run measures the certified levers and does not claim closure of full launch fusion or the remaining sparsity search.',
        backward_sha256=hashlib.sha256((ROOT/'backward.py').read_bytes()).hexdigest())
    (OUT/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
