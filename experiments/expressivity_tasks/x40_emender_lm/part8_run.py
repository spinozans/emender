"""Attended, sequential lease runner; keep each failed envelope receipt."""
import json
from pathlib import Path
import subprocess
import os
from .part8_surface import ROOT, OUT, ARMS, SHAPES, envelope_specs


def main():
    env=os.environ.copy()
    env.update({k:'1' for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_STABLE_MAPPING','X40_FUSED_INIT')})
    env.update(PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    lease=str(ROOT/'run_leased.sh')
    OUT.mkdir(parents=True,exist_ok=True)
    # Fresh pressure-inclusive parity precedes every pure timing.
    for context in (512,2048):
        dest=OUT/f'production_T{context}.json'
        with (OUT/f'production_T{context}.log').open('w') as log:
            r=subprocess.run([lease,'-m','experiments.expressivity_tasks.x40_emender_lm.part5_gate','--batch','1','--context',str(context),'--receipt',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'production parity failed T{context}')
    surface=OUT/'surface';surface.mkdir(exist_ok=True)
    for scale in SHAPES:
        for context in (512,2048):
            for batch in (64,128):
                for arm in ARMS:
                    name=f'{scale}_{arm}_B{batch}_T{context}'
                    dest=surface/f'{name}.json'
                    if dest.exists():
                        print(name,'retained completed receipt',flush=True)
                        continue
                    for index,spec in enumerate(envelope_specs(batch,arm)):
                        path=OUT/'attempts'/name/f'{index:02d}.json';path.parent.mkdir(parents=True,exist_ok=True)
                        if not path.exists():
                            with path.with_suffix('.log').open('w') as log:
                                r=subprocess.run([lease,'-m','experiments.expressivity_tasks.x40_emender_lm.part8_surface','--scale',scale,'--arm',arm,'--context',str(context),'--batch',str(batch),'--envelope-index',str(index)],env=env,stdout=log,stderr=subprocess.STDOUT)
                        if not path.exists():
                            result=dict(status='failed',error=f'runner exit {r.returncode} without receipt',scale=scale,arm=arm,batch=batch,context=context,envelope=spec)
                        else:
                            result=json.loads(path.read_text())
                            assert result['envelope']==spec
                        print(name,spec['name'],result['status'],flush=True)
                        if result['status']!='oom':break
                    result['attempt_receipts']=[str(p.relative_to(ROOT)) for p in sorted(path.parent.glob('*.json'))]
                    result['selection_policy']='first fit in declared ordering; escalate only after CUDA OOM'
                    dest.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
