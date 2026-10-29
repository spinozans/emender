"""Retry missing-preflight-receipt failures after the sequential surface ends.

Keep the failed preflight separately. Never retry native/numerical failures
or infer that a failed preflight is a capacity failure.
"""
import json
import os
import shutil
import subprocess
from .part8_surface import ROOT,OUT,envelope_specs


def missing_gate_preflight(row):
    detail=row.get('error','')+'\n'+row.get('traceback','')
    return row['status']=='failed' and 'FileNotFoundError' in detail and 'checkpoint_gate_' in detail


def main():
    env=os.environ.copy()
    env.update({k:'1' for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_STABLE_MAPPING','X40_FUSED_INIT')})
    env.update(PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for path in sorted((OUT/'surface').glob('*.json')):
        row=json.loads(path.read_text())
        if not missing_gate_preflight(row):continue
        archive=OUT/'preflight_failures'/path.stem;archive.mkdir(parents=True,exist_ok=True)
        attempts=OUT/'attempts'/path.stem
        start=row['envelope_index']
        for old in (path,attempts/f'{start:02d}.json',attempts/f'{start:02d}.log'):
            shutil.move(str(old),str(archive/old.name))
        for index,spec in enumerate(envelope_specs(row['batch'],row['arm'])):
            if index<start:continue
            target=attempts/f'{index:02d}.json'
            with target.with_suffix('.log').open('w') as log:
                done=subprocess.run([str(ROOT/'run_leased.sh'),'-m','experiments.expressivity_tasks.x40_emender_lm.part8_surface',
                    '--scale',row['scale'],'--arm',row['arm'],'--batch',str(row['batch']),'--context',str(row['context']),'--envelope-index',str(index)],env=env,stdout=log,stderr=subprocess.STDOUT)
            if not target.exists():raise RuntimeError(f'retry failed without receipt, exit={done.returncode}')
            result=json.loads(target.read_text())
            if result['status']!='oom':break
        result['attempt_receipts']=[str(p.relative_to(ROOT)) for p in sorted(attempts.glob('*.json'))]
        result['selection_policy']='first fit in declared ordering; escalate only after CUDA OOM'
        result['preflight_failure_archive']=str(archive.relative_to(ROOT))
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(path.stem,'preflight retry',result['status'],flush=True)

if __name__=='__main__':main()
