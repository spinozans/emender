#!/usr/bin/env python3
"""Separate historical behavior binding from repeatability of a pinned policy."""
import argparse
import json
import math
import struct
from pathlib import Path
import numpy as np
from scripts.eval_e97_native_execution import publish,sha
from ndm.recurrent_precision import FIXED_RECURRENT_KERNEL


def row_metrics(rows,selected):
    if len(rows)!=len(selected) or not rows:raise ValueError('row coverage')
    paired=[];historical=[];ce=[]
    for row,item in zip(rows,selected):
        if (row['id'],row['turn'],row['recorded'])!=(item['id'],item['turn'],item['recorded_logprobs']):
            raise ValueError('historical reference identity')
        n=len(item['generated'])
        if not n or any(len(row[k])!=n for k in ('actor_replay','teacher','recorded')):raise ValueError('token coverage')
        if not all(math.isfinite(v) for k in ('actor_replay','teacher','recorded') for v in row[k]) or not math.isfinite(row['ce_mean']):
            raise ValueError('nonfinite measurement')
        paired.extend(abs(a-b) for a,b in zip(row['actor_replay'],row['teacher']))
        historical.extend(abs(a-b) for a,b in zip(row['actor_replay'],row['recorded']))
        ce.append(abs(row['ce_mean']+sum(row['teacher'])/n))
    return dict(tokens=len(paired),pair_max=max(paired),pair_p99=float(np.quantile(paired,.99)),
                historical_replay_max=max(historical),ce_max=max(ce))


def repeated_exactly(a,b):
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if (x['id'],x['turn'],x['ce_mean'])!=(y['id'],y['turn'],y['ce_mean']):return False
        for key in ('actor_replay','teacher'):
            if len(x[key])!=len(y[key]):return False
            if any(struct.pack('!f',u)!=struct.pack('!f',v) for u,v in zip(x[key],y[key])):return False
    return True


def audit(root,*,expected_recipe_sha='82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9',expected_kernel_cases=10,numerical_policy=None):
    import xml.etree.ElementTree as ET
    recipe_path=root/'authority/recipe-private.json';recipe_sha=sha(recipe_path)
    if recipe_sha!=expected_recipe_sha:raise ValueError('frozen input recipe')
    recipe=json.loads(recipe_path.read_text());plan=json.loads((root/'plan.json').read_text())
    if plan['kernel']!=FIXED_RECURRENT_KERNEL or plan['workers']!=2:raise ValueError('fixed policy plan')
    if numerical_policy is not None and plan.get('numerical_policy')!=numerical_policy:raise ValueError('composite policy plan')
    tests=ET.parse(root/'kernel-tests.xml').findall('.//testcase')
    kernel_pass=len(tests)==expected_kernel_cases and all(not any(x.find(t) is not None for t in ('skipped','failure','error')) for x in tests)
    rows=[];summaries=[];metrics=[];checks={'kernel_cases':kernel_pass}
    for i in (1,2):
        r=root/f'worker-{i}';m=json.loads((r/'measurements-private.json').read_text());s=json.loads((r/'summary.json').read_text())
        if m['recipe_sha256']!=recipe_sha or s['recipe_sha256']!=recipe_sha:raise ValueError('worker recipe identity')
        rows.append(m['records']);summaries.append(s);metrics.append(row_metrics(m['records'],recipe['selected']))
        checks[f'worker_{i}']=bool(s['recurrent_kernel']==FIXED_RECURRENT_KERNEL and s['legacy_autotune_cache_entries']==0
            and s['recurrent_state_precision']=='fp32' and s['checks']['parameters_unchanged'] and s['checks']['finite']
            and s['optimizer_updates']==0 and len(m['records'])==57 and metrics[-1]['tokens']==2711
            and metrics[-1]['pair_max']<=recipe['teacher_abs_max'] and metrics[-1]['pair_p99']<=recipe['teacher_abs_p99']
            and metrics[-1]['ce_max']<=recipe['ce_mean_delta_max'])
        if numerical_policy is not None:
            checks[f'worker_{i}_linear_policy']=bool(s.get('numerical_policy')==numerical_policy and s.get('linear_modules_fp32')==163
                and s.get('head_dtypes')==dict(actor=['torch.float32'],teacher=['torch.float32'])
                and s['peak_hbm_allocated']<=plan['max_hbm_allocated'])
    checks['fresh_process_repeat_exact']=repeated_exactly(*rows)
    checks['same_parameters']=summaries[0]['parameter_sha256']==summaries[1]['parameter_sha256']=='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd'
    result=dict(schema='e97-fixed-recurrent-kernel-audit-v1',checks=checks,current_policy_numerics_passed=all(checks.values()),
        historical_probability_path_passed=[s['probability_path_passed'] for s in summaries],metrics=metrics,
        kernel=FIXED_RECURRENT_KERNEL,optimizer_updates=0,rl_optimizer_ready=False,training_eligible=False,
        scope='Pinned-policy forced-token numerical assay, not historical behavior-probability replacement or training/restart qualification',
        hashes={str(p.relative_to(root)):sha(p) for p in [root/'kernel-tests.xml']+[root/f'worker-{i}'/n for i in (1,2) for n in ('summary.json','measurements-private.json')]})
    if numerical_policy is not None:
        result.update(schema='e97-composite-numerical-policy-audit-v1',numerical_policy=numerical_policy)
    publish(root/'audit.json',result);print(json.dumps(result,sort_keys=True),flush=True)
    if not result['current_policy_numerics_passed']:raise SystemExit('pinned-policy numerical gate failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
