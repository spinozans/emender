#!/usr/bin/env python3
"""Reference-bound head arithmetic experiment on the pinned FP32 recurrent policy."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.eval_e97_native_execution import publish,sha
from ndm.recurrent_precision import FIXED_RECURRENT_KERNEL

ROOT=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
REFERENCE=ROOT/'fixed-recurrent-kernel-v1'
REFERENCE_SHA='6745d29adda5c7b537e9daa615822aa8abdf87cc42938075e6c20c8ffd8698fc'
REFERENCE_SUMMARY_SHA='0ef8d58e4cf00f3b28e2c17451ba0f91544d5cb2eaadc4f016f398ca16da8220'
BASE_RECIPE_SHA='82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9'
PROBE_RECIPE_SHA='eb99ad750df1722d7cd58c36ab8844d4c9e9655d76ae9774eed88c069186cf96'
SCOPE_SUFFIX='; read-only counterfactual FP32 head projection, not a changed-policy qualification; CPU FP64 selected dots and counterfactual BF16 re-quantization'


def reference():
    for i in (1,2):
        if sha(REFERENCE/f'worker-{i}/measurements-private.json')!=REFERENCE_SHA or sha(REFERENCE/f'worker-{i}/summary.json')!=REFERENCE_SUMMARY_SHA:
            raise ValueError('pinned reference identity')
    p=REFERENCE/'authority/recipe-private.json'
    if sha(p)!=BASE_RECIPE_SHA:raise ValueError('pinned input recipe identity')
    return json.loads(p.read_text()),json.loads((REFERENCE/'worker-1/measurements-private.json').read_text())['records']


def freeze(args):
    from scripts.qualify_e97_native_rl_logprobs import freeze as freeze_assay
    base,_=reference()
    freeze_assay(SimpleNamespace(output=args.output/'assay',head_probe=True,head_numeric_audit=True,recurrent_state_precision='fp32'))
    p=args.output/'assay/recipe-private.json';recipe=json.loads(p.read_text())
    stripped=dict(recipe);probe=stripped.pop('head_probe')
    if probe!=dict(vocab_chunk=4096,max_rows=128,returns_original_outputs=True,numeric_audit=True):raise ValueError('numeric probe recipe')
    if stripped['scope']!=base['scope']+SCOPE_SUFFIX:raise ValueError('probe scope')
    stripped['scope']=base['scope']
    if stripped!=base:raise ValueError('non-observer recipe changed')
    if sha(p)!=PROBE_RECIPE_SHA:raise ValueError('frozen probe recipe identity')
    publish(args.output/'plan.json',dict(schema='e97-pinned-head-numeric-audit-v1',kernel=FIXED_RECURRENT_KERNEL,
        recipe_sha256=sha(p),reference_sha256=REFERENCE_SHA,reference_summary_sha256=REFERENCE_SUMMARY_SHA,
        observer_binding_abs_max=1e-4,pair_abs_max=.05,pair_abs_p99=.02,max_hbm_allocated=12*1024**3,
        workers=1,worker_seconds=3600,outer_seconds=3900,teardown_seconds=30,
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,
        scope='Read-only FP32 head and CPU FP64 selected-dot analysis; no changed-policy qualification'))
    print('PINNED_HEAD_AUDIT_FROZEN',sha(p),flush=True)


def compare(rows,old):
    if len(rows)!=len(old) or not rows:raise ValueError('head audit turn coverage')
    binding=[];details=[]
    for row,ref in zip(rows,old):
        if (row['id'],row['turn'],row['recorded'],row['padded_tokens'])!=(ref['id'],ref['turn'],ref['recorded'],ref['padded_tokens']):
            raise ValueError('head audit reference identity')
        n=len(ref['recorded'])
        head=row['head_probe']
        if not n or any(len(row[k])!=n or len(ref[k])!=n for k in ('actor_replay','teacher')) or any(len(head[k])!=n for k in ('actor','teacher')):
            raise ValueError('head audit token coverage')
        for key in ('actor_replay','teacher'):
            binding.extend(abs(a-b) for a,b in zip(row[key],ref[key]))
        binding.append(abs(row['ce_mean']-ref['ce_mean']))
        if not all(math.isfinite(x) for x in binding):raise ValueError('nonfinite reference binding')
        for i,(a,t) in enumerate(zip(head['actor'],head['teacher'])):
            if not all(math.isfinite(v) for d in (a,t) for v in d.values()):raise ValueError('nonfinite head diagnostic')
            binding.extend((abs(a['native_logprob']-row['actor_replay'][i]),abs(t['native_logprob']-row['teacher'][i])))
            if any(d['fp32_vs_fp64_selected_abs']!=abs(d['fp32_selected_logit']-d['fp64_selected_logit']) for d in (a,t)):
                raise ValueError('selected-dot error identity')
            details.append(dict(id=row['id'],turn=row['turn'],position=i,
                actual_gap=abs(row['actor_replay'][i]-row['teacher'][i]),
                fp32_gap=abs(a['fp32_logprob']-t['fp32_logprob']),
                requantized_gap=abs(a['rounded_fp32_logprob']-t['rounded_fp32_logprob']),
                bf16_selected_delta=abs(a['bf16_selected_logit']-t['bf16_selected_logit']),
                fp64_selected_delta=abs(a['fp64_selected_logit']-t['fp64_selected_logit']),
                fp32_vs_fp64_error=max(a['fp32_vs_fp64_selected_abs'],t['fp32_vs_fp64_selected_abs']),
                native_vs_rounded_logits_max=max(a['native_vs_rounded_fp32_logit_abs_max'],t['native_vs_rounded_fp32_logit_abs_max']),
                hidden_relative_l2=t['hidden_relative_l2'],hidden_absolute_max=t['hidden_max_absolute']))
    return max(binding),details


def audit(args):
    _,old=reference();root=args.output;plan=json.loads((root/'plan.json').read_text())
    p=root/'assay/measurements-private.json';measurements=json.loads(p.read_text());summary=json.loads((root/'assay/summary.json').read_text())
    if plan['schema']!='e97-pinned-head-numeric-audit-v1' or plan['kernel']!=FIXED_RECURRENT_KERNEL or plan['reference_sha256']!=REFERENCE_SHA:
        raise ValueError('head experiment identity')
    if not (sha(root/'assay/recipe-private.json')==measurements['recipe_sha256']==summary['recipe_sha256']==plan['recipe_sha256']==PROBE_RECIPE_SHA):raise ValueError('head recipe identity')
    binding,details=compare(measurements['records'],old)
    checks=dict(reference_bound=binding<=plan['observer_binding_abs_max'],coverage=len(old)==57 and len(details)==2711,
        fixed_recurrence=summary['recurrent_kernel']==FIXED_RECURRENT_KERNEL and summary['legacy_autotune_cache_entries']==0,
        parameters_unchanged=summary['checks']['parameters_unchanged'] and summary['parameter_sha256']=='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd',
        finite=summary['checks']['finite'],no_updates=summary['optimizer_updates']==0,
        bounded_memory=summary['peak_hbm_allocated']<=plan['max_hbm_allocated'])
    stats={k:dict(max=max(d[k] for d in details),p99=float(np.quantile([d[k] for d in details],.99))) for k in ('actual_gap','fp32_gap','requantized_gap')}
    ready=all(checks.values())
    result=dict(schema='e97-pinned-head-numeric-result-v1',checks=checks,diagnostic_ready=ready,
        observer_binding_max=binding,statistics=stats,
        fp32_shadow_within_existing_pair_limits=bool(ready and stats['fp32_gap']['max']<=.05 and stats['fp32_gap']['p99']<=.02),
        fp32_vs_fp64_selected_error_max=max(d['fp32_vs_fp64_error'] for d in details),
        hidden_relative_l2_max=max(d['hidden_relative_l2'] for d in details),
        current_outliers=[d for d in details if d['actual_gap']>.05],
        shadow_outliers=[d for d in details if d['fp32_gap']>.05],
        measurements_sha256=sha(p),summary_sha256=sha(root/'assay/summary.json'),
        actual_probability_path_passed=summary['probability_path_passed'],peak_hbm_allocated=summary['peak_hbm_allocated'],
        optimizer_updates=0,training_eligible=False,changed_policy_qualified=False,
        scope='Counterfactual diagnostics only; actual logits, behavior probabilities and failed gates are not replaced')
    publish(root/'details-private.json',dict(details=details));publish(root/'audit.json',result)
    print(json.dumps(result,sort_keys=True),flush=True)
    if not ready:raise SystemExit('head diagnostic reference/coverage/safety gate failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','audit'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
