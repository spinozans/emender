#!/usr/bin/env python3
"""Frozen no-update qualification of the composite FP32-linear candidate."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from ndm.numerical_policy import POLICY
from ndm.recurrent_precision import FIXED_RECURRENT_KERNEL
from scripts.eval_e97_native_execution import publish,sha

BASE=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/fixed-recurrent-kernel-v1/authority/recipe-private.json')
BASE_SHA='82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9'
RECIPE_SHA='916b1e87207a161d17ecf61881e5be3952a4fcd23ac3e6e6a6fe81d282045d15'
SCOPE='; candidate FP32 linear arithmetic, BF16 hidden stores, FP32 readout'


def freeze(args,*,policy=POLICY,expected_recipe_sha=RECIPE_SHA,kernel_cases=13):
    from scripts.qualify_e97_native_rl_logprobs import freeze as prepare
    if sha(BASE)!=BASE_SHA:raise ValueError('base authority identity')
    prepare(SimpleNamespace(output=args.output/'authority',numerical_policy=policy,recurrent_state_precision=None,head_probe=False))
    p=args.output/'authority/recipe-private.json';recipe=json.loads(p.read_text());base=json.loads(BASE.read_text())
    stripped=dict(recipe)
    if stripped.pop('numerical_policy')!=policy or stripped['scope'].count(SCOPE)!=1:raise ValueError('candidate policy scope')
    stripped['scope']=stripped['scope'].replace(SCOPE,'')
    if stripped!=base:raise ValueError('candidate changed inputs or limits')
    if sha(p)!=expected_recipe_sha:raise ValueError('candidate recipe identity')
    publish(args.output/'plan.json',dict(schema='e97-fp32-linear-qualification-v1',kernel=FIXED_RECURRENT_KERNEL,numerical_policy=policy,
        recipe_sha256=sha(p),base_recipe_sha256=BASE_SHA,workers=2,kernel_cases=kernel_cases,kernel_seconds=900,
        worker_seconds=1800,outer_seconds=5100,teardown_seconds=30,max_hbm_allocated=16*1024**3,
        max_single_fp32_weight_bytes=1024**3,expected_linear_modules=163,
        repeat_requirement='exact actual actor/teacher scores and CE means across fresh processes',
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,
        scope='Actual composite numerical-policy forward assay; historical behavior retained separately; not4B backward/64K/restart qualification',
        **(dict(teacher_reference_sha256='64ba00df08cf48bc87e87ca1afc9c5257bfb88bb25addb63b03cedd88812e34e',
            teacher_invariance='exact FP32 scores, CE and padded lengths versus composite v1') if policy=='fp32-linear-v2' else {})))
    print('FP32_LINEAR_CANDIDATE_FROZEN',sha(p),flush=True)


def audit(args,*,policy=POLICY,expected_recipe_sha=RECIPE_SHA,kernel_cases=13):
    from scripts.audit_e97_fixed_recurrent_kernel import audit as check
    plan=json.loads((args.output/'plan.json').read_text())
    if plan['schema']!='e97-fp32-linear-qualification-v1' or plan['numerical_policy']!=policy or plan['recipe_sha256']!=expected_recipe_sha or plan['kernel_cases']!=kernel_cases:raise ValueError('candidate plan')
    check(args.output,expected_recipe_sha=plan['recipe_sha256'],expected_kernel_cases=kernel_cases,numerical_policy=policy)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','audit'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
