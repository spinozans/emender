#!/usr/bin/env python3
"""Same activation-trace actor with controlled diagnostic-only per-step work."""
import argparse
import json
import os
from pathlib import Path
from scripts.diagnose_e97_activation_alignment import evaluate,PROFILES,KEYS,SOURCE,SOURCE_SHA,ROOT
from scripts.diagnose_e97_observer_effect import TRACE,TRACE_SHA,delta
from scripts.eval_e97_native_execution import publish,sha

MODES=('none','read','allocate','copy-allocate','synchronize')


class StepProbe:
    def __init__(self,mode):
        if mode not in MODES[1:]:raise ValueError('step probe mode')
        self.mode=mode;self.calls=0;self.values=[]

    def __call__(self,logits,token):
        import torch
        if logits.ndim!=1 or not 0<logits.numel()<=65536 or logits.dtype!=torch.bfloat16:
            raise ValueError('bounded BF16 cache logits required')
        if type(token) is not int or not 0<=token<logits.numel():raise ValueError('cache token range')
        self.calls+=1
        if self.mode=='read':
            self.values.append(float(torch.log_softmax(logits.float(),-1)[token].item()))
        elif self.mode in ('allocate','copy-allocate'):
            # The model never consumes either temporary. Match the two FP32
            # vocabulary-sized allocations of conversion plus log-softmax.
            first=(logits.float() if self.mode=='copy-allocate' else torch.empty(logits.shape,device=logits.device,dtype=torch.float32))
            second=torch.empty_like(first)
            del second,first
        elif logits.device.type=='cuda':torch.cuda.synchronize(logits.device)


def freeze(args):
    if sha(SOURCE)!=SOURCE_SHA or sha(TRACE)!=TRACE_SHA:raise ValueError('source identity')
    source=json.loads(SOURCE.read_text());table={(r['id'],r['turn']):r for r in source['selected']}
    old={(r['id'],r['turn']):r for r in json.loads(TRACE.read_text())['reports']};selected=[]
    for key in KEYS:
        previous=next(r for r in old[key]['profiles'] if r['profile']['name']=='actor-cache')['logprobs']
        item=dict(table[key]);delta(previous,item['recorded_logprobs']);item['previous_trace_logprobs']=previous;selected.append(item)
    recipe=dict(schema='emender-e97-readback-effect-v1',selected=selected,model=source['model'],args_json=source['args_json'],
                args_sha256=source['args_sha256'],source_recipe_sha256=SOURCE_SHA,previous_trace_sha256=TRACE_SHA,
                modes=MODES,repetitions=2,workers=1,actor_profile=PROFILES[0],limit=1e-4,
                parameter_sha256='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd',
                optimizer_updates=0,training_eligible=False,automatic_expansion=False,
                scope='Forty same-trace forced actor replays; readback/temporary-allocation/synchronization controls, not numerical fixes or RL qualification')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False);publish(args.output/'recipe-private.json',recipe)
    print('READBACK_DIAGNOSTIC_FROZEN',sha(args.output/'recipe-private.json'),flush=True)


def run(args):
    import torch
    from ndm.e97 import load_e97_checkpoint
    from scripts.audit_e97_live_actor_capture import fingerprint,runtime
    if sha(args.recipe)!=args.recipe_sha:raise ValueError('recipe identity')
    recipe=json.loads(args.recipe.read_text());target=recipe['model'];local=int(os.environ.get('LOCAL_RANK','0'))
    if recipe['modes']!=list(MODES) or recipe['actor_profile']!=PROFILES[0] or recipe['repetitions']!=2 or recipe['limit']!=1e-4:
        raise ValueError('execution recipe mismatch')
    if recipe['workers']!=1 or int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single worker required')
    if [(r['id'],r['turn']) for r in recipe['selected']]!=KEYS:raise ValueError('selection mismatch')
    for path,digest in ((target['checkpoint'],target['sha256']),(recipe['args_json'],recipe['args_sha256'])):
        if sha(path)!=digest:raise ValueError('model identity')
    torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':raise ValueError('precision defaults')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=recipe['args_json'],device=torch.device('cuda',local),dtype=torch.bfloat16,
                               weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model;before=fingerprint(model)
    if before['parameters']!=recipe['parameter_sha256'] or any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('effective parameters')
    if len(model.layers)!=18 or any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()):raise ValueError('deterministic model required')
    publish(args.output/'preflight.json',dict(recipe_sha256=args.recipe_sha,fingerprint=before,runtime=runtime(loaded,local)))
    private=args.output/'profiles-private';private.mkdir(mode=0o700,exist_ok=False);rows=[];receipts={}
    for item in recipe['selected']:
        if not 0<len(item['prefix'])<=16384 or not 0<len(item['generated'])<=512:raise ValueError('input bounds')
        for repetition in range(2):
            for mode in (MODES if repetition==0 else tuple(reversed(MODES))):
                torch.manual_seed(974223);probe=None if mode=='none' else StepProbe(mode)
                bank,values=evaluate(loaded,item,recipe['actor_profile'],step_probe=probe)
                if probe is not None and probe.calls!=len(item['generated']):raise ValueError('per-step coverage')
                row=dict(id=item['id'],turn=item['turn'],mode=mode,repetition=repetition,logprobs=values,
                         direct_read_logprobs=probe.values if mode=='read' else None,
                         read_vs_observer_max=delta(values,probe.values) if mode=='read' else None,
                         recorded_actor_max=delta(values,item['recorded_logprobs']),previous_trace_max=delta(values,item['previous_trace_logprobs']))
                del bank
                path=private/f"{item['id']}-turn-{item['turn']}-{mode}-r{repetition}.json"
                publish(path,dict(recipe_sha256=args.recipe_sha,measurement=row));receipts[path.name]=sha(path);rows.append(row)
                print('READBACK_PROFILE_COMPLETE',item['id'],item['turn'],mode,repetition,row['recorded_actor_max'],row['previous_trace_max'],flush=True)
    after=fingerprint(model)
    if before!=after or any(p.grad is not None for p in model.parameters()):raise ValueError('state/gradient mutation')
    comparisons=[]
    for item in recipe['selected']:
        table={(r['mode'],r['repetition']):r for r in rows if (r['id'],r['turn'])==(item['id'],item['turn'])}
        if len(table)!=10:raise ValueError('profile coverage')
        for mode in MODES:
            comparisons.append(dict(id=item['id'],turn=item['turn'],mode=mode,
                repeat_max=delta(table[(mode,0)]['logprobs'],table[(mode,1)]['logprobs']),
                mode_vs_none_max=max(delta(table[(mode,r)]['logprobs'],table[('none',r)]['logprobs']) for r in range(2))))
    publish(args.output/'measurements-private.json',dict(recipe_sha256=args.recipe_sha,rows=rows,comparisons=comparisons))
    summary=dict(status='diagnostic-measurements-complete',recipe_sha256=args.recipe_sha,fingerprint=before,receipts=receipts,comparisons=comparisons,
                 optimizer_updates=0,training_eligible=False,automatic_expansion=False,rl_optimizer_ready=False,original_probability_gate='failed, unchanged',
                 peak_hbm_allocated=torch.cuda.max_memory_allocated(local),
                 modes={mode:dict(recorded_actor_max=max(r['recorded_actor_max'] for r in rows if r['mode']==mode),
                                 previous_trace_max=max(r['previous_trace_max'] for r in rows if r['mode']==mode)) for mode in MODES},
                 read_vs_observer_max=max(r['read_vs_observer_max'] for r in rows if r['mode']=='read'),
                 baseline_trace_binding_passed=max(r['previous_trace_max'] for r in rows if r['mode']=='none')<=recipe['limit'])
    publish(args.output/'summary.json',summary);print('READBACK_DIAGNOSTIC_COMPLETE',sha(args.output/'summary.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('run');q.add_argument('--output',type=Path,required=True);q.add_argument('--recipe',type=Path,required=True);q.add_argument('--recipe-sha',required=True)
    a=p.parse_args();globals()[a.command](a)
