#!/usr/bin/env python3
"""Compare native cache probabilities to observers before trusting activation traces."""
import argparse
import contextlib
import json
import os
from pathlib import Path
from scripts.diagnose_e97_activation_alignment import Taps,KEYS,SOURCE,SOURCE_SHA,ROOT
from scripts.eval_e97_native_execution import publish,sha

MODES=('native','attributes','disabled-autocast','head-observer','full-observer')
FIELDS=('gradient_checkpointing','gradient_checkpoint_group_size','loss_chunk_size','loss_logits_fp32','checkpoint_loss_chunks')
TRACE=ROOT/'native-activation-alignment-v2/measurements-private.json'
TRACE_SHA='9f3ef99fd200ea44d70d344848c2bdda0458cf3ac43e4dcc4dbef7309c85e7d3'


def attributes(model):
    return dict(training=model.training,model={k:dict(present=hasattr(model,k),value=getattr(model,k,None)) for k in FIELDS},
                mlp=[dict(present=hasattr(l.mlp,'checkpoint_chunk_size'),value=getattr(l.mlp,'checkpoint_chunk_size',None)) for l in model.layers])


def restore(model,snapshot):
    model.train(snapshot['training'])
    for module,values in [(model,snapshot['model'])]+[(l.mlp,{'checkpoint_chunk_size':v}) for l,v in zip(model.layers,snapshot['mlp'])]:
        for key,entry in values.items():
            if entry['present']:setattr(module,key,entry['value'])
            elif hasattr(module,key):delattr(module,key)


def configure(model):
    model.eval();model.gradient_checkpointing=False;model.gradient_checkpoint_group_size=3
    model.loss_chunk_size=128;model.loss_logits_fp32=True;model.checkpoint_loss_chunks=True
    for layer in model.layers:layer.mlp.checkpoint_chunk_size=0


def delta(a,b):
    import math
    if not a or len(a)!=len(b) or any(not math.isfinite(x) for x in a+b):raise ValueError('probability coverage/nonfinite')
    return max(abs(x-y) for x,y in zip(a,b))


def collect(loaded,item,mode,*,segment=None,step=None):
    import torch
    if mode not in MODES:raise ValueError('observer mode')
    if segment is None and step is None:
        from ndm.e97 import advance_e97_cache_segment as segment,advance_e97_cache as step
    elif segment is None or step is None:raise ValueError('cache primitive pair')
    taps=Taps(loaded.model) if mode in ('head-observer','full-observer') else None
    if mode=='head-observer':
        for handle in taps.handles[:-1]:handle.remove()
        taps.handles=taps.handles[-1:]
    device=next(loaded.model.parameters()).device
    context=(torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=False)
             if mode in ('disabled-autocast','head-observer','full-observer') else contextlib.nullcontext())
    values=[];generated=item['generated'];prefix=item['prefix']
    try:
        with torch.no_grad(),context:
            if taps is not None:taps.begin([len(prefix)-1],[generated[0]])
            cache=segment(loaded,prefix)
            for i,token in enumerate(generated):
                # Authoritative measurement is exactly the native one-dimensional
                # cache log-softmax, never the observer's gathered row tensor.
                values.append(float(torch.log_softmax(cache.next_logits.float(),-1)[token].item()))
                if taps is not None:
                    taps.begin([0] if i+1<len(generated) else [],[generated[i+1]] if i+1<len(generated) else [])
                cache=step(loaded,[token],cache)
        observed=None
        if taps is not None:
            observed=taps.logprobs
            if mode=='full-observer':taps.finish(len(generated))
            elif set(taps.bank)!={'head-input'} or sum(len(x) for x in taps.bank['head-input'])!=len(generated):
                raise ValueError('head observer row coverage')
            delta(values,observed)
        return dict(native_logprobs=values,observer_logprobs=observed,
                    observer_vs_native_max=delta(values,observed) if observed is not None else None)
    finally:
        if taps is not None:taps.close()


def freeze(args):
    if sha(SOURCE)!=SOURCE_SHA or sha(TRACE)!=TRACE_SHA:raise ValueError('source identity')
    source=json.loads(SOURCE.read_text());table={(r['id'],r['turn']):r for r in source['selected']}
    old={(r['id'],r['turn']):r for r in json.loads(TRACE.read_text())['reports']}
    selected=[]
    for key in KEYS:
        previous=next(p for p in old[key]['profiles'] if p['profile']['name']=='actor-cache')['logprobs']
        item=dict(table[key]);delta(previous,item['recorded_logprobs']);item['previous_observer_logprobs']=previous;selected.append(item)
    recipe=dict(schema='emender-e97-observer-effect-v1',model=source['model'],args_json=source['args_json'],args_sha256=source['args_sha256'],
                selected=selected,modes=MODES,repetitions=2,workers=1,source_recipe_sha256=SOURCE_SHA,source_trace_sha256=TRACE_SHA,
                parameter_sha256='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd',
                limit=1e-4,maximum_prefix_tokens=16384,maximum_generated_tokens=512,
                optimizer_updates=0,training_eligible=False,automatic_expansion=False,
                scope='Forty forced actor replays, diagnostic only; reverse mode order on second repetition; no model numerical changes or RL qualification')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False);publish(args.output/'recipe-private.json',recipe)
    print('OBSERVER_DIAGNOSTIC_FROZEN',sha(args.output/'recipe-private.json'),flush=True)


def run(args):
    import torch
    from ndm.e97 import load_e97_checkpoint
    from scripts.audit_e97_live_actor_capture import fingerprint,runtime
    if sha(args.recipe)!=args.recipe_sha:raise ValueError('recipe identity')
    recipe=json.loads(args.recipe.read_text())
    if recipe['modes']!=list(MODES) or recipe['repetitions']!=2 or recipe['workers']!=1 or recipe['limit']!=1e-4:
        raise ValueError('execution recipe mismatch')
    if [(r['id'],r['turn']) for r in recipe['selected']]!=KEYS:raise ValueError('selection mismatch')
    target=recipe['model'];local=int(os.environ.get('LOCAL_RANK','0'))
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single fixed worker')
    for path,digest in ((target['checkpoint'],target['sha256']),(recipe['args_json'],recipe['args_sha256'])):
        if sha(path)!=digest:raise ValueError('model identity')
    torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':raise ValueError('precision defaults')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=recipe['args_json'],device=torch.device('cuda',local),dtype=torch.bfloat16,
                               weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model;initial=attributes(model);before=fingerprint(model)
    if before['parameters']!=recipe['parameter_sha256'] or any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('effective model identity')
    if len(model.layers)!=18 or any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()):raise ValueError('deterministic model required')
    publish(args.output/'preflight.json',dict(recipe_sha256=args.recipe_sha,runtime=runtime(loaded,local),attributes=initial,fingerprint=before))
    rows=[];receipts={};private=args.output/'profiles-private';private.mkdir(mode=0o700,exist_ok=False)
    for item in recipe['selected']:
        if not 0<len(item['prefix'])<=recipe['maximum_prefix_tokens'] or not 0<len(item['generated'])<=recipe['maximum_generated_tokens']:
            raise ValueError('input bounds')
        for repetition in range(2):
            for mode in (MODES if repetition==0 else tuple(reversed(MODES))):
                restore(model,initial)
                if mode!='native':configure(model)
                torch.manual_seed(974223)
                result=collect(loaded,item,mode)
                result.update(id=item['id'],turn=item['turn'],mode=mode,repetition=repetition,attributes=attributes(model),
                              recorded_actor_max=delta(result['native_logprobs'],item['recorded_logprobs']),
                              previous_trace_max=delta(result['native_logprobs'],item['previous_observer_logprobs']))
                path=private/f"{item['id']}-turn-{item['turn']}-{mode}-r{repetition}.json"
                publish(path,dict(recipe_sha256=args.recipe_sha,measurement=result));receipts[path.name]=sha(path);rows.append(result)
                print('OBSERVER_PROFILE_COMPLETE',item['id'],item['turn'],mode,repetition,result['recorded_actor_max'],result['observer_vs_native_max'],flush=True)
    restore(model,initial);after=fingerprint(model)
    if before!=after or attributes(model)!=initial or any(p.grad is not None for p in model.parameters()):raise ValueError('state/gradient mutation')
    comparisons=[]
    for item in recipe['selected']:
        table={(r['mode'],r['repetition']):r for r in rows if (r['id'],r['turn'])==(item['id'],item['turn'])}
        if len(table)!=10:raise ValueError('profile coverage')
        for mode in MODES:
            comparisons.append(dict(id=item['id'],turn=item['turn'],mode=mode,
                repeat_max=delta(table[(mode,0)]['native_logprobs'],table[(mode,1)]['native_logprobs']),
                mode_vs_native_max=max(delta(table[(mode,r)]['native_logprobs'],table[('native',r)]['native_logprobs']) for r in range(2))))
    publish(args.output/'measurements-private.json',dict(recipe_sha256=args.recipe_sha,rows=rows,comparisons=comparisons))
    summary=dict(status='diagnostic-measurements-complete',recipe_sha256=args.recipe_sha,fingerprint=before,receipts=receipts,
                 optimizer_updates=0,training_eligible=False,automatic_expansion=False,rl_optimizer_ready=False,original_probability_gate='failed, unchanged',
                 peak_hbm_allocated=torch.cuda.max_memory_allocated(local),comparisons=comparisons,
                 modes={mode:dict(recorded_actor_max=max(r['recorded_actor_max'] for r in rows if r['mode']==mode),
                                 observer_vs_native_max=max((r['observer_vs_native_max'] for r in rows if r['mode']==mode and r['observer_vs_native_max'] is not None),default=None)) for mode in MODES})
    publish(args.output/'summary.json',summary);print('OBSERVER_DIAGNOSTIC_COMPLETE',sha(args.output/'summary.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('run');q.add_argument('--output',type=Path,required=True);q.add_argument('--recipe',type=Path,required=True);q.add_argument('--recipe-sha',required=True)
    a=p.parse_args();globals()[a.command](a)
