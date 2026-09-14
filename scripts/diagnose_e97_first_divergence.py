#!/usr/bin/env python3
"""One failing turn, two actual paths, exact binding, coarse then local traces."""
import argparse
import json
import math
import os
import struct
from pathlib import Path
from scripts.eval_e97_native_execution import publish,sha
from scripts.e97_first_divergence import Rows,compare,digest

ROOT=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/fp32-linear-candidate-v1')
RECIPE_SHA='916b1e87207a161d17ecf61881e5be3952a4fcd23ac3e6e6a6fe81d282045d15'
MEASUREMENT_SHA='64ba00df08cf48bc87e87ca1afc9c5257bfb88bb25addb63b03cedd88812e34e'
SUMMARY_SHA='490092c0e3aa4f9154aee5437350aec0a9b548889205c4ab9abeccd56fb4c7c8'
KEY=('onpolicy-task-004-sample-0',1);POSITION=23


def reference():
    p=ROOT/'authority/recipe-private.json'
    if sha(p)!=RECIPE_SHA:raise ValueError('current recipe identity')
    base=json.loads(p.read_text());rows=None
    for i in (1,2):
        p=ROOT/f'worker-{i}/measurements-private.json';s=ROOT/f'worker-{i}/summary.json'
        if sha(p)!=MEASUREMENT_SHA or sha(s)!=SUMMARY_SHA:raise ValueError('current repeated reference identity')
        rows=json.loads(p.read_text())['records']
    item=next(x for x in base['selected'] if (x['id'],x['turn'])==KEY)
    row=next(x for x in rows if (x['id'],x['turn'])==KEY)
    if len(item['generated'])<=POSITION or len(item['prefix'])+POSITION>4607:raise ValueError('causal trace length')
    return base,item,row


def freeze(args):
    base,item,_=reference()
    plan=dict(schema='e97-first-divergence-v1',base_recipe_sha256=RECIPE_SHA,measurements_sha256=MEASUREMENT_SHA,
        key=list(KEY),generated_position=POSITION,causal_rows=len(item['prefix'])+POSITION,
        numerical_policy=base['numerical_policy'],workers=1,repetitions=2,maximum_evaluations=8,
        profiles=['actor','teacher'],phases=['coarse','first-differing-layer'],
        reference_binding='exact FP32 scores and exact CE mean, otherwise stop without interpretation',
        repeat_binding='exact captured tensor bytes',trace_payload_per_bank=2*1024**3,
        maximum_live_trace_payload=8*1024**3,max_hbm_allocated=16*1024**3,
        worker_seconds=1200,outer_seconds=1800,teardown_seconds=30,optimizer_updates=0,
        automatic_expansion=False,training_eligible=False,
        scope='Module-boundary localization on current reference; not a new precision policy or packed64K qualification')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=True);publish(args.output/'plan.json',plan)
    print('FIRST_DIVERGENCE_FROZEN',sha(args.output/'plan.json'),flush=True)


def exact_scores(values,expected):
    if len(values)!=len(expected) or not values or not all(math.isfinite(v) for v in values+expected):return False
    return struct.pack('<'+'f'*len(values),*values)==struct.pack('<'+'f'*len(expected),*expected)


def evaluate(loaded,item,base,profile,layer):
    import torch
    from ndm.e97 import advance_e97_cache,advance_e97_cache_segment
    from scripts.qualify_e97_native_rl_logprobs import turn_layout
    model=loaded.model;actor=profile=='actor';model.train(not actor)
    model.gradient_checkpointing=not actor;model.gradient_checkpoint_group_size=1 if actor else base['checkpoint_group']
    model.loss_chunk_size=base['loss_chunk'];model.loss_logits_fp32=not actor;model.checkpoint_loss_chunks=not actor
    for m in model.layers:m.mlp.checkpoint_chunk_size=0 if actor else base['mlp_chunk']
    trace=Rows(model,len(item['prefix'])+POSITION,layer);handle=None;values=[];ce=None;padded=None
    try:
        with torch.no_grad():
            if actor:
                cache=advance_e97_cache_segment(loaded,item['prefix'])
                for token in item['generated']:
                    values.append(float(torch.log_softmax(cache.next_logits.float(),-1)[token].item()))
                    cache=advance_e97_cache(loaded,[token],cache)
            else:
                device=next(model.parameters()).device
                tokens,valid,reset,mask=turn_layout(item['prefix'],item['generated'],device,base['alignment'])
                labels=tokens[:,1:];offset=0;padded=tokens.shape[1]
                def head(_module,_inputs,logits):
                    nonlocal offset
                    width=logits.shape[1];selected=mask[:,offset:offset+width]
                    if bool(selected.any()):
                        target_ids=labels[:,offset:offset+width][selected]
                        lp=torch.log_softmax(logits[selected].float(),-1).gather(1,target_ids[:,None]).squeeze(1)
                        values.extend(lp.cpu().tolist())
                    offset+=width;return None
                handle=model.lm_head.register_forward_hook(head)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model(tokens,return_loss=True,loss_mask=mask,valid_mask=valid,reset_before=reset,loss_reduction='sum')
                if offset!=padded-1 or len(values)!=len(item['generated']):raise ValueError('score coverage')
                ce=float(loss.item())/len(values)
        bank=trace.finish()
        return bank,dict(scores=values,ce_mean=ce,padded_tokens=padded,trace_payload_bytes=trace.bytes,site_rows=trace.counts,calls=trace.calls)
    finally:
        if handle is not None:handle.remove()
        trace.close()


def run(args):
    import torch
    from ndm.e97 import load_e97_checkpoint
    from ndm.numerical_policy import configure_numerical_policy
    from scripts.audit_e97_live_actor_capture import fingerprint
    from ndm.triton.e88_triton_forward import _AUTOTUNE_CACHE
    if sha(args.output/'plan.json')!=args.plan_sha:raise ValueError('plan identity')
    plan=json.loads((args.output/'plan.json').read_text());base,item,ref=reference()
    if plan['schema']!='e97-first-divergence-v1' or plan['key']!=list(KEY) or plan['generated_position']!=POSITION:raise ValueError('selection identity')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single worker required')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    target=base['model']
    for path,expected in ((target['checkpoint'],target['sha256']),(base['args_json'],base['args_sha256'])):
        if sha(path)!=expected:raise ValueError('model/argument identity')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=base['args_json'],device=torch.device('cuda',local),dtype=torch.bfloat16,
        weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model;configure_numerical_policy(model,base['numerical_policy'])
    if len(model.layers)!=18 or any(m.mixer.projection_chunk_size!=512 for m in model.layers):raise ValueError('architecture/chunks')
    if any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('persistent dtype')
    before=fingerprint(model)
    if before['parameters']!='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd':raise ValueError('effective parameters')
    results=[];layer=None
    for phase in ('coarse','fine'):
        banks={}
        for profile in ('actor','teacher'):
            for repeat in range(2):
                bank,receipt=evaluate(loaded,item,base,profile,layer)
                receipt['tensor_sha256']={k:digest(v) for k,v in bank.items()}
                expected=ref['actor_replay' if profile=='actor' else 'teacher']
                bound=exact_scores(receipt['scores'],expected)
                if profile=='teacher':bound=bound and receipt['ce_mean']==ref['ce_mean'] and receipt['padded_tokens']==ref['padded_tokens']
                receipt.update(reference_bound=bound,phase=phase,profile=profile,repeat=repeat)
                publish(args.output/f'{phase}-{profile}-{repeat}-private.json',receipt)
                if repeat==0:
                    path=args.output/f'{phase}-{profile}-tensors-private.pt'
                    with path.open('xb') as f:torch.save(dict(plan_sha256=args.plan_sha,bank=bank),f)
                    path.chmod(0o400);banks[profile]=bank
                else:
                    if receipt['tensor_sha256']!={k:digest(v) for k,v in banks[profile].items()}:raise ValueError('trace repeat differs')
                    del bank
                if not bound:raise ValueError('observer/reference binding failed; trace not interpretable')
        rows=compare(banks['actor'],banks['teacher']);first=next((x for x in rows if not x['exact']),None)
        result=dict(phase=phase,layer=layer,first=first,sites=rows,reference_bound=True,repeat_exact=True)
        publish(args.output/f'{phase}-comparison.json',result);results.append(result)
        print('FIRST_DIVERGENCE_BOUND',json.dumps(dict(phase=phase,first=first)),flush=True)
        del banks
        if phase=='coarse':
            if first is None or '.block.' not in first['site']:break
            layer=int(first['site'].split('.')[0])
    after=fingerprint(model)
    if before!=after or _AUTOTUNE_CACHE or any(p.grad is not None for p in model.parameters()):raise ValueError('state/gradient/tuner mutation')
    peak=torch.cuda.max_memory_allocated(local)
    if peak>plan['max_hbm_allocated']:raise ValueError('HBM bound')
    publish(args.output/'summary.json',dict(schema='e97-first-divergence-result-v1',localization_ready=True,plan_sha256=args.plan_sha,
        parameter_fingerprint=after,peak_hbm_allocated=peak,phases=[dict(phase=x['phase'],layer=x['layer'],first=x['first']) for x in results],
        all_references_exact=True,all_traces_repeat_exact=True,optimizer_updates=0,numerical_gate_still_failed=True,
        training_eligible=False,scope=plan['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'));p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-sha')
    a=p.parse_args()
    try:globals()[a.command](a)
    except Exception as e:
        if a.output.exists():publish(a.output/'failure.json',dict(error_type=type(e).__name__,message=str(e),localization_ready=False))
        raise
