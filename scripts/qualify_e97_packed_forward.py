#!/usr/bin/env python3
"""Frozen actual64K-pack forward/reset assay. Never calls backward or an optimizer."""
import argparse
import inspect
import json
import math
import os
import time
from pathlib import Path
import torch
from scripts.e97_packed_forward_probe import assemble,probes,variants,HeadRows,state_hashes
from scripts.eval_e97_native_execution import publish,sha

ROOT=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
DATA=ROOT/'grounding-correction-v1-data'
AUTH_SHA='bf2645fc1c41fbb013bd11c9b8fe539aba6ac88b618e5f03dd5e759a7f7d2a92'
PACK_SHA='41c47a919b0cd82dc6a0486329fbf34592d60496f26fd9efe045c1a1d9f887ae'
LOADER_SHA='3d49215c5aad63186aecbc0fdf699c26776e8167bde2f3db4aa4e55f1e84889c'
REFERENCE=ROOT/'uniform-workspace-v2-r2/authority/recipe-private.json'
REFERENCE_SHA='3ab8a9237cd2102b34c83f125ed529758982f74a39aa6beef52c069c5811e364'
PARAM_SHA='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd'


def exact(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def save_private(path,value):
    with path.open('xb') as f:torch.save(value,f)
    path.chmod(0o400)


def freeze(args):
    from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset,SFTSamplerIdentity
    if sha(inspect.getfile(MaskedSFTPackedDataset))!=LOADER_SHA:raise ValueError('use immutable committed loader export; unrelated dirty loader is not this authority')
    if sha(REFERENCE)!=REFERENCE_SHA:raise ValueError('uniform-policy authority')
    args.output.mkdir(parents=True,exist_ok=True)
    identity=SFTSamplerIdentity(AUTH_SHA,PACK_SHA,0,8,65536)
    d=MaskedSFTPackedDataset(DATA/'authority',DATA/'packs',identity=identity,rank=0,
        verify_payload_hashes=True,sampler_mode='epoch-permutation')
    try:
        t,m,v,r,length,targets,_=d.pack_at_with_boundaries(0)
        first=int(d.packs[0]['record_offset']);count=int(d.packs[0]['record_count'])
        order=[int(x) for x in d.pack_record_ids[first:first+count]];records={}
        for rid in order:
            entry=d.records[rid];offset=int(entry['offset']);n=int(entry['tokens'])
            records[rid]=dict(tokens=torch.tensor(d.tokens[offset:offset+n].astype('int64')),
                mask=torch.tensor(d.masks[offset:offset+n].astype('bool')))
        del entry
        rebuilt=assemble(records,order)
        if any(not torch.equal(a,b) for a,b in zip((t,m,v,r),(rebuilt[k] for k in ('tokens','mask','valid','reset')))):raise ValueError('materialization differs from actual trainer pack')
    finally:d.close()
    if length!=60893 or targets!=7894 or len(order)!=24:raise ValueError('pack0 geometry changed')
    candidates=[]
    for rid in order:
        try:
            p=probes(records[rid])
            if len(records[rid]['tokens'])<=8192:candidates.append((rid,p))
        except ValueError:continue
    available=dict(candidates)
    sentinels=[]
    for position in (0,32768,65536):
        choices=[x for x in order if x in available and x not in sentinels]
        if not choices:raise ValueError('sentinel coverage')
        sentinels.append(min(choices,key=lambda x:abs(rebuilt['starts'][x]-position)))
    if sentinels[0]!=order[0]:raise ValueError('first authority record must be usable')
    layouts=variants(records,order,sentinels)
    probe_map={rid:available[rid] for rid in sentinels}
    payload=dict(records={k:records[k] for k in sentinels},sentinels=sentinels,probes=probe_map,layouts=layouts)
    path=args.output/'inputs-private.pt';save_private(path,payload)
    ref=json.loads(REFERENCE.read_text())
    plan=dict(schema='e97-packed-forward-v1',model=ref['model'],args_json=ref['args_json'],args_sha256=ref['args_sha256'],
        numerical_policy='fp32-linear-v2',reference_recipe_sha256=REFERENCE_SHA,
        authority_sha256=AUTH_SHA,pack_sha256=PACK_SHA,loader_sha256=LOADER_SHA,inputs_sha256=sha(path),
        pack_id=0,records=24,real_tokens=length,tail_tokens=65537-length,targets=targets,context=65536,
        sentinel_ids=sentinels,placements={k:layouts[k]['starts'][sentinels[0]] for k in ('original','middle','late')},
        repeats=2,actor_spans=12,standalone_forwards=6,packed_forwards=12,
        projection_chunk=512,checkpoint_interval=16,checkpoint_group=3,mlp_chunk=4096,loss_chunk=128,
        abs_max=.05,abs_p99=.02,ce_mean_delta_max=.0001,
        isolation='bitwise selected logits/hidden/scores under changed predecessors at fixed position with resets',
        padding='bitwise selected logits/hidden/scores, final states and CE under changed invalid tail values',
        negative_control='removing only anchor reset under changed predecessors must change boundary logits',
        max_hbm_allocated=40*1024**3,worker_seconds=3600,outer_seconds=4200,teardown_seconds=30,
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,
        scope='One actual historical pack and whole-record rotations; diagnostic prefix/tail/reset interventions. Forward only; not new data admission, gradients, eight-rank, cache/restart or behavioral qualification.')
    publish(args.output/'plan.json',plan)
    print('PACKED_FORWARD_FROZEN',sha(args.output/'plan.json'),flush=True)


def row_map(batch,probe_map,ids):
    return {f'{rid}.{kind}':[batch['starts'][rid]+p for p in positions]
        for rid in ids for kind,positions in probe_map[rid].items()}


def teacher(model,batch,probe_map,ids,device):
    model.train();model.gradient_checkpointing=True;model.gradient_checkpoint_group_size=3
    model.loss_chunk_size=128;model.loss_logits_fp32=True;model.checkpoint_loss_chunks=True
    mlps=[m for m in model.modules() if hasattr(m,'checkpoint_chunk_size')]
    if len(mlps)!=18:raise ValueError('MLP coverage')
    for m in mlps:m.checkpoint_chunk_size=4096
    b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    observer=HeadRows(b,row_map(b,probe_map,ids));handle=model.lm_head.register_forward_hook(observer.hook)
    try:
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            loss,states=model(b['tokens'][None],return_loss=True,loss_mask=b['mask'][None],
                valid_mask=b['valid'][None],reset_before=b['reset'][None],loss_reduction='sum',return_prev_hiddens=True)
        result=observer.finish(float(loss.item()));result['state_sha256']=state_hashes(states[0])
        if any(not torch.equal(b[k].cpu(),batch[k]) for k in ('tokens','mask','valid','reset')):raise ValueError('forward mutated input/control tensors')
    finally:handle.remove()
    return result


def actor(loaded,record,positions):
    from ndm.e97 import advance_e97_cache_segment,advance_e97_cache
    if positions!=list(range(positions[0],positions[-1]+1)):raise ValueError('contiguous replay span')
    loaded.model.eval();tokens=record['tokens'].tolist();start=positions[0];values=[]
    with torch.no_grad():
        cache=advance_e97_cache_segment(loaded,tokens[:start+1])
        for p in positions:
            if cache.next_logits.dtype!=torch.float32:raise ValueError('actor actual head dtype')
            values.append(float(torch.log_softmax(cache.next_logits,-1)[tokens[p+1]].item()))
            cache=advance_e97_cache(loaded,[tokens[p+1]],cache)
    if not all(math.isfinite(x) for x in values):raise ValueError('actor nonfinite')
    return values


def compare(values,reference,plan):
    import numpy as np
    if set(values)!=set(reference):raise ValueError('comparison coverage')
    gaps=[]
    for k in values:
        a=values[k];b=reference[k]
        if not a or len(a)!=len(b):raise ValueError('score coverage')
        gaps.extend(abs(x-y) for x,y in zip(a,b))
    if not all(math.isfinite(x) for x in gaps):raise ValueError('nonfinite comparison')
    return dict(count=len(gaps),abs_max=max(gaps),abs_p99=float(np.quantile(gaps,.99)),
        passed=max(gaps)<=plan['abs_max'] and float(np.quantile(gaps,.99))<=plan['abs_p99'])


def run(args):
    from ndm.e97 import load_e97_checkpoint
    from ndm.numerical_policy import configure_numerical_policy,RecomputedFP32Linear
    from scripts.qualify_e97_response_gradients import parameter_digest
    from ndm.triton.e88_triton_forward import _AUTOTUNE_CACHE
    if sha(args.output/'plan.json')!=args.plan_sha:raise ValueError('plan identity')
    plan=json.loads((args.output/'plan.json').read_text());path=args.output/'inputs-private.pt'
    if sha(path)!=plan['inputs_sha256']:raise ValueError('inputs identity')
    data=torch.load(path,map_location='cpu',weights_only=True)
    target=plan['model']
    if sha(target['checkpoint'])!=target['sha256'] or sha(plan['args_json'])!=plan['args_sha256']:raise ValueError('model identity')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single leased worker')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local)
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=plan['args_json'],device=device,dtype=torch.bfloat16,
        weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model;configure_numerical_policy(model,plan['numerical_policy'])
    before=parameter_digest(model)
    if before!=PARAM_SHA or any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('parameter identity/dtype')
    if sum(isinstance(m,RecomputedFP32Linear) for m in model.modules())!=163 or sum(bool(getattr(m,'uniform_recurrent_workspace',False)) for m in model.modules())!=18:raise ValueError('composite policy coverage')
    mixers=[m for m in model.modules() if hasattr(m,'recurrent_state_precision')]
    if len(mixers)!=18 or any((m.n_heads,m.n_state,m.head_v_dim,m.projection_chunk_size,m.checkpoint_interval)!=(60,64,64,512,16) for m in mixers):raise ValueError('production geometry')
    if (model.lm_head.in_features,model.lm_head.out_features)!=(3840,50281):raise ValueError('readout geometry')
    records=data['records'];ids=data['sentinels'];pm=data['probes'];actors={};standalones={};measurements={};checks={};timings={};started=time.monotonic()
    for rid in ids:
        for kind,positions in pm[rid].items():
            name=f'{rid}.{kind}';repeats=[]
            for repeat in range(2):
                result=actor(loaded,records[rid],positions);repeats.append(result)
                publish(args.output/f'actor-{rid}-{kind}-{repeat}-private.json',dict(plan_sha256=args.plan_sha,logprobs=result))
            if not exact(repeats[0],repeats[1]):raise ValueError('actor repeat mismatch')
            actors[name]=repeats[0]
        length=len(records[rid]['tokens']);context=((length-2)//16+1)*16
        batch=assemble(records,[rid],context);previous=None
        for repeat in range(2):
            result=teacher(model,batch,pm,[rid],device)
            publish(args.output/f'standalone-{rid}-{repeat}-private.json',dict(plan_sha256=args.plan_sha,**result))
            if previous is not None and not exact(previous,result):raise ValueError('standalone repeat mismatch')
            previous=result
        standalones.update({k:v['logprobs'] for k,v in previous['probes'].items()})
        checks[f'standalone_ce_{rid}']=previous['ce_delta']<=plan['ce_mean_delta_max']
    metrics={'standalone_vs_actor':compare(standalones,actors,plan)}
    checks['standalone_vs_actor']=metrics['standalone_vs_actor']['passed']
    if not all(checks.values()):
        publish(args.output/'baseline-failed.json',dict(checks=checks,metrics=metrics));raise ValueError('standalone gate failed before packed runs')
    print('PACKED_BASELINES_BOUND',json.dumps(metrics['standalone_vs_actor'],sort_keys=True),flush=True)
    for name,batch in data['layouts'].items():
        selected=[ids[0]] if name in ('predecessor_changed','reset_removed') else ids
        previous=None;timings[name]=[]
        for repeat in range(2):
            torch.cuda.synchronize(local);t0=time.monotonic()
            result=teacher(model,batch,pm,selected,device)
            torch.cuda.synchronize(local);timings[name].append(time.monotonic()-t0)
            publish(args.output/f'{name}-{repeat}-private.json',dict(plan_sha256=args.plan_sha,**result))
            if previous is not None and not exact(previous,result):raise ValueError('packed repeat mismatch')
            previous=result
            if torch.cuda.max_memory_allocated(local)>plan['max_hbm_allocated']:raise ValueError('allocated HBM ceiling')
            print('PACKED_FORWARD_MEASURED',name,repeat,flush=True)
        measurements[name]=previous;checks[f'{name}_ce']=previous['ce_delta']<=plan['ce_mean_delta_max']
        checks[f'{name}_targets']=previous['targets']==plan['targets']
        if name in ('original','middle','late'):
            values={k:v['logprobs'] for k,v in previous['probes'].items()}
            for ref,values_ref in (('actor',actors),('standalone',standalones)):
                key=f'{name}_vs_{ref}';metrics[key]=compare(values,values_ref,plan);checks[key]=metrics[key]['passed']
    anchor=ids[0];keys=[f'{anchor}.{kind}' for kind in pm[anchor]]
    checks['predecessor_isolation_exact']=all(exact(measurements['late']['probes'][k],measurements['predecessor_changed']['probes'][k]) for k in keys)
    checks['reset_negative_control_sensitive']=measurements['predecessor_changed']['probes'][f'{anchor}.boundary']['logits_sha256']!=measurements['reset_removed']['probes'][f'{anchor}.boundary']['logits_sha256']
    checks['padding_values_inert_exact']=exact(measurements['original'],measurements['padding_changed'])
    checks['parameters_unchanged']=parameter_digest(model)==before
    checks['no_gradients']=all(p.grad is None for p in model.parameters())
    checks['no_autotune']=not _AUTOTUNE_CACHE
    checks['memory']=torch.cuda.max_memory_allocated(local)<=plan['max_hbm_allocated']
    summary=dict(schema='e97-packed-forward-result-v1',plan_sha256=args.plan_sha,checks=checks,metrics=metrics,
        passed=all(checks.values()),repeats_exact=True,parameter_sha256=before,numerical_policy=plan['numerical_policy'],
        head_dtype='torch.float32',instrumented_packed_forward_seconds=timings,peak_hbm_allocated=torch.cuda.max_memory_allocated(local),elapsed_seconds=time.monotonic()-started,
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,
        scope=plan['scope'],remaining='full4B gradients, broader packs/positions, eight-rank, cache and restart')
    publish(args.output/'summary.json',summary);print(json.dumps(summary,sort_keys=True),flush=True)
    if not summary['passed']:raise SystemExit('packed forward/reset numerical gate failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'));p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-sha')
    a=p.parse_args()
    try:globals()[a.command](a)
    except Exception as exc:
        if a.output.exists():publish(a.output/f'{a.command}-failure.json',dict(error=type(exc).__name__,message=str(exc),training_eligible=False,optimizer_updates=0))
        raise
