#!/usr/bin/env python3
"""Repeat eight original live episodes with actor fingerprints and same-process replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from scripts.e97_native_onpolicy_canary import repair
from scripts.eval_e97_native_execution import episode,publish,sha

SOURCE=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-onpolicy-canary-v2')
PANEL_SHA='795ef4e090d6606ddc099c0f630b6e876c39833dc9ae8494263429ac1fea7be8'
SUMMARY_SHA='8278969f8ff65afd6f49251abb8a05c758ef6586c748824d3cdbc8345e187e6d'


def freeze(args):
    if sha(SOURCE/'panel.json')!=PANEL_SHA or sha(SOURCE/'results/summary.json')!=SUMMARY_SHA:
        raise ValueError('historical identity')
    panel=json.loads((SOURCE/'panel.json').read_text());summary=json.loads((SOURCE/'results/summary.json').read_text())
    selected=[r for r in summary['results'] if r['family'] in ('lookup','edit') and r['id'].endswith('-sample-0')]
    if len(selected)!=8:raise ValueError('eight original live episodes required')
    for row in selected:
        path=SOURCE/'results'/row['id']/'episode-private.json'
        if sha(path)!=row['episode_sha256']:raise ValueError('historical episode identity')
    recipe=dict(schema='emender-live-actor-capture-audit-v1',panel=panel,selected=selected,
                source_panel_sha256=PANEL_SHA,source_summary_sha256=SUMMARY_SHA,
                workers=2,optimizer_updates=0,training_eligible=False,automatic_expansion=False,
                scope='Eight repeated original training episodes, not new independent evaluation or training',
                replay_abs_max=1e-4)
    args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    publish(args.output/'recipe-private.json',recipe)
    print('LIVE_CAPTURE_AUDIT_FROZEN',sha(args.output/'recipe-private.json'),flush=True)


def load(args):
    if sha(args.recipe)!=args.recipe_sha:raise ValueError('audit recipe identity')
    return json.loads(args.recipe.read_text())


def fingerprint(model):
    # Local implementation avoids adding numerical-probe imports to the actor path.
    import torch
    digest=hashlib.sha256();buffers=hashlib.sha256()
    for target,items in ((digest,model.named_parameters()),(buffers,model.named_buffers())):
        for name,value in items:
            target.update(json.dumps([name,list(value.shape),str(value.dtype)]).encode())
            flat=value.detach().reshape(-1)
            for start in range(0,flat.numel(),1048576):
                target.update(flat[start:start+1048576].cpu().view(torch.uint8).numpy().tobytes())
    return dict(parameters=digest.hexdigest(),buffers=buffers.hexdigest())


def runtime(loaded,local):
    import torch
    return dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),local_rank=local,
                device_name=torch.cuda.get_device_name(local),python_hash_seed=os.environ.get('PYTHONHASHSEED'),
                matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),threads=torch.get_num_threads(),
                default_dtype=str(torch.get_default_dtype()),grad_enabled=torch.is_grad_enabled(),
                model_training=loaded.model.training,weight_mode=loaded.weight_mode,
                configuration=loaded.config,
                mlp_chunks=[m.checkpoint_chunk_size for m in loaded.model.modules() if hasattr(m,'checkpoint_chunk_size')])


def run(args):
    import torch
    import tiktoken
    from ndm.e97 import load_e97_checkpoint,advance_e97_cache_segment,advance_e97_cache
    recipe=load(args);panel=recipe['panel'];rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('two fixed audit actors')
    target=panel['models'][0]
    if sha(target['checkpoint'])!=target['sha256'] or sha(panel['args_json'])!=panel['args_sha256']:
        raise ValueError('model identity')
    torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=panel['args_json'],device=torch.device('cuda',local),
                               dtype=torch.bfloat16,weight_mode=target['mode'],use_triton=True,mmap=True)
    loaded.model.eval();enc=tiktoken.get_encoding('p50k_base');before=fingerprint(loaded.model)
    publish(args.output/f'actor-{rank}-before-private.json',dict(fingerprint=before,runtime=runtime(loaded,local)))
    results=[];fresh={};historical={}
    family=('lookup','edit')[rank]
    for row in recipe['selected']:
        if row['family']!=family:continue
        case=next(c for c in panel['cases'] if c['id']==row['task_id'])
        case={**case,'id':row['id']};out=args.output/row['id']
        torch.manual_seed(row['seed'])
        def continuation(sandbox,messages,original):
            return repair(sandbox,messages,original,case,panel,enc,out)
        actual=episode(loaded,case,panel,enc,out,continuation=continuation)
        fresh[row['id']]=actual
        original=SOURCE/'results'/row['id']/'episode-private.json'
        if sha(original)!=row['episode_sha256']:raise ValueError('historical episode changed')
        historical[row['id']]=json.loads(original.read_text())
        results.append(dict(id=row['id'],reward=int(actual['autonomous_success']),episode_sha256=sha(out/'episode-private.json')))
        print('LIVE_CAPTURE_EPISODE_RECORDED',rank,row['id'],flush=True)
    after_collection=fingerprint(loaded.model);records=[]
    # Replay only after all live episodes, so audit calls cannot perturb later
    # sampling/RNG sequences within the original live collection order.
    with torch.no_grad():
        for origin,episodes in (('fresh',fresh),('historical',historical)):
            for identity,record in episodes.items():
                for turn in record['generations']:
                    ids=turn['token_ids'];trace=turn['sampling']
                    if not ids:continue
                    if len(ids)!=len(trace['selected_logprobs']):raise ValueError('sampling coverage')
                    cache=advance_e97_cache_segment(loaded,trace['prompt_token_ids']);values=[]
                    for token in ids:
                        values.append(float(torch.log_softmax(cache.next_logits.float(),-1)[token].item()))
                        cache=advance_e97_cache(loaded,[token],cache)
                    delta=max(abs(a-b) for a,b in zip(values,trace['selected_logprobs']))
                    records.append(dict(origin=origin,id=identity,turn=turn['turn'],max_delta=delta,
                                        replay=values,recorded=trace['selected_logprobs']))
    after=fingerprint(loaded.model)
    publish(args.output/f'rank-{rank}-private.json',dict(rank=rank,recipe_sha256=args.recipe_sha,
        fingerprint_before=before,fingerprint_after_collection=after_collection,fingerprint_after=after,
        runtime_after=runtime(loaded,local),records=records,results=results))


def aggregate(args):
    import math
    recipe=load(args);workers=[];maxima={}
    for rank in range(2):
        r=json.loads((args.output/f'rank-{rank}-private.json').read_text())
        if r['rank']!=rank or r['recipe_sha256']!=args.recipe_sha:raise ValueError('worker identity')
        expected={x['id'] for x in recipe['selected'] if x['family']==('lookup','edit')[rank]}
        if {x['id'] for x in r['results']}!=expected or len(r['results'])!=4:raise ValueError('episode coverage')
        if r['fingerprint_before']!=r['fingerprint_after_collection'] or r['fingerprint_before']!=r['fingerprint_after']:
            raise ValueError('actor model/buffer mutation')
        for row in r['results']:
            out=args.output/row['id']
            if sha(out/'episode-private.json')!=row['episode_sha256']:raise ValueError('fresh episode binding')
            if not json.loads((out/'cleanup.json').read_text())['removed']:raise ValueError('agent cleanup')
        for record in r['records']:
            if any(not math.isfinite(x) for k in ('replay','recorded') for x in record[k]):raise ValueError('nonfinite probabilities')
        workers.append(r)
    for origin in ('fresh','historical'):
        maxima[origin]=max(x['max_delta'] for r in workers for x in r['records'] if x['origin']==origin)
    result=dict(status='diagnostic-measurements-complete',recipe_sha256=args.recipe_sha,
        replay_abs_max=maxima,fresh_capture_consistent=maxima['fresh']<=recipe['replay_abs_max'],
        actor_fingerprints=[r['fingerprint_before'] for r in workers],
        worker_metadata=[r['runtime_after'] for r in workers],
        matches_control_parameter_hash=all(r['fingerprint_before']['parameters']=='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd' for r in workers),
        optimizer_updates=0,rl_optimizer_ready=False,training_eligible=False,automatic_expansion=False,
        original_probability_gate='failed, unchanged')
    publish(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='worker_metadata'},sort_keys=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--output',type=Path,required=True)
    for command in ('run','aggregate'):
        q=s.add_parser(command);q.add_argument('--recipe',type=Path,required=True);q.add_argument('--recipe-sha',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
