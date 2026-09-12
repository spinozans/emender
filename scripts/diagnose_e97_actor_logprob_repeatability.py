#!/usr/bin/env python3
"""Focused no-update repeatability diagnostic; never relabels the failed assay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from scripts.eval_e97_native_execution import publish,sha

SOURCE=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-rl-logprob-qualification-v1/recipe-private.json')
SOURCE_SHA='ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f'
KEYS=[('onpolicy-task-000-sample-0',0),('onpolicy-task-004-sample-0',1),
      ('onpolicy-task-010-sample-0',1),('onpolicy-task-014-sample-0',4)]
MODES=('before-step','after-step','sampling-shaped-after-step','full-generator-forced')


def freeze(args):
    if sha(SOURCE)!=SOURCE_SHA:raise ValueError('source assay identity')
    source=json.loads(SOURCE.read_text());selected=[]
    for identity,turn in KEYS:
        matches=[r for r in source['selected'] if (r['id'],r['turn'])==(identity,turn)]
        if len(matches)!=1:raise ValueError('diagnostic selection coverage')
        selected.extend(matches)
    recipe=dict(schema='emender-actor-logprob-repeatability-v1',model=source['model'],
        args_json=source['args_json'],args_sha256=source['args_sha256'],selected=selected,
        source_recipe_sha256=SOURCE_SHA,modes=MODES,repetitions=2,workers=2,
        selection='One stable control plus three observed outliers; diagnostic, not independent validation',
        optimizer_updates=0,automatic_expansion=False,training_eligible=False,
        scope='Forced recorded tokens; optional discarded sampling draw tests capture/RNG execution shape; no new rollouts')
    args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    publish(args.output/'recipe-private.json',recipe)
    print('REPEATABILITY_DIAGNOSTIC_FROZEN',sha(args.output/'recipe-private.json'),flush=True)


def load(args):
    if sha(args.recipe)!=args.recipe_sha:raise ValueError('recipe identity')
    recipe=json.loads(args.recipe.read_text())
    if tuple(recipe['modes'])!=MODES or recipe['workers']!=2 or recipe['repetitions']!=2:
        raise ValueError('diagnostic execution recipe mismatch')
    return recipe


def run(args):
    import torch
    import tiktoken
    from unittest.mock import patch
    from scripts.eval_e97_native_execution import generate_turn
    from ndm.e97 import load_e97_checkpoint,advance_e97_cache_segment,advance_e97_cache,_sample_token
    from scripts.qualify_e97_response_gradients import parameter_digest
    recipe=load(args);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('two fixed diagnostic workers')
    target=recipe['model']
    if sha(target['checkpoint'])!=target['sha256'] or sha(recipe['args_json'])!=recipe['args_sha256']:
        raise ValueError('model identity')
    torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=recipe['args_json'],device=torch.device('cuda',local),
                               dtype=torch.bfloat16,weight_mode=target['mode'],use_triton=True,mmap=True)
    loaded.model.eval();before=parameter_digest(loaded.model);records=[]
    metadata=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),local_rank=local,
        device_name=torch.cuda.get_device_name(local),matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    def measure(logits,token):
        values=logits.float()
        return dict(logp=float(torch.log_softmax(values,-1)[token].item()),
                    selected_logit=float(values[token].item()),log_normalizer=float(torch.logsumexp(values,-1).item()),
                    logits_sha256=hashlib.sha256(logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    with torch.no_grad():
        for repetition in range(2):
            for item in recipe['selected']:
                # Reverse mode order on the second pass to expose order dependence.
                for mode in (MODES if repetition==0 else tuple(reversed(MODES))):
                    torch.manual_seed(974223)
                    if mode=='full-generator-forced':
                        import time
                        enc=tiktoken.get_encoding('p50k_base');stream=iter(item['generated']);trace={}
                        def forced(logits,**kwargs):
                            _sample_token(logits,**kwargs)
                            return next(stream)
                        with patch('ndm.e97._sample_token',forced):
                            _,ids,_=generate_turn(loaded,enc.decode(item['prefix']),enc,len(item['generated']),
                                                 time.monotonic()+300,sampling_trace=trace)
                        if ids!=item['generated'] or trace['prompt_token_ids']!=item['prefix']:
                            raise ValueError('full generator forced-token coverage')
                        steps=[dict(logp=p,selected_logit=None,log_normalizer=None,logits_sha256=None)
                               for p in trace['selected_logprobs']]
                        records.append(dict(id=item['id'],turn=item['turn'],mode=mode,repetition=repetition,steps=steps))
                        print('REPEATABILITY_MEASURED',rank,repetition,item['id'],item['turn'],mode,flush=True)
                        continue
                    cache=advance_e97_cache_segment(loaded,item['prefix']);steps=[]
                    for token in item['generated']:
                        logits=cache.next_logits
                        if mode=='before-step':value=measure(logits,token)
                        if mode=='sampling-shaped-after-step':
                            _sample_token(logits,temperature=1.,top_k=0,top_p=0.)
                        cache=advance_e97_cache(loaded,[token],cache)
                        if mode!='before-step':value=measure(logits,token)
                        steps.append(value)
                    records.append(dict(id=item['id'],turn=item['turn'],mode=mode,repetition=repetition,steps=steps))
                    print('REPEATABILITY_MEASURED',rank,repetition,item['id'],item['turn'],mode,flush=True)
    after=parameter_digest(loaded.model)
    publish(args.output/f'rank-{rank}-private.json',dict(rank=rank,recipe_sha256=args.recipe_sha,
            parameter_before=before,parameter_after=after,metadata=metadata,records=records))


def compare(a,b):
    if len(a['steps'])!=len(b['steps']):raise ValueError('token coverage')
    return dict(logprob_max=max(abs(x['logp']-y['logp']) for x,y in zip(a['steps'],b['steps'])),
                selected_logit_max=(max(abs(x['selected_logit']-y['selected_logit']) for x,y in zip(a['steps'],b['steps']))
                    if all(x['selected_logit'] is not None for x in a['steps']+b['steps']) else None),
                differing_full_logit_digests=(sum(x['logits_sha256']!=y['logits_sha256'] for x,y in zip(a['steps'],b['steps']))
                    if all(x['logits_sha256'] is not None for x in a['steps']+b['steps']) else None))


def aggregate(args):
    import math
    recipe=load(args);workers=[];tables=[]
    for rank in range(2):
        r=json.loads((args.output/f'rank-{rank}-private.json').read_text())
        if r['rank']!=rank or r['recipe_sha256']!=args.recipe_sha:raise ValueError('worker identity')
        if r['parameter_before']!=r['parameter_after']:raise ValueError('parameter mutation')
        table={(v['id'],v['turn'],v['mode'],v['repetition']):v for v in r['records']}
        expected={(x['id'],x['turn'],m,n) for x in recipe['selected'] for m in MODES for n in range(2)}
        if set(table)!=expected or len(table)!=len(r['records']):raise ValueError('worker coverage')
        if any(not math.isfinite(v) for item in table.values() for step in item['steps'] for k,v in step.items() if k!='logits_sha256' and v is not None):
            raise ValueError('nonfinite diagnostic value')
        workers.append(r);tables.append(table)
    if workers[0]['parameter_before']!=workers[1]['parameter_before']:raise ValueError('different worker parameters')
    comparisons=[]
    for item in recipe['selected']:
        key=(item['id'],item['turn'])
        for rank in range(2):
            for mode in MODES:
                comparisons.append(dict(kind='repeat',rank=rank,id=key[0],turn=key[1],mode=mode,
                    **compare(tables[rank][(*key,mode,0)],tables[rank][(*key,mode,1)])))
            for mode in MODES[1:]:
                comparisons.append(dict(kind='capture-shape',rank=rank,id=key[0],turn=key[1],mode=mode,
                    **compare(tables[rank][(*key,MODES[0],0)],tables[rank][(*key,mode,0)])))
        for mode in MODES:
            comparisons.append(dict(kind='worker',id=key[0],turn=key[1],mode=mode,
                **compare(tables[0][(*key,mode,0)],tables[1][(*key,mode,0)])))
    result=dict(status='diagnostic-measurements-complete',recipe_sha256=args.recipe_sha,
        parameter_sha256=workers[0]['parameter_before'],optimizer_updates=0,
        maximum_logprob_deltas={k:max(r['logprob_max'] for r in comparisons if r['kind']==k) for k in ('repeat','capture-shape','worker')},
        comparisons=comparisons,worker_metadata=[r['metadata'] for r in workers],
        rl_optimizer_ready=False,original_probability_gate='failed, unchanged',automatic_expansion=False)
    publish(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},sort_keys=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--output',type=Path,required=True)
    for mode in ('run','aggregate'):
        q=s.add_parser(mode);q.add_argument('--recipe',type=Path,required=True);q.add_argument('--recipe-sha',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
