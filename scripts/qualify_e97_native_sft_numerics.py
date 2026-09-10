#!/usr/bin/env python3
"""No-update, full-model native-trajectory gradient/chunk qualification.

Uses the trainer's explicit precision policy. This does not admit the source,
construct a training optimizer, execute source tools, or produce a checkpoint.
Parent train-y restoration may use a temporary Schedule-Free loader object.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import torch
import tiktoken
from ndm.e97 import load_e97_checkpoint
from scripts.qualify_e97_response_gradients import (
    aligned_length, comparison, parameter_digest, publish, sha,
)
from scripts.train_e97_4b_pi_sft import configure_precision

SCHEMA='emender-e97-native-sft-numerics-v1'
BINS=((12289,16384),(60000,65536))
INDEX=struct.Struct('<QQQB7x')


def select_panel(rows):
    selected=[]
    for low,high in BINS:
        candidates=[row for row in rows if row['split']==0 and low<=row['tokens']<=high
                    and row['assistant_units']>=3]
        if not candidates:raise ValueError('missing predeclared trajectory-length bin')
        selected.append(min(candidates,key=lambda row:hashlib.sha256(
            ('native-numerics-v1\0'+row['trajectory_identity']).encode()).hexdigest()))
    return selected


def record_arrays(root,row):
    with (root/'records.idx').open('rb') as stream:
        stream.seek(INDEX.size*row['record_index']);entry=stream.read(INDEX.size)
    if INDEX.unpack(entry)!=(row['offset'],row['tokens'],row['targets'],row['split']):
        raise ValueError('record index disagreement')
    arrays=[]
    for name,dtype,width in [('tokens.bin','<u4',4),('loss_mask.bin','u1',1)]:
        with (root/name).open('rb') as stream:
            stream.seek(width*row['offset']);raw=stream.read(width*row['tokens'])
        if len(raw)!=width*row['tokens']:raise ValueError('short record')
        arrays.append(np.frombuffer(raw,dtype=dtype).copy())
    tokens,mask=arrays
    if not np.isin(mask,[0,1]).all() or int(mask.sum())!=row['targets']:
        raise ValueError('record mask disagreement')
    starts=np.flatnonzero((mask==1)&np.concatenate(([True],mask[:-1]==0)))
    if len(starts)!=row['assistant_units'] or len(starts)<3 or starts[0]==0:
        raise ValueError('assistant boundary count')
    if not np.all(tokens[starts]==32750):raise ValueError('non-Analysis opening')
    selected=[int(starts[i]) for i in (0,len(starts)//2,len(starts)-1)]
    return tokens,selected


def inputs(tokens,positions,device):
    n=len(tokens);padded=aligned_length(n-1,16)+1
    if not 1<n<=65536 or len(set(positions))!=3 or any(not 0<p<n for p in positions):
        raise ValueError('invalid native diagnostic positions')
    ids=torch.zeros(1,padded,dtype=torch.long,device=device)
    ids[0,:n]=torch.as_tensor(np.asarray(tokens,dtype=np.int64),device=device)
    valid=torch.zeros_like(ids,dtype=torch.bool);valid[:,:n]=True
    reset=torch.zeros_like(valid);reset[:,0]=True
    mask=torch.zeros(1,padded-1,dtype=torch.bool,device=device)
    mask[0,[p-1 for p in positions]]=True
    return ids,valid,reset,mask


def freeze(args):
    for digest in (args.authority_sha256,args.checkpoint_sha256,args.args_sha256):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('expected a full lowercase SHA-256 digest')
    root=args.authority.resolve();manifest=root/'manifest.json'
    if sha(manifest)!=args.authority_sha256:raise ValueError('authority identity')
    data=json.loads(manifest.read_text())
    if data['schema']!='emender-open-swe-source-native-candidate-v1' or data['training_eligible'] is not False:
        raise ValueError('expected unadmitted source-native diagnostic authority')
    identities={str(manifest):args.authority_sha256}
    for name in ('tokens.bin','loss_mask.bin','records.idx','records.jsonl'):
        entry=data['outputs'][name];path=root/name
        if entry['path']!=name or sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:
            raise ValueError('authority payload identity')
        identities[str(path)]=entry['sha256']
    selected=select_panel([json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()])
    encoder=tiktoken.get_encoding('p50k_base')
    for row in selected:
        tokens,positions=record_arrays(root,row)
        if hashlib.sha256(encoder.decode(tokens.tolist()).encode()).hexdigest()!=row['sha256']:
            raise ValueError('record text identity')
        row['opening_positions']=positions
    for path,digest in ((args.checkpoint,args.checkpoint_sha256),(args.args_json,args.args_sha256)):
        if sha(path)!=digest:raise ValueError('model input identity: '+str(path))
        identities[str(path.resolve())]=digest
    recipe={'schema':SCHEMA,'scope':'no-update numerical qualification; no source admission or tool execution',
            'authority':str(root),'checkpoint':str(args.checkpoint.resolve()),'args_json':str(args.args_json.resolve()),
            'input_sha256':identities,'panel':selected,'length_bins':BINS,
            'parameter_relative_l2_max':.05,'mean_loss_delta_max':.05,'ce_nll_delta_max':1e-4,
            'configuration':dict(optimizer_precision='bf16-sr-candidate',sr_seed=927413,
                offload_schedulefree_state=True,loss_logits_fp32=True,loss_chunk_size=128,
                checkpoint_loss_chunks=True,disable_bf16_reduced_precision_reduction=True,
                gradient_checkpoint_group_size=3,mlp_checkpoint_chunk_size=4096,
                lr=0.,weight_decay=.01,warmup_steps=0),
            'learning_rate_note':'zero is a no-update assay placeholder, not a selected training LR',
            'cases':[{'name':'unchunked-mlp-reference','mlp_chunk':0},
                     {'name':'trainer-mlp4096-candidate','mlp_chunk':4096}],
            'training_optimizer_constructed':False,'optimizer_steps':0,'training_eligible':False}
    publish(args.output,recipe)


def measure(model,row,arrays,policy,case,baseline):
    tokens,valid,reset,mask=arrays
    model.zero_grad(set_to_none=True)
    options=SimpleNamespace(**{**policy,'mlp_checkpoint_chunk_size':case['mlp_chunk']})
    effective=configure_precision(model,options)
    model.gradient_checkpointing=True;model.gradient_checkpoint_group_size=options.gradient_checkpoint_group_size
    modules=[m for m in model.modules() if hasattr(m,'checkpoint_chunk_size')]
    if len(modules)!=18:raise ValueError('expected 18 SwiGLU modules')
    for module in modules:module.checkpoint_chunk_size=case['mlp_chunk']
    embedding=[];head=[];offset=0
    def embedding_hook(_module,_inputs,out):
        def backward(gradient):
            last=row['opening_positions'][-1]
            embedding.append({'future_nonzero':int(torch.count_nonzero(gradient[:,last:])),
                              'prefix_nonzero':int(torch.count_nonzero(gradient[:,:last]))})
        out.register_hook(backward)
    def head_hook(_module,_inputs,out):
        nonlocal offset
        for p in row['opening_positions']:
            if offset<=p-1<offset+out.shape[1]:
                logits=out[0,p-1-offset].detach().float()
                head.append({'position':p,'target_nll':float(-logits.log_softmax(-1)[32750])})
        offset+=out.shape[1]
    hooks=[model.embedding.register_forward_hook(embedding_hook),model.lm_head.register_forward_hook(head_hook)]
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(tokens,return_loss=True,loss_mask=mask,valid_mask=valid,
                       reset_before=reset,loss_reduction='mean')
        loss.backward();torch.cuda.synchronize()
    finally:
        for hook in hooks:hook.remove()
    if not torch.isfinite(loss) or len(embedding)!=1 or len(head)!=3:
        raise ValueError('incomplete gradient/head observation')
    metrics={};saved={} if baseline is None else baseline
    for name,p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise ValueError('missing/nonfinite gradient: '+name)
        if baseline is None:saved[name]=p.grad.detach().cpu().clone()
        else:metrics[name]=comparison(p.grad,baseline[name])
    return {'case':case,'effective_policy':effective,'trajectory_identity':row['trajectory_identity'],
            'record_tokens':row['tokens'],'padded_prediction_tokens':tokens.shape[1]-1,
            'supervised_openings':row['opening_positions'],'loss':float(loss.detach()),
            'ce_nll_delta':abs(float(loss.detach())-sum(x['target_nll'] for x in head)/3),
            'head':head,'embedding_gradient':embedding[0],'parameter_gradients':metrics,
            'peak_hbm_allocated':torch.cuda.max_memory_allocated(),
            'peak_hbm_reserved':torch.cuda.max_memory_reserved()},saved


def run(args):
    if sha(args.recipe)!=args.recipe_sha256:raise ValueError('recipe identity')
    recipe=json.loads(args.recipe.read_text())
    if recipe['schema']!=SCHEMA or recipe['optimizer_steps']!=0 or recipe['training_eligible'] is not False:
        raise ValueError('not a no-update diagnostic recipe')
    for path,digest in recipe['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('input identity: '+path)
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    loaded=load_e97_checkpoint(recipe['checkpoint'],args_json=recipe['args_json'],device=device,
                              dtype=torch.bfloat16,weight_mode='train',use_triton=True,mmap=True)
    model=loaded.model.train()
    if sum(p.numel() for p in model.parameters())!=4045972080 or any(p.dtype!=torch.bfloat16 for p in model.parameters()):
        raise ValueError('not the BF16 E97 4B model')
    initial=parameter_digest(model);reports=[]
    for row in recipe['panel']:
        raw,positions=record_arrays(Path(recipe['authority']),row)
        if positions!=row['opening_positions']:raise ValueError('opening identity')
        arrays=inputs(raw,positions,device);baseline=None;reference=None
        for case in recipe['cases']:
            report,baseline=measure(model,row,arrays,recipe['configuration'],case,baseline)
            passed=(report['embedding_gradient']['future_nonzero']==0 and
                    report['embedding_gradient']['prefix_nonzero']>0 and
                    report['ce_nll_delta']<=recipe['ce_nll_delta_max'])
            if reference is not None:
                report['mean_loss_delta']=abs(report['loss']-reference['loss'])
                passed=passed and report['mean_loss_delta']<=recipe['mean_loss_delta_max']
                passed=passed and all(x['relative_l2'] is not None and x['relative_l2']<=recipe['parameter_relative_l2_max']
                                      for x in report['parameter_gradients'].values())
            report['passed']=bool(passed)
            publish(args.output/f"record-{row['record_index']}-{case['name']}.json",report)
            reports.append(report)
            if not passed:raise ValueError('native numerical gate failed; retained per-case measurements')
            reference=report
            print('NATIVE_NUMERICS_CASE_PASS '+str(row['record_index'])+' '+case['name'],flush=True)
        model.zero_grad(set_to_none=True);del baseline,arrays;gc.collect();torch.cuda.empty_cache()
    if parameter_digest(model)!=initial:raise ValueError('parameters changed in no-update assay')
    for path,digest in recipe['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('input changed during assay: '+path)
    publish(args.output/'summary.json',{'schema':SCHEMA,'status':'passed','recipe_sha256':args.recipe_sha256,
        'parameter_digest_before_after':initial,'cases':len(reports),'optimizer_steps':0,
        'training_optimizer_constructed':False,'training_eligible':False,'full_world_restart_qualified':False,
        'scope':recipe['scope'],'torch':torch.__version__,'cuda':torch.version.cuda,
        'max_parameter_relative_l2':max(x['relative_l2'] for r in reports for x in r['parameter_gradients'].values()),
        'max_mean_loss_delta':max(r.get('mean_loss_delta',0) for r in reports),
        'peak_hbm_allocated':max(r['peak_hbm_allocated'] for r in reports)})


def main():
    parser=argparse.ArgumentParser();subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('freeze')
    for name in ('authority','checkpoint','args-json','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('authority-sha256','checkpoint-sha256','args-sha256'):p.add_argument('--'+name,required=True)
    p=subs.add_parser('run');p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--recipe-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args)
    else:
        # Create outside the handler: an existing/concurrently claimed output
        # must never receive a failure receipt from this attempt.
        args.output.mkdir(parents=True,exist_ok=False)
        try:run(args)
        except Exception as error:
            if args.output.is_dir():
                publish(args.output/'failure.json',{'schema':SCHEMA,'status':'failed','type':type(error).__name__,
                    'error':str(error),'recipe_sha256':args.recipe_sha256,'optimizer_steps':0,'training_eligible':False})
            raise


if __name__=='__main__':main()
