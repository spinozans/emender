#!/usr/bin/env python3
"""Eight-GPU E97 update/checkpoint fixture: produce and resume in fresh jobs.

Authored numerical token packs only: no Open-SWE records, source commands,
source admission, LR selection, sustained training, or model promotion.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ndm.e97 import load_e97_checkpoint
from scripts.qualify_e97_response_gradients import publish, sha
from scripts.train_e97_4b_pi_sft import (
    SCHEMA as SFT_SCHEMA, atomic_save, build_optimizer, configure_precision,
    load_resume_optimizer, packed_objective, validate_precision_world,
)

SCHEMA='emender-e97-full-sft-restart-qualification-v1'
POLICY=dict(optimizer_precision='bf16-sr-candidate',sr_seed=927413,
    offload_schedulefree_state=True,schedulefree_offload_pin_memory=1,
    schedulefree_offload_release_gradients=1,schedulefree_offload_bucket_numel=262144,
    loss_logits_fp32=True,loss_chunk_size=128,checkpoint_loss_chunks=True,
    disable_bf16_reduced_precision_reduction=True,disable_diloco_merge=True,
    gradient_checkpoint_group_size=3,mlp_checkpoint_chunk_size=4096,
    lr=5e-5,weight_decay=.01,warmup_steps=0)


def fixture(rank,step,device):
    """Two 129-token documents, a non-16-aligned reset, and padded tail."""
    if type(rank) is not int or not 0<=rank<8 or type(step) is not int or not 1<=step<=3:
        raise ValueError('fixture rank/step outside frozen domain')
    alphabet=(15496,11,995,0,198,464,2068,7586)
    values=[alphabet[(i+rank+step*3+(i//129)*2)%len(alphabet)] for i in range(258)]
    tokens=torch.zeros(1,513,dtype=torch.long,device=device)
    tokens[0,:258]=torch.tensor(values,device=device)
    valid=torch.zeros_like(tokens,dtype=torch.bool);valid[:,:258]=True
    reset=torch.zeros_like(valid);reset[0,[0,129]]=True
    loss=torch.zeros(1,512,dtype=torch.bool);loss[0,:128]=True;loss[0,129:257]=True
    return tokens,loss.to(device),valid,reset


def state_digest(model,optimizer):
    """Exact model/optimizer identity with bounded BF16 host transfer slices."""
    digest=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            digest.update(str((value.dtype,tuple(value.shape))).encode())
            flat=value.detach().reshape(-1)
            for start in range(0,flat.numel(),1048576):
                digest.update(flat[start:start+1048576].cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):digest.update(str(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        else:digest.update(repr(value).encode())
    visit(model.state_dict());visit(optimizer.state_dict())
    return digest.hexdigest()


def consensus(digest):
    peers=[None]*dist.get_world_size();dist.all_gather_object(peers,digest)
    if len(set(peers))!=1:raise ValueError('DDP replica state disagreement')
    return digest


def step(model,core,opt,rank,index,device):
    opt.zero_grad(set_to_none=True)
    tokens,mask,valid,reset=fixture(rank,index,device)
    local_input_tokens=int(valid.sum())
    if local_input_tokens!=258:raise ValueError('fixture input accounting')
    count=torch.tensor(int(mask.sum()),dtype=torch.int64,device=device)
    dist.all_reduce(count)
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    loss,observed=packed_objective(model,tokens,mask,valid,reset,count,8)
    if observed!=256 or int(count)!=2048:raise ValueError('fixture target accounting')
    norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.)
    if not torch.isfinite(norm):raise ValueError('nonfinite gradient norm')
    opt.step();opt.assert_state_offloaded();torch.cuda.synchronize()
    stats=dict(opt.last_step_stats)
    if stats['y_sr_changed']<=0 or stats['y_nonzero_proposals']<=0:
        raise ValueError('no effective model update')
    if any(p.dtype!=torch.bfloat16 for p in core.parameters()):raise ValueError('parameter dtype changed')
    return {'step':index,'rank':rank,'local_loss_sum':float(loss),'global_targets':int(count),
            'global_input_tokens':local_input_tokens*8,
            'grad_norm':float(norm),'seconds':time.monotonic()-begin,'optimizer':stats,
            'peak_hbm_allocated':torch.cuda.max_memory_allocated(),
            'peak_hbm_reserved':torch.cuda.max_memory_reserved()}


def validate_recipe(recipe):
    if (recipe.get('schema')!=SCHEMA or recipe.get('policy')!=POLICY
            or recipe.get('world_size')!=8 or recipe.get('saved_update')!=2
            or recipe.get('continuation_update')!=3 or recipe.get('training_eligible') is not False
            or recipe.get('continuation_acceptance')!='bitwise-full-model-and-optimizer-state'):
        raise ValueError('unrecognized frozen update/restart recipe')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('produce','resume'),required=True)
    parser.add_argument('--recipe',type=Path,required=True);parser.add_argument('--recipe-sha256',required=True)
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    if sha(args.recipe)!=args.recipe_sha256:raise ValueError('recipe identity')
    recipe=json.loads(args.recipe.read_text());validate_recipe(recipe)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('requires exactly eight ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=300))
    phase_root=args.root/args.phase
    if rank==0:
        phase_root.mkdir(parents=True,exist_ok=False)
        for path,digest in recipe['input_sha256'].items():
            if sha(path)!=digest:raise ValueError('input identity: '+path)
    dist.barrier()
    if args.phase=='produce':path=Path(recipe['checkpoint']);mode='train'
    else:
        expected=json.loads((args.root/'produce'/'terminal.json').read_text())
        if (expected['schema']!=SCHEMA or expected['status']!='passed' or expected['phase']!='produce'
                or expected['recipe_sha256']!=args.recipe_sha256):
            raise ValueError('missing passing exact-recipe predecessor')
        committed=json.loads((args.root/'produce'/'committed.json').read_text())
        if committed['recipe_sha256']!=args.recipe_sha256:raise ValueError('committed recipe identity')
        path=args.root/'checkpoints'/'latest.pt';mode='saved'
        if path.resolve()!=Path(committed['checkpoint']).resolve():raise ValueError('atomic checkpoint pointer mismatch')
        if rank==0 and sha(path)!=committed['checkpoint_sha256']:raise ValueError('checkpoint bytes mismatch')
        dist.barrier()
    loaded=load_e97_checkpoint(path,args_json=recipe['args_json'],device=device,dtype=torch.bfloat16,
                              weight_mode=mode,use_triton=True,mmap=True)
    core=loaded.model.train()
    if sum(p.numel() for p in core.parameters())!=4045972080:raise ValueError('not E97 4B')
    if any(isinstance(m,torch.nn.Dropout) and m.p for m in core.modules()):
        raise ValueError('fixture requires dropout-free continuation')
    options=SimpleNamespace(**POLICY);validate_precision_world(options,8)
    identity=configure_precision(core,options)
    core.gradient_checkpointing=True;core.gradient_checkpoint_group_size=3
    modules=[m for m in core.modules() if hasattr(m,'checkpoint_chunk_size')]
    if len(modules)!=18:raise ValueError('expected 18 MLPs')
    for module in modules:module.checkpoint_chunk_size=4096
    model=DDP(core,device_ids=[local],output_device=local,gradient_as_bucket_view=True,
              find_unused_parameters=False)
    opt=build_optimizer(core.parameters(),options,named_parameters=core.named_parameters())
    if args.phase=='produce':opt.initialize_state_()
    else:
        clocks=load_resume_optimizer(path,opt,{'sft_precision':identity,
            'qualification_recipe_sha256':args.recipe_sha256,'qualification_only':True})
        if clocks!={'updates':2,'total_tokens':4128,'assistant_target_tokens':4096}:
            raise ValueError('restored token/update clocks')
    opt.train();opt.assert_state_offloaded()
    reports=[]
    if args.phase=='produce':
        for index in (1,2):
            report=step(model,core,opt,rank,index,device);reports.append(report)
            publish(phase_root/f'rank-{rank}-step-{index}.json',report)
            if rank==0:print('FULL_SFT_UPDATE '+str(index),flush=True)
        committed_digest=consensus(state_digest(core,opt))
        dist.barrier()
        if rank==0:
            opt.eval()
            checkpoint=args.root/'checkpoints'/'checkpoint_numerical_u000002.pt'
            payload={'schema':SFT_SCHEMA,'qualification_only':True,'training_eligible':False,
                'qualification_recipe_sha256':args.recipe_sha256,'sft_precision':identity,
                'model_state_dict':core.state_dict(),'optimizer_state_dict':opt.state_dict(),
                'sft_updates':2,'sft_total_tokens':4128,'assistant_target_tokens':4096,
                'weight_mode':'saved-eval-x'}
            atomic_save(checkpoint,payload);del payload
            checkpoint.chmod(0o400);checkpoint.parent.chmod(0o500)
            publish(phase_root/'committed.json',{'recipe_sha256':args.recipe_sha256,
                'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':sha(checkpoint),
                'train_state_digest':committed_digest,'qualification_only':True,'training_eligible':False})
            opt.train()
        dist.barrier()
        if consensus(state_digest(core,opt))!=committed_digest:
            raise ValueError('checkpoint basis transition altered live training state')
    else:
        restored=consensus(state_digest(core,opt))
        if restored!=committed['train_state_digest']:raise ValueError('fresh restore differs from committed live state')
        publish(phase_root/f'rank-{rank}-restore.json',{'rank':rank,'exact':True,'state_digest':restored})
    report=step(model,core,opt,rank,3,device);reports.append(report)
    publish(phase_root/f'rank-{rank}-step-3.json',report)
    terminal=consensus(state_digest(core,opt))
    if args.phase=='resume':
        if expected['recipe_sha256']!=args.recipe_sha256 or terminal!=expected['state_digest']:
            raise ValueError('fresh continuation is not bitwise identical; no tolerance fallback')
    if rank==0:
        if args.phase=='resume' and sha(args.root/'checkpoints'/'latest.pt')!=committed['checkpoint_sha256']:
            raise ValueError('checkpoint changed during resume')
        for path,digest in recipe['input_sha256'].items():
            if sha(path)!=digest:raise ValueError('input changed: '+path)
        publish(phase_root/'terminal.json',{'schema':SCHEMA,'status':'passed','phase':args.phase,
            'recipe_sha256':args.recipe_sha256,'state_digest':terminal,'world_size':8,
            'fixture':'authored two-document numeric pack; not Open-SWE or admitted SFT data',
            'qualification_only':True,'training_eligible':False,'sustained_training':False,
            'torch':torch.__version__,'cuda':torch.version.cuda})
        print('FULL_SFT_'+args.phase.upper()+'_PASS',flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
