"""Part-six scale probe: recorded checkpoint, bounded projection, token-equivalent accumulation.

Production models and pilot policy budgets are unchanged. Every point requires
an independent checkpoint exactness receipt for its context before timing.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import torch
from torch.utils.checkpoint import checkpoint
from unittest.mock import patch
from .scale_surface import SurfaceLM, SHAPES, counts, VOCAB
from .part6_checkpoint import checkpoint_block
from .part6_journal import packed_journals
from . import backward
from ..x36_tape_perf.loss import ChunkedLoss

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/part6'


def features(model, ids, arm, use_checkpoint=True, legacy=False, offload=False):
    x=model.embed(ids);gates=[]
    for block in model.blocks:
        if arm=='pure':
            if legacy:
                x=checkpoint(block,x,ids,use_reentrant=False);gate=block.cell.gate_mean
            elif use_checkpoint:
                if offload:
                    from .part6_offload import offloaded_checkpoint
                    x,gate=offloaded_checkpoint(block,x,ids,memory=True)
                else:x,gate=checkpoint_block(block,x,ids)
            else:
                x=block(x,ids);gate=block.cell.gate_mean;block.cell.gate_mean=None
            gates.append(gate)
        else:
            if offload:
                from .part6_offload import offloaded_checkpoint
                x=offloaded_checkpoint(block,x,ids,memory=False)
            else:x=checkpoint(block,x,ids,use_reentrant=True) if use_checkpoint else block(x,ids)
    return model.norm(x),gates


def main():
    p=argparse.ArgumentParser();p.add_argument('--scale',choices=SHAPES,required=True)
    p.add_argument('--arm',choices=['pure','transformer'],required=True)
    p.add_argument('--batch',type=int,choices=[64,128],required=True)
    p.add_argument('--context',type=int,choices=[512,2048],required=True)
    p.add_argument('--require-expandable',action='store_true');p.add_argument('--offload-checkpoint',action='store_true');p.add_argument('--warmups',type=int,default=3);p.add_argument('--legacy-checkpoint',action='store_true');p.add_argument('--microbatch',type=int,default=32);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--packed',action='store_true');p.add_argument('--label',default='surface');p.add_argument('--no-checkpoint',action='store_true')
    a=p.parse_args();assert a.microbatch>0 and a.batch%a.microbatch==0
    assert a.steps>=1 and a.warmups>=0
    assert not (a.legacy_checkpoint and a.no_checkpoint)
    assert not a.offload_checkpoint or not (a.legacy_checkpoint or a.no_checkpoint)
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ['GPU_LEASE_HELD']
    gate=json.loads((OUT/f'checkpoint_gate_B1_T{a.context}{"_packed" if a.packed else ""}{"_offload" if a.offload_checkpoint else ""}{"_expandable" if a.require_expandable else ""}.json').read_text());assert gate['status']=='passed'
    assert gate['helper_sha256']==hashlib.sha256((ROOT/'part6_checkpoint.py').read_bytes()).hexdigest()
    for key,value in gate['flags'].items():assert os.environ.get(key,'0')==value
    for name in ('model.py','forward.py','backward.py','kernels.py','part6_checkpoint.py','part6_journal.py','scale_surface.py'):
        assert gate['sources'][name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name
    if a.offload_checkpoint:assert gate['sources']['part6_offload.py']==hashlib.sha256((ROOT/'part6_offload.py').read_bytes()).hexdigest()
    allocator_settings=torch.cuda.memory._snapshot().get('allocator_settings',{})
    if a.require_expandable:assert allocator_settings['expandable_segments']
    production_name=f'production_T{a.context}'+('_expandable' if allocator_settings.get('expandable_segments') else '')+'.json'
    production=json.loads((OUT/production_name).read_text())
    assert production['status']=='passed' and production['pressure_gate']
    for key,value in production['flags'].items():assert os.environ.get(key,'0')==value
    torch.set_num_threads(1);torch.manual_seed(3401)
    out=OUT/a.label;out.mkdir(parents=True,exist_ok=True)
    name=f'{a.scale}_{a.arm}_B{a.batch}_T{a.context}'
    result=dict(status='failed',scale=a.scale,arm=a.arm,batch=a.batch,context=a.context,
                measurement_kind='sustained-training' if a.steps>=20 and a.warmups>=3 else 'memory-diagnostic',microbatch=a.microbatch,accumulation_steps=a.batch//a.microbatch,
                tokens_per_update=a.batch*a.context,processed_microbatch_tokens=(a.batch//a.microbatch)*a.microbatch*a.context,
                checkpoint=not a.no_checkpoint,checkpoint_input_offload=a.offload_checkpoint,allocator_expandable=bool(allocator_settings.get('expandable_segments')),allocator_settings=allocator_settings,legacy_checkpoint=a.legacy_checkpoint,warmups=a.warmups,measured_updates=a.steps,
                allocator_conf=os.environ.get('PYTORCH_ALLOC_CONF',os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),flags={k:os.environ.get(k,'0') for k in gate['flags']},
                physical_gpu=os.environ['GPU_LEASE_HELD'],python=os.sys.executable,
                protocol='bf16 dense blocks; fp32 native stores and bounded vocabulary projection; mean losses weighted by microbatch tokens, clip and AdamW once per effective batch',
                sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
    result['checkpoint_input_tensor_bytes_per_microbatch']=SHAPES[a.scale][1]*a.microbatch*a.context*(SHAPES[a.scale][0]*4+8)
    result['checkpoint_input_storage']='pinned CPU' if a.offload_checkpoint else 'GPU'
    result['checkpoint_input_transfer_bytes_per_update']=2*(a.batch//a.microbatch)*result['checkpoint_input_tensor_bytes_per_microbatch'] if a.offload_checkpoint else 0
    phases=[];journal_records=[]
    def memory(label):
        torch.cuda.synchronize()
        phases.append(dict(phase=label,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated()))
    try:
        with torch.device('meta'):reference=SurfaceLM(a.scale,'pure');capacity=counts(reference)
        del reference
        model=SurfaceLM(a.scale,a.arm,capacity['total']).cuda().train()
        result.update(parameters=counts(model),pure_parameters=capacity,ff_hidden=model.hidden)
        assert abs(result['parameters']['total']/capacity['total']-1)<.001
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        ids=torch.randint(1,VOCAB,(a.batch,a.context+1),device='cuda')
        memory('weights')
        samples=[];losses=[];replay_counts=[]
        native_backward=backward.Replay.backward
        for it in range(a.steps+a.warmups):
            calls=0
            def replay(*args):
                nonlocal calls
                calls+=1
                return native_backward(*args)
            torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
            total=0.
            for begin in range(0,a.batch,a.microbatch):
                tokens=ids[begin:begin+a.microbatch]
                with packed_journals(journal_records) if a.packed else nullcontext(), torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden,gates=features(model,tokens[:,:-1],a.arm,not a.no_checkpoint,a.legacy_checkpoint,a.offload_checkpoint)
                loss=ChunkedLoss.apply(hidden.float(),model.head.weight,tokens[:,1:],1024)
                if gates:loss=loss+.01*torch.stack(gates).mean()
                loss=loss*(a.microbatch/a.batch)
                if it==0:memory('micro_forward')
                with packed_journals(journal_records) if a.packed else nullcontext(), patch.object(backward.Replay,'backward',staticmethod(replay)):loss.backward()
                total+=float(loss.detach())
                del loss,hidden,gates
                if it==0:memory('micro_backward')
            if a.arm=='pure':assert calls==4*SHAPES[a.scale][1]*(a.batch//a.microbatch),(calls,a.scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if it==0:memory('optimizer_initialized')
            if it>=a.warmups:samples.append(elapsed);losses.append(total);replay_counts.append(calls)
            print(name,it,elapsed,total,calls,flush=True)
        result.update(status='passed',seconds=samples,losses=losses,replays_per_update=replay_counts,
                      tokens_per_second=a.batch*a.context*len(samples)/sum(samples))
    except Exception as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:
        result.update(journal_records=journal_records,packed=a.packed,memory_phases=phases,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        (out/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
