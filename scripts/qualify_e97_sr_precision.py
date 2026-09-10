#!/usr/bin/env python3
"""Local CUDA/NCCL precision qualification; small models, no SFT or Frontier claim.

Run produce then resume as separate torchrun jobs. Every replica shares SR keys;
independent islands diverge only through their actual local gradients/moments.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import gc
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import schedulefree
from ndm.e97_atomic import publish_bytes_no_replace
from ndm.schedulefree_offload import CPUOffloadAdamWScheduleFree
from ndm.schedulefree_sr_candidate import ScheduleFreeSRCandidate, counter_round
from train import diloco_merge


def publish(path, value):
    publish_bytes_no_replace(path,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)


def digest_state(model,opt):
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            h.update(str((value.dtype,tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):
                h.update(str(key).encode()); visit(value[key])
        elif isinstance(value,(tuple,list)):
            for item in value: visit(item)
        else: h.update(repr(value).encode())
    visit(model.state_dict()); visit(opt.state_dict())
    return h.hexdigest()


def numerical_and_memory(root):
    torch.cuda.synchronize()
    size=65536
    p=torch.nn.Parameter(torch.ones(size,device='cuda',dtype=torch.bfloat16))
    nearest=torch.nn.Parameter(p.detach().clone())
    reference=torch.nn.Parameter(p.detach().float())
    sr=ScheduleFreeSRCandidate([('norm',p)],lr=2e-6,betas=(0.9,0.95),bucket_numel=32768)
    old=CPUOffloadAdamWScheduleFree([nearest],lr=2e-6,betas=(0.9,0.95),bucket_numel=32768)
    ref=schedulefree.AdamWScheduleFree([reference],lr=2e-6,betas=(0.9,0.95),foreach=False)
    for optimizer in (sr,old,ref): optimizer.train()
    start=time.monotonic()
    stalls=proposals=0
    for _ in range(64):
        for parameter in (p,nearest,reference): parameter.grad=torch.ones_like(parameter)
        sr.step(); old.step(); ref.step()
        stalls+=sr.last_step_stats['y_rne_would_stall']
        proposals+=sr.last_step_stats['y_nonzero_proposals']
    torch.cuda.synchronize()
    result={'sr_y_mean':float(p.detach().double().mean()),'reference_y_mean':float(reference.detach().double().mean()),
            'legacy_y_mean':float(nearest.detach().double().mean()),
            'sr_changed_coordinates':int((p!=1).sum()),'legacy_changed_coordinates':int((nearest!=1).sum()),
            'y_nonzero_proposals':proposals,'y_rne_would_stall':stalls,'elapsed_seconds':time.monotonic()-start}
    assert result['legacy_changed_coordinates']==0
    assert result['sr_changed_coordinates']>0
    assert abs(result['sr_y_mean']-result['reference_y_mean'])<2e-5
    # Actual CPU/CUDA counter-store equality and boundary independence.
    values=torch.linspace(-0.02003,0.02007,65536)
    kwargs=dict(seed=99,step=71,parameter_id=123,stream=0x10001)
    expected=counter_round(values,offset=0,**kwargs)
    actual=torch.cat([counter_round(values[i:i+997].cuda(),offset=i,**kwargs).cpu() for i in range(0,len(values),997)])
    assert torch.equal(expected,actual)
    result['counter_cpu_cuda_and_bucket_parity']=True
    del p,nearest,reference,sr,old,ref,parameter,optimizer
    gc.collect(); torch.cuda.empty_cache()
    memory=[]
    for count in (1048576,16777216):
        p=torch.nn.Parameter(torch.full((count,),0.01,device='cuda',dtype=torch.bfloat16))
        opt=ScheduleFreeSRCandidate([('large',p)],lr=2e-6,betas=(0.9,0.95),bucket_numel=262144)
        opt.train(); opt.initialize_state_(); p.grad=torch.ones_like(p)
        torch.cuda.synchronize(); baseline=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        start=time.monotonic(); opt.step(); torch.cuda.synchronize()
        scratch=torch.cuda.max_memory_allocated()-baseline
        assert opt.offloaded_state_bytes()==4*count
        assert all(v.dtype==torch.bfloat16 and v.device.type=='cpu' and v.is_pinned() for state in opt.state.values() for v in state.values())
        memory.append({'coordinates':count,'bucket_numel':262144,'scratch_peak_bytes':scratch,
                       'optimizer_state_bytes':opt.offloaded_state_bytes(),'step_seconds':time.monotonic()-start})
        del p,opt; gc.collect(); torch.cuda.empty_cache()
    assert memory[1]['scratch_peak_bytes']<=memory[0]['scratch_peak_bytes']+(32<<20)
    assert max(m['scratch_peak_bytes'] for m in memory)<256<<20
    publish(root/'cuda-arithmetic-memory.json',{'schema':'emender-sr-cuda-arithmetic-memory-v1',
            'status':'passed','arithmetic':result,'memory':memory,
            'scope':'constant-gradient synthetic update and bounded scratch only; no E97 convergence claim'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['produce','resume'],required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    assert world>=4 and world%2==0
    local_rank=int(os.environ['LOCAL_RANK'])
    assert 0<=local_rank<torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    assert torch.cuda.current_device()==local_rank
    device_identity={'local_rank':local_rank,'current_cuda_device':torch.cuda.current_device(),
                     'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                     'device_uuid':str(getattr(torch.cuda.get_device_properties(local_rank),'uuid','unavailable'))}
    print('SR_CUDA_DEVICE '+json.dumps({'rank':rank,**device_identity}),flush=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl',timeout=timedelta(seconds=900))
    root=args.output_root
    merge_args=SimpleNamespace(optimizer='schedulefree',diloco_outer_optimizer='avg',diloco_outer_lr=1.,
       diloco_outer_beta=0.,diloco_export_basis='x',diloco_merge_bucket_numel=73,
       diloco_merge_topology='global',diloco_merge_completion_barrier=1,
       _diloco_merge_groups=None,diloco_merge_debug=0,diloco_merge_debug_ranks='0')
    reports=[]
    for mode in ('ddp','diloco','hybrid'):
        island_size={'ddp':world,'diloco':1,'hybrid':world//2}[mode]
        group=None
        for start in range(0,world,island_size):
            ranks=list(range(start,start+island_size)); created=dist.new_group(ranks)
            if rank in ranks: group=created
        torch.manual_seed(18)
        model=torch.nn.Linear(16,16,bias=False,device='cuda',dtype=torch.bfloat16)
        model.weight.data.fill_(0.01)
        opt=ScheduleFreeSRCandidate(model.named_parameters(),lr=3e-4,betas=(0.9,0.95),seed=99,
                                    bucket_numel=59 if args.phase=='produce' else 113)
        checkpoint_path=root/f'{mode}-rank-{rank:02d}.pt'
        if args.phase=='resume':
            payload=checkpoint_path.read_bytes()
            evidence=json.loads((root/f'{mode}-rank-{rank:02d}-expected.json').read_text())
            assert hashlib.sha256(payload).hexdigest()==evidence['checkpoint_sha256']
            saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
            assert saved['schema']=='emender-sr-toy-checkpoint-v1' and saved['world_size']==world and saved['rank']==rank and saved['step']==3
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        opt.train()
        net=DDP(model,device_ids=[local_rank],process_group=group) if island_size>1 else model
        for step in range(0 if args.phase=='produce' else 3,6):
            generator=torch.Generator(device='cuda').manual_seed(881+rank*100+step)
            x=torch.randn(4,16,device='cuda',generator=generator).bfloat16()
            target=torch.randn(4,16,device='cuda',generator=generator)
            (net(x).float()-target).square().mean().backward(); opt.step()
            replicas=[torch.empty_like(model.weight) for _ in range(island_size)]
            dist.all_gather(replicas,model.weight,group=group)
            assert all(torch.equal(replicas[0],value) for value in replicas)
            if mode!='ddp' and (step+1)%3==0:
                opt.eval(); mean_x=model.weight.detach().float().clone()
                mean_z=opt.state[model.weight]['z'].cuda().float(); opt.train()
                dist.all_reduce(mean_x); mean_x.div_(world)
                dist.all_reduce(mean_z); mean_z.div_(world)
                counter=dict(seed=99,step=opt.merge_count+1,parameter_id=opt.ids[model.weight],offset=0)
                expected_x=counter_round(mean_x,stream=0x6000b,**counter)
                expected_z=counter_round(mean_z,stream=0x7000d,**counter)
                expected_y=counter_round(expected_x.float().lerp(expected_z.float(),0.1),stream=0x50009,**counter)
                moment=opt.state[model.weight]['exp_avg_sq'].clone()
                clock=opt.param_groups[0]['k']
                diloco_merge(model,opt,merge_args,world,None,step=step+1,merge_index=opt.merge_count+1)
                assert torch.equal(model.weight,expected_y)
                assert torch.equal(opt.state[model.weight]['z'],expected_z.cpu())
                assert torch.equal(opt.state[model.weight]['exp_avg_sq'],moment)
                assert opt.param_groups[0]['k']==clock
                all_ranks=[torch.empty_like(model.weight) for _ in range(world)]
                dist.all_gather(all_ranks,model.weight)
                assert all(torch.equal(all_ranks[0],value) for value in all_ranks)
            if args.phase=='produce' and step==2:
                live_y=model.weight.detach().clone(); opt.eval()
                buffer=io.BytesIO()
                torch.save({'schema':'emender-sr-toy-checkpoint-v1','model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            'optimizer':opt.state_dict(),'world_size':world,'rank':rank,'step':3},buffer)
                publish_bytes_no_replace(checkpoint_path,buffer.getvalue(),mode=0o600)
                opt.train(); assert torch.equal(model.weight,live_y)
        final_digest=digest_state(model,opt)
        if args.phase=='produce':
            publish(root/f'{mode}-rank-{rank:02d}-expected.json',{'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    'final_state_sha256':final_digest})
        else:
            assert final_digest==evidence['final_state_sha256'], f'cold-resume state differs for {mode} rank {rank}'
        reports.append({'mode':mode,'island_size':island_size,'final_state_sha256':final_digest,
                        'merge_count':opt.merge_count,'status':'passed'})
        dist.barrier()
        del net,model,opt; gc.collect(); torch.cuda.empty_cache()
    if args.phase=='produce' and rank==0:
        numerical_and_memory(root)
    dist.barrier()
    publish(root/f'{args.phase}-rank-{rank:02d}.json',{'schema':'emender-sr-cuda-distributed-v1','status':'passed',
            'phase':args.phase,'rank':rank,'world_size':world,'reports':reports,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'device_identity':device_identity,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'small-model CUDA/NCCL DDP, avg DiLoCo and hybrid; cold resume with different buckets; not Frontier or full E97 training'})
    print(f'SR_CUDA_PHASE_PASSED phase={args.phase} rank={rank}',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':
    main()
