"""Actual fixed-world Gloo/DDP/DiLoCo combinations; CPU qualification only."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ndm.schedulefree_sr_candidate import ScheduleFreeSRCandidate, counter_round


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,timeout=timedelta(seconds=90))
    from train import diloco_merge
    args=SimpleNamespace(optimizer='schedulefree',diloco_outer_optimizer='avg',diloco_outer_lr=1.,
                         diloco_outer_beta=0.,diloco_export_basis='x',diloco_merge_bucket_numel=17,
                         diloco_merge_topology='global',diloco_merge_completion_barrier=1,
                         _diloco_merge_groups=None,diloco_merge_debug=0,diloco_merge_debug_ranks='0')
    for mode in ('ddp','diloco','hybrid'):
        island_size={'ddp':4,'diloco':1,'hybrid':2}[mode]
        group=None
        for start in range(0,4,island_size):
            ranks=list(range(start,start+island_size)); created=dist.new_group(ranks)
            if rank in ranks: group=created
        model=torch.nn.Linear(8,8,bias=False).bfloat16()
        model.weight.data.fill_(0.01)
        opt=ScheduleFreeSRCandidate(model.named_parameters(),lr=5e-4,betas=(0.9,0.95),seed=99,
                                     bucket_numel=19,pin_memory=False)
        opt.train()
        net=DDP(model,process_group=group) if island_size>1 else model
        for step in range(6):
            generator=torch.Generator().manual_seed(771+rank*100+step)
            x=torch.randn(4,8,generator=generator).bfloat16()
            target=torch.randn(4,8,generator=generator)
            (net(x).float()-target).square().mean().backward(); opt.step()
            replicas=[torch.empty_like(model.weight) for _ in range(island_size)]
            dist.all_gather(replicas,model.weight,group=group)
            assert all(torch.equal(replicas[0],p) for p in replicas)
            if mode!='ddp' and (step+1)%3==0:
                opt.eval(); mean_x=model.weight.detach().float().clone()
                mean_z=opt.state[model.weight]['z'].float().clone(); opt.train()
                dist.all_reduce(mean_x); mean_x.div_(4)
                dist.all_reduce(mean_z); mean_z.div_(4)
                counter=dict(seed=99,step=opt.merge_count+1,parameter_id=opt.ids[model.weight],offset=0)
                expected_x=counter_round(mean_x,stream=0x6000b,**counter)
                expected_z=counter_round(mean_z,stream=0x7000d,**counter)
                expected_y=counter_round(expected_x.float().lerp(expected_z.float(),0.1),stream=0x50009,**counter)
                moment=opt.state[model.weight]['exp_avg_sq'].clone()
                clocks=(opt.param_groups[0]['k'],opt.param_groups[0]['weight_sum'])
                diloco_merge(model,opt,args,4,None,step=step+1,merge_index=opt.merge_count+1)
                assert torch.equal(model.weight,expected_y)
                assert torch.equal(opt.state[model.weight]['z'],expected_z)
                assert torch.equal(moment,opt.state[model.weight]['exp_avg_sq'])
                assert clocks==(opt.param_groups[0]['k'],opt.param_groups[0]['weight_sum'])
                all_ranks=[torch.empty_like(model.weight) for _ in range(4)]
                dist.all_gather(all_ranks,model.weight)
                assert all(torch.equal(all_ranks[0],p) for p in all_ranks)
        # Averaged checkpoint preserves exact y and all stochastic counters.
        before=model.weight.detach().clone(); opt.eval(); checkpoint=deepcopy(opt.state_dict()); opt.train()
        resumed=ScheduleFreeSRCandidate(model.named_parameters(),lr=5e-4,betas=(0.9,0.95),seed=99,bucket_numel=31,pin_memory=False)
        resumed.load_state_dict(checkpoint); resumed.train()
        assert torch.equal(model.weight,before)
        assert resumed.merge_count==(0 if mode=='ddp' else 2)
        dist.barrier()
    dist.destroy_process_group()


def test_real_ddp_diloco_and_hybrid_with_precision_candidate(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),),nprocs=4,join=True)
