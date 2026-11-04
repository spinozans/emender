"""Receipt isolated, real-cache training steps without changing any trained arm."""
import argparse,cProfile,json,pstats,time
from pathlib import Path
import numpy as np
import torch
from . import c14_b_native as native
from . import c14_b_train as train
from .c14_b_model import StoreIntegrated
from .c14_fast_features import load_decoder

OUT=Path(__file__).with_name('receipts')/'c14_b_v2'
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=200);p.add_argument('--candidate',action='store_true');p.add_argument('--label');a=p.parse_args()
    label=a.label or ('candidate' if a.candidate else 'baseline')
    assert not (OUT/f'{label}_profile.json').exists(), 'refuse to overwrite profile receipts'
    torch.set_num_threads(1);torch.manual_seed(3401)
    source,shards,gate=train.preflight(check_implementation=not a.candidate)
    raw=np.memmap(source,dtype=np.uint8,mode='r');rows=iter(train.verified_batches(raw))
    r=shards[0];data=(train.CACHE/r['file']).read_bytes()
    identities,lengths,values,contexts,_=train.oracle_shard(data,r['windows'],load_decoder())
    model=StoreIntegrated().cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    rng=torch.Generator(device='cuda').manual_seed(3401)
    times={};counts={}
    hashes={name:train.sha(train.ROOT/name) for name in (*train.CODE,
        "c14_b_batched_replay.py","c14_b_replay_graph.py","c14_b_profile.py")}
    def wrap(name,module=native):
        original=getattr(module,name)
        def measured(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                torch.cuda.synchronize();times[name]=times.get(name,0)+time.perf_counter()-start;counts[name]=counts.get(name,0)+1
        setattr(module,name,measured)
    for name in ('eager','native_forward','native_reports'):wrap(name)
    if a.candidate:
        from . import c14_b_replay_graph
        wrap('backward',c14_b_replay_graph)
    profile=cProfile.Profile();curve=[]
    for i in range(a.steps):
        begin=(i*16)%len(values)
        if begin==0:rows=iter(train.verified_batches(raw))
        batch=next(rows);xx,yy,mask=train.tensors(batch)
        ff=torch.as_tensor(values[begin:begin+16],device='cuda').clone()
        ff*= (torch.rand((16,128),device='cuda',generator=rng)>=.5)[...,None]
        ll=torch.as_tensor(contexts[begin:begin+16],device='cuda')
        torch.cuda.synchronize();start=time.perf_counter();profile.enable()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lp,_=model(xx,ff,ll,windows=[(d,o) for d,o,_ in batch],valid_lengths=lengths[begin:begin+16].tolist())
            loss=-lp.gather(-1,yy[...,None])[...,0][mask].mean()
        torch.cuda.synchronize();forward=time.perf_counter()-start;at=time.perf_counter()
        loss.backward();torch.cuda.synchronize();backward=time.perf_counter()-at;at=time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        update=time.perf_counter()-at;profile.disable()
        curve.append(dict(step=i+1,forward=forward,backward=backward,update=update,loss=float(loss)))
        result=dict(status='running' if i+1<a.steps else 'completed',steps=i+1,shape=[16,128],seed=3401,
            gpu=__import__('os').environ.get('GPU_LEASE_HELD'),source_sha256=hashes,component_seconds=times,calls=counts,curve=curve,
            instrumentation='cProfile + synchronizing wall-clock wrappers; totals overlap checkpoint recomputation; no checkpoint/telemetry journal in measured region')
        (OUT/f'{label}_profile.json').write_text(json.dumps(result,indent=2)+'\n')
        profile.dump_stats(str(OUT/f'{label}_profile.pstats'))
        with (OUT/f'{label}_profile.txt').open('w') as f:pstats.Stats(profile,stream=f).sort_stats('cumtime').print_stats(70)
        print(json.dumps(curve[-1]),flush=True)
if __name__=='__main__':main()
