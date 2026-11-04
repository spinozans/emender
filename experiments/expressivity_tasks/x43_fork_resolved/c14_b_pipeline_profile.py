"""200 real sequential steps including cache read/decode, assembly and telemetry.

No profiling-window repetition, registered training budget or checkpoint. This
is a separate fresh model and its optimizer state is discarded after measuring.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .c14_b_v2 import configure
from . import c14_b_train as train
from .c14_fast_features import load_decoder


def main():
    configure();source,shards,gate=train.preflight()
    torch.set_num_threads(1);torch.manual_seed(3401)
    model=train.StoreIntegrated().cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    rng=torch.Generator(device='cuda').manual_seed(3401);decoder=load_decoder()
    rows=iter(train.verified_batches(np.memmap(source,dtype=np.uint8,mode='r')))
    totals={};curve=[];telemetry={};steps=0
    def at():torch.cuda.synchronize();return time.perf_counter()
    for r in shards:
        start=at();data=(train.CACHE/r['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256']
        identities,lengths,values,contexts,_=train.oracle_shard(data,r['windows'],decoder)
        load=at()-start
        for begin in range(0,len(values),16):
            start=at();batch=next(rows)
            for j,(doc,offset,bb) in enumerate(batch):assert (doc,offset,len(bb)-1)==(*identities[begin+j],lengths[begin+j])
            xx,yy,mask=train.tensors(batch)
            ff=torch.as_tensor(values[begin:begin+16],device='cuda').clone()
            ff*=(torch.rand((16,128),device='cuda',generator=rng)>=.5)[...,None]
            ll=torch.as_tensor(contexts[begin:begin+16],device='cuda');assembly=at()-start
            start=at();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                lp,_=model(xx,ff,ll,windows=[(d,o) for d,o,_ in batch],valid_lengths=lengths[begin:begin+16].tolist())
                loss=-lp.gather(-1,yy[...,None])[...,0][mask].mean()
            forward=at()-start;start=at();loss.backward();backward=at()-start;start=at()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();update=at()-start;start=at()
            train.accumulate(telemetry,model.telemetry);report=at()-start
            step=dict(step=steps+1,load=load if begin==0 else 0.,assembly=assembly,forward=forward,backward=backward,update=update,telemetry=report,targets=int(mask.sum()),loss=float(loss))
            for k in ('load','assembly','forward','backward','update','telemetry'):totals[k]=totals.get(k,0.)+step[k]
            curve.append(step);steps+=1
            if steps==200:
                result=dict(status='completed',steps=steps,shape=[16,128],seed=3401,targets=sum(x['targets'] for x in curve),
                    component_seconds=totals,measured_seconds=sum(totals.values()),steps_per_measured_second=steps/sum(totals.values()),
                    source_sha256=gate['code_sha256'],shards_consumed=1+next(i for i,x in enumerate(shards) if x is r),curve=curve,
                    timing='synchronized wall-clock component measurements; graph capture included; preflight initialization and final receipt serialization excluded; not registered training',
                    gpu=__import__('os').environ['GPU_LEASE_HELD'])
                (train.OUT/'pipeline_profile.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({k:v for k,v in result.items() if k!='curve'},indent=2));return
    raise RuntimeError('source exhausted before 200-step profile')
if __name__=='__main__':main()
