"""R4 stage event attribution; explicit diagnostic, synthetic concentration only."""
import json
import sys
from pathlib import Path
import torch
from . import c14_phase5r4_stream_pages as stages
from . import c14_phase5r4_probe as probe

records=[]
snapshots=[]
class Timed:
    def __init__(self,name,jit): self.name,self.jit=name,jit
    def __getitem__(self,grid):
        def launch(*a,**kw):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();result=self.jit[grid](*a,**kw);end.record()
            records.append((self.name,start,end))
            if self.name=='normalize' and a[6] in (256,512,1023):
                snapshots.append((a[6],a[3].detach().clone()))
            return result
        return launch

class Profile:
    calls=0
    def __getitem__(self,grid):
        def launch(*a,**kw):
            self.calls+=1
            if self.calls!=2: return original[grid](*a,**kw)
            saved={name:getattr(stages,name) for name in ('initialize','control','score_pages','normalize','read_page')}
            try:
                for name,jit in saved.items():setattr(stages,name,Timed(name,jit))
                result=original[grid](*a,**kw)
                torch.cuda.synchronize()
                by={}
                for name,start,end in records:
                    row=by.setdefault(name,dict(calls=0,event_ms=0.))
                    row['calls']+=1;row['event_ms']+=start.elapsed_time(end)
                concentration=[]
                for t,p in snapshots:
                    p=p.double().flatten();total=p.sum().item()
                    concentration.append(dict(t=t,nonzero=int((p!=0).sum()),total=total,top16=float(p.topk(16).values.sum()/total),top32=float(p.topk(32).values.sum()/total)))
                output.write_text(json.dumps(dict(stages=by,synthetic_concentration=concentration,limitations='CUDA events include instrumentation/enqueue gaps; not hardware counters. Random fixture is not trained PB3 telemetry.'),indent=2)+'\n')
                return result
            finally:
                for name,jit in saved.items():setattr(stages,name,jit)
        return launch

if __name__=='__main__':
    output=Path(sys.argv[1]);assert not output.exists()
    original=probe.kernel
    probe.kernel=Profile()
    sys.argv=[sys.argv[0],'--only1024','--matched-r2','--output',str(output.with_name('profile_sweep.json'))]
    probe.main()
