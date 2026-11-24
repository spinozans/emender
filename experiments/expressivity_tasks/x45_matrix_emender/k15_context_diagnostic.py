"""No labels or scoring: localize the batch-shape veto on one consumed prefix."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .k15_context_run import OUT,START,ARMS,OutcomeRows,evidence,mass,Resident,setup,load_net,j,put

DEST=Path('output/k15N-diagnostic')


class TraceResident(Resident):
    def __init__(self,raw):
        self.digest=hashlib.sha256(); super().__init__(j.EXE,'backoff')
        for i in range(0,START,32): self.append(raw[i:i+32])

    def rpc(self,cmd):
        reply=super().rpc(cmd); self.digest.update(json.dumps([cmd,reply],sort_keys=True).encode()); return reply


@torch.inference_mode()
def predict(net,prefix):
    ids=torch.tensor([list(s) for s in prefix],device='cuda'); assert ids.shape[1]==128
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=net(ids,doors=False)[0][:,-1].float(); return logits.cpu().numpy(),logits.softmax(-1).cpu().numpy()


class Observer:
    def __init__(self,raw):
        self.state=TraceResident(raw); self.rows=OutcomeRows(); self.write=(0.,0.)

    def append(self,b):
        self.rows.observe(self.key,self.base,self.hist,b); self.write=self.rows.rows.get(self.key,(0.,0.)); self.state.append(bytes([b]))

    def read(self,pair):
        self.key=self.state.raw[-2:]; logits,self.base=pair; self.hist,f=evidence(self.base,self.state.read()); c,hit=self.rows.read(self.key)
        return dict(native=self.state.digest.hexdigest(),key=self.key.hex(),hist=self.hist,features=f,logits=logits,local=self.base,credit=c,hit=int(hit),write=self.write)


def mixed(row,arm,w):
    a=mass(row['features'],row['credit'],arm,w); return a,(1-a)*row['local']+a*row['hist']


def diff(result,left,right,i):
    for field in left:
        a,b=left[field],right[field]; err=float(a!=b) if isinstance(a,str) else float(np.abs(np.array(a)-b).max()); r=result.setdefault(field,dict(first=None,max=0.))
        if err and r['first'] is None: r['first']=i
        r['max']=max(r['max'],err)


def main():
    stream,_=setup(); ck=OUT/'checkpoint.json'; assert j.digest(ck)=='c20d7661eead343438329c9113c04f2866cdbc0928818a3a7f0e3c98eeca8b15'; w=json.loads(ck.read_text())['weights']; d=json.loads((OUT/'panels.json').read_text())['dev'][0]; assert d['doc']==100626487343
    with stream.source.open('rb') as f: f.seek(d['offset']); raw=f.read(3871)
    net=load_net().eval(); batched=[]
    for t in (START,START+16): batched.extend(zip(*predict(net,[raw[i-128:i] for i in range(t,t+16)])))
    comparisons=[{},{}]; trace=[Observer(raw) for _ in range(3)]
    try:
        for i in range(32):
            rows=[]
            for n,o in enumerate(trace):
                if i: o.append(raw[START+i-1]);
                pair=batched[i] if n==2 else [arr[0] for arr in predict(net,[o.state.raw[-128:]])]; r=o.read(pair)
                for a in ARMS: r['gate_'+a],r['mix_'+a]=mixed(r,a,w[a])
                rows.append(r)
            for n in (1,2): diff(comparisons[n-1],rows[0],rows[n],START+i)
            if i in (0,15,31): put(DEST/'comparison.json',comparisons); j.emit(step=i+1)
    finally:
        for o in trace: o.state.session.close()
    states={}; timing={}
    for a in ARMS:
        tic=time.perf_counter(); states[a]=Observer(raw); timing[a]=dict(prefill=time.perf_counter()-tic,latencies=[])
    try:
        for i in range(32):
            for jx in range(len(ARMS)):
                a=ARMS[(i+jx)%len(ARMS)]; tic=time.perf_counter(); o=states[a]
                if i: o.append(raw[START+i-1])
                r=o.read([arr[0] for arr in predict(net,[o.state.raw[-128:]])]); _,dist=mixed(r,a,w[a]); timing[a]['latencies'].append(time.perf_counter()-tic); assert abs(dist.sum()-1)<1e-5
            if i%4==0: j.emit(timing_step=i+1)
    finally:
        for o in states.values(): o.state.session.close()
    put(DEST/'timing.json',timing); j.emit(complete=True)

if __name__=='__main__': main()
