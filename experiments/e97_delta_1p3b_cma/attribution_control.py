"""fuse-2kernel attribution control: separate "linear-state quality loss" from
"chunked-kernel/floor numerics". Same candidate config (dim 2112, 21 gdn-neg + 43
e97_delta), 720s, seed 0, REAL Pile. Two arms:
  A  identity-state + SEQUENTIAL kernel (use_chunked_e97_delta=False)
  B  tanh-state     + SEQUENTIAL kernel (the PRIOR decisive cell — anchors 2.071)
Compare held-out BPB to the head-to-head's identity-CHUNKED (2.25/2.38):
  A ~ 2.25  => linear state itself kills quality (kernel exonerated; the quality loss is in the state, not fixable in the kernel)
  A ~ 2.07  => the chunked kernel/floor kills quality (fixable)
"""
import os, sys, json, time, datetime
_THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_THIS)
RESULTS=os.path.join(_THIS,'results')
from final_headtohead import run_jobs_perseed
from fused_headtohead import DELTA_CFG

def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}',flush=True)

A=dict(DELTA_CFG); A['e97_state_nonlin']='identity'; A['use_chunked_e97_delta']=False
B=dict(DELTA_CFG); B['e97_state_nonlin']='tanh';     B['use_chunked_e97_delta']=False
for c in (A,B): c.pop('params_b',None); c.pop('counts',None)

if __name__=='__main__':
    gpus=[int(x) for x in (sys.argv[1] if len(sys.argv)>1 else '2,3').split(',')]
    jobs=[('A_identity_seq',A,0),('B_tanh_seq',B,0)]
    t0=time.time()
    res=run_jobs_perseed(jobs,gpus,720.0,None,1,1400.0)
    def bpb(t): r=res.get(t,{}); return (r.get('heldout') or {}).get('heldout_bpb') if isinstance(r,dict) else None
    def toks(t): r=res.get(t,{}); return r.get('tokens') if isinstance(r,dict) else None
    def ts(t): r=res.get(t,{}); return r.get('sustained_tok_s') if isinstance(r,dict) else None
    out=dict(task='fuse-2kernel-attribution',A_cfg=A,B_cfg=B,results=res,
             wallclock_minutes=round((time.time()-t0)/60,1),
             timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(out,open(os.path.join(RESULTS,'attribution_control.json'),'w'),indent=2)
    log('=== ATTRIBUTION ===')
    log(f'A identity+SEQ : bpb={bpb("A_identity_seq")} tok={toks("A_identity_seq")} tok/s={ts("A_identity_seq")}')
    log(f'B tanh+SEQ     : bpb={bpb("B_tanh_seq")} tok={toks("B_tanh_seq")} tok/s={ts("B_tanh_seq")}')
    log('(compare vs head-to-head identity+CHUNKED: 2.25/2.38)')
