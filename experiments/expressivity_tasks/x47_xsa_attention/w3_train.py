"""One natural-LM pass per matched head, with shared frozen-backbone work and an unopened test gate."""
import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .workbench_run import check_gpu_owner,dump,sha
from .workbench_witness import digest_state,require
from .w3_data import OUT,T,SHARD,N
from .w3_model import Head,ARMS,ANCHOR_SHA,base_model,backbone,likelihood


def batches(stage):
    folder=OUT/stage; rec=json.loads((folder/'receipt.json').read_text()); require(rec['panel_sha']==sha(OUT/'panels.json'),'panel hash')
    require(len(rec['shards'])*SHARD==N[stage],'incomplete split')
    for shard in rec['shards']:
        p=folder/shard['file']; require(sha(p)==shard['sha256'],'cache changed')
        with np.load(p,allow_pickle=False) as z: ts={k:torch.from_numpy(z[k]).to('cuda') for k in z.files}
        require(list(ts['raw'].shape)==[SHARD,T+1],'partial window')
        for j in range(0,SHARD,4):
            b={k:v[j:j+4] for k,v in ts.items() if k!='table'}
            b['children']=ts['table'][b.pop('ci').long()]; raw=b.pop('raw').long()
            b['x'],b['y']=raw[:,:-1],raw[:,1:]; b['ptr']=b['ptr'].long(); b['codes']=b['codes'].long()
            yield b


def encode(net,b):
    h,q=backbone(net,b['x']); ny=q.gather(-1,b['y'][...,None]).squeeze(-1)
    args=(h,b['x'],b['feat'],b['ptr'],b['codes'],b['prior'],b['children'],b['on'])
    return args,ny


def losses(head,mode,b,args,ny):
    result=head(*args,mode); p=likelihood(result,ny,b['target_mass'],b['codes'],b['y'])
    return -p.log(),result


def save_ck(run,heads,opts,step,elapsed):
    path=run/'latest.pt'; tmp=run/'partial.pt'; torch.save(dict(step=step,heads={k:h.state_dict() for k,h in heads.items()},
        opts={k:o.state_dict() for k,o in opts.items()},panel_sha=sha(OUT/'panels.json'),anchor=ANCHOR_SHA),tmp); tmp.replace(path)
    dump(run/'progress.json',dict(step=step,targets_per_arm=step*4*T,elapsed_s=elapsed,checkpoint_sha=sha(path)))


@torch.no_grad()
def qualify(net,b):
    # Same batch shape and precision, arbitrary future suffix replacement. Checks
    # causal neural hidden and probabilities including a point after a block reset.
    x=b['x']; h,p=backbone(net,x); y=x.clone(); cut=177; y[:,cut:]=255-y[:,cut:]; h2,p2=backbone(net,y)
    err=max((h[:,:cut]-h2[:,:cut]).abs().max().item(),(p[:,:cut]-p2[:,:cut]).abs().max().item()); require(err<=1e-5,'future leak in base')
    with torch.autocast('cuda',dtype=torch.bfloat16): q=net(x.reshape(-1,128)).float().reshape_as(p).softmax(-1)
    err2=(p-q).abs().max().item(); require(err2<=1e-5,'backbone hook changed distribution')
    ptr=b['ptr']; lag=torch.arange(T,device=x.device)[None,:]-ptr
    require(((ptr<0)|(lag>=16)).all().item(),'future or recent writer')
    return dict(trunk_future_max=err,stock_probability_max=err2,max_prob_sum_err=(p.sum(-1)-1).abs().max().item())


def fit(net,heads,run):
    opts={k:torch.optim.AdamW(h.parameters(),lr=.002,weight_decay=.01) for k,h in heads.items()}
    tic=time.monotonic(); curve=[]; sums={k:0. for k in heads}; started=time.perf_counter(); core=0.
    for step,b in enumerate(batches('train'),1):
        if step==1: dump(run/'qualify.json',qualify(net,b))
        torch.cuda.synchronize(); t=time.perf_counter(); args,ny=encode(net,b); torch.cuda.synchronize(); core+=time.perf_counter()-t
        for k,head in heads.items():
            opt=opts[k]; opt.zero_grad(set_to_none=True); ell,_=losses(head,k,b,args,ny); loss=ell.mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); opt.step(); sums[k]+=float(loss.detach())
        if step==1 or step%16==0:
            sec=time.monotonic()-tic; rec=dict(step=step,elapsed_s=sec,nll_so_far={k:v/step for k,v in sums.items()})
            curve.append(rec); print(json.dumps(rec),flush=True)
            dump(run/'curve.json',curve); save_ck(run,heads,opts,step,sec)
        if step==32:
            require((time.monotonic()-tic)*16<1000,'projected stage exceeds GPU time budget')
    require(step==512,'incorrect exposure'); return dict(seconds=time.perf_counter()-started,shared_backbone_s=core,steps=step,
                  targets_per_arm=step*4*T,online_train_mean={k:v/step for k,v in sums.items()},arms=list(heads))


def interval(v):
    v=np.asarray(v); r=np.random.default_rng(4703); draws=r.integers(0,len(v),(10000,len(v)))
    return [float(v.mean()),*np.quantile(v[draws].mean(1),[.025,.975]).tolist()]


@torch.no_grad()
def assess(net,heads,stage,run):
    keys=['neural',*ARMS,'full_as_blind','full_erased']; docs=[]; begin=time.perf_counter()
    for b in batches(stage):
        args,ny=encode(net,b); active=b['on']; y=b['y']; codes=b['codes']; ptr=b['ptr']; lag=torch.arange(T,device=y.device)[None,:]-ptr
        stats={'neural':-ny.log()}; weights={}
        for key in keys[1:]:
            k=key if key in ARMS else 'full'; mode=key if key in ARMS else ('blind' if key=='full_as_blind' else 'erase')
            stats[key],res=losses(heads[k],mode,b,args,ny); weights[key]=res[0][...,-1]
        for j in range(len(y)):
            mask=active[j]; n=int(mask.sum()); eligible=ptr[j]>=0
            docs.append(dict(targets=T,addressed=int(eligible.sum()),active=n,hit=int(((codes[j]==y[j,:,None]).any(-1)&mask).sum()),
                lag_sum=int(lag[j][mask].sum()),nll={k:float(v[j].double().mean()) for k,v in stats.items()},
                active_loss_sum={k:float(v[j][mask].double().sum()) for k,v in stats.items()},
                join_weight_sum={k:float(v[j][mask].double().sum()) for k,v in weights.items()}))
    require(len(docs)==N[stage],'eval denominator'); y={k:np.array([d['nll'][k] for d in docs]) for k in keys}
    gain={k:interval(v-y['full']) for k,v in y.items() if k!='full'}; passed=all(gain[k][0]>=.001 and gain[k][1]>0 for k in ARMS if k!='full')
    for a,d in zip(json.loads((OUT/'panels.json').read_text())['panels'][stage],docs,strict=True): d['doc']=a['doc']
    row=dict(stage=stage,seconds=time.perf_counter()-begin,docs=docs,means={k:float(v.mean()) for k,v in y.items()},
        control_minus_full=gain,pass_gate=passed,checkpoint_sha=sha(run/'latest.pt'),panel_sha=sha(OUT/'panels.json'))
    dump(OUT/f'{stage}.json',row); print(json.dumps({k:v for k,v in row.items() if k not in ('docs',)}),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['train','confirm']); ap.add_argument('run',type=Path); a=ap.parse_args()
    require(os.environ.get('OMP_NUM_THREADS')=='4','set OMP_NUM_THREADS=4'); torch.set_num_threads(4); torch.manual_seed(4703)
    torch.backends.mha.set_fastpath_enabled(False); torch.set_float32_matmul_precision('highest')
    lease=check_gpu_owner(); net=base_model().cuda(); proto=Head().cuda(); heads={k:copy.deepcopy(proto) for k in ARMS}
    if a.stage=='train':
        require(not (a.run/'latest.pt').exists(),'no accidental checkpoint overwrite')
        dump(a.run/'identity.json',dict(lease=lease,torch=str(torch.__version__),cuda=torch.version.cuda,host=platform.node(),
             gpu=torch.cuda.get_device_name(),birth=digest_state(proto),params=sum(p.numel() for p in proto.parameters()),
             base_parameters=sum(p.numel() for p in net.parameters()),anchor_sha=ANCHOR_SHA))
        dump(a.run/'train.json',fit(net,heads,a.run)); assess(net,heads,'dev',a.run)
    else:
        dev=json.loads((OUT/'dev.json').read_text()); require(dev['pass_gate'] and sha(a.run/'latest.pt')==dev['checkpoint_sha'],'confirmation locked')
        ck=torch.load(a.run/'latest.pt',weights_only=True); require(ck['step']==512 and ck['panel_sha']==sha(OUT/'panels.json'),'invalid checkpoint')
        for k,h in heads.items(): h.load_state_dict(ck['heads'][k]); h.eval()
        assess(net,heads,'confirm',a.run)


if __name__=='__main__': main()
