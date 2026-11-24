"""Train delayed recall on frozen, native counterfactual block losses."""
import json,sys,time
from pathlib import Path
import torch
from .k15_bank_pilot import setup,extract
from .k15_historical_training import load_net,j,ANCHOR_SHA
from .k15_handle_bank import BankPort,Recall,DecodePort,decode_last
from .k15_coupled_training import save
from .k15_coupled_summary import interval
from .k15_relation_training import window


def gains(item,model,fixed):
    h,m,g,baseline,e,dg,n=item; idx=torch.arange(len(h)); chosen=model(h,m).argmax(1); count=m[:,:,3].sum(1).long().clamp_min(1); shuffled=torch.where(chosen>0,1+chosen%count,0)
    return {k:[float(g[idx,a].sum()),float(dg[idx,a].sum())] for k,a in [('current',0),('fifo',1),('fixed',fixed),('adaptive',chosen),('reassign',shuffled)]}


def main(out):
    pilot=out.with_name('k15L-pilot'); p=json.loads((pilot/'panels.json').read_text()); assert json.loads((pilot/'console').read_text().splitlines()[-2])['passed']; stream,ident=setup(); assert p['identity']==ident; ph=j.digest(pilot/'panels.json'); torch.set_num_threads(1); net=load_net().eval(); net.port=BankPort(); models={}; opts={}; cache=[]; train_gain=torch.zeros(5); curve=[]
    def collect(docs,group):
        result=[]
        with (out/(group+'.jsonl')).open('x') as f:
            for d in docs:
                item,row=extract(net,stream,d); print(json.dumps(row),file=f,flush=True); result.append(item)
        return result
    def assess(group,items,keys):
        report={}
        for key in keys:
            values=[gains(item,models[key],fixed) for item in items]; means={k:interval([v[k][0]/1024 for v in values]) for k in values[0]}; paired={k:interval([(v['adaptive'][0]-v[k][0])/1024 for v in values]) for k in ['fifo','fixed','reassign']}; conditional={k:[sum(v[k][0] for v in values)/max(1,sum(i[4] for i in items)),sum(v[k][1] for v in values)/max(1,sum(i[6] for i in items))] for k in means}; report[str(key)]=dict(gain=means,adaptive_vs=paired,conditional=conditional,eligible=sum(i[4] for i in items),conflict=sum(i[6] for i in items),base=sum(i[3] for i in items)/len(items)/1024)
        (out/(group+'.json')).write_text(json.dumps(report)); return report
    with (out/'train.jsonl').open('x') as f:
        for step,d in enumerate(p['train'],1):
            item,row=extract(net,stream,d); print(json.dumps(row),file=f,flush=True)
            if step<=64: cache.append(item); train_gain+=item[2].sum(0)
            for seed in [3414,3415]:
                for mode in ['fresh','cached']:
                    key=(seed,mode); h,m,g,*_=item if mode=='fresh' else cache[(step-1)%64]
                    if step==1:
                        torch.manual_seed(seed); models[key]=Recall(h.shape[-1]); opts[key]=torch.optim.Adam(models[key].parameters(),lr=.003)
                    model=models[key]; opt=opts[key]; loss=-(model(h,m).softmax(1)*g).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step==64: fixed=int(train_gain.argmax()); j.emit(fixed=fixed,rank_gain=train_gain.tolist())
            if step in (1,64,256,1024):
                path=out/f'dose{step}.pt'; save(path,dict(step=step,panels=ph,anchor=ANCHOR_SHA,models={str(k):v.state_dict() for k,v in models.items()},optimizers={str(k):v.state_dict() for k,v in opts.items()})); j.emit(step=step,sha=j.digest(path)); assert torch.load(path,weights_only=True)['step']==step
                if step>=64:
                    if not curve: curve=collect(p['curve'],'curve');
                    assess(f'curve{step}',curve,models)
            if step%16==0: j.emit(train_step=step)
    keys=[(s,'fresh') for s in (3414,3415)]; dev=assess('dev_scores',collect(p['dev'],'dev'),keys); passed=all(r['adaptive_vs'][k][1]>0 for r in dev.values() for k in ('fixed','reassign'))
    j.emit(dev_pass=passed)
    if passed: assess('locked_scores',collect(p['locked'],'locked'),keys)
    x,_=window(stream,p['train'][0]); profile=[]
    for key in [None,keys[0]]:
        net.port=DecodePort(models[key].cuda() if key else None); net.eval()
        for end in range(127,135):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.cuda.synchronize(); start=time.perf_counter(); z,selected,work,picks=net(x[:,:end],pad_to=1024); dist=decode_last(z,selected); torch.cuda.synchronize(); elapsed=time.perf_counter()-start; assert abs(float(dist.sum())-1)<1e-5
            profile.append(dict(arm=str(key),end=end,seconds=elapsed,work=work,rank=picks[-1],copy=int(selected.flatten()[end-32])))
    (out/'decode.json').write_text(json.dumps(profile)); j.emit(complete=True)

if __name__=='__main__': main(Path(sys.argv[1]))
