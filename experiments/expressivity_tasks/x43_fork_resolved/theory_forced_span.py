"""Self-indexed training-corpus recall diagnostic, registered before training."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .c11 import Boosted
from .c14_documents import DocumentWindows,sha
from .c14_paired_eval import ROOT,DATA

OUT=ROOT/'receipts/theory_forced_span'


def write(name,obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def context_counts(documents):
    counts=collections.defaultdict(collections.Counter)
    for raw in documents:
        for target in range(1,len(raw)):
            for width in range(1,min(4,target)+1):
                counts[raw[target-width:target]][raw[target]]+=1
    return counts


def labels(raw,start,n,counts):
    bins=[];entropy=[]
    for i in range(n):
        target=start+i+1
        context=raw[max(start,target-4):target]
        c=counts[context];total=sum(c.values())
        assert total>0 and c[raw[target]]>0
        h=-sum(v/total*math.log(v/total) for v in c.values())
        bins.append(0 if len(c)==1 else 1 if h<=math.log(2) else 2);entropy.append(h)
    return bins,entropy


def forced_runs(bins):
    start=0
    while start<len(bins):
        if bins[start]!=0:start+=1;continue
        end=start+1
        while end<len(bins) and bins[end]==0:end+=1
        if end-start>=4:yield start,end
        start=end


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('5','6')
    torch.set_num_threads(1);torch.manual_seed(4301);torch.use_deterministic_algorithms(True)
    span=json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    assert sha(DATA/'pile_offsets.jsonl')==span['offsets_sha256']
    assert sha(Path(span['documents_index']))==span['documents_index_sha256']
    source=Path(span['source']);stat=source.stat()
    assert (stat.st_size,stat.st_ino,stat.st_mtime_ns)==(span['source_size'],span['source_inode'],span['source_mtime_ns'])
    raw=np.memmap(source,dtype=np.uint8,mode='r');docs=[];receipts=[]
    for line in (DATA/'pile_offsets.jsonl').open():
        d=json.loads(line)
        if d['split']!='train' or d.get('truncated') or d['byte_length']<2:continue
        bb=bytes(raw[d['byte_offset']:d['byte_offset']+d['byte_length']])
        assert hashlib.sha256(bb).hexdigest()==d['sha256']
        docs.append(bb);receipts.append(d)
        if len(docs)==16:break
    assert len(docs)==16
    selected={d['source_index']:d['sha256'] for d in receipts};matched={}
    for line in Path(span['documents_index']).open():
        d=json.loads(line)
        if d['source_index'] in selected:
            assert d['sha256']==selected[d['source_index']]
            matched[d['source_index']]=d['sha256']
        if len(matched)==16:break
    assert matched==selected
    counts=context_counts(docs);sampler=DocumentWindows(128)
    rows=[];class_labels=[];entropies=[];panels=[]
    for j,(bb,d) in enumerate(zip(docs,receipts)):
        for w in sampler.windows(bb,d):
            assert w.inputs==bb[w.start:w.start+len(w.inputs)] and w.targets==bb[w.start+1:w.start+len(w.targets)+1]
            rows.append(w)
            bins,h=labels(bb,w.start,len(w.targets),counts)
            class_labels.append(bins);entropies.append(h);panels.append('calibration' if j<8 else 'final_test')
    xx=np.zeros((len(rows),128),np.int64);yy=np.zeros_like(xx);mm=np.zeros_like(xx,dtype=bool)
    for j,w in enumerate(rows):
        n=len(w.targets);xx[j,:n]=list(w.inputs);yy[j,:n]=list(w.targets);mm[j,:n]=True
    index_hash=hashlib.sha256()
    for key,c in sorted(counts.items()):
        index_hash.update(bytes([len(key)])+key)
        index_hash.update(np.array(sorted(c.items()),dtype='<i8').tobytes())
    result=dict(status='prepared',protocol_sha256=sha(ROOT/'PREREGISTRATION_FORCED_SPAN.md'),
        runner_sha256=sha(__file__),physical_gpu=os.environ['GPU_LEASE_HELD'],seed=4301,
        documents=receipts,documents_index_sha256=span['documents_index_sha256'],
        offsets_sha256=span['offsets_sha256'],exact_index_sha256=index_hash.hexdigest(),
        rows=len(rows),corpus_targets=int(mm.sum()),cross_document_windows=0,padding_targets_masked=int((~mm).sum()),
        calibration_documents=[d['source_index'] for d in receipts[:8]],final_test_documents=[d['source_index'] for d in receipts[8:]],
        scope='training-corpus recall; both disjoint reporting panels were in training; no consultation features')
    write('preflight.json',result)
    x=torch.as_tensor(xx,device='cuda');y=torch.as_tensor(yy,device='cuda');mask=torch.as_tensor(mm,device='cuda')
    model=Boosted('emender',2000000).cuda().train()
    result['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    rng=torch.Generator(device='cuda').manual_seed(4301)
    features=torch.zeros(16,128,260,device='cuda');lengths=torch.zeros(16,128,device='cuda')
    torch.cuda.synchronize();start=time.monotonic();steps=targets=0;curve=[]
    while time.monotonic()-start<180:
        at=torch.randint(0,len(rows),(16,),device='cuda',generator=rng)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lp,_=model(x[at],features,lengths)
            nll=-lp.gather(-1,y[at,...,None])[...,0]
            loss=nll[mask[at]].mean()
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        steps+=1;targets+=int(mask[at].sum())
        torch.cuda.synchronize()
        if steps==1 or steps%100==0:
            row=dict(step=steps,seconds=time.monotonic()-start,nll=float(loss.detach()))
            curve.append(row);print(row,flush=True)
    result.update(steps=steps,actual_targets=targets,exposure_epochs=targets/result['corpus_targets'],
                  active_seconds=time.monotonic()-start,curve=curve,status='trained')
    write('training.json',result)
    model.eval();tables={};span_stats={p:dict(spans=0,exact=0,bytes=0) for p in ('calibration','final_test')}
    predictions=[]
    with torch.inference_mode():
        for lo in range(0,len(rows),16):
            size=min(16,len(rows)-lo)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                lp,_=model(x[lo:lo+size],features[:size],lengths[:size])
            nll=-lp.gather(-1,y[lo:lo+size,...,None])[...,0]
            pred=lp.argmax(-1);correct=(pred==y[lo:lo+size]).cpu().numpy();loss=nll.cpu().numpy()
            for j in range(size):
                idx=lo+j;n=len(rows[idx].targets);panel=panels[idx];bins=np.array(class_labels[idx]);h=np.array(entropies[idx])
                predictions.append(dict(document=rows[idx].document,start=rows[idx].start,
                    prediction_sha256=hashlib.sha256(pred[j,:n].cpu().numpy().tobytes()).hexdigest()))
                for cls,name in enumerate(('forced','low','high')):
                    sel=bins==cls;key=panel+'/'+name
                    table=tables.setdefault(key,dict(targets=0,correct=0,nll_sum=0.,entropy_sum=0.))
                    table['targets']+=int(sel.sum());table['correct']+=int(correct[j,:n][sel].sum())
                    table['nll_sum']+=float(loss[j,:n][sel].astype(np.float64).sum());table['entropy_sum']+=float(h[sel].sum())
                for a,b in forced_runs(bins):
                    s=span_stats[panel];s['spans']+=1;s['bytes']+=b-a;s['exact']+=int(correct[j,a:b].all())
    for row in tables.values():
        n=row['targets'];row.update(accuracy=row['correct']/n if n else None,nll=row['nll_sum']/n if n else None,mean_entropy=row['entropy_sum']/n if n else None)
    for s in span_stats.values():s['accuracy']=s['exact']/s['spans'] if s['spans'] else None
    acc=[tables['final_test/'+name]['accuracy'] for name in ('forced','low','high')]
    result.update(status='complete',table=tables,forced_runs=span_stats,predictions=predictions,
        PB_L1=('unresolved_empty_stratum' if any(a is None for a in acc) or span_stats['final_test']['accuracy'] is None
               else 'supported' if span_stats['final_test']['accuracy']>=.99 and acc[0]>=acc[1]>=acc[2]
               else 'refuted'))
    write('results.json',result)
    print(json.dumps({k:result[k] for k in ('steps','actual_targets','active_seconds','PB_L1','table','forced_runs')}),flush=True)

if __name__=='__main__':main()
