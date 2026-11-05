"""Registered bounded A diagnostic; writes only the new A receipt directory."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .c14_documents import sha
from .c14_b_model import StoreIntegrated
from .c11 import Boosted
from .operator_rounds import IterativeForward,fingerprint
from .c14_paired_eval import ROOT,CACHE,DATA,JOURNAL
from .c14_fast_features import load_decoder
from .c14_b_eval_decode import load_context_decoder,eval_shard

OUT=ROOT/'receipts/theory_rounds'


def write(name,obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('3','5')
    torch.set_num_threads(1);torch.manual_seed(4301);torch.use_deterministic_algorithms(True)
    assert sha(CACHE/'journal.jsonl')==JOURNAL
    journal=[r for line in (CACHE/'journal.jsonl').read_text().splitlines() if (r:=json.loads(line))['event']=='shard_complete']
    selected=[journal[0],journal[len(journal)//2]]
    span=json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    assert sha(DATA/'pile_offsets.jsonl')==span['offsets_sha256']
    offsets={}
    wanted=[];decoded=[]
    decoder=load_decoder();cd=load_context_decoder()
    for shard in selected:
        path=CACHE/shard['file'];assert sha(path)==shard['sha256']
        ids,lengths,features,contexts=eval_shard(path.read_bytes(),shard['windows'],decoder,cd)
        decoded.append((ids[:2],lengths[:2],features[:2],contexts[:2]))
        wanted.extend(ids[:2,0].tolist())
    for line in (DATA/'pile_offsets.jsonl').open():
        row=json.loads(line)
        if row['source_index'] in wanted:offsets[row['source_index']]=row
        if len(offsets)==len(set(wanted)):break
    raw=np.memmap(span['source'],dtype=np.uint8,mode='r')
    construction=json.loads((ROOT/'receipts/c14_dense/eval_construction.json').read_text())
    assert sha(Path(construction['windows_path']))==construction['windows_sha256']
    sample_indices={s['first_window_index']+j for s in selected for j in range(2)}
    panels={i:json.loads(line)['half'] for i,line in enumerate(Path(construction['windows_path']).open()) if i in sample_indices}
    batches=[];documents=[]
    for ids,lengths,features,contexts in decoded:
        xx=np.zeros((len(ids),128),np.int64);yy=np.zeros_like(xx)
        for j,((doc,at),length) in enumerate(zip([tuple(row) for row in ids.tolist()],lengths.tolist())):
            d=offsets[doc];bb=raw[d['byte_offset']:d['byte_offset']+d['byte_length']]
            assert __import__('hashlib').sha256(bb).hexdigest()==d['sha256']
            assert at+length<d['byte_length']
            xx[j,:length]=bb[at:at+length];yy[j,:length]=bb[at+1:at+length+1]
            documents.append(dict(document=doc,start=at,targets=length,sha256=d['sha256']))
        batches.append((torch.as_tensor(xx,device='cuda'),torch.as_tensor(yy,device='cuda'),
                        torch.as_tensor(features,device='cuda'),torch.as_tensor(contexts,device='cuda'),[tuple(row) for row in ids.tolist()],lengths.tolist()))
    result=dict(protocol_sha256=sha(ROOT/'PREREGISTRATION_OPERATOR_ROUNDS.md'),
        implementation_sha256=sha(ROOT/'operator_rounds.py'),runner_sha256=sha(__file__),
        physical_gpu=os.environ['GPU_LEASE_HELD'],precision='FP32',documents=documents,
        panels=panels,source_receipt_sha256=sha(ROOT/'receipts/c14_dense/pile_span.json'),
        journal_sha256=JOURNAL,shards=selected,rows=[],gates=[],live_RPCs=0)
    write('progress.json',result)
    for arm,path,expected in [('B1_v2',DATA/'emender_c14_b1_v2_resume.pt','9c87a6a306b708608d7f466dbd62c70c53263c5322f24963b91b94afc8f4def8'),
                               ('dense',DATA/'emender_dense.pt','7a2598c8c1e77f349be6be7af1d59ece7300a5345cb3dab1420e0274d2523019')]:
        assert sha(path)==expected,(arm,'checkpoint hash mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=(StoreIntegrated() if arm=='B1_v2' else Boosted('emender',2000000)).cuda().eval()
        model.load_state_dict(saved['model'] if arm=='B1_v2' else saved)
        runner=IterativeForward(model,store_arm=arm=='B1_v2')
        for k in (1,2,4):
            losses=[];all_losses=[];records=[];entropies=[];queries=0;start=time.monotonic()
            for x,y,a,c,windows,lengths in batches:
                # Fork labels use the exact cached byte distribution, including
                # positive support that may underflow in an entropy feature.
                forks=(a[...,:256]>0).sum(-1)>1
                p=a[...,:256];entropy=-(p*torch.where(p>0,p.log(),torch.zeros_like(p))).sum(-1)
                with torch.inference_mode():
                    input_answers=a if arm=="B1_v2" else torch.zeros_like(a)
                    if k==1:
                        reference,_=model(x,input_answers,c,windows=windows,valid_lengths=lengths,checkpointed=False) if arm=='B1_v2' else model(x,input_answers,c)
                    lp,_=runner(x,input_answers,c,windows=windows,valid_lengths=lengths,forks=forks,k=k)
                    nll=-lp.gather(-1,y[...,None])[...,0]
                    if k==1:
                        ref_loss=-reference.gather(-1,y[...,None])[...,0]
                        gate=dict(arm=arm,logits=fingerprint(lp),standard_logits=fingerprint(reference),losses=fingerprint(nll),standard_losses=fingerprint(ref_loss))
                        assert gate['logits']==gate['standard_logits'] and gate['losses']==gate['standard_losses']
                        result['gates'].append(gate)
                for j,n in enumerate(lengths):
                    losses.extend(nll[j,:n][forks[j,:n]].cpu().tolist());all_losses.extend(nll[j,:n].cpu().tolist())
                if k==1:queries+=sum(lengths)*len(model.head.base.blocks)
                else:
                    queries+=sum(r['branch_reads'] for r in runner.telemetry)
                    for r in runner.telemetry:
                        if r['fork']:
                            r=dict(r,entropy=float(entropy[r['batch'],r['position']]))
                            records.append(r)
            assert losses and np.isfinite(losses).all()
            terminal=[r for j,r in enumerate(records) if j+1==len(records) or
                      (r['layer'],r['batch'],r['position'])!=(records[j+1]['layer'],records[j+1]['batch'],records[j+1]['position'])]
            bins={}
            for name,predicate in [('low',lambda h:h<=np.log(2)),('high',lambda h:h>np.log(2))]:
                rr=[r for r in terminal if predicate(r['entropy'])]
                bins[name]=dict(n=len(rr),mean_rounds=float(np.mean([r['round'] for r in rr])) if rr else None,
                    histogram={str(n):sum(r['round']==n for r in rr) for n in range(1,k+1)})
            row=dict(arm=arm,k=k,checkpoint_sha256=expected,fork_targets=len(losses),fork_nll=float(np.mean(losses)),
                total_targets=len(all_losses),total_nll=float(np.mean(all_losses)),logical_branch_reads=queries,
                seconds=time.monotonic()-start,halting=bins,op_telemetry=records)
            result['rows'].append(row);write('progress.json',result)
            print({n:v for n,v in row.items() if n!='op_telemetry'},flush=True)
        del runner,model;torch.cuda.empty_cache()
    result['status']='complete';write('results.json',result)

if __name__=='__main__':main()
