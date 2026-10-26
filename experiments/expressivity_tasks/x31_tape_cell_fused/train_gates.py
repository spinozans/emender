"""Registered MQAR then 2500-update ladder; JSON receipts only, no .pt files.

Requires the complete parity receipt supplied by --parity-receipt. The receipt
is checked for a successful full suite, not inferred from forward-only tests.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .cell_fused import TapeCell, TapeHeadConfig
from ..x28e_memory_head.generators import stage1_mqar as mqar
from ..x28e_memory_head.generators import stage3_ladder as ladder

ROOT=Path(__file__).resolve().parent


def require_parity(path):
    receipt=Path(path).resolve()
    if not receipt.is_relative_to(ROOT):raise ValueError('receipt must be inside X31')
    data=json.loads(receipt.read_text())
    if data.get('full_parity') is not True:raise RuntimeError('full parity gate is not passed')
    from .record_parity import source_hashes
    if data.get('source_sha256')!=source_hashes():raise RuntimeError('parity source hashes are stale')


def save(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2)+'\n')


class WordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell=TapeCell(256,TapeHeadConfig(n_heads=4,budget=736))
        self.W_o=nn.Linear(512,256)
        self.binary=nn.Linear(256,2)
        self.embed=nn.Embedding(65,256);self.embed.requires_grad_(False)
        self.classifier=nn.Linear(256,64)

    def features(self,ids):
        x=self.embed(ids)
        read,_,_,_=self.cell(x,ids)
        return x+self.W_o(read.flatten(2))

    def forward(self,ids):return self.classifier(self.features(ids))

    def ladder_logits(self,ids,lengths):
        x=self.features(ids)
        return self.binary(x[torch.arange(len(ids),device=ids.device),lengths-1])


@torch.no_grad()
def evaluate_mqar(model,pool):
    model.eval();xs,ys,mask,classes=pool
    true=[];pred=[]
    for start in range(0,len(xs),64):
        x,y,m=(a[start:start+64] for a in (xs,ys,mask));t=128
        output=model(torch.from_numpy(x[:,:t]).cuda()).argmax(-1).cpu().numpy()+1
        true.extend(y[:,:t][m[:,:t]].tolist());pred.extend(output[m[:,:t]].tolist())
    return float(mqar.mean_per_class_recall(np.array(pred),np.array(true),classes))


@torch.no_grad()
def evaluate_ladder(model,pool):
    model.eval();xs,ys,lengths=pool[:3];pred=[]
    for start in range(0,len(xs),8):
        ln=lengths[start:start+8]
        words=xs[start:start+8]
        words=np.pad(words,((0,0),(0,386-words.shape[1])))
        ids=torch.from_numpy(words).cuda()
        pred.extend(model.ladder_logits(ids,torch.from_numpy(ln).cuda()).argmax(-1).cpu().tolist())
    return ladder.binary_report(ys,np.array(pred))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parity-receipt',required=True)
    args=p.parse_args();require_parity(args.parity_receipt)
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in tuple(str(i) for i in range(7)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
        raise RuntimeError('single broker lease on physical cards 0..6 required')
    torch.set_num_threads(1);torch.manual_seed(0)
    model=WordModel().cuda();rng=np.random.default_rng(0)
    provenance=dict(physical_gpu=held,torch=torch.__version__,seed=0,
                    budget=736,heads=4,batch=64,lr=1e-4)
    train=mqar.generate_pool(128,70000,mqar.STAGE1_SPLITS['train'].seed)
    val=mqar.generate_pool(128,20000,mqar.STAGE1_SPLITS['validation'].seed)
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=1e-4,weight_decay=1e-4)
    rows=[];best=-1.;best_weights=None;begin=time.perf_counter()
    for update in range(1,5001):
        model.train();idx=rng.choice(len(train[0]),64,replace=False)
        x,y,mask=(a[idx] for a in train[:3]);t=128
        ids=torch.from_numpy(x[:,:t]).cuda();labels=torch.from_numpy(y[:,:t]-1).cuda()
        m=torch.from_numpy(mask[:,:t]).cuda();opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(ids)[m],labels[m]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if update%10==0:print(json.dumps(dict(stage='mqar',update=update,loss=float(loss.detach()),seconds=time.perf_counter()-begin)),flush=True)
        if update%500==0:
            score=evaluate_mqar(model,val);rows.append(dict(update=update,recall=score))
            if score>=best:best=score;best_weights=copy.deepcopy(model.state_dict())
            save('mqar_part5.json',dict(**provenance,updates=update,best=best,gate=best>=.99,validation=rows,
                train_hash=mqar.stage1_hash('train',*train[:3]),validation_hash=mqar.stage1_hash('validation',*val[:3])))
            if best>=.99:break
    model.load_state_dict(best_weights)
    # Same stage-1 warm start and reported omission of stage-2 as X30's pilot.
    model.embed=nn.Embedding(5,256).cuda();model.embed.requires_grad_(False)
    rng=np.random.default_rng(0)
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=1e-4,weight_decay=1e-4)
    train=ladder.ladder_train_pool();val=ladder.ladder_validation_pool();rows=[]
    for update in range(1,2501):
        model.train();idx=rng.choice(len(train[0]),64,replace=False)
        idx=idx[np.argsort(train[2][idx])];opt.zero_grad(set_to_none=True);total=0.
        for start in range(0,64,64):
            mb=idx[start:start+64];ln=train[2][mb]
            words=train[0][mb]
            words=np.pad(words,((0,0),(0,386-words.shape[1])))
            ids=torch.from_numpy(words).cuda()
            labels=torch.from_numpy(train[1][mb]).cuda()
            logits=model.ladder_logits(ids,torch.from_numpy(ln).cuda())
            loss=F.binary_cross_entropy_with_logits(logits[:,1]-logits[:,0],labels.float(),reduction='sum')/64
            loss.backward();total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if update%10==0:print(json.dumps(dict(stage='ladder',update=update,loss=total,seconds=time.perf_counter()-begin)),flush=True)
        if update%500==0:
            report=evaluate_ladder(model,val);rows.append(dict(update=update,report=report))
            save('ladder_part5.json',dict(**provenance,updates=update,validation=rows,
                 gate=report['balanced_accuracy']>.9,
                 validation_hash=ladder.ladder_hash('validation',*val[:3]),
                 deviations=['stage-2 occupancy curriculum omitted','full physical batch64','full right padding to 386; EOS supervision unchanged']))
    return 0


if __name__=='__main__':raise SystemExit(main())
