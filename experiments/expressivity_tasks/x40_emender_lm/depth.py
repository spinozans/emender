"""X28E final-EOS ladder supervision on pretrained LM hidden features.

The original stage3 contract uses 30000 updates (not the X31 2500 pilot).
An identified matched transformer receipt is required for comparative claims.
"""
from dataclasses import replace
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..x28e_memory_head.generators import stage3_ladder as ladder
from .model import require_replays
from .runner import write


class Probe(nn.Module):
    def __init__(self,lm):
        super().__init__()
        self.lm=lm
        # X28E stage-local scalar-token embedding; no BPE merging of a/b.
        self.lm.embed=nn.Embedding(5,256).cuda()
        self.binary=nn.Linear(256,2).cuda()
        for block in lm.blocks:
            block.cell.configs=tuple(replace(c,budget=2944) for c in block.cell.configs)

    def forward(self,ids,lengths):
        features=self.lm.features(ids)
        return self.binary(features[torch.arange(len(ids),device=ids.device),lengths-1])


@torch.no_grad()
def evaluate(probe,pool):
    probe.eval();pred=[]
    xs,ys,lengths=pool[:3]
    for start in range(0,len(xs),8):
        ln=lengths[start:start+8]
        ids=torch.from_numpy(xs[start:start+8,:int(ln.max())].copy()).cuda()
        logits=probe(ids,torch.from_numpy(ln).cuda())
        pred.extend((logits[:,1]>logits[:,0]).long().cpu().tolist())
    return ladder.binary_report(ys,np.asarray(pred))


def fine_tune(lm):
    from .model import POLICIES
    from .parity import certify
    policies=tuple(replace(c,budget=2944) for c in POLICIES)
    write('depth_parity.json',certify(policies=policies))
    torch.manual_seed(0)
    probe=Probe(lm)
    train=ladder.ladder_train_pool();val=ladder.ladder_validation_pool()
    rng=np.random.default_rng(0)
    opt=torch.optim.AdamW(probe.parameters(),lr=1e-4,weight_decay=1e-4)
    best=-1.;weights=None;selected=None;rows=[];update=0
    # X28E epoch permutations, B64, final-only summed BCE normalized per batch.
    while update<30000:
        permutation=rng.permutation(len(train[0]))
        for start in range(0,len(permutation)-63,64):
            idx=permutation[start:start+64]
            ln=train[2][idx]
            ids=torch.from_numpy(train[0][idx,:int(ln.max())].copy()).cuda()
            labels=torch.from_numpy(train[1][idx]).cuda()
            probe.train();opt.zero_grad(set_to_none=True)
            logits=probe(ids,torch.from_numpy(ln).cuda())
            loss=F.binary_cross_entropy_with_logits(logits[:,1]-logits[:,0],labels.float())
            require_replays(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(),1.,error_if_nonfinite=True)
            opt.step();update+=1
            if update%500==0:
                report=evaluate(probe,val);metric=report['balanced_accuracy']
                rows.append(dict(update=update,report=report))
                if metric>=best:
                    best=metric;selected=update
                    weights={k:v.detach().cpu().clone() for k,v in probe.state_dict().items()}
                write('depth_validation.json',dict(updates=update,validation=rows))
            if update>=30000:break
    probe.load_state_dict(weights)
    bins={name:evaluate(probe,ladder.ladder_bin_pool(name)) for name in ladder.BIN_ORDER}
    write('depth.json',dict(status='completed',updates=30000,selected_update=selected,
        bins=bins,supervision='final EOS binary margin BCE',budget=2944,
        train_hash=ladder.ladder_hash('train',*train[:3]),
        comparison_status='not established: no matched fine-tuned transformer receipt identified',
        deviations=['LM pretraining replaces X28E MQAR and occupancy warmup',
                    'new scalar-token embedding and binary head on pretrained six-block hidden states']))
