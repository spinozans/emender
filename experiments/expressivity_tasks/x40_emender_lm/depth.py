"""X28E final-EOS ladder supervision on pretrained LM hidden features.

The original stage3 contract uses 30000 updates (not the X31 2500 pilot).
An identified matched transformer receipt is required for comparative claims.
"""
from dataclasses import replace
import copy
import os
import hashlib
import random
import time
from unittest.mock import patch
from ..x42_pile_scale.resume_probe import state_hash
from pathlib import Path
from ..x42_pile_scale.checkpoint import save,restore
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..x28e_memory_head.generators import stage3_ladder as ladder
from .model import require_replays
from . import model as memory_model
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


def depth_profile(state,trace,pressure):
    """Token event density, without X39's zero-entry cumulative live ledger.

    Live occupancy is not reconstructed from events here. The cell also emits
    its X40 gate, match and error means and existing routing metrics.
    """
    return dict(telemetry_contract='x40-depth-token-events-v1',
        allocation_rate_per_token_head=(trace.alloc_target>=0).float().mean((0,1)).tolist(),
        write_enabled_fraction_per_head=trace.w_on.float().mean((0,1)).tolist())


@torch.no_grad()
@patch.object(memory_model,'profile',depth_profile)
@patch.dict(os.environ, X40_BATCHED_READS='0')
def evaluate(probe,pool):
    probe.eval();pred=[]
    xs,ys,lengths=pool[:3]
    for start in range(0,len(xs),8):
        ln=lengths[start:start+8]
        ids=torch.from_numpy(xs[start:start+8,:int(ln.max())].copy()).cuda()
        logits=probe(ids,torch.from_numpy(ln).cuda())
        pred.extend((logits[:,1]>logits[:,0]).long().cpu().tolist())
    return ladder.binary_report(ys,np.asarray(pred))


@patch.dict(os.environ, X40_STABLE_MAPPING='1', X40_FRESH_POOL='1', X40_FUSED_INIT='1', X40_BATCHED_READS='1')
def fine_tune(lm,checkpoint_dir=None,resume=False,*,updates=30000,checkpoint_every=32,validation_every=500):
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
    binding=dict(environment={k:os.environ[k] for k in ('X40_STABLE_MAPPING','X40_FRESH_POOL','X40_FUSED_INIT','X40_BATCHED_READS')},stage='depth',updates=updates,seed=0,checkpoint_every=checkpoint_every,validation_every=validation_every,train_hash=ladder.ladder_hash('train',*train[:3]))
    permutation=None;start=0
    checkpoint=Path(checkpoint_dir)/'depth.ckpt' if checkpoint_dir else None
    if resume and checkpoint is not None and checkpoint.exists():
        state=restore(checkpoint,probe,opt,binding)
        update=state['step'];rows=state['rows'];cursor=state['cursor']
        rng.bit_generator.state=cursor['rng'];permutation=cursor['permutation'];start=cursor['start']
        best=cursor['best'];weights=cursor['weights'];selected=cursor['selected']
    trace_path=Path(os.environ.get('X40_RECEIPT_DIR','.'))/'depth_trace.jsonl'
    if trace_path.exists():
        import json
        retained=[line for line in trace_path.read_text().splitlines() if json.loads(line)['update']<=update]
        trace_path.write_text(''.join(line+'\n' for line in retained))
    started=time.monotonic();initial_update=update
    # X28E epoch permutations, B64, final-only summed BCE normalized per batch.
    while update<updates:
        if permutation is None:permutation=rng.permutation(len(train[0]))
        for start in range(start,len(permutation)-63,64):
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
            if update%validation_every==0:
                report=evaluate(probe,val);metric=report['balanced_accuracy']
                rows.append(dict(update=update,report=report))
                if metric>=best:
                    best=metric;selected=update
                    weights={k:v.detach().cpu().clone() for k,v in probe.state_dict().items()}
                write('depth_validation.json',dict(updates=update,validation=rows))
            # Persist sampler position and best-validation weights at update boundaries.
            if checkpoint is not None and (update%checkpoint_every==0 or update==updates):
                cursor=dict(rng=rng.bit_generator.state,permutation=permutation,start=start+64,
                            best=best,weights=weights,selected=selected)
                save(checkpoint,probe,opt,binding,update,cursor,rows,0.)
                write('depth_checkpoint.json',dict(update=update,path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),seconds_per_update=(time.monotonic()-started)/(update-initial_update),environment=binding['environment']))
            if update%checkpoint_every in (1,2):
                import json
                digest=hashlib.sha256()
                for value in probe.state_dict().values():digest.update(value.detach().cpu().numpy().tobytes())
                with trace_path.open('a') as trace:trace.write(json.dumps(dict(update=update,loss=float(loss.detach()),model_sha256=digest.hexdigest(),optimizer_sha256=state_hash(opt.state_dict()),torch_rng_sha256=state_hash(torch.get_rng_state()),cuda_rng_sha256=state_hash(torch.cuda.get_rng_state() if torch.cuda.is_available() else None),numpy_rng_sha256=state_hash(rng.bit_generator.state),python_rng_sha256=state_hash(random.getstate()),sampler_sha256=state_hash((permutation,start+64))))+'\n')
            if update>=updates:break
        permutation=None;start=0
    probe.load_state_dict(weights)
    bins={name:evaluate(probe,ladder.ladder_bin_pool(name)) for name in ladder.BIN_ORDER}
    write('depth.json',dict(status='completed',updates=updates,selected_update=selected,
        bins=bins,supervision='final EOS binary margin BCE',budget=2944,
        train_hash=ladder.ladder_hash('train',*train[:3]),
        comparison_status='not established: no matched fine-tuned transformer receipt identified',
        deviations=['LM pretraining replaces X28E MQAR and occupancy warmup',
                    'new scalar-token embedding and binary head on pretrained six-block hidden states']))
