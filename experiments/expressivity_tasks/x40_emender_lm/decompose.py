"""Actual initial per-policy inputs: journal ablation and compiler receipts."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from . import model, forward
from .performance import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):raise RuntimeError('lease required')
    # This diagnostic consumes the preceding production parity receipt.
    parity=json.loads((a.output/'parity.json').read_text());assert parity['status']=='passed'
    torch.set_num_threads(1);torch.manual_seed(3401);np.random.seed(3401)
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    lm=model.LM(tok.get_vocab_size(),ignition=True).cuda().train()
    x,_=data.batch('train',0,batch_size=64);x=x.cuda()
    captured=[]
    def capture(*args,**kw):
        if len(captured)<4:captured.append((args[:6],args[7],args[8],kw.copy()))
        return original(*args,**kw)
    original=model.wired_forward
    with torch.no_grad(),patch.object(model,'wired_forward',capture):lm(x)
    del lm
    rows=[]
    class Launcher:
        def __getitem__(self,grid):
            def launch(*args,**kw):
                compiled=kernel[grid](*args,**kw)
                stats.update(registers=compiled.n_regs,spills=compiled.n_spills,
                    shared=compiled.metadata.shared,warps=compiled.metadata.num_warps)
                return compiled
            return launch
    kernel=forward.scan_kernel
    for inputs,cfg,coef,kwargs in captured:
        row=dict(slots=cfg.slots,decay=cfg.decay_rate)
        for record in (False,True):
            stats={};samples=[]
            with torch.no_grad(),patch.object(forward,'scan_kernel',Launcher()):
                for it in range(7):
                    torch.cuda.synchronize();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                    # beta is an in/out wire: recreate its nonpadding input each time.
                    inputs=(*inputs[:3],(x!=0).float().unsqueeze(-1),*inputs[4:])
                    s.record();result=original(*inputs,None,cfg,coef,**{**kwargs,'record_all':record});e.record();e.synchronize()
                    if it>=2:samples.append(s.elapsed_time(e))
                    if record:
                        counts=result[1].all_field_journal.counts
                        row['journal_count_min_max']=[int(counts.min()),int(counts.max())]
                    del result
            row['journal' if record else 'no_journal']=dict(milliseconds=samples,median_ms=float(np.median(samples)),compiler=stats)
        row['journal_increment_ms']=row['journal']['median_ms']-row['no_journal']['median_ms']
        rows.append(row)
    write(a.output/'ablations.json',dict(policies=rows,scope='separate forward-only ablations, initial bottom-block inputs; not additive wall components'))

if __name__=='__main__':main()
