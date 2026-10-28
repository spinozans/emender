"""Broker-only preflight and ordered TinyStories run; fail closed on parity."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from ..x34_lm_pilot.cli import validate
from .model import LM, policy_receipt, parameter_receipt, require_replays
from .parity import certify

ROOT=Path(__file__).resolve().parent
RECEIPT_ROOT=ROOT/'receipts'


def write(name,value):
    # evaluate/depth import this module separately when runner runs as __main__.
    target=Path(os.environ.get('X40_RECEIPT_DIR',str(RECEIPT_ROOT)))/name
    target.write_text(json.dumps(value,indent=2)+'\n')


def fingerprint():
    paths=[]
    for folder in ('x30_tape_cell','x31_tape_cell_fused','x34_lm_pilot','x36_tape_perf',
                   'x37_head_differentiation','x39_memory_usage','x40_emender_lm'):
        paths.extend((ROOT.parent/folder).glob('*.py'))
    paths.append(ROOT/'RECEIPTS.md')
    return {str(p.relative_to(ROOT.parent)):data.sha(p) for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',action='store_true')
    p.add_argument('--ignition-v1',action='store_true')
    p.add_argument('--receipt-dir',type=Path)

    args=p.parse_args()
    global RECEIPT_ROOT
    if args.ignition_v1 and args.receipt_dir is None:
        p.error('--ignition-v1 requires a fresh --receipt-dir')
    if args.receipt_dir is not None:
        RECEIPT_ROOT=args.receipt_dir.resolve()
        RECEIPT_ROOT.mkdir(parents=True,exist_ok=False)
    os.environ['X40_RECEIPT_DIR']=str(RECEIPT_ROOT)
    gate_cost=0. if args.ignition_v1 else .01
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in ('0','1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
        raise RuntimeError('single broker lease required on 0/1/5/6')
    torch.set_num_threads(1)
    fp=fingerprint()
    registration=dict(sources=fp,physical_gpu=held,ignition_v1=args.ignition_v1,gate_cost=gate_cost,policies=policy_receipt(),
        preregistration_sha256=data.sha(ROOT/'RECEIPTS.md'),torch=torch.__version__)
    write('registration.json',registration)
    try:
        command=[sys.executable,"-m","pytest","-q","-p",
            "experiments.expressivity_tasks.x36_tape_perf.oracle_adapter",
            str(ROOT.parent/"x31_tape_cell_fused/tests"),
            "--junitxml="+str(RECEIPT_ROOT/"oracle.xml")]
        with (RECEIPT_ROOT/"oracle.log").open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        parity=certify()
        write('parity.json',parity)
    except BaseException as exc:
        write('parity.json',dict(status='failed',error=repr(exc)))
        raise
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    torch.manual_seed(3401);np.random.seed(3401)
    model=LM(tok.get_vocab_size(),ignition=args.ignition_v1)
    write('parameters.json',parameter_receipt(model,tok.get_vocab_size()))
    model=model.cuda().train()
    x,y=(a.cuda() for a in data.batch('train',0,batch_size=64))
    logits,_=model(x)
    ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
    loss=ce+gate_cost*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
    count=require_replays(loss);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    write('smoke.json',dict(status='passed',replay_nodes=count,loss=float(loss.detach()),batch=64,context=512))
    del logits,loss,ce,x,y
    if not args.train:return
    # Reset initialization after smoke: the smoke is never counted as exposure.
    del model
    torch.manual_seed(3401);np.random.seed(3401)
    model=LM(tok.get_vocab_size(),ignition=args.ignition_v1).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    registration['data']={name:data.sha(data.RUNS/name) for name in ('train.bin','valid.bin','tokenizer.json')}
    write('registration.json',registration)
    begin=time.monotonic();step=0
    try:
        with (RECEIPT_ROOT/'curve.jsonl').open('x',buffering=1) as log:
            for step in range(1,3053):
                opt.zero_grad(set_to_none=True)
                x,y=(a.cuda() for a in data.batch('train',step-1,batch_size=64))
                logits,_=model(x)
                ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
                loss=ce+gate_cost*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
                require_replays(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                value=float(loss.detach())
                gates=[float(b.cell.gate_mean.detach()) for b in model.blocks]
                if args.ignition_v1 and any(g<=.5 for g in gates):
                    raise AssertionError('ignition-v1 enable invariant failed')
                if not math.isfinite(value):raise FloatingPointError('nonfinite loss')
                log.write(json.dumps(dict(step=step,tokens=step*32768,loss=value,ce=float(ce.detach()),gate_per_block=gates,grad_norm=float(norm),replay_nodes=24))+'\n')
                del logits,loss,ce
                if step%256==0 or step in (611,3052):
                    if fingerprint()!=fp:raise RuntimeError('source changed during training')
                    write(f'quality_{step}.json',dict(step=step,tokens=step*32768,validation=validate(model,'cuda')))
                    print(json.dumps(dict(step=step,seconds=time.monotonic()-begin)),flush=True)
        from .evaluate import evaluate
        evaluate(model)
        from .depth import fine_tune
        fine_tune(model)
        write('training.json',dict(status='completed',steps=3052,tokens=100007936,seconds=time.monotonic()-begin))
    except BaseException as exc:
        write('training.json',dict(status='failed',last_attempted_step=step,error=repr(exc),seconds=time.monotonic()-begin))
        raise

if __name__=='__main__':main()
