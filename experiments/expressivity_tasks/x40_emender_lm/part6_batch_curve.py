"""Matched-token physical-batch curve against an accumulated, certified reference."""
import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from ..x36_tape_perf.loss import ChunkedLoss
from .model import LM
from .part6_surface import features
from .part6_journal import packed_journals
from .part6_deferred_journal import packed_cells
from .part6_stream_probe import overlap
from .part6_register_probe import RegisterLaunch
from .part6_pool_fusion import fused_pool
from . import backward

ROOT=Path(__file__).resolve().parent/'receipts/part6'


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);a=p.parse_args()
    proof=json.loads((ROOT/a.receipt).read_text());assert proof['status']=='passed'
    batch=proof['batch'];assert batch%32==0
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    reference=LM(tok.get_vocab_size(),ignition=True).cuda().train();candidate=copy.deepcopy(reference)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1) for m in (reference,candidate)]
    steps=max(20,math.ceil(2_000_000/(batch*512)))
    out=ROOT/('matched_'+Path(a.receipt).stem);out.mkdir(exist_ok=False)
    original=backward.reverse_kernel;native=backward.Replay.backward;records=[];rows=[]
    with (out/'curve.jsonl').open('x',buffering=1) as log:
        for step in range(steps):
            x,y=(v.cuda() for v in data.batch('train',step,batch_size=batch))
            values=[];counts=[]
            for model,opt,is_candidate in ((reference,optimizers[0],False),(candidate,optimizers[1],True)):
                opt.zero_grad(set_to_none=True);total=0.;calls=0
                def replay(*args):
                    nonlocal calls
                    calls+=1;return native(*args)
                physical=batch if is_candidate else 32
                reg=proof['registers'] if is_candidate else 0
                streams=proof['head_streams'] and is_candidate
                deferred=proof.get('deferred_pack',False) and is_candidate
                checkpoint=proof['checkpoint'] if is_candidate else True
                packed=proof.get('packed',False) if is_candidate else True
                with patch.object(backward,'reverse_kernel',RegisterLaunch(original,reg)) if reg else nullcontext(),overlap() if streams else nullcontext(),packed_cells(records) if deferred else nullcontext(),fused_pool() if is_candidate and proof.get('fused_pool') else nullcontext():
                    for begin in range(0,batch,physical):
                        with packed_journals(records) if packed else nullcontext(),patch.object(backward.Replay,'backward',staticmethod(replay)):
                            hidden,gates=features(model,x[begin:begin+physical],'pure',checkpoint)
                            loss=(ChunkedLoss.apply(hidden,model.head.weight,y[begin:begin+physical],proof.get('loss_chunk',1024) if is_candidate else 1024)+0.*torch.stack(gates).mean())*(physical/batch)
                            loss.backward();total+=float(loss.detach());del hidden,gates,loss
                assert calls==24*(batch//physical)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                values.append(total);counts.append(calls)
            row=dict(step=step+1,tokens=(step+1)*batch*512,reference_loss=values[0],candidate_loss=values[1],absolute_difference=abs(values[0]-values[1]),reference_replays=counts[0],candidate_replays=counts[1])
            rows.append(row);log.write(json.dumps(row)+'\n');print(row,flush=True)
    maximum=max(r['absolute_difference'] for r in rows)
    result=dict(status='passed' if maximum<=1e-5 else 'failed',candidate_receipt=a.receipt,batch=batch,context=512,
                reference_microbatch=32,candidate_physical_batch=batch,steps=steps,tokens=steps*batch*512,
                max_loss_difference=maximum,tolerance=1e-5,
                scope='Identical initialization/data/effective batch, weighted microbatch means, one clip and AdamW per update. Reference uses recorded checkpoint and exact journal packing, original stream/register policy.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise AssertionError(maximum)


if __name__=='__main__':main()
