"""Paired production pilot: exact per-step losses, gradients and parameters."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import numpy as np
from tokenizers import Tokenizer
from experiments.expressivity_tasks.x34_lm_pilot import data
from experiments.expressivity_tasks.x36_tape_perf.loss import ChunkedLoss
from experiments.expressivity_tasks.x40_emender_lm.model import LM
from experiments.expressivity_tasks.x40_emender_lm.part6_surface import features
from experiments.expressivity_tasks.x40_emender_lm.part6_journal import packed_journals

root=Path(__file__).resolve().parent
assert sys.executable=='/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3','4')
torch.set_num_threads(1);torch.manual_seed(3401);np.random.seed(3401)
arm=sys.argv[1]
reference=json.loads((root/'curve_base.json').read_text()) if arm=='promoted' else None
tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
def digest(grad=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());x=p.grad if grad else p
        if x is not None:h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for name in ('x31_tape_cell_fused','x36_tape_perf','x40_emender_lm') for p in (Path('experiments/expressivity_tasks')/name).glob('*.py')}
result=dict(status='running',arm=arm,initial=digest(),rows=[],sources=sources,scope='pilot LM, B64/T512, two B32 accumulation envelopes, 64 updates, FP32, checkpoint and packed journals')
if reference:assert result['initial']==reference['initial']
try:
    for step in range(64):
        opt.zero_grad(set_to_none=True)
        x,y=(v.cuda() for v in data.batch('train',step,batch_size=64))
        total=0.
        for begin in (0,32):
            with packed_journals([]):
                hidden,gates=features(model,x[begin:begin+32],'pure')
                loss=(ChunkedLoss.apply(hidden,model.head.weight,y[begin:begin+32],1024)+0.*torch.stack(gates).mean())*.5
                loss.backward();total+=float(loss.detach())
                del hidden,gates,loss
        gradient=digest(True)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        row=dict(step=step+1,tokens=(step+1)*32768,loss=total,norm=float(norm),gradient=gradient,parameters=digest())
        result['rows'].append(row);print(json.dumps(row),flush=True)
        if reference:assert row==reference['rows'][step],dict(observed=row,expected=reference['rows'][step])
    result.update(status='passed',tokens=64*32768)
except BaseException as exc:
    result.update(status='failed',error=repr(exc));raise
finally:
    (root/f'curve_{arm}.json').write_text(json.dumps(result,indent=2)+'\n')
