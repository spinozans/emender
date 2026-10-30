import json
import os
import sys
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from experiments.expressivity_tasks.x34_lm_pilot import data
from experiments.expressivity_tasks.x40_emender_lm.model import LM
from experiments.expressivity_tasks.x40_emender_lm.graphs import certify_training_graph
assert sys.executable=='/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3','4')
torch.set_num_threads(1);torch.manual_seed(3401)
tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
x,y=(v.cuda() for v in data.batch('train',0,batch_size=64))
graph,result=certify_training_graph(model,x,y)
samples=[dict(result,batch_index=0)]
x,y=(v.cuda() for v in data.batch('train',1,batch_size=64))
for block in model.blocks:block.cell.use_chunk_graphs=False
model.zero_grad(set_to_none=True)
logits,_=model(x)
loss=F.cross_entropy(logits.flatten(0,1),y.flatten())+0.*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
loss.backward();reference=float(loss.detach())
gradients=[None if p.grad is None else p.grad.detach().cpu() for p in model.parameters()]
del logits,loss
for block in model.blocks:block.cell.use_chunk_graphs=True
observed=graph.replay(x,y)
torch.testing.assert_close(observed,observed.new_tensor(reference),rtol=0,atol=0)
for param,expected in zip(model.parameters(),gradients):
    if expected is None:assert param.grad is None
    else:torch.testing.assert_close(param.grad.cpu(),expected,rtol=0,atol=0)
samples.append(dict(status='passed',batch_index=1,batch=list(x.shape),loss=reference,gradients='bitwise eager versus reused per-chunk graphs with new inputs'))
print(json.dumps(dict(status='passed',gradients='bitwise',samples=samples)),flush=True)
