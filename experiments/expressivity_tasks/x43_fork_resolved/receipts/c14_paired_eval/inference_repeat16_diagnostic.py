import json, torch, numpy as np
from pathlib import Path
from experiments.expressivity_tasks.x43_fork_resolved.c14_paired_eval import ROOT,CACHE,DATA,decode_shard,rows,write
from experiments.expressivity_tasks.x43_fork_resolved.c11 import Boosted
r=[r for r in rows(CACHE/'journal.jsonl') if r['event']=='shard_complete'][0]
ids,lengths,ff=decode_shard((CACHE/r['file']).read_bytes(),r['windows'])
docs={}
with (DATA/'pile_offsets.jsonl').open() as f:
 for line in f:
  d=json.loads(line)
  if d['source_index'] in ids[:,0]: docs[d['source_index']]=d
  if len(docs)==len(set(ids[:,0])): break
raw=np.memmap('/mnt/nvme2n1/erikg/pile.txt',mode='r',dtype=np.uint8)
x=np.zeros((len(ids),128),np.int64);y=np.zeros_like(x)
for j,(doc,at) in enumerate(ids):
 d=docs[int(doc)];off=d['byte_offset']+int(at);n=int(lengths[j]);x[j,:n]=raw[off:off+n];y[j,:n]=raw[off+1:off+n+1]
x=torch.tensor(x,device='cuda');y=torch.tensor(y,device='cuda');ff=torch.tensor(ff,device='cuda');zz=torch.zeros_like(ff)
x=x[:16];y=y[:16];ff=ff[:16];zz=zz[:16]
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1); result=[]; models={}
for label,p in [('dense',DATA/'emender_dense.pt'),('v1',DATA/'emender_consultation_resume.pt'),('v2',DATA/'emender_consultation_v2_resume.pt')]:
 s=torch.load(p,map_location='cpu',weights_only=False);m=Boosted('emender',2000000).cuda().eval();m.load_state_dict(s if label=='dense' else s['model']);models[label]=m
with torch.inference_mode():
 for label,m in models.items():
  baseline=None
  for rep in range(8):
   for block in m.head.base.blocks: block.cell.training=bool(rep%2)
   f=zz if label=='dense' else ff
   with torch.autocast('cuda',dtype=torch.bfloat16): lp,_=m(x,f,torch.zeros(x.shape,device='cuda'))
   loss=(-lp.gather(-1,y[...,None])[...,0]).float().cpu().numpy()
   if baseline is None:baseline=loss.copy()
   entry=dict(label=label,rep=rep,nonfinite=int((~np.isfinite(loss)).sum()),bit_differences=int((loss.view(np.uint32)!=baseline.view(np.uint32)).sum()),max_abs_difference=float(np.nanmax(np.abs(loss-baseline))),first_bad=np.argwhere(~np.isfinite(loss))[:10].tolist(),first=float(loss[0,0]))
   print(json.dumps(entry),flush=True);result.append(entry)
write('inference_repeat16_diagnostic.json',dict(results=result,scope='first-shard all models resident; alternating normal eval and telemetry suppression; exploratory only'))
