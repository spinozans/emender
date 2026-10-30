import os,json,torch
from pathlib import Path
from experiments.expressivity_tasks.x40_emender_lm import depth
from experiments.expressivity_tasks.x40_emender_lm.model import LM
from unittest.mock import patch
from experiments.expressivity_tasks.x40_emender_lm import kernels,model
from experiments.expressivity_tasks.x31_tape_cell_fused.lazy_decay import advance_exact,death_exact
import x40_part15_reducers as diag
for func,replacements in ((kernels.scan_kernel,[('tl.sum(r_live.to(tl.int32),0)','tl.reduce(r_live.to(tl.int32),0,combine_live)'),('tl.sum(((hist>0) & (hist!=4)).to(tl.int32), 0)','tl.reduce(((hist>0) & (hist!=4)).to(tl.int32),0,combine_requests)'),('tl.sum((hist == 5).to(tl.int32), 0)','tl.reduce((hist == 5).to(tl.int32),0,combine_caps)')]),(advance_exact,[('tl.sum((remaining > 0).to(tl.int32), 0)','tl.reduce((remaining > 0).to(tl.int32),0,combine_remaining)')]),(death_exact,[('tl.sum(active.to(tl.int32), 0)','tl.reduce(active.to(tl.int32),0,combine_active)')])):
 src=func.src
 for old,new in replacements:
  assert old in src;src=src.replace(old,new)
 for name in ('combine_live','combine_requests','combine_caps','combine_remaining','combine_active'):
  func.fn.__globals__[name]=getattr(diag,name)
 func._unsafe_update_src(src)
kernels.scan_kernel.hash=None
original_scan=model.scan
call=0
def logged_scan(*args,**kwargs):
 global call
 call+=1;print('SCAN',call,'shape',tuple(args[0].shape),'config',args[6],flush=True)
 return original_scan(*args,**kwargs)
model.scan=logged_scan
os.environ.update(X40_STABLE_MAPPING='1',X40_FRESH_POOL='1',X40_FUSED_INIT='1',X40_BATCHED_READS='0')
torch.set_num_threads(1)
p=Path('experiments/expressivity_tasks/x40_emender_lm/receipts/part12_checkpoint_managed/training/runs/depth.ckpt')
s=torch.load(p,map_location='cpu',weights_only=False)
probe=depth.Probe(LM(s['model']['lm.head.weight'].shape[0],ignition=True).cuda())
probe.load_state_dict(s['cursor']['weights']);probe.eval()
pool=depth.ladder.ladder_bin_pool('bin_9x')
with torch.no_grad(),patch.object(depth.memory_model,'profile',depth.depth_profile):
 for start in range(80,88,8):
  ln=pool[2][start:start+8]; print(json.dumps(dict(start=start,lengths=ln.tolist(),gpu=os.environ['GPU_LEASE_HELD'])),flush=True)
  ids=torch.from_numpy(pool[0][start:start+8,:int(ln.max())].copy()).cuda()
  out=probe(ids,torch.from_numpy(ln).cuda());torch.cuda.synchronize()
  print('passed',start,flush=True)
