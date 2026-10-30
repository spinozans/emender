import os,json,torch
from pathlib import Path
from experiments.expressivity_tasks.x40_emender_lm import depth
from experiments.expressivity_tasks.x40_emender_lm.model import LM
from unittest.mock import patch
from experiments.expressivity_tasks.x40_emender_lm import model
from experiments.expressivity_tasks.x40_emender_lm import kernels
mode=os.environ.get('PART15_MODE','live')
old={'live':'r_live.to(tl.int32)','requests':'((hist>0) & (hist!=4)).to(tl.int32)','caps':'(hist == 5).to(tl.int32)'}[mode]
assert old in kernels.scan_kernel.src
kernels.scan_kernel._unsafe_update_src(kernels.scan_kernel.src.replace(old,old.replace('tl.int32','tl.int64')))
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
