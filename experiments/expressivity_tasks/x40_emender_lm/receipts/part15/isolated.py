import os,sys,json,torch,hashlib
from experiments.expressivity_tasks.x40_emender_lm import kernels,model
from experiments.expressivity_tasks.x31_tape_cell_fused.lazy_decay import advance_exact,death_exact
mode=sys.argv[1]
if mode=='tagged':
 import x40_part15_reducers_pure as diag
 for func,replacements in ((kernels.scan_kernel,[('tl.sum(r_live.to(tl.int32),0)','tl.reduce(r_live.to(tl.int32),0,combine_live)'),('tl.sum(((hist>0) & (hist!=4)).to(tl.int32), 0)','tl.reduce(((hist>0) & (hist!=4)).to(tl.int32),0,combine_requests)'),('tl.sum((hist == 5).to(tl.int32), 0)','tl.reduce((hist == 5).to(tl.int32),0,combine_caps)')]),(advance_exact,[('tl.sum((remaining > 0).to(tl.int32), 0)','tl.reduce((remaining > 0).to(tl.int32),0,combine_remaining)')]),(death_exact,[('tl.sum(active.to(tl.int32), 0)','tl.reduce(active.to(tl.int32),0,combine_active)')])):
  src=func.src
  for old,new in replacements:
   assert old in src;src=src.replace(old,new)
  for name in ('combine_live','combine_requests','combine_caps','combine_remaining','combine_active'):func.fn.__globals__[name]=getattr(diag,name)
  func._unsafe_update_src(src)
 kernels.scan_kernel.hash=None
os.environ.update(X40_STABLE_MAPPING='1',X40_FRESH_POOL='1',X40_FUSED_INIT='1',X40_BATCHED_READS='0')
torch.set_num_threads(1)
args,kwargs=torch.load('/tmp/x40_part15_scan10.pt',weights_only=False)
args=tuple(x.cuda() if torch.is_tensor(x) else x for x in args)
print('mode',mode,'shape',args[0].shape,'cfg',args[6],flush=True)
with torch.no_grad():out=model.scan(*args,**kwargs);torch.cuda.synchronize()
print('PASSED',flush=True)
