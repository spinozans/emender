"""Process-local diagnostics; no production kernel substitutions."""
import os, sys, json, runpy
from pathlib import Path
import torch
from experiments.expressivity_tasks.x40_emender_lm import model, kernels
mode = sys.argv[1]
os.environ.update(X40_STABLE_MAPPING=os.environ.get('PART16_STABLE','1'), X40_FRESH_POOL='1',X40_FUSED_INIT='1',X40_BATCHED_READS='0')
torch.set_num_threads(1)
held=[]
original=model.scan
call=0
def scan(*args,**kwargs):
    global call
    call+=1
    padding=int(os.environ.get('PART16_PADDING','0'))
    if padding:
        held.append(torch.empty(padding,device='cuda',dtype=torch.uint8))
    print(json.dumps(dict(event='scan',call=call,shape=list(args[0].shape),stable=os.environ['X40_STABLE_MAPPING'],padding=padding)),flush=True)
    result=original(*args,**kwargs)
    torch.cuda.synchronize()
    print(json.dumps(dict(event='scan_pass',call=call)),flush=True)
    return result
model.scan=scan
if os.environ.get('PART16_ADDRESSES')=='1':
    run=kernels.scan_kernel.run
    def logged_run(*args,**kwargs):
        rows=[]
        for name,x in list(zip(kernels.scan_kernel.arg_names,args))+list(kwargs.items()):
            if torch.is_tensor(x):
                ptr=x.data_ptr()
                rows.append(dict(name=name,base=hex(ptr),storage_base=hex(x.untyped_storage().data_ptr()),offset=x.storage_offset(),mod256=ptr%256,mod4096=ptr%4096,shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),bytes=x.numel()*x.element_size()))
        print(json.dumps(dict(event='kernel_inputs',call=call,inputs=rows)),flush=True)
        return run(*args,**kwargs)
    kernels.scan_kernel.run=logged_run
if mode=='isolated':
    args,kwargs=torch.load('/tmp/x40_part15_scan10.pt',weights_only=False)
    args=tuple(x.cuda() if torch.is_tensor(x) else x for x in args)
    with torch.no_grad():
        for i in range(100):
            result=model.scan(*args,**kwargs)
            del result
    print('ISOLATED_100_PASSED',flush=True)
else:
    # Reuse the exact production batch setup, removing only its fixed flag override
    # and its scan wrapper so this harness can vary flags and record launches.
    source=Path('experiments/expressivity_tasks/x40_emender_lm/receipts/part15/single.py').read_text()
    source=source[source.index('torch.set_num_threads(1)'):]
    from experiments.expressivity_tasks.x40_emender_lm import depth
    from experiments.expressivity_tasks.x40_emender_lm.model import LM
    from unittest.mock import patch
    exec(compile(source,'part15_single_setup','exec'))
