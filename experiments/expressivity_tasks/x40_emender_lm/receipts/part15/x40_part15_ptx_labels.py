import re,subprocess,hashlib,json
from pathlib import Path
from experiments.expressivity_tasks.x40_emender_lm import forward
original=forward.scan_kernel
out=Path('experiments/expressivity_tasks/x40_emender_lm/receipts/part15')
class Labels:
 def __getitem__(self,grid):
  def launch(*args,**kwargs):
   kernel=original.warmup(*args,grid=grid,**kwargs)
   if not getattr(kernel,'part15_labeled',False):
    assert kernel.module is None
    ptx=kernel.asm['ptx']
    def label(m):
     label=f'P15{int(m[1]):04d}'
     assert len(label)==7
     return '.global .align 1 .b8 assertFunc_'+m[1]+'[8] = {'+', '.join(str(ord(c)) for c in label)+'};'
    changed=re.sub(r'\.global \.align 1 \.b8 assertFunc_(\d+)\[8\] = \{[^}]+\};',label,ptx)
    normalize=lambda s:re.sub(r'\.global \.align 1 \.b8 assertFunc_\d+\[8\] = \{[^}]+\};','ASSERTFUNC',s)
    assert normalize(ptx)==normalize(changed)
    root=Path('/tmp/x40_part15_'+kernel.hash)
    root.with_suffix('.ptx').write_text(changed)
    subprocess.run(['/home/erikg/emender/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas','-lineinfo','--fmad=false','--gpu-name=sm_89',str(root.with_suffix('.ptx')),'-o',str(root.with_suffix('.cubin'))],check=True)
    kernel.kernel=root.with_suffix('.cubin').read_bytes();kernel.part15_labeled=True
    locations=[]
    lines=ptx.splitlines()
    for i,line in enumerate(lines):
     m=re.search(r'mov.u64\s+%rd\d+, assertFunc_(\d+)',line)
     if m:locations.append(dict(label='P15'+m[1].zfill(4),ptx_line=i+1,context=lines[max(0,i-15):i+4]))
    (out/('ptx_labels_'+kernel.hash[:12]+'.json')).write_text(json.dumps(dict(kernel_hash=kernel.hash,original_ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),labeled_ptx_sha256=hashlib.sha256(changed.encode()).hexdigest(),instruction_text_unchanged=True,locations=locations),indent=2)+'\n')
   return original[grid](*args,**kwargs)
  return launch
forward.scan_kernel=Labels()
