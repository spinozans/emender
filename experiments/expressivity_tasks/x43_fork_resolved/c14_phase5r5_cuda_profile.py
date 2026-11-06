"""Actual CUDA kernel-duration attribution via CUPTI, separate from wall samples."""
import json
import sys
from pathlib import Path
import torch
from . import c14_phase5r5_probe as probe

class Profile:
    calls=0
    def __getattr__(self,name):return getattr(original,name)
    def __getitem__(self,grid):
        def launch(*a,**kw):
            self.calls+=1
            if self.calls!=2:return original[grid](*a,**kw)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                result=original[grid](*a,**kw)
                torch.cuda.synchronize()
            by={}
            for event in prof.events():
                if event.device_type.name=='CUDA':
                    item=by.setdefault(event.name,dict(calls=0,device_us=0.))
                    item['calls']+=1;item['device_us']+=event.device_time_total
            output.write_text(json.dumps(dict(arm=arm,kernels=by,limitations='CUPTI profiled iteration; kernel times exclude gaps; uninstrumented wall times in separate matched sweep.'),indent=2)+'\n')
            assert by, 'No CUDA kernel events: profiling failed, not a zero-time kernel'
            return result
        return launch

if __name__=='__main__':
    arm=sys.argv[1];output=Path(sys.argv[2]);assert not output.exists()
    if arm=='r4':
        from .c14_phase5r4_stream_pages import kernel as original
    else:
        assert arm=='r5'
        original=probe.kernel
    probe.kernel=Profile()
    sys.argv=[sys.argv[0],'--only1024','--matched-r2','--output',str(output.with_name(f'cuda_{arm}_sweep.json'))]
    probe.main()
