"""Repeat historical failing arms with allocator history; preserve original receipts."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import torch
from . import scale_surface
from .part6_allocator import summarize


def main():
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ['GPU_LEASE_HELD']
    root=Path(__file__).resolve().parent/'receipts/part6/baseline'
    root.mkdir(exist_ok=True)
    for name in ('production_parity.json','production_T2048.json','reads_parity.json'):
        shutil.copyfile(scale_surface.ROOT/name,root/name)
    scale_surface.ROOT=root
    torch.cuda.memory._record_memory_history(enabled='all',context='all',stacks='python',max_entries=200000)
    try:
        scale_surface.main()
    finally:
        snapshot=torch.cuda.memory._snapshot()
        result=summarize(snapshot)
        result.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                      physical_gpu=os.environ.get('GPU_LEASE_HELD'),python=sys.executable,command=sys.argv,
                      analyzer_sha256=hashlib.sha256((Path(__file__).parent/'part6_allocator.py').read_bytes()).hexdigest())
        args=sys.argv
        name='_'.join((args[args.index('--scale')+1],args[args.index('--arm')+1],
                       'B'+args[args.index('--batch')+1],'T'+args[args.index('--context')+1]))
        (root/f'{name}_allocator.json').write_text(json.dumps(result,indent=2)+'\n')
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__=='__main__':main()
