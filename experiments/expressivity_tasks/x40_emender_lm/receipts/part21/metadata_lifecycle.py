"""Extend the 32 real lifecycle fixtures with allocation-sentinel neutrality.

At the very first launch, replace due/terminal/birth initialization with
finite sentinels, then a NaN terminal sentinel, then candidate zeros. All
later chunks retain their native metadata and exercise POP, frees and reuse.
This proves defined-state neutrality; the separate backward matrix is stronger
and remains a separate gate rather than being silently replaced by this test.
"""
import hashlib
import json
from pathlib import Path
import sys
import torch
from candidate import install

assert sys.executable=='/home/erikg/emender/.venv/bin/python'
root=Path(__file__).resolve().parent
print(json.dumps({'candidate_sources':install()}),flush=True)
source=(root.parent/'part20/lifecycle.py').read_text()
original=source
old='def __init__(self, value):\n        self.value = value'
assert source.count(old)==1
source=source.replace(old,'def __init__(self, value, variant):\n        self.value = value\n        self.variant = variant')
old='''            return kernels.scan_kernel[grid](*args, **kwargs)'''
assert source.count(old)==1
new='''            names=kernels.scan_kernel.arg_names
            if args[names.index('Fresh')] and args[names.index('Token')]==0:
                integer=0 if self.variant==2 else 1234567
                for name in ('Due','BirthStore'):
                    args[names.index(name)].fill_(integer)
                args[names.index('Terminal')].fill_(0. if self.variant==2 else self.value)
            return kernels.scan_kernel[grid](*args, **kwargs)'''
source=source.replace(old,new)
assert source.count('Poison(poison)')==1
source=source.replace('Poison(poison)','Poison(poison,variant)')
namespace={'__name__':'metadata_lifecycle_fixture'}
exec(compile(source,str(root.parent/'part20/lifecycle.py'),'exec'),namespace)
print(json.dumps({'original_fixture_sha256':hashlib.sha256(original.encode()).hexdigest(),
                  'extended_fixture_sha256':hashlib.sha256(source.encode()).hexdigest()}),flush=True)
result={target:namespace['validate'](target) for target in ('x40','x31')}
print(json.dumps({'status':'passed','fixtures':sum(map(len,result.values())),'details':result}),flush=True)
