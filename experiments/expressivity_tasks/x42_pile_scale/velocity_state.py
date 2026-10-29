"""Atomic generation-level CMA-ES state; candidates restart from scratch."""
import json
import os
from pathlib import Path
import numpy as np
from .cmaes import CMAES


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def save_generation(path,es,registration_hash,completed):
    state={}
    for key,value in vars(es).items():
        if key=="rng":continue
        state[key]=value.tolist() if isinstance(value,np.ndarray) else value
    state['rng']=es.rng.bit_generator.state
    atomic_json(path,dict(version=1,registration_hash=registration_hash,cma=state,completed=completed))


def restore_generation(path,registration_hash):
    saved=json.loads(Path(path).read_text())
    if saved['version']!=1 or saved['registration_hash']!=registration_hash:raise ValueError('generation identity mismatch')
    s=saved['cma'];es=CMAES(s['mean'],population=s['population'])
    array_fields={'mean','w','C','pc','ps','B','D','y','x'}
    for key,value in s.items():
        if key=='rng':es.rng.bit_generator.state=value
        else:setattr(es,key,np.asarray(value) if key in array_fields else value)
    return es,saved['completed']
