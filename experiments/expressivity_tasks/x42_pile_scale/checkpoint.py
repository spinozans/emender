"""Atomic update-boundary checkpoints, including optimizer, RNG and sampler cursor."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def identity(registration):
    return hashlib.sha256(json.dumps(registration,sort_keys=True).encode()).hexdigest()


def save(path, model, optimizer, registration, step, cursor, rows, elapsed):
    path=Path(path)
    state=dict(version=1,identity=identity(registration),model=model.state_dict(),
               optimizer=optimizer.state_dict(),step=step,cursor=cursor,rows=rows,elapsed=elapsed,
               torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
               python_rng=random.getstate(),numpy_rng=np.random.get_state())
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        torch.save(state,f); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def restore(path, model, optimizer, registration):
    # Only locally produced trusted checkpoints are accepted. Never load remote artifacts.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['version']!=1 or state['identity']!=identity(registration):
        raise ValueError('checkpoint registration mismatch')
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state(state['cuda_rng'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    return {k:state[k] for k in ('step','cursor','rows','elapsed')}
