"""Fixed-exposure evaluator used by search; selected model factory owns policies."""
import copy
import json
import time
from pathlib import Path
from .train import train


def evaluate(candidate, config, output, seeds, tokens, wall_seconds):
    """No architecture selection here: X40 verdict and adapter are mandatory inputs."""
    cfg = copy.deepcopy(config)
    if not cfg.get('model_factory'):
        raise RuntimeError('search requires X40-selected factory with policy/occupancy interface')
    cfg['search_constants'] = candidate
    cfg['tokens'] = tokens
    cfg['lr'] = candidate['lr']
    cfg['warmup_steps'] = candidate['warmup_steps']
    count = candidate['lane_count']
    center = round(candidate['lane_placement_fraction']*(cfg['layers']-1))
    cfg['lane_positions'] = sorted({(center+i)%cfg['layers'] for i in range(count)})
    result=[]
    start=time.monotonic()
    for i,seed in enumerate(seeds):
        remaining=wall_seconds-(time.monotonic()-start)
        if remaining <= 0:
            raise RuntimeError('candidate budget exhausted')
        cfg['wall_seconds']=remaining
        cfg['seed']=seed
        seed_output=Path(output)/f'seed{seed}'
        try:
            row=train(cfg,config['data'],seed_output,config['verdict'])
        except Exception as exc:
            if seed_output.exists():
                (seed_output/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(exc))))
            raise
        if 'occupancy_length_slope' not in row:
            raise RuntimeError('factory omitted occupancy profile; cannot score candidate')
        result.append(row)
    return dict(tokens_per_seed=[r['tokens'] for r in result],
                valid_nll=[r['valid_nll'] for r in result],
                perplexity=[r['perplexity'] for r in result],
                occupancy_length_slope=sum(r['occupancy_length_slope'] for r in result)/len(result),
                seconds=time.monotonic()-start)
