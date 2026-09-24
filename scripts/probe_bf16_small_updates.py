#!/usr/bin/env python3
"""Synthetic constant-gradient writeback experiment, not an Adam/training run."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from ndm.bf16_rounding_probe import stochastic_round_bf16
from ndm.e97_atomic import publish_bytes_no_replace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    records=[]
    for magnitude in (0.001,0.01,1.0):
        for lr in (2e-6,5e-5):
            initial=torch.tensor(magnitude,dtype=torch.bfloat16)
            nearest=initial.repeat(65536)
            stochastic=nearest.clone()
            rng=torch.Generator().manual_seed(927413)
            for _ in range(100):
                nearest.add_(-lr)
                stochastic=stochastic_round_bf16(stochastic.float()-lr,generator=rng)
            target=float(initial)-100*lr
            records.append({'initial_bf16':float(initial),'lr':lr,'steps':100,
                            'normalized_gradient':1.0,'coordinates':nearest.numel(),
                            'ideal_real_arithmetic_endpoint':target,
                            'rne_mean':float(nearest.double().mean()),
                            'stochastic_mean':float(stochastic.double().mean()),
                            'rne_unchanged_fraction':float((nearest==initial).double().mean()),
                            'stochastic_unchanged_fraction':float((stochastic==initial).double().mean())})
    result={'schema':'emender-bf16-small-update-probe-v1','device':'cpu',
            'production_optimizer_changed':False,'persistent_fp32_master_weights':False,
            'scope':'synthetic constant normalized-gradient subtraction only; not Schedule-Free or model convergence',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records}
    publish_bytes_no_replace(args.output,(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print(json.dumps(result))

if __name__=='__main__':
    main()
