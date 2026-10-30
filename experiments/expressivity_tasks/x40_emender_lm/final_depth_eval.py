"""Attended, checkpoint-bound final evaluation; never performs training updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import torch
from . import depth
from .model import LM
from ..x42_pile_scale.checkpoint import identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_binding(state, checkpoint_sha, registration, amendment=None):
    """Accept exact registration or an explicit artifact-bound amendment only."""
    current=identity(registration)
    if state['version'] != 1:
        raise ValueError('checkpoint version mismatch')
    if state['identity']==current:
        return 'current'
    if (not amendment or amendment.get('checkpoint_sha256')!=checkpoint_sha
            or amendment.get('recorded_identity')!=state['identity']
            or amendment.get('rebound_identity')!=current
            or amendment.get('checkpoint_update')!=state['step']
            or not amendment.get('cause_chain')):
        raise ValueError('checkpoint registration mismatch: amendment absent or not covered')
    return 'amendment-covered'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    failure=args.output/'depth_eval_failure.json'
    try:
        held=os.environ.get('GPU_LEASE_HELD')
        if held not in ('0','1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
            raise RuntimeError('single broker lease required on 0/1/5/6')
        torch.set_num_threads(1)
        digest=sha(args.checkpoint)
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        registration=json.loads(args.registration.read_text())
        amendment=json.loads(args.amendment.read_text())
        binding=validate_binding(state,digest,registration,amendment)
        if state['step']!=30000:raise ValueError('fine-tune incomplete')
        for path,expected in registration['sources'].items():
            if sha(path)!=expected:raise ValueError('evaluation source drift: '+path)
        os.environ.update(X40_STABLE_MAPPING='1',X40_FRESH_POOL='1',X40_FUSED_INIT='1',X40_BATCHED_READS='1')
        probe=depth.Probe(LM(state['model']['lm.head.weight'].shape[0],ignition=True).cuda())
        # Match the original fine_tune selection rule, not the last optimizer weights.
        probe.load_state_dict(state['cursor']['weights'])
        bins={}
        for name in depth.ladder.BIN_ORDER:
            bins[name]=depth.evaluate(probe,depth.ladder.ladder_bin_pool(name))
            print(json.dumps(dict(bin=name,report=bins[name])),flush=True)
        report=dict(status='completed',updates=state['step'],selected_update=state['cursor']['selected'],
            selected_validation=state['cursor']['best'],bins=bins,checkpoint_sha256=digest,
            registration_sha256=sha(args.registration),amendment_sha256=sha(args.amendment),
            binding=binding,physical_gpu=held,supervision='final EOS binary margin BCE',budget=2944,
            deviations=['LM pretraining replaces X28E MQAR and occupancy warmup',
                        'new scalar-token embedding and binary head on pretrained six-block hidden states'])
        (args.output/'depth.json').write_text(json.dumps(report,indent=2)+'\n')
    except BaseException as exc:
        failure.write_text(json.dumps(dict(status='failed',error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n')
        raise


if __name__=='__main__':main()
