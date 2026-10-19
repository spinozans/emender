#!/usr/bin/env python3
"""Independently verify W3's wash-1 x -> FP32 y -> fresh-state seed recipe."""
import argparse
import json
from pathlib import Path

import torch

from scripts.w3_matcheddose_preflight import ROOT, PARENT_SHA, checked, file_sha


def verify(root):
    directory = root / 'wash-v1/checkpoints'
    original = directory / 'checkpoint_agent_sft_u000256_loss_0.5701.pt'
    materialized = directory / 'wash_u256_y_materialized_seed.pt'
    fresh = directory / 'wash_u256_y_seed_fresh_state.pt'
    bindings = [checked(original, '2f2b209d3b13f5ed5a7205aefa15188091cfd7fd1bc366d0ea414f2e012457e0'),
                checked(materialized, '4132b84afdd2cff9c06a88723a8361f9af31644917fbac58d1b27fcb2289f1f2'),
                checked(fresh, PARENT_SHA)]
    x, y, seed = [torch.load(p, map_location='cpu', mmap=True, weights_only=False)
                  for p in (original, materialized, fresh)]
    groups = x['optimizer_state_dict']['param_groups']
    state = x['optimizer_state_dict']['state']
    if len(groups) != 1 or groups[0]['train_mode'] is not False or groups[0]['betas'] != (.9, .95):
        raise ValueError('original SF state semantics')
    from ndm.e97 import build_e97_model
    config = json.loads(Path('/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json').read_text())
    with torch.device('meta'):
        layout = build_e97_model(config, vocab_size=50281, use_triton=False)
    names = [name for name, _ in layout.named_parameters()]
    parameter_ids = groups[0]['params']
    if len(parameter_ids) != len(names):
        raise ValueError('model/optimizer layout mismatch')
    # The production head is tied to embedding; state_dict contains both names.
    if layout.lm_head.weight is not layout.embedding.weight:
        raise ValueError('expected tied production head')
    rows = []
    total = 0
    for name, parameter_id in zip(names, parameter_ids):
        stored_x = x['model_state_dict'][name]
        # The historical facade built FP32 parameters, copied BF16 x, then
        # load_state_dict cast z to FP32 before SF.train() used lerp_(z, .1).
        reconstructed = stored_x.float().lerp(state[parameter_id]['z'].float(), 1 - .9)
        if not torch.equal(reconstructed, y['model_state_dict'][name]):
            raise ValueError(f'x->y reconstruction mismatch: {name}')
        if not torch.equal(y['model_state_dict'][name], seed['model_state_dict'][name]):
            raise ValueError(f'materialized->fresh seed weights differ: {name}')
        total += stored_x.numel()
        rows.append(dict(name=name, shape=list(stored_x.shape), original_dtype=str(stored_x.dtype),
                         materialized_dtype=str(y['model_state_dict'][name].dtype),
                         seed_dtype=str(seed['model_state_dict'][name].dtype), equal=True))
    for checkpoint in (x, y, seed):
        if not torch.equal(checkpoint['model_state_dict']['embedding.weight'], checkpoint['model_state_dict']['lm_head.weight']):
            raise ValueError('tied head storage mismatch')
    fresh_state = seed['optimizer_state_dict']
    if fresh_state['state'] or len(fresh_state['param_groups']) != 1:
        raise ValueError('seed optimizer is not fresh')
    group = fresh_state['param_groups'][0]
    if any(group[k] != v for k, v in dict(k=0, warmup_steps=0, train_mode=False,
                                         weight_sum=0., lr_max=-1., scheduled_lr=0.,
                                         lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01).items()):
        raise ValueError('seed fresh optimizer group mismatch')
    if total != 4045972080:
        raise ValueError('production parameter count')
    return dict(schema='emender-w3-parent-conversion-v1', status='passed', files=bindings,
                script_sha256=file_sha(Path(__file__)), parameters=total, tensors=rows,
                recipe='load original saved x into FP32 model; cast z to FP32; per parameter x.lerp_(z, 1-beta1) with beta1=.9; replace model_state_dict; discard old state; fresh SF group k=0/warmup=0/state={}',
                fresh_group={k: v for k, v in group.items() if k != 'params'},
                metadata=seed['checkpoint_metadata'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = verify(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print('W3_PARENT_CONVERSION passed: 236 unique tensors (tied head verified), 4,045,972,080 parameters, fresh state empty')


if __name__ == '__main__':
    main()
