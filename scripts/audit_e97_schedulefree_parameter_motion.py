#!/usr/bin/env python3
"""Read-only endpoint-motion audit. Unchanged endpoints do not prove lost updates.

Recover train/y using the stock BF16 lerp convention on CPU; do not claim this
is a GPU writeback trace or a measurement of historical per-step gradients.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from ndm.e97_atomic import publish_bytes_no_replace


def load(path, expected):
    with path.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('checkpoint hash mismatch')
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model = checkpoint['model_state_dict']
    optimizer = checkpoint['optimizer_state_dict']
    # These dense checkpoints have only parameter tensors, with tied head aliases.
    # Validate alias-deduplicated parameter order against every optimizer slot.
    unique, seen = [], set()
    for name, tensor in model.items():
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
        if key not in seen:
            unique.append((name, tensor)); seen.add(key)
    groups = [(pid, group) for group in optimizer['param_groups'] for pid in group['params']]
    if len(groups) != len(unique):
        raise ValueError('cannot prove dense parameter/optimizer mapping')
    mapped = {}
    for (name, x), (pid, group) in zip(unique, groups, strict=True):
        state = optimizer['state'][pid]
        z = state['z']
        if x.dtype != torch.bfloat16 or z.dtype != x.dtype or z.shape != x.shape:
            raise ValueError('parameter/optimizer layout mismatch')
        if group['train_mode']:
            raise ValueError('audit requires saved/x checkpoints')
        mapped[name] = (x, z, float(group['betas'][0]))
    return checkpoint, mapped


def stats():
    return {'elements': 0, 'unchanged': 0, 'max_abs_delta': 0.0, 'delta_squared_sum': 0.0}


def update(result, value, baseline):
    if not torch.isfinite(value).all() or not torch.isfinite(baseline).all():
        raise ValueError('nonfinite parameter')
    result['elements'] += value.numel()
    result['unchanged'] += int(torch.count_nonzero(value == baseline))
    delta = value.float() - baseline.float()
    result['max_abs_delta'] = max(result['max_abs_delta'], float(delta.abs().max()))
    result['delta_squared_sum'] += float(delta.square().sum(dtype=torch.float64))


def finish(result):
    return {**result, 'unchanged_fraction': result['unchanged']/result['elements'],
            'rms_delta': (result['delta_squared_sum']/result['elements'])**0.5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--candidate-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError(args.output)
    parent, before = load(args.parent, args.parent_sha256)
    candidate, after = load(args.candidate, args.candidate_sha256)
    if before.keys() != after.keys():
        raise ValueError('parameter names differ')
    total = {mode: stats() for mode in ('saved_x', 'reconstructed_train_y', 'z')}
    parameters = {}
    for name, (old_x, old_z, old_beta) in before.items():
        x, z, beta = after[name]
        if x.shape != old_x.shape:
            raise ValueError('parameter shapes differ')
        local = {mode: stats() for mode in total}
        for start in range(0, x.numel(), 1 << 20):
            sl = slice(start, start+(1 << 20))
            baseline = old_x.reshape(-1)[sl].clone().lerp_(old_z.reshape(-1)[sl], 1-old_beta)
            sx, sz = x.reshape(-1)[sl], z.reshape(-1)[sl]
            y = sx.clone().lerp_(sz, 1-beta)
            for mode, value in (('saved_x', sx), ('reconstructed_train_y', y), ('z', sz)):
                update(local[mode], value, baseline)
        for mode, r in local.items():
            for k in ('elements','unchanged','delta_squared_sum'):
                total[mode][k] += r[k]
            total[mode]['max_abs_delta'] = max(total[mode]['max_abs_delta'], r['max_abs_delta'])
        parameters[name] = {mode: finish(r) for mode,r in local.items()}
    result = {'schema':'emender-e97-schedulefree-endpoint-motion-v1',
              'parent_sha256':args.parent_sha256, 'candidate_sha256':args.candidate_sha256,
              'candidate_updates':candidate['sft_updates'], 'candidate_lr':candidate['learning_rate'],
              'candidate_targets':candidate['assistant_target_tokens'],
              'baseline':'parent train/y reconstructed by stock BF16 lerp on CPU',
              'limitations':['endpoint equality does not distinguish rounding, small gradients or cancellation',
                             'CPU basis reconstruction, not a GPU per-step writeback audit'],
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'total':{mode:finish(r) for mode,r in total.items()}, 'parameters':parameters}
    publish_bytes_no_replace(args.output,(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print(json.dumps({k:v for k,v in result.items() if k!='parameters'}),flush=True)

if __name__=='__main__':
    main()
