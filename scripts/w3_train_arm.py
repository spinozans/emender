#!/usr/bin/env python3
"""Fixed W3 production arm: exact 512-update historical replay, no evaluation."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import schedulefree
import torch

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
from scripts.train_e97_4b_pi_sft import atomic_save
from scripts.w3_adamw_viability import production_model
from scripts.w3_fix_regime_branch import X9B, fix_branch
from scripts.w3_matcheddose_preflight import ROOT, REPO, PARENT_SHA, AUTHORITY_SHA, PACK_SHA, checked, file_sha, digest


def learning_rate(update, regime):
    if not 1 <= update <= 512 or regime not in ('sf-constant', 'sf-cosine', 'adamw-cosine'):
        raise ValueError('unregistered W3 schedule')
    return 1e-5 if regime == 'sf-constant' else 1e-7 + .5 * (1e-5 - 1e-7) * (1 + math.cos(math.pi * (update - 1) / 511))


def publish(path, value):
    # Evidence is create-once. Model checkpoint publication uses atomic_save.
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def pin(source, directory, expected):
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    checked(source, expected)
    target = directory / source.name
    subprocess.run(['cp', '--reflink=auto', str(source), str(target)], check=True)
    checked(target, expected)
    publish(directory / 'pin-receipt.json', dict(schema='emender-w3-pin-v1',
            checkpoint=target.name, sha256=expected, source=str(source),
            source_sha256=expected, pinned_before_gate_read=True))
    return target


def verify_sources(freeze):
    for binding in freeze['source_bindings']:
        checked(binding['path'], binding['sha256'])


def verify_batch(data, index, row):
    values = data.pack_at_with_boundaries(data.pack_id_at(index))
    tokens, loss, valid, reset, length, targets, name = values
    observed = (name, length, targets, [data.sample_id(index)],
                digest(tokens.numpy().astype('<u4').tobytes()), digest(loss.numpy().tobytes()),
                digest(valid.numpy().tobytes()), digest(reset.numpy().tobytes()))
    expected = (row['pack_id'], row['inputs'], row['targets'], row['sample_ids'], row['tokens_sha256'],
                row['loss_mask_sha256'], row['valid_mask_sha256'], row['reset_mask_sha256'])
    if observed != expected:
        raise ValueError(f'W3 identity/dose divergence before update {index+1}; invalid/inconclusive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--branch', type=Path, required=True)
    parser.add_argument('--x9b-root', type=Path, default=X9B)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--viability', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    branch = json.loads(args.branch.read_text())
    if branch != fix_branch(args.x9b_root):
        raise ValueError('branch does not bind verified X9b final receipt')
    prereg = json.loads(args.preregistration.read_text())
    if prereg['status'] != 'frozen-before-production-results':
        raise ValueError('missing preregistration')
    for binding in prereg['bindings']:
        checked(binding['path'], binding['sha256'])
    # Preregistration bytes themselves must be committed, not just a receipt
    # naming an unrelated commit. Branch receipt is separately frozen prelaunch.
    for receipt_path in (args.preregistration, args.branch):
        relative = str(receipt_path.resolve().relative_to(REPO))
        committed = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=REPO)
        if digest(committed) != file_sha(receipt_path):
            raise ValueError('preregistration/branch receipt is not committed')
    viability = json.loads(args.viability.read_text())
    if viability['status'] != 'passed' or viability['point_hashes']['train'] != viability['point_hashes']['saved']:
        raise ValueError('ordinary AdamW production viability missing')
    preflight = json.loads(args.preflight.read_text())
    freeze = json.loads(args.freeze.read_text())
    if freeze['preflight_sha256'] != file_sha(args.preflight):
        raise ValueError('runtime freeze/recount binding mismatch')
    verify_sources(freeze)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    parent = ROOT / 'wash-v1/checkpoints/wash_u256_y_seed_fresh_state.pt'
    entry = pin(parent, args.output / 'entry-pinned', PARENT_SHA)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    model, precision = production_model(entry)
    regime = 'sf-constant' if args.arm == 'A' else branch['regime']
    ordinary = regime == 'adamw-cosine'
    if ordinary:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .95),
                                     eps=1e-8, weight_decay=.01, foreach=False, fused=False)
    else:
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-5,
                    betas=(.9, .95), eps=1e-8, weight_decay=.01, warmup_steps=0,
                    r=0., weight_lr_power=2., foreach=True)
    if optimizer.state:
        raise ValueError('arm optimizer is not fresh')
    if not ordinary:
        optimizer.train()
    identity = SFTSamplerIdentity.from_metadata(freeze['sampler_identity'])
    data = MaskedSFTPackedDataset(ROOT / 'wash-v2/authority', ROOT / 'wash-v2/packs',
                                 identity=identity, rank=0, sampler_mode='epoch-permutation')
    spec = importlib.util.spec_from_file_location('w3_historical_step', REPO / 'scripts/rl-loop-v1/rl_train_step.py')
    step = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step)
    publish(args.output / 'start-receipt.json', dict(schema='emender-w3-arm-start-v1', arm=args.arm,
                regime=regime, branch_sha256=file_sha(args.branch), preregistration_sha256=file_sha(args.preregistration),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                preflight_sha256=file_sha(args.preflight), runtime_freeze_sha256=file_sha(args.freeze),
                parent_sha256=PARENT_SHA, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu=torch.cuda.get_device_name(0), world_size=1, context_size=12288,
                precision=precision, requested_steps=512, no_retry=True, no_threshold_stop=True,
                no_best_checkpoint_selection=True, grad_clip=1., entry_pinned=str(entry)))
    total_tokens = total_targets = 0
    cumulative_weighted_exposure = cumulative_lr_weighted_exposure = 0.
    losses = []
    with (args.output / 'training.jsonl').open('x', buffering=1) as log:
        for update, expected in enumerate(preflight['exposure']['sequence'], 1):
            verify_batch(data, update - 1, expected)
            lr = learning_rate(update, regime)
            for group in optimizer.param_groups:
                group['lr'] = lr
            began = time.monotonic()
            result = step.train_pack_update(model, optimizer, data, sample_count=1,
                                            device=torch.device('cuda:0'), grad_clip=1.)
            if (result['targets'], result['tokens'], result['sample_ids']) != (
                    expected['targets'], expected['inputs'], expected['sample_ids']):
                raise ValueError('actual update dose/identity divergence; invalid/inconclusive')
            actual_lr = optimizer.param_groups[0]['lr' if ordinary else 'scheduled_lr']
            if actual_lr != lr or not math.isfinite(result['loss']):
                raise ValueError('actual LR mismatch/nonfinite loss; invalid/inconclusive')
            total_tokens += result['tokens']
            total_targets += result['targets']
            cumulative_weighted_exposure += result['targets'] * expected['loss_weight']
            cumulative_lr_weighted_exposure += actual_lr * result['targets'] * expected['loss_weight']
            losses.append(result['loss'])
            log.write(json.dumps(dict(update=update, scheduled_lr=lr, actual_lr=actual_lr,
                      cumulative_targets=total_targets, cumulative_inputs=total_tokens,
                      target_objective_weight=expected['loss_weight'],
                      cumulative_objective_weighted_exposure=cumulative_weighted_exposure,
                      cumulative_lr_weighted_exposure=cumulative_lr_weighted_exposure,
                      seconds=time.monotonic()-began, **result), sort_keys=True) + '\n')
    if (total_tokens, total_targets, data.next_absolute_rank_sample_index) != (3571182, 154386, 512):
        raise ValueError('final exposure mismatch; invalid/inconclusive')
    verify_sources(freeze)
    if not ordinary:
        optimizer.eval()
    point = 'live-nonaveraged' if ordinary else 'saved-eval-x'
    checkpoint = args.output / 'checkpoints/fixed-final-u000512.pt'
    atomic_save(checkpoint, dict(schema='emender-e97-4b-pi-masked-sft-v1', model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), weight_mode=point, sft_precision=precision,
                optimizer_identity=dict(schema='emender-e97-optimizer-identity-v1',
                  **{'class': 'torch.optim.AdamW' if ordinary else 'schedulefree.AdamWScheduleFree'},
                  version=torch.__version__ if ordinary else version('schedulefree'), weight_point=point),
                sft_updates=512, sft_total_tokens=total_tokens, assistant_target_tokens=total_targets,
                loss=sum(losses)/512, parent_checkpoint_sha256=PARENT_SHA,
                authority_manifest_sha256=AUTHORITY_SHA, pack_manifest_sha256=PACK_SHA,
                sampler_key=identity.sampler_key, sampler_cursor=512, data_world_size=1,
                context_size=12288, branch_sha256=file_sha(args.branch)))
    checkpoint_sha = file_sha(checkpoint)
    pinned = pin(checkpoint, args.output / 'final-pinned', checkpoint_sha)
    publish(args.output / 'complete-receipt.json', dict(schema='emender-w3-arm-complete-v1', status='complete',
                arm=args.arm, regime=regime, updates=512, inputs=total_tokens, targets=total_targets,
                cumulative_objective_weighted_exposure=cumulative_weighted_exposure,
                cumulative_lr_weighted_exposure=cumulative_lr_weighted_exposure,
                checkpoint=str(pinned), checkpoint_sha256=checkpoint_sha,
                training_log_sha256=file_sha(args.output / 'training.jsonl'),
                start_receipt_sha256=file_sha(args.output / 'start-receipt.json'),
                max_hbm_allocated=torch.cuda.max_memory_allocated(), max_hbm_reserved=torch.cuda.max_memory_reserved()))
    data.close()
    print('W3_ARM_COMPLETE', args.arm, checkpoint_sha, flush=True)


if __name__ == '__main__':
    main()
