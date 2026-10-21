#!/usr/bin/env python3
"""W4 chill modulation: fixed 512-update replay, ordinary AdamW, no evaluation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
from scripts.train_e97_4b_pi_sft import atomic_save
from scripts.w3_adamw_viability import production_model
from scripts.w3_matcheddose_preflight import ROOT, REPO, PARENT_SHA, AUTHORITY_SHA, PACK_SHA, checked, file_sha, digest
from scripts.w4_chillwash import verify_registration, pin, publish
from scripts.w3_train_arm import verify_batch, verify_sources


def learning_rate(update):
    if not 1 <= update <= 512:
        raise ValueError('unregistered W4 update')
    return 2e-8 + .5 * (2e-6 - 2e-8) * (1 + math.cos(math.pi * (update - 1) / 511))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_registration(args.registration)
    args.preflight = REPO / 'receipts/w3-matcheddose/preflight.json'
    args.freeze = REPO / 'receipts/w3-matcheddose/runtime-freeze-v2.json'
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
    # The reused production helper fixes geometry, but its historical LR is metadata.
    precision = dict(precision, learning_rate=2e-6)
    regime = 'adamw-chill-cosine'
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(.9, .95),
                                 eps=1e-8, weight_decay=.01, foreach=False, fused=False)
    if optimizer.state:
        raise ValueError('W4 optimizer is not fresh')
    identity = SFTSamplerIdentity.from_metadata(freeze['sampler_identity'])
    data = MaskedSFTPackedDataset(ROOT / 'wash-v2/authority', ROOT / 'wash-v2/packs',
                                 identity=identity, rank=0, sampler_mode='epoch-permutation')
    spec = importlib.util.spec_from_file_location('w3_historical_step', REPO / 'scripts/rl-loop-v1/rl_train_step.py')
    step = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step)
    publish(args.output / 'start-receipt.json', dict(schema='emender-w4-start-v1',
                regime=regime, registration_sha256=file_sha(args.registration),
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
            lr = learning_rate(update)
            for group in optimizer.param_groups:
                group['lr'] = lr
            began = time.monotonic()
            result = step.train_pack_update(model, optimizer, data, sample_count=1,
                                            device=torch.device('cuda:0'), grad_clip=1.)
            if (result['targets'], result['tokens'], result['sample_ids']) != (
                    expected['targets'], expected['inputs'], expected['sample_ids']):
                raise ValueError('actual update dose/identity divergence; invalid/inconclusive')
            actual_lr = optimizer.param_groups[0]['lr']
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
    verify_registration(args.registration)
    point = 'live-nonaveraged'
    checkpoint = args.output / 'checkpoints/fixed-final-u000512.pt'
    atomic_save(checkpoint, dict(schema='emender-e97-4b-pi-masked-sft-v1', model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), weight_mode=point, sft_precision=precision,
                optimizer_identity=dict(schema='emender-e97-optimizer-identity-v1',
                  **{'class': 'torch.optim.AdamW'},
                  version=torch.__version__, weight_point=point),
                sft_updates=512, sft_total_tokens=total_tokens, assistant_target_tokens=total_targets,
                loss=sum(losses)/512, parent_checkpoint_sha256=PARENT_SHA,
                authority_manifest_sha256=AUTHORITY_SHA, pack_manifest_sha256=PACK_SHA,
                sampler_key=identity.sampler_key, sampler_cursor=512, data_world_size=1,
                context_size=12288, registration_sha256=file_sha(args.registration)))
    checkpoint_sha = file_sha(checkpoint)
    pinned = pin(checkpoint, args.output / 'final-pinned', checkpoint_sha)
    publish(args.output / 'complete-receipt.json', dict(schema='emender-w4-complete-v1', status='complete',
                regime=regime, updates=512, inputs=total_tokens, targets=total_targets,
                cumulative_objective_weighted_exposure=cumulative_weighted_exposure,
                cumulative_lr_weighted_exposure=cumulative_lr_weighted_exposure,
                checkpoint=str(pinned), checkpoint_sha256=checkpoint_sha,
                training_log_sha256=file_sha(args.output / 'training.jsonl'),
                start_receipt_sha256=file_sha(args.output / 'start-receipt.json'),
                max_hbm_allocated=torch.cuda.max_memory_allocated(), max_hbm_reserved=torch.cuda.max_memory_reserved()))
    data.close()
    print('W4_TRAINING_COMPLETE', checkpoint_sha, flush=True)


if __name__ == '__main__':
    main()
