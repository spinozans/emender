#!/usr/bin/env python3
"""One disposable full-4B AdamW update at wash-2 precision, not an experiment arm."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
from ndm.e97 import load_e97_checkpoint
from ndm.models.e97 import E97SplitEditLayer
from scripts.train_e97_4b_pi_sft import configure_precision, atomic_save
from scripts.w3_matcheddose_preflight import ROOT, PARENT_SHA, AUTHORITY_SHA, PACK_SHA, REPO, checked, file_sha

ARGS_PATH = Path('/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json')


def production_model(parent):
    loaded = load_e97_checkpoint(parent, args_json=ARGS_PATH, device='cuda:0',
                                 dtype=torch.bfloat16, weight_mode='train', use_triton=True, mmap=True)
    model = loaded.model.train()
    args = SimpleNamespace(optimizer_precision='legacy', lr=1e-5, weight_decay=.01, warmup_steps=0,
                           loss_logits_fp32=True, checkpoint_loss_chunks=True, loss_chunk_size=2048,
                           disable_bf16_reduced_precision_reduction=True, gradient_checkpoint_group_size=3,
                           mlp_checkpoint_chunk_size=16384, projection_chunk_size=2048)
    precision = configure_precision(model, args)
    model.gradient_checkpointing = True
    model.gradient_checkpoint_group_size = 3
    mlps, scans = 0, 0
    for module in model.modules():
        if hasattr(module, 'checkpoint_chunk_size'):
            module.checkpoint_chunk_size = 16384
            mlps += 1
        if isinstance(module, E97SplitEditLayer):
            if module.projection_chunk_size <= 0:
                raise ValueError('inherited chunk geometry missing')
            module.projection_chunk_size = 2048
            scans += 1
    if (mlps, scans) != (18, 18) or sum(p.numel() for p in model.parameters()) != 4045972080:
        raise ValueError('production architecture mismatch')
    return model, precision


def point_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    preflight = json.loads(args.preflight.read_text())
    parent = ROOT / 'wash-v1/checkpoints/wash_u256_y_seed_fresh_state.pt'
    checked(parent, PARENT_SHA)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    began = time.monotonic()
    model, precision = production_model(parent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .95),
                                 eps=1e-8, weight_decay=.01, foreach=False, fused=False)
    data = MaskedSFTPackedDataset(ROOT / 'wash-v2/authority', ROOT / 'wash-v2/packs',
                                 identity=SFTSamplerIdentity(AUTHORITY_SHA, PACK_SHA, 970001, 1, 12288),
                                 rank=0, sampler_mode='epoch-permutation')
    if data.pack_id_at(0) != int(preflight['exposure']['sequence'][0]['pack_id'].removeprefix('pack-')):
        raise ValueError('historical sampler differs')
    spec = importlib.util.spec_from_file_location('w3_historical_step', REPO / 'scripts/rl-loop-v1/rl_train_step.py')
    step = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step)
    result = step.train_pack_update(model, optimizer, data, sample_count=1,
                                    device=torch.device('cuda:0'), grad_clip=1.)
    state_dtypes = set()
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError('nonfinite AdamW production parameter')
        state = optimizer.state[p]
        for key in ('exp_avg', 'exp_avg_sq'):
            state_dtypes.add(str(state[key].dtype))
            if not torch.isfinite(state[key]).all():
                raise ValueError('nonfinite AdamW production moment')
    if state_dtypes != {'torch.bfloat16'}:
        raise ValueError('unexpected AdamW state precision')
    if (result['tokens'], result['targets']) != (6677, 482):
        raise ValueError('first-pack exposure differs')
    checkpoint = args.output / 'viability-only.pt'
    atomic_save(checkpoint, dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                weight_mode='live-nonaveraged', sft_precision=precision,
                optimizer_identity=dict(schema='emender-e97-optimizer-identity-v1',
                 **{'class': 'torch.optim.AdamW'}, version=torch.__version__, weight_point='live-nonaveraged')))
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del model, optimizer
    data.close()
    torch.cuda.empty_cache()
    hashes = {}
    for mode in ('train', 'saved'):
        loaded = load_e97_checkpoint(checkpoint, args_json=ARGS_PATH, device='cpu',
                                     dtype=torch.bfloat16, weight_mode=mode, use_triton=False)
        hashes[mode] = point_hash(loaded.model)
        del loaded
    if hashes['train'] != hashes['saved']:
        raise ValueError('AdamW live point differs across read modes')
    receipt = dict(schema='emender-w3-adamw-viability-v1', status='passed',
                   scope='one disposable preflight update; not arm B; no gate reads',
                   parent_sha256=PARENT_SHA, preflight_sha256=file_sha(args.preflight),
                   script_sha256=file_sha(Path(__file__)), args=checked(ARGS_PATH, '4e3563462d345e6c9f76daf24ca544c565ecb43aaf433a2b1ee5b70c07625b4c'),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0), physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                   precision=precision, parameters=4045972080, optimizer='torch.optim.AdamW',
                   betas=[.9, .95], eps=1e-8, weight_decay=.01, lr=1e-5,
                   foreach=False, fused=False, state_dtypes=sorted(state_dtypes),
                   result=result, point_hashes=hashes, max_hbm_allocated=peak_allocated,
                   max_hbm_reserved=peak_reserved, checkpoint=dict(path=str(checkpoint), sha256=file_sha(checkpoint)),
                   seconds=time.monotonic()-began)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
