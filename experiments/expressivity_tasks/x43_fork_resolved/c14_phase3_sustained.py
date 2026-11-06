"""Single-arm sustained training receipt; never promotes or overwrites a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from . import c14_b_train as train
from .c14_b_unbudgeted_profile import configure
from .c14_fast_features import load_decoder
from .c14_phase2_staged import install_rowkeys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=600)
    a = p.parse_args()
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('3', '5', '6')
    assert a.seconds >= 600 and not a.output.exists()
    a.output.mkdir(parents=True)
    wall = time.monotonic()
    with (a.output/'registration_check.json').open('w') as out:
        subprocess.run([sys.executable, 'scripts/registration_guard.py', 'check',
                        str(train.ROOT/'receipts/scalability_phase2/registration.json')], stdout=out, check=True)
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    random.seed(3401)
    np.random.seed(3401)
    configure()
    source, shards, _ = train.preflight(check_implementation=False)
    checkpoint_hash = train.sha(train.CHECKPOINT)
    assert checkpoint_hash == json.loads((train.OUT/'HARVEST.json').read_text())['checkpoint_sha256_verified']
    install_rowkeys()
    saved = torch.load(train.CHECKPOINT, map_location='cpu', weights_only=False)
    model = train.StoreIntegrated().cuda().train()
    model.load_state_dict(saved['model'])
    del saved
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = torch.Generator(device='cuda').manual_seed(3401)
    raw = np.memmap(source, dtype=np.uint8, mode='r')
    rows = iter(train.verified_batches(raw))
    decoder = load_decoder()
    steps = targets = cursor = 0
    active = excluded = 0.
    telemetry = {}
    receipt = dict(status='running', checkpoint_sha256=checkpoint_hash,
                   offsets_sha256=train.sha(train.OFFSETS), physical_gpu=os.environ['GPU_LEASE_HELD'],
                   shape=dict(batch=16, context=128, cap=257, ops='on', vocabulary=256),
                   trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
                   total_parameters=sum(x.numel() for x in model.parameters()),
                   clock='active includes shard loading/hash/decode, document verification, assembly, transfer, telemetry, update and synchronized completion; excludes receipt/checkpoint writes; first 20 updates retained as warmup')
    (a.output/'config.json').write_text(json.dumps(receipt, indent=2)+'\n')
    active_start = time.monotonic()
    try:
        with (a.output/'steps.jsonl').open('x') as log:
            for shard in shards:
                data = (train.CACHE/shard['file']).read_bytes()
                assert hashlib.sha256(data).hexdigest() == shard['sha256']
                identities, lengths, values, contexts, _ = train.oracle_shard(data, shard['windows'], decoder)
                for begin in range(0, len(values), 16):
                    batch = next(rows)
                    n = len(batch)
                    assert n == 16 and shard['first_window_index']+begin == cursor
                    for j, (doc, offset, bb) in enumerate(batch):
                        assert (doc, offset, len(bb)-1) == (*identities[begin+j], lengths[begin+j])
                    xx, yy, mask = train.tensors(batch)
                    ff = torch.as_tensor(values[begin:begin+n], device='cuda').clone()
                    ff *= (torch.rand((n,128), device='cuda', generator=rng) >= .5)[...,None]
                    ll = torch.as_tensor(contexts[begin:begin+n], device='cuda')
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        lp, _ = model(xx, ff, ll, windows=[(d,o) for d,o,_ in batch], valid_lengths=lengths[begin:begin+n].tolist())
                        loss = -lp.gather(-1, yy[...,None])[...,0][mask].mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    torch.cuda.synchronize()
                    train.accumulate(telemetry, model.telemetry)
                    assert all(h['capacity_merges'] == 0 and h['refused_insertions'] == 0 for layer in model.telemetry for h in layer)
                    steps += 1
                    cursor += n
                    targets += int(mask.sum())
                    active = time.monotonic()-active_start-excluded
                    at = time.monotonic()
                    row = dict(step=steps, targets=targets, next_window=cursor, active_seconds=active,
                               loss=float(loss.detach()), telemetry=model.telemetry)
                    log.write(json.dumps(row)+'\n')
                    log.flush()
                    if steps % 25 == 0:
                        print(json.dumps(dict(step=steps, targets=targets, active_seconds=active)), flush=True)
                    if steps % 500 == 0 or active >= a.seconds:
                        path = Path('/home/erikg/c14_dense_bytes/emender_phase3_sustained.pt')
                        assert not path.exists() or steps > 500, 'refuse previous-run checkpoint'
                        payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), dropout_rng=rng.get_state(),
                                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                                       python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                                       steps=steps, targets=targets, next_window=cursor, telemetry=telemetry, active_seconds=active)
                        tmp = path.with_suffix('.tmp.pt')
                        torch.save(payload, tmp)
                        tmp.replace(path)
                        receipt['last_checkpoint'] = dict(path=str(path), sha256=train.sha(path), steps=steps, targets=targets, next_window=cursor)
                    excluded += time.monotonic()-at
                    if active >= a.seconds:
                        break
                if active >= a.seconds:
                    break
        assert active >= a.seconds
        assert train.sha(train.CHECKPOINT) == checkpoint_hash
        receipt['status'] = 'completed_active_budget'
    except BaseException as exc:
        receipt.update(status='failed', error=repr(exc))
        raise
    finally:
        receipt.update(steps=steps, targets=targets, active_seconds=active, wall_seconds=time.monotonic()-wall,
                       excluded_receipt_checkpoint_seconds=excluded, telemetry=telemetry, next_window=cursor)
        (a.output/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
