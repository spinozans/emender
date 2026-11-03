"""Registered time-budget C14 consultation training; immutable source-order cache."""
import argparse
import datetime
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import numpy as np
import torch
from .c11 import Boosted
from .c6 import count
from .c14_dense_train import batches, tensors, OFFSETS
from .c14_documents import sha
from .c14_cache_scale import read_windows
from .c14_fast_features import decode_shard, load_decoder

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/c14_consultation_emender_v2'
CACHE = Path('/home/erikg/c14_consultation_features/registered_emender')
CHECKPOINT = Path('/home/erikg/c14_dense_bytes/emender_consultation_v2_resume.pt')


def write(name, data):
    p = OUT/name
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('2', '4')
    global OUT, CHECKPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-steps', type=int, default=0)
    parser.add_argument('--oracle-profile', action='store_true')
    args = parser.parse_args()
    if args.profile_steps:
        OUT = ROOT/'receipts/c14_consultation_emender_v2_profile'
    if args.profile_steps:
        CHECKPOINT = CHECKPOINT.with_name('emender_consultation_v2_profile.pt')
    if args.oracle_profile:
        CHECKPOINT = CHECKPOINT.with_name('emender_consultation_v2_oracle_profile.pt')
        assert args.profile_steps > 0
        OUT = ROOT/'receipts/c14_consultation_emender_v2_oracle_profile'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'training.json').exists(), 'refuse existing training receipts'
    decoder = load_decoder()
    wall_start = time.monotonic()
    decision = ROOT/'receipts/c14_registered_scale/stream_audit.json'
    audit = json.loads(decision.read_text())
    assert audit['status'] == 'resolved_stopping_rule'
    for path, digest in audit['receipt_hashes'].items():
        assert sha(ROOT.parents[2]/path) == digest, path
    preflight = json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    assert sha(OFFSETS) == preflight['offsets_sha256']
    source = Path(preflight['source']); stat = source.stat()
    assert (stat.st_size, stat.st_ino, stat.st_mtime_ns) == (preflight['source_size'], preflight['source_inode'], preflight['source_mtime_ns'])
    raw = np.memmap(source, dtype=np.uint8, mode='r')
    validation = json.loads((ROOT/'receipts/c14_emender_closed/cache_validation.json').read_text())
    assert sha(CACHE/'journal.jsonl') == validation['journal_sha256']
    records = [json.loads(line) for line in (CACHE/'journal.jsonl').read_text().splitlines()]
    shards = [r for r in records if r['event'] == 'shard_complete']
    windows = positions = 0
    for r in shards:
        assert r['first_window_index'] == windows and r['full_compact_row_parity']
        p = CACHE/r['file']
        assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
        windows += r['windows']; positions += r['positions']
    assert (windows, positions) == (1785536, 225880702)
    shapes = json.loads((ROOT/'receipts/c14_byte/model_shapes.json').read_text())
    for path, digest in shapes['sources'].items():
        assert sha(ROOT.parents[2]/path) == digest
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    model = Boosted('emender', 2000000).cuda().train()
    assert count(model) == 2133700
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = torch.Generator(device='cuda').manual_seed(3401)
    config = dict(seed=3401, dropout_seed=3401, batch=16, context=128, precision='BF16 autocast',
        optimizer='AdamW lr=3e-4 decay=.01 clip=1', parameters=count(model), budget_active_seconds=2700,
        dropout='independent Bernoulli keep=.5 per position; multiply all 260 scalars, no rescaling',
        resets='independent forward per window; document tails right-padded and loss-masked',
        decision_sha256=sha(decision), cache_journal_sha256=validation['journal_sha256'],
        cache_windows=windows, cache_targets=positions, source_receipt_sha256=sha(ROOT/'receipts/c14_dense/pile_span.json'),
        code_sha256=sha(__file__), physical_gpu=os.environ['GPU_LEASE_HELD'], interpreter=sys.executable,
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        timer='includes source sampling, native shard decode, assembly, transfers and training; excludes checkpoint stalls and lease wait',
        feature_decoder_sha256=sha(ROOT/'c14_fast_features.c'), dense_reorganization_disk_bytes=0,
        amendment='v2 primary paired arm; v1 retained; compare actual bytes/steps at registered active time',
        dense_reference_steps_per_second=41.3, eval='pending live registered_eval phase; no eval performed')
    write('config.json', config)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    rows_iter = iter(batches(raw, 'train'))
    curve = []; steps = targets = kept = seen = 0
    timing = dict(shard_read_decode=0., sampling_transfer=0., feature_transfer_dropout=0., training=0.)
    excluded = []; excluded_seconds = 0.
    torch.cuda.synchronize(); start = time.monotonic(); last_save = 0.
    def active():
        return time.monotonic()-start-excluded_seconds
    def checkpoint(status):
        nonlocal excluded_seconds
        torch.cuda.synchronize(); elapsed = active(); at = time.monotonic()
        if shutil.disk_usage(CHECKPOINT.parent).free < 150_000_000_000:
            raise RuntimeError('STOP: less than 150 GB free')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), dropout_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), next_window=steps*16,
            steps=steps, targets=targets, active_seconds=elapsed, config=config), CHECKPOINT)
        checkpoint_sha256 = sha(CHECKPOINT)
        duration = time.monotonic()-at; excluded_seconds += duration
        excluded.append(dict(kind='checkpoint_save_and_hash', seconds=duration, after_step=steps))
        result = dict(status=status, steps=steps, windows=steps*16, actual_bytes_targets=targets,
            active_seconds=elapsed, wall_seconds=time.monotonic()-wall_start, excluded_stalls=excluded,
            lease_wait_seconds=0, lease_wait_note='acquired by wrapper before process; see lease log timestamps',
            steps_per_active_second=steps/elapsed, throughput_ratio_to_dense_41_3=(steps/elapsed)/41.3,
            keep_fraction=kept/seen if seen else None, step_breakdown_seconds=timing, checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_sha256,
            next_window_index=steps*16, curve=curve, eval='pending; no eval performed')
        write('training.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('curve','excluded_stalls')}), flush=True)
    try:
        for r in shards:
            if shutil.disk_usage(CACHE).free < 150_000_000_000:
                raise RuntimeError('STOP: less than 150 GB free')
            at = time.monotonic()
            data = (CACHE/r['file']).read_bytes()
            assert __import__('hashlib').sha256(data).hexdigest() == r['sha256']
            if args.oracle_profile:
                import io
                from .c14_branch_serve import features
                windows = list(read_windows(io.BytesIO(data)))
                ids = np.array([[d,o] for d,o,_ in windows])
                lengths = np.array([len(rr) for _,_,rr in windows])
                values = np.zeros((len(windows),128,260),np.float32)
                for j,(_,_,rr) in enumerate(windows):
                    values[j,:len(rr)] = [features(v) for v in rr]
            else:
                ids, lengths, values = decode_shard(data, r['windows'], decoder)
            timing['shard_read_decode'] += time.monotonic()-at
            for begin in range(0, len(values), 16):
                    at = time.monotonic()
                    rows = next(rows_iter)
                    assert len(rows) == 16
                    for j,(doc,offset,b) in enumerate(rows):
                        assert (doc,offset,len(b)-1) == (*ids[begin+j],lengths[begin+j])
                    f = values[begin:begin+16]
                    x,y,mask = tensors(rows)
                    torch.cuda.synchronize()
                    timing['sampling_transfer'] += time.monotonic()-at
                    at = time.monotonic()
                    feature_tensor = torch.as_tensor(f, device='cuda')
                    keep = torch.rand((16,128),device='cuda',generator=rng) >= .5
                    feature_tensor *= keep[...,None]
                    kept += int((keep & mask).sum()); seen += int(mask.sum())
                    torch.cuda.synchronize()
                    timing['feature_transfer_dropout'] += time.monotonic()-at
                    at = time.monotonic()
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        lp,_ = model(x,feature_tensor,feature_tensor[...,259])
                        loss = -lp.gather(-1,y[...,None])[...,0][mask].mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step(); steps += 1; targets += int(mask.sum())
                    torch.cuda.synchronize()
                    timing['training'] += time.monotonic()-at
                    if steps == 1 or steps % 100 == 0:
                        entry = dict(step=steps, active_seconds=active(), targets=targets, nll=float(loss.detach()))
                        curve.append(entry); print(json.dumps(entry),flush=True)
                    if active() >= 2700 or stop or (args.profile_steps and steps >= args.profile_steps): break
                    if active()-last_save >= 120:
                        checkpoint('running'); last_save=active()
            if active() >= 2700 or stop or (args.profile_steps and steps >= args.profile_steps): break
        torch.cuda.synchronize()
        checkpoint('trained_registered_budget' if active() >= 2700 else 'interrupted' if stop else 'profile_complete' if args.profile_steps and steps >= args.profile_steps else 'failed_cache_exhausted')
    except BaseException as error:
        checkpoint('failed: '+repr(error))
        raise

if __name__ == '__main__':
    main()
