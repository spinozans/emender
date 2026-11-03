"""Registered time-budget C14 consultation training; immutable source-order cache."""
import datetime
import itertools
import json
import os
from pathlib import Path
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
from .c14_branch_serve import features

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/c14_consultation_emender'
CACHE = Path('/home/erikg/c14_consultation_features/registered_emender')
CHECKPOINT = Path('/home/erikg/c14_dense_bytes/emender_consultation_resume.pt')


def write(name, data):
    p = OUT/name
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('2', '4')
    OUT.mkdir(exist_ok=False)
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
        timer='includes source sampling, sparse decode, assembly, transfers and training; excludes checkpoint stalls and lease wait',
        dense_reference_steps_per_second=41.3, eval='pending live registered_eval phase; no eval performed')
    write('config.json', config)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    rows_iter = iter(batches(raw, 'train'))
    curve = []; steps = targets = kept = seen = 0
    excluded = []; excluded_seconds = 0.
    torch.cuda.synchronize(); start = time.monotonic(); last_save = 0.
    def active():
        return time.monotonic()-start-excluded_seconds
    def checkpoint(status):
        nonlocal excluded_seconds
        torch.cuda.synchronize(); elapsed = active(); at = time.monotonic()
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
            keep_fraction=kept/seen if seen else None, checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_sha256,
            next_window_index=steps*16, curve=curve, eval='pending; no eval performed')
        write('training.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('curve','excluded_stalls')}), flush=True)
    try:
        for r in shards:
            with (CACHE/r['file']).open('rb') as stream:
                cached = iter(read_windows(stream))
                while True:
                    chunk = list(itertools.islice(cached, 16))
                    if not chunk: break
                    assert len(chunk) == 16
                    rows = next(rows_iter)
                    f = np.zeros((16,128,260), dtype=np.float32)
                    for j, ((doc, offset, feature_rows), (actual_doc, actual_offset, b)) in enumerate(zip(chunk, rows)):
                        assert (doc, offset, len(feature_rows)) == (actual_doc, actual_offset, len(b)-1)
                        f[j,:len(feature_rows)] = [features(v) for v in feature_rows]
                    x,y,mask = tensors(rows)
                    feature_tensor = torch.as_tensor(f, device='cuda')
                    keep = torch.rand((16,128),device='cuda',generator=rng) >= .5
                    feature_tensor *= keep[...,None]
                    kept += int((keep & mask).sum()); seen += int(mask.sum())
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        lp,_ = model(x,feature_tensor,feature_tensor[...,259])
                        loss = -lp.gather(-1,y[...,None])[...,0][mask].mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step(); steps += 1; targets += int(mask.sum())
                    if steps == 1 or steps % 100 == 0:
                        entry = dict(step=steps, active_seconds=active(), targets=targets, nll=float(loss.detach()))
                        curve.append(entry); print(json.dumps(entry),flush=True)
                    if active() >= 2700 or stop: break
                    if active()-last_save >= 120:
                        checkpoint('running'); last_save=active()
            if active() >= 2700 or stop: break
        torch.cuda.synchronize()
        checkpoint('trained_registered_budget' if active() >= 2700 else 'interrupted' if stop else 'failed_cache_exhausted')
    except BaseException as error:
        checkpoint('failed: '+repr(error))
        raise

if __name__ == '__main__':
    main()
