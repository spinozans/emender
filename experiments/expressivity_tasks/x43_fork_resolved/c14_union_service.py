"""Managed frozen C14 union extraction; CPU only, no training side effects.

The transformer continuation shares the immutable Emender prefix: no duplicate
extraction of those positions. Phase boundary receipts do not authorize training
until its complete paired eval cache and registered launch gates are available.
"""
import argparse
import fcntl
import hashlib
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import shutil
import time
from . import c14_registered_cache as legacy
from .c14_registered_cache import journal, guard, checked_resume, slow_check, FLOOR
from .c14_branch_serve import BranchServer, ROOT, XSA, select, source_windows
from .c14_cache_scale import MAGIC, QueryReuse, read_windows, write_window
from .c14_feature_extract import digest, INDEX


def fsync_directory(directory):
    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def recover(directory):
    """Retain crash remnants outside the committed shard namespace; replay them.

    An incomplete last JSONL write cannot commit a shard. Preserve it, truncate
    only that uncommitted suffix, and retain any intent-bound orphan for audit.
    Never adopt uncommitted data or hide an unexpected orphan/corrupt commit.
    """
    log = directory/'journal.jsonl'
    if not log.exists():
        return
    raw = log.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary = raw.rfind(b'\n')+1
        saved = directory/f'crash_journal_{time.time_ns()}.partial'
        with saved.open('xb') as out:
            out.write(raw[boundary:]); out.flush(); os.fsync(out.fileno())
        with log.open('r+b') as out:
            out.truncate(boundary); out.flush(); os.fsync(out.fileno())
    records = [json.loads(line) for line in log.read_text().splitlines()]
    committed = {r['file'] for r in records if r['event'] == 'shard_complete'}
    intents = {r['file'] for r in records if r['event'] == 'write_intent'}
    for path in sorted(directory.glob('*.sparse')):
        if path.name in committed:
            continue
        if path.name not in intents:
            raise ValueError('unexpected orphan retained; manual inspection required')
        retained = path.with_name(path.name+f'.crash_{time.time_ns()}')
        path.rename(retained)
        fsync_directory(directory)
        journal(log, dict(event='crash_orphan_retained', file=path.name,
                          retained=retained.name, bytes=retained.stat().st_size,
                          sha256=digest(retained)))


def phase_target(audit, phase):
    if phase == 'emender':
        return audit['arms'][0]
    if phase == 'eval':
        return audit['eval']
    a, b = audit['arms']
    return dict(windows=b['windows']-a['windows'], targets=b['targets']-a['targets'])


def eval_windows(audit):
    receipt = json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    construction = json.loads((ROOT/'receipts/c14_dense/eval_construction.json').read_text())
    offsets, windows = Path(receipt['offsets_path']), Path(construction['windows_path'])
    if digest(offsets) != receipt['offsets_sha256'] or digest(windows) != audit['eval']['windows_sha256']:
        raise ValueError('frozen eval/offset hash mismatch')
    source = Path(receipt['source'])
    st = source.stat()
    if (st.st_size, st.st_ino, st.st_mtime_ns) != (receipt['source_size'], receipt['source_inode'], receipt['source_mtime_ns']):
        raise ValueError('source identity changed')
    count = positions = 0
    with offsets.open() as index, source.open('rb') as raw, windows.open() as frozen:
        for line in index:
            doc = json.loads(line)
            if doc['split'] != 'valid':
                continue
            raw.seek(doc['byte_offset']); data = raw.read(doc['byte_length'])
            if hashlib.sha256(data).hexdigest() != doc['sha256']:
                raise ValueError('eval document hash mismatch')
            for start in range(0, len(data)-1, 128):
                length = min(128, len(data)-1-start)
                expected = json.loads(next(frozen))
                if (expected['document'], expected['start'], expected['targets']) != (doc['source_index'], start, length):
                    raise ValueError('paired window identity mismatch')
                count += 1; positions += length
                yield doc['source_index'], start, [data[start:start+i] for i in range(1, length+1)]
        if frozen.read() or (count, positions) != (audit['eval']['windows'], audit['eval']['targets']):
            raise ValueError('paired eval cardinality mismatch')


def phase_windows(audit, phase):
    if phase == 'eval':
        return eval_windows(audit)
    if phase == 'emender':
        return source_windows(audit['arms'][0]['windows'])
    return itertools.islice(source_windows(audit['arms'][1]['windows']), audit['arms'][0]['windows'], None)


def run_phase(directory, phase, jobs=128):
    if directory.resolve().is_relative_to(Path('/mnt/nvme2n1')):
        raise ValueError('read-only source mount cannot hold caches')
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT/'receipts/c14_registered_scale/stream_audit.json'
    audit = json.loads(audit_path.read_text())
    for relative, expected in audit['receipt_hashes'].items():
        if digest(Path(relative)) != expected:
            raise ValueError(f'frozen receipt changed: {relative}')
    selector = ROOT/'c14_branch_serve.py'
    identity = dict(stream_audit_sha256=digest(audit_path), selector_sha256=digest(selector),
        native_sha256=digest(XSA), runner_sha256=digest(legacy.__file__) if phase == 'emender' else digest(__file__), arm=phase,
        representation='C14SPARSE1 uncompressed')
    target = phase_target(audit, phase)
    recover(directory)
    done, positions = checked_resume(directory, identity)
    start_done = done
    log = directory/'journal.jsonl'
    begin = time.monotonic()
    journal(log, dict(event='phase_start', identity=identity, windows=done, positions=positions,
        expected_windows=target['windows'], expected_positions=target['targets'],
        projected_bytes_at_113=target['targets']*113, free_bytes=shutil.disk_usage(directory).free,
        floor_bytes=FLOOR, python=__import__('sys').executable, gpu_used=False))
    server = None
    try:
        guard(directory, 0)
        server = BranchServer(directory/f'native_{time.time_ns()}.stderr', jobs=jobs)
        reuse = QueryReuse(server.query, 250000)
        windows = itertools.islice(phase_windows(audit, phase), done, None)
        while done < target['windows']:
            guard(directory, 0)
            chunk = list(itertools.islice(windows, 128))
            if not chunk:
                raise ValueError('source exhausted before frozen prefix')
            lefts = [s for _, _, ss in chunk for s in ss]
            rows, rates, _ = select(lefts, reuse)
            samples = slow_check(lefts, rows, server.query)
            stream = BytesIO(); stream.write(MAGIC)
            at = 0
            for doc, offset, ss in chunk:
                write_window(stream, doc, offset, rows[at:at+len(ss)])
                at += len(ss)
            payload = stream.getvalue()
            # Full row and arithmetic parity for every newly persisted position.
            restored = [r for _, _, rr in read_windows(BytesIO(payload)) for r in rr]
            if restored != rows:
                raise ValueError('compact encoding parity failure')
            name = f'{done:08d}.sparse'
            available = guard(directory, len(payload))
            journal(log, dict(event='write_intent', file=name, first_window_index=done,
                bytes=len(payload), sha256=__import__('hashlib').sha256(payload).hexdigest(), free_bytes=available, floor_bytes=FLOOR))
            with (directory/name).open('xb') as out:
                os.posix_fallocate(out.fileno(), 0, len(payload))
                out.write(payload); out.flush(); os.fsync(out.fileno())
            fsync_directory(directory)
            guard(directory, 0)
            record = dict(event='shard_complete', identity=identity, file=name,
                first_window_index=done, windows=len(chunk), positions=len(rows),
                first_identity=list(chunk[0][:2]), last_identity=list(chunk[-1][:2]),
                bytes=len(payload), sha256=digest(directory/name), support=rates,
                slow_samples=samples, full_compact_row_parity=True,
                covered=sum(r['coverage'] for r in rows))
            journal(log, record)
            done += len(chunk); positions += len(rows)
            print(json.dumps(dict(windows=done, positions=positions,
                seconds=time.monotonic()-begin, phase=phase)), flush=True)
        complete = done == target['windows'] and positions == target['targets']
        if done == target['windows'] and not complete:
            raise ValueError('frozen target cardinality mismatch')
        journal(log, dict(event='phase_complete' if complete else 'bounded_pause',
            windows=done, positions=positions, next_window_index=done,
            seconds=time.monotonic()-begin, new_windows=done-start_done,
            status='complete' if complete else 'incomplete', no_training_started=True))
    except BaseException as error:
        journal(log, dict(event='failure', error=repr(error), windows=done,
            positions=positions, next_window_index=done, retained_all_files=True))
        raise
    finally:
        if server is not None:
            server.close()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/erikg/c14_consultation_features'))
    parser.add_argument('--jobs', type=int, default=128)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'union.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = args.root/'union_service.jsonl'
        journal(log, dict(event='service_start', time=time.time(), pid=os.getpid(),
            python=__import__('sys').executable, code_sha256=digest(__file__), gpu_used=False,
            baseline_targets_per_second=5884.178857234429, floor_bytes=FLOOR))
        try:
            # The full source-index binding is frozen in the attended resume receipt.
            resume = json.loads((ROOT/'receipts/c14_registered_resume/resume.json').read_text())
            for phase in ('emender', 'eval', 'transformer'):
                if digest(INDEX) != resume['source_index']['sha256']:
                    raise ValueError('native source index changed')
                oracle = json.loads((ROOT/'receipts/c14_assembly/fixture.json').read_text())
                if digest(XSA) != oracle['xsa_sha256']:
                    raise ValueError('native executable differs from frozen oracle')
                journal(log, dict(event='phase_start', phase=phase, time=time.time(),
                    free_bytes=shutil.disk_usage(args.root).free))
                run_phase(args.root/('registered_'+phase), phase, args.jobs)
                journal(log, dict(event='phase_boundary', phase=phase, time=time.time(),
                    cache_bytes=sum(p.stat().st_size for p in (args.root/('registered_'+phase)).glob('*.sparse')),
                    journal_sha256=digest(args.root/('registered_'+phase)/'journal.jsonl'),
                    training_pending=True))
        except BaseException as error:
            journal(log, dict(event='service_failure', time=time.time(), error=repr(error),
                free_bytes=shutil.disk_usage(args.root).free, retained_all_files=True))
            raise
        journal(log, dict(event='union_complete', time=time.time(), training_pending=True))


if __name__ == '__main__':
    main()
