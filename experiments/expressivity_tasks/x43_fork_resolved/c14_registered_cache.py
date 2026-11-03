"""Disk-guarded, resumable extraction of the exact frozen Emender prefix.

Each immutable shard is committed by a hash-bound journal row after fsync.
Unreceipted files cause a closed failure and are never deleted or overwritten.
This runner does not train, amend the protocol, or certify a complete eval cache.
"""
import argparse
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import shutil
import time

from .c14_branch_serve import BranchServer, ROOT, XSA, features, select, source_windows
from .c14_cache_scale import MAGIC, QueryReuse, read_windows, write_window
from .c14_feature_extract import digest

FLOOR = 150_000_000_000


def journal(path, record):
    with path.open('a') as out:
        out.write(json.dumps(record, sort_keys=True)+'\n')
        out.flush()
        os.fsync(out.fileno())


def guard(directory, size, free=None):
    available = shutil.disk_usage(directory).free if free is None else free
    if available-size < FLOOR:
        raise OSError(f'disk guard: {available} free, {size} requested, {FLOOR} reserved')
    return available


def checked_resume(directory, identity):
    path = directory/'journal.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = [r for r in records if r['event'] == 'shard_complete']
    windows = positions = 0
    for r in done:
        if r['identity'] != identity or r['first_window_index'] != windows:
            raise ValueError('resume identity/order mismatch')
        p = directory/r['file']
        if p.stat().st_size != r['bytes'] or digest(p) != r['sha256']:
            raise ValueError('resume shard corruption')
        windows += r['windows']; positions += r['positions']
    known = {r['file'] for r in done}
    if {p.name for p in directory.glob('*.sparse')} != known:
        raise ValueError('unreceipted cache retained; inspect before resume')
    return windows, positions


def slow_check(lefts, rows, query):
    """Exhaust every suffix at deterministic first/middle/last sample positions."""
    samples = sorted({0, len(lefts)//2, len(lefts)-1})
    receipts = []
    for i in samples:
        s = lefts[i].rsplit(b'\x1e', 1)[-1][-1024:]
        patterns = [s[-k:] for k in range(1, len(s)+1)]
        replies = query(patterns) if patterns else []
        k = max((j+1 for j, r in enumerate(replies) if r['width']), default=0)
        # Rebuild via the registered selector on the exact maximal suffix, then
        # retain original censoring. Exhaustive search is independent of bisection.
        expected, _, _ = select([s[-k:] if k else b''], query)
        expected[0]['censored'] = bool(k and k == len(s))
        if expected[0] != rows[i] or features(expected[0]) != features(rows[i]):
            raise ValueError(f'slow suffix parity failed at sample {i}')
        import hashlib
        receipts.append(dict(index=i, context_sha256=hashlib.sha256(s).hexdigest(),
            maximal_suffix_bytes=k, feature_sha256=hashlib.sha256(
                json.dumps(features(rows[i])).encode()).hexdigest()))
    return receipts


def run(directory, max_seconds, jobs=128):
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
        native_sha256=digest(XSA), runner_sha256=digest(__file__), arm='emender',
        representation='C14SPARSE1 uncompressed')
    target = audit['arms'][0]
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
        windows = itertools.islice(source_windows(target['windows']), done, None)
        while done < target['windows']:
            if time.monotonic()-begin >= max_seconds:
                break
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
                bytes=len(payload), free_bytes=available, floor_bytes=FLOOR))
            with (directory/name).open('xb') as out:
                os.posix_fallocate(out.fileno(), 0, len(payload))
                out.write(payload); out.flush(); os.fsync(out.fileno())
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
                seconds=time.monotonic()-begin)), flush=True)
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True,
        help='Attended extraction time slice; never a training budget change')
    parser.add_argument('--jobs', type=int, default=128)
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.jobs < 1:
        parser.error('positive extraction slice and jobs required')
    run(args.output, args.max_seconds, args.jobs)


if __name__ == '__main__':
    main()
