"""CPU-only, frozen beyond-boundary flagship extraction; no training or leases."""
import argparse, fcntl, gzip, hashlib, itertools, json, os, resource, shutil, time
from pathlib import Path
from io import BytesIO
from .c14_union_service import recover, fsync_directory
from .c14_registered_cache import journal, guard, checked_resume, slow_check, FLOOR
from . import c14_branch_serve as serving
from .c14_branch_serve import BranchServer, ROOT, select
from .flagship_cache import MAGIC, QueryReuse, read_windows, write_window
from .c14_feature_extract import digest
from .c14_documents import DocumentWindows


def cpu_seconds():
    return sum(getattr(resource.getrusage(w), k) for w in
               (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN) for k in ('ru_utime','ru_stime'))


def verify(audit):
    for key in ('source','index','binary'):
        row=audit[key]; st=Path(row['path']).stat()
        assert (st.st_size,st.st_ino,st.st_mtime_ns)==(row['size'],row['inode'],row['mtime_ns']), key
    assert digest(serving.XSA)==audit['binary_sha256']
    assert digest(Path(audit['manifest']))==audit['manifest_sha256']
    assert digest(ROOT/'receipts/p100_query_verification/verdict.json')==audit['query_verdict_sha256']
    serving.N=100_000_000_001


def oracle_check(server):
    reg=json.loads((ROOT/'receipts/p100_query_verification/registration.json').read_text())
    oracle=json.loads((ROOT/'receipts/p100_query_verification/native.json').read_text())
    got=server.query([bytes.fromhex(p['hex']) for p in reg['patterns']])
    assert json.loads(json.dumps(got))==oracle, '17-context brute-force oracle parity failed'


def phase_windows(audit, phase):
    sampler=DocumentWindows(audit['context'])
    with gzip.open(audit['manifest'],'rt') as manifest, Path(audit['source']['path']).open('rb') as source:
        for line in manifest:
            doc=json.loads(line)
            if doc['split']!=phase: continue
            source.seek(doc['byte_offset']); raw=source.read(doc['byte_length'])
            receipt=dict(source_index=doc['document'],sha256=doc['sha256'],split='train' if phase=='train' else 'valid')
            for w in sampler.windows(raw,receipt):
                yield w.document,w.start,[sampler.causal_context(raw,w.start+i,w.start) for i in range(1,len(w.targets)+1)]


def run_phase(directory, phase, jobs=128):
    if directory.resolve().is_relative_to(Path('/mnt/nvme2n1')):
        raise ValueError('read-only source mount cannot hold caches')
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT/'receipts/flagship_stream/registration.json'
    audit = json.loads(audit_path.read_text())
    verify(audit)
    identity = dict(registration_sha256=digest(audit_path),
        code_sha256=digest(__file__), dependencies={p:digest(ROOT/p) for p in
        ('c14_union_service.py','c14_branch_serve.py','c14_registered_cache.py','c14_cache_scale.py','c14_documents.py','flagship_cache.py','c14_feature_extract.py')},
        arm=phase, representation='FLAGSHIPSPARSE2 260-scalar-equivalent')
    target = audit['splits'][phase]
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
        server = BranchServer(directory/f'native_{time.time_ns()}.stderr', jobs=jobs, index=Path(audit['index']['path']))
        oracle_check(server)
        journal(log,dict(event='oracle_parity',contexts=17,full_region_bruteforce_reference=True))
        reuse = QueryReuse(server.query, 250000)
        windows = itertools.islice(phase_windows(audit, phase), done, None)
        while done < target['windows']:
            guard(directory, 16*2048*2310)
            chunk = list(itertools.islice(windows, 16))
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
            print(json.dumps(dict(time=time.time(), cpu_seconds=cpu_seconds(), windows=done, positions=positions,
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
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('/home/erikg/flagship_consultation_features'));p.add_argument('--jobs',type=int,default=128);a=p.parse_args()
    assert __import__('sys').executable=='/home/erikg/emender/.venv/bin/python'
    assert not a.root.resolve().is_relative_to(Path('/mnt/nvme2n1'))
    a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'union.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        log=a.root/'union_service.jsonl'
        journal(log,dict(event='service_start',time=time.time(),pid=os.getpid(),gpu_used=False,cpu_seconds=cpu_seconds()))
        try:
            audit=json.loads((ROOT/'receipts/flagship_stream/registration.json').read_text())
            verify(audit)
            assert digest(Path(audit['index']['path']))==audit['index']['sha256'], 'index hash changed'
            guard(a.root,audit['extraction']['storage_reservation_bytes'])
            for phase in ('calibration','final_test','train'):
                run_phase(a.root/phase,phase,a.jobs)
                journal(log,dict(event='phase_boundary',phase=phase,time=time.time(),cpu_seconds=cpu_seconds()))
        except BaseException as e:
            journal(log,dict(event='service_failure',time=time.time(),error=repr(e),cpu_seconds=cpu_seconds()))
            raise
        journal(log,dict(event='union_complete',time=time.time(),cpu_seconds=cpu_seconds()))

if __name__=='__main__': main()
