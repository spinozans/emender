"""Attended read-only cache verification; failures stop the union service first.
Run from the repo root with .venv/bin/python: before, parity (default), after, integrity.
The parity mode reads the existing before snapshot. No rate tuning or training.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import BranchServer, ROOT, XSA, features
from experiments.expressivity_tasks.x43_fork_resolved.c14_cache_scale import read_windows
from experiments.expressivity_tasks.x43_fork_resolved.c14_feature_extract import digest

OUT = Path(__file__).resolve().parent
CACHE = Path('/home/erikg/c14_consultation_features')
UNIT = 'emender-velocity-c14-union.service'

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')

def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

def snapshot(label):
    stamp = time.time()
    status = command('systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'SubState', '-p', 'NRestarts', '-p', 'MainPID', '-p', 'ExecMainStartTimestamp', '-p', 'CPUUsageNSec', '-p', 'MemoryCurrent', '-p', 'ControlGroup')
    (OUT/f'{label}_systemd.txt').write_text(status)
    (OUT/f'{label}_host.txt').write_text(command('uptime')+command('free', '-b')+command('ps', '-eo', 'pid,ppid,pcpu,pmem,nlwp,etime,args', '--sort=-pcpu')[:16000]+''.join(f'\n{p}\n'+Path(p).read_text() for p in ['/proc/pressure/cpu','/proc/pressure/memory','/proc/pressure/io']))
    for source, target in [('union_console.jsonl','console.jsonl'),('union_service.jsonl','service_journal.jsonl'),('registered_emender/journal.jsonl','extraction_journal.jsonl'),('union_stderr.txt','stderr.txt')]:
        data=(CACHE/source).read_bytes()
        # Discard only an in-flight final line in a read-only snapshot.
        if source.endswith('.jsonl'):
            data=data[:data.rfind(b'\n')+1]
        (OUT/f'{label}_{target}').write_bytes(data)
    (OUT/f'{label}_system_journal.txt').write_text(command('journalctl','--user','-u',UNIT,'--since','2026-11-03 06:30:19','--no-pager'))
    free=shutil.disk_usage(CACHE).free
    write(f'{label}_snapshot.json',dict(time=stamp,free_bytes=free,floor_bytes=150000000000,files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in OUT.glob(f'{label}_*') if p.is_file() and p.name != f'{label}_snapshot.json'}))
    assert 'ActiveState=active' in status and 'SubState=running' in status, 'service unhealthy'
    assert 'NRestarts=0' in status, 'restart count changed'
    assert free>=150000000000, 'disk floor crossed'
    service=[json.loads(s) for s in (OUT/f'{label}_service_journal.jsonl').read_text().splitlines()]
    journal=[json.loads(s) for s in (OUT/f'{label}_extraction_journal.jsonl').read_text().splitlines()]
    assert not any(r['event'] in ('failure','service_failure') for r in service+journal), 'failure in journal'
    assert not (OUT/f'{label}_stderr.txt').stat().st_size, 'service stderr nonempty'
    return journal

def parity(records):
    shards=[r for r in records if r['event']=='shard_complete']
    assert digest(XSA)==shards[-1]['identity']['native_sha256'], 'native binding changed'
    assert digest(ROOT/'c14_branch_serve.py')==shards[-1]['identity']['selector_sha256'], 'selector binding changed'
    assert digest(ROOT/'c14_union_service.py')==json.loads((OUT/'before_service_journal.jsonl').read_text().splitlines()[-2])['code_sha256'], 'service binding changed'
    # Snapshot only committed rows; live pending files are neither adopted nor rejected.
    windows=positions=cache_bytes=0
    for r in shards:
        p=CACHE/'registered_emender'/r['file']
        assert r['first_window_index']==windows
        assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'], 'committed shard corrupt'
        assert r['full_compact_row_parity'] and len(r['slow_samples'])==3
        windows+=r['windows']; positions+=r['positions']; cache_bytes+=r['bytes']
    newly=[r for r in shards if r['first_window_index']>=19968]
    assert len(newly)>=3
    chosen=[newly[0],newly[len(newly)//2],newly[-1]]
    span=json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    offsets=Path(span['offsets_path']); source=Path(span['source']); st=source.stat()
    assert digest(offsets)==span['offsets_sha256']
    assert (st.st_size,st.st_ino,st.st_mtime_ns)==(span['source_size'],span['source_inode'],span['source_mtime_ns'])
    needed={}
    for r in chosen:
        with (CACHE/'registered_emender'/r['file']).open('rb') as f:
            decoded=list(read_windows(f))
        assert len(decoded)==r['windows'] and sum(len(rows) for _,_,rows in decoded)==r['positions']
        needed[r['file']]=decoded[len(decoded)//2]
    ids={d for d,_,_ in needed.values()}
    documents={}
    with offsets.open() as f,source.open('rb') as raw:
        for line in f:
            d=json.loads(line)
            if d['source_index'] not in ids:
                continue
            assert d['split']=='train'
            raw.seek(d['byte_offset']); data=raw.read(d['byte_length'])
            assert hashlib.sha256(data).hexdigest()==d['sha256']
            documents[d['source_index']]=(d,data)
            if len(documents)==len(ids): break
    # Exhaustive suffix search, without binary search, QueryReuse, or select().
    results=[]
    begin=time.time()
    server=BranchServer(OUT/'native_spotcheck.stderr',jobs=128)
    try:
        for r in chosen:
            doc,start,rows=needed[r['file']]; d,data=documents[doc]
            assert len(rows)==min(128,len(data)-1-start)
            samples=[]
            for pos in sorted({1,min(64,len(rows)),len(rows)}):
                context=data[start:start+pos].rsplit(b'\x1e',1)[-1][-1024:]
                replies=server.query([context[-k:] for k in range(1,len(context)+1)]) if context else []
                k=max((i+1 for i,x in enumerate(replies) if x['width']),default=0)
                native=replies[k-1] if k else dict(width=0,eof=0,branches=[])
                pairs=native['branches']; total=sum(c for b,c in pairs if b!=30)
                expected=dict(context_bytes=k,censored=bool(k and k==len(context)),context_count=native['width'],eof=native['eof'],terminal_count=sum(c for b,c in pairs if b==30),branches=pairs,support_size=total,distinct_branches=sum(b!=30 for b,c in pairs),entropy=-sum(c/total*math.log(c/total) for b,c in pairs if b!=30) if total else 0.,coverage=bool(total))
                assert expected==rows[pos-1], f'oracle row mismatch: {r["file"]} {pos}'
                a=struct.pack('<260d',*features(expected)); b=struct.pack('<260d',*features(rows[pos-1]))
                assert a==b
                samples.append(dict(position=pos,context_sha256=hashlib.sha256(context).hexdigest(),maximal_suffix_bytes=k,feature_bits_sha256=hashlib.sha256(a).hexdigest(),exact_row_and_feature_bits=True))
            results.append(dict(file=r['file'],first_window_index=r['first_window_index'],shard_sha256=r['sha256'],document=doc,start=start,targets=len(rows),document_sha256=d['sha256'],samples=samples))
    finally:
        server.close()
    write('parity.json',dict(status='passed',snapshot_windows=windows,snapshot_positions=positions,snapshot_bytes=cache_bytes,all_committed_shards_hash_checked=len(shards),new_shards_since_launch=len(newly),native_sha256=digest(XSA),offsets_sha256=digest(offsets),spotchecks=results,oracle='exhaustive native suffix oracle, independent of production binary search/reuse; frozen native fixture binding unchanged',verifier_start=begin,verifier_end=time.time(),jobs=128,scope='sampled newly extracted windows; incomplete cache, no training'))


def integrity():
    records = [json.loads(s) for s in (OUT/'after_extraction_journal.jsonl').read_text().splitlines()]
    shards = [r for r in records if r['event'] == 'shard_complete']
    windows = positions = size = 0
    for r in shards:
        path = CACHE/'registered_emender'/r['file']
        assert r['first_window_index'] == windows
        assert path.stat().st_size == r['bytes'] and digest(path) == r['sha256']
        assert r['full_compact_row_parity'] and len(r['slow_samples']) == 3
        windows += r['windows']; positions += r['positions']; size += r['bytes']
    for label in ('before', 'after'):
        manifest = json.loads((OUT/f'{label}_snapshot.json').read_text())
        for name, expected in manifest['files'].items():
            path = OUT/name
            assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256']
    accounting = json.loads((OUT/'accounting.json').read_text())
    assert (windows, positions, size) == (accounting['snapshot_windows'], accounting['snapshot_positions'], accounting['disk']['all_committed_cache_bytes'])
    write('final_integrity.json', dict(status='passed', time=time.time(),
        snapshot_windows=windows, snapshot_positions=positions, snapshot_bytes=size,
        all_committed_shards_hash_checked=len(shards),
        journal_sha256=digest(OUT/'after_extraction_journal.jsonl'),
        snapshot_manifests_verified=True,
        all_committed_shards_have_full_compact_parity_and_three_slow_oracle_samples=True,
        scope='immutable committed prefix in after snapshot; later live writes are outside this receipt'))

if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1] in ('before', 'after'):
            snapshot(sys.argv[1])
        elif len(sys.argv)>1 and sys.argv[1]=='integrity':
            integrity()
        else:
            parity([json.loads(s) for s in (OUT/'before_extraction_journal.jsonl').read_text().splitlines()])
    except BaseException as error:
        # Required fail-closed action precedes writing the failure receipt.
        stopped=subprocess.run(['systemctl','--user','stop',UNIT],capture_output=True,text=True)
        write('FAILURE.json',dict(error=repr(error),stop_returncode=stopped.returncode,stop_stdout=stopped.stdout,stop_stderr=stopped.stderr,time=time.time(),unit_state=command('systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState')))
        raise
