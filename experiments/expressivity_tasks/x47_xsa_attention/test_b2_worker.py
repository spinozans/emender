import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

from .b1_mount import Archive, ROOT, HERE
from .b2_worker import WorkerRegistry

EXE = Path(os.environ.get('B2_EXE', Path.home()/'.cache/xsa-b2-target/release/xsa'))
MANIFEST = json.loads((HERE/'b1_mounts.json').read_text())
pytestmark = pytest.mark.skipif(not EXE.exists(), reason='requires local B2 binary and registered mounts')


def test_whole_mount_binary_and_crossing():
    ref = Archive('F6')
    r = WorkerRegistry({'new-source':MANIFEST['F6'], 'sibling':MANIFEST['F6']},ROOT,EXE,{'new-source','sibling'})
    try:
        for name,kw in [('F6',{}),('new-source',{'excluded_docs':[1]}),('sibling',{'whole_bank':False})]:
            with pytest.raises(PermissionError): r.open(name,**kw)
        assert not r.workers
        w=r.open('new-source'); raw=ref.text.read_bytes(); cut=int(ref.starts[1]); crossing=raw[cut-8:cut+8]
        pats=[*ref.patterns,b'\x00\x1e\xff',crossing]
        spans,answer=w.request(pats)
        for i,p in enumerate(pats):
            row=answer['rows'][i]
            if i==2:
                assert row['raw']>8 and row['status']=='overcap' and not row['hits'] and not row['lf']; continue
            expected=[]; at=raw.find(p); count=0
            while at>=0:
                count+=1; d=int(ref.starts.searchsorted(at,'right'))-1
                if at+len(p)<=(ref.starts[d+1] if d+1<len(ref.starts) else ref.size): expected.append(at)
                at=raw.find(p,at+1)
            assert row['raw']==count and sorted(h[0] for h in row['hits'])==expected
            if i==4: assert count>len(expected)
        assert not spans[3] and answer['rows'][3]['status']=='complete'
        with pytest.raises(ValueError): w.request([b''])
        sibling=r.open('sibling').request(pats[:1])[0][0]
        assert spans[0] and sibling and spans[0][0].source=='new-source' and sibling[0].source=='sibling'
    finally: r.close()


@pytest.mark.parametrize('wire',[b'\x04',struct.pack('<I',260)+b'\x01\x01\x01'+bytes(257),struct.pack('<I',4)+bytes([17,1,0,0]),struct.pack('<I',4130),struct.pack('<I',4)+b'\x01\x01',struct.pack('<I',4)+b'\x01\x00\x00x',struct.pack('<I',4)+b'\x02\x01\x00A'])
def test_terminal_frame_no_partial_answer(wire):
    files=MANIFEST['F6']['files']; args=[EXE,'positional-worker',*(ROOT/f['path'] for f in files[:2])]
    proc=subprocess.run(args,input=wire,capture_output=True,timeout=30)
    assert proc.returncode and len(proc.stdout.splitlines())==1 and json.loads(proc.stdout)['ready']==1


def test_archived_counts_positions_and_compacted_votes():
    from .b1_head import pack, transfer
    from .b1_mount import EXE_SHA, XSA, digest
    assert digest(XSA.read_bytes())==EXE_SHA
    ref=Archive('P100'); a,b=ref.patterns
    r=WorkerRegistry(MANIFEST,ROOT,EXE,{'P100'})
    try:
        # Deliberately duplicate the first vote; verify per-key against B1.
        rows=r.resolve('P100',[[a,a],[b]])
        old=ref.lookup([a,b]); assert [len(old[a]),len(old[b])]==[2,1]
        assert rows==[old[a]+old[a],old[b]]
        packed=transfer(pack(rows,r.open('P100'),'P100'),'cpu')
        for ids,row in zip(packed.checked()[2].ids.tolist(),rows,strict=True):
            assert [packed.slot_spans[i] for i in ids if i>=0]==[s for s,_ in row]
    finally: r.close()


def test_missing_m15_refuses_before_ready(tmp_path):
    import zlib
    # Header/directory only, zero-width members; no corpus/index build.
    ids=[1,4,5,6,10,11,12,13,20]
    directory=b''.join(struct.pack('<IIQQQII',i,i,64,0,0,0,0) for i in ids)
    size=64+len(directory); h=bytearray(64)
    struct.pack_into('<4sIQQQIIQQ',h,0,b'SXI2',10,2,0,1,len(ids),size,size,1)
    struct.pack_into('<I',h,56,zlib.crc32(h+directory)); path=tmp_path/'no-map'; path.write_bytes(h+directory)
    proc=subprocess.run([EXE,'positional-worker',path,path],capture_output=True,input=b'')
    assert proc.returncode and not proc.stdout and b'without member m15' in proc.stderr
