"""B1 exact matched-span providers; no continuation or structural END value."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess
import time
import tempfile
import uuid
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/nvme2n1/erikg')
XSA = Path.home() / 'sxgc/xsa/target/release/xsa'
SESSION_EXE = Path.home() / 'sxgc/xsa/target/debug/xsa'
SESSION_SHA = '87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
EXE_SHA = 'fdf5f97cb23d35e427161b60f9d7be47f5eeed6fd0bbf1a4bf4b81dbb6c0f1f0'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class Span:
    source: str
    version: str
    doc: int
    start: int  # document relative
    stop: int


class Registry:
    def __init__(self, allowed):
        self.allowed = frozenset(allowed)
        require(self.allowed <= {'context', 'P100', 'F6'}, 'unknown namespace')
        self.mounts = {}

    def open(self, name, *, excluded_docs=(), whole_bank=True):
        if name not in self.allowed or excluded_docs or not whole_bank:
            raise PermissionError('only entire authorized banks are implemented')
        if name not in self.mounts:
            exe,sha = (SESSION_EXE,SESSION_SHA) if name=='context' else (XSA,EXE_SHA)
            require(digest(exe.read_bytes()) == sha, 'native identity')
            self.mounts[name] = Session() if name == 'context' else Archive(name)
        return self.mounts[name]

    def resolve(self, name, rows, *, excluded_docs=(), whole_bank=True):
        require(1 <= len(rows) <= 8 and all(1 <= len(r) <= 2 for r in rows), 'batch capacity')
        mount = self.open(name, excluded_docs=excluded_docs, whole_bank=whole_bank)
        patterns = list(dict.fromkeys(p for row in rows for p in row))
        hits = mount.lookup(patterns)
        return [[item for p in row for item in hits[p]] for row in rows]

    def close(self):
        if 'context' in self.mounts:
            self.mounts['context'].close()


class ArchiveDescriptor:
    """Validate the B1 immutable v10 + XSEP descriptor without a query policy."""
    def __init__(self, name, conf, root):
        self.name = name
        files = conf["files"]; paths = [root / r["path"] for r in files]
        require(len(files) == 4 and "directory_sha256" in files[0] and all("sha256" in f for f in files[2:]), "descriptor identities")
        self.index, self.text, seps, _ = paths
        self.version = 'span8-v1:' + digest(json.dumps(conf, sort_keys=True).encode())
        for r, path in zip(files, paths):
            st = path.stat()
            require((st.st_ino, st.st_size, st.st_mtime_ns) == (r['inode'], r['size'], r['mtime_ns']), 'source identity')
            if 'sha256' in r:
                require(digest(path.read_bytes()) == r['sha256'], 'sidecar identity')
        with self.index.open('rb') as f:
            h = f.read(64); f.seek(-400, 2); directory = f.read(400)
        size = self.text.stat().st_size
        require(h[:8] == b'SXI2\n\0\0\0' and struct.unpack_from('<Q', h, 8)[0] == size+1, 'SXI2 v10')
        require(struct.unpack_from('<IIQ', h, 32) == (10,464,files[0]['size']), 'directory length')
        require(digest(h+directory) == files[0]['directory_sha256'] and zlib.crc32(h[:56]+bytes(4)+h[60:]+directory) == struct.unpack_from('<I',h,56)[0], 'directory identity/CRC')
        members = list(struct.iter_unpack('<IIQQQII', directory))
        require([m[0] for m in members] == [1,4,5,7,10,11,12,13,15,20], 'unsupported members')
        end = 64
        for off, n in sorted((m[2],m[3]) for m in members):
            require(off == (end+7)//8*8, 'member extents'); end = off+n
        require(end+400 == files[0]['size'], 'directory extent')
        data = seps.read_bytes()
        magic,v,_,stride,docs,length,anchors,base = struct.unpack_from('<4sIIIQQQQ',data)
        require((magic,v,docs,length,base,stride)==(b'XSEP',1,conf['docs'],size,0,64) and anchors == (docs+63)//64 and len(data)==48+docs*4+anchors*8,'XSEP layout')
        delta = np.frombuffer(data,'<u4',docs,48)
        self.starts = np.cumsum(delta,dtype=np.int64)
        require(docs>0 and delta[0]==0 and delta[1:].all() and self.starts[-1]<size and np.array_equal(self.starts[::64],np.frombuffer(data,'<u8',anchors,48+4*docs)), 'XSEP starts')
        self.size = size


class Archive(ArchiveDescriptor):
    def __init__(self, name):
        super().__init__(name, json.loads((HERE/"b1_mounts.json").read_text())[name], ROOT)
        if name == 'P100':
            reg = HERE.parent/'x43_fork_resolved/receipts/p100_query_verification/registration.json'
            pats = json.loads(reg.read_text())['patterns']; self.patterns = tuple(bytes.fromhex(pats[i]['hex']) for i in (2,6))
        else:
            with self.text.open('rb') as f:
                first = f.read(24)
            self.patterns = (first[:8],first[16:],b' the quick brown fox ')

    def lookup(self, patterns):
        require(all(p in self.patterns for p in patterns),'unregistered pattern')
        t=time.perf_counter()
        with tempfile.NamedTemporaryFile() as f:
            f.write(b''.join(p.hex().encode()+b'\n' for p in patterns)); f.flush()
            out = subprocess.run([XSA,'search',self.index,'--patterns',f.name,'--hex','--jsonl','--max-occ','8','--corpus',self.text], capture_output=True, text=True, timeout=30, check=True)
        self.cli_ms=1000*(time.perf_counter()-t)
        if 'SEARCH_PATTERN_REFUSED' in out.stderr:
            raise OverflowError('native raw interval exceeds 8; no page')
        result = {p:[] for p in patterns}; seen=set()
        summary = re.search(r'SEARCH_BOUNDARY mode=no-cross-default structure=member-m15 patterns=(\d+) absent=\d+ refused=0 excluded_cross_doc=(\d+) matches_kept=(\d+) verified=(\d+)',out.stderr)
        self.search_ms=1000*float(re.search(r' wall_s=([0-9.]+) law=',out.stderr)[1])
        lines = out.stdout.splitlines()
        require(summary and tuple(map(int,summary.groups())) == (len(patterns),int(summary[2]),len(lines),len(lines)+int(summary[2])), 'incomplete native result')
        for line in lines:
            hit=json.loads(line); i,pos=hit['id'],hit['pos']; require(0<=i<len(patterns),'pattern id'); p=patterns[i]
            d=int(self.starts.searchsorted(pos,side='right'))-1
            end=int(self.starts[d+1]) if d+1<len(self.starts) else self.size
            require(d>=0 and pos+len(p)<=end and not hit['cross'] and hit['len']==len(p) and (i,pos) not in seen and len(result[p])<8,'invalid hit')
            seen.add((i,pos)); start=pos-int(self.starts[d]); addr=Span(self.name,self.version,d,start,start+len(p))
            value=tuple((1+(p[j%len(p)]+start+j)%256)/256 for j in range(8))
            result[p].append((addr,value))
        return result


class Session:
    def __init__(self):
        self.proc=subprocess.Popen([SESSION_EXE,'session','16'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        self.identity=uuid.uuid4().hex; self.generation=0; self.values=[]; self.starts=[0]

    @property
    def version(self):
        return f'activation8-v1:{self.identity}:{self.generation}:{len(self.values)}'

    def command(self,s):
        self.proc.stdin.write(s+'\n'); self.proc.stdin.flush()
        line=self.proc.stdout.readline()
        require(line,'session failed'); return json.loads(line)

    def observe(self,data,values):
        require(len(data)>0 and np.shape(values)==(len(data),8) and np.isfinite(values).all(),'observed activation shape')
        require(self.command('append '+data.hex())['visible']==len(self.values)+len(data),'watermark')
        self.values.extend(map(tuple,values)); self.generation+=1

    def end(self):
        require(self.command('end')['ended'],'end'); self.starts.append(len(self.values)); self.generation+=1

    def seal(self):
        require(self.command('seal')['sealed']==len(self.values),'seal'); self.generation+=1

    def lookup(self,patterns):
        out={}
        for p in patterns:
            require(1<=len(p)<=16,'live depth')
            positions=self.command(f'locate {p.hex()} 0 {len(self.values)} 8')['positions']; out[p]=[]
            for pos in positions:
                d=int(np.searchsorted(self.starts,pos,'right'))-1; stop=pos+len(p)
                require(0<=pos and stop<=(self.starts+[len(self.values)])[d+1],'live bounds')
                out[p].append((Span('context',self.version,d,pos-self.starts[d],stop-self.starts[d]),self.values[stop-1]))
        return out

    def close(self):
        self.proc.stdin.close(); self.proc.wait(timeout=3); self.proc.stdout.close()
