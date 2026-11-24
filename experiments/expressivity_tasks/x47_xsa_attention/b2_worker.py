"""Trusted local whole-mount adapter. Callers own namespace admission, not native ACLs."""
import json
import struct
import subprocess
from time import perf_counter_ns as clock

from .b1_mount import ArchiveDescriptor, Span, require


class WorkerRegistry:
    def __init__(self, manifest, root, executable, allowed):
        self.manifest, self.root, self.executable = manifest, root, executable
        self.allowed = frozenset(allowed)
        self.workers = {}

    def open(self, name, *, excluded_docs=(), whole_bank=True):
        if name not in self.allowed or excluded_docs or not whole_bank:
            raise PermissionError('only an admitted whole mount can be opened')
        if name not in self.workers:
            desc = ArchiveDescriptor(name, self.manifest[name], self.root)
            self.workers[name] = PositionalWorker(desc, self.executable)
        return self.workers[name]

    def resolve(self, name, rows, *, excluded_docs=(), whole_bank=True):
        require(1 <= len(rows) <= 8 and all(1 <= len(row) <= 2 for row in rows), 'batch capacity')
        w = self.open(name, excluded_docs=excluded_docs, whole_bank=whole_bank)
        patterns = list(dict.fromkeys(p for row in rows for p in row))
        spans, record = w.request(patterns)
        if any(row['status'] != 'complete' for row in record['rows']):
            raise OverflowError('native raw interval exceeds 8; no page')
        votes = {p: [(a, tuple((1+(p[j % len(p)]+a.start+j)%256)/256 for j in range(8)))
                     for a in addresses] for p, addresses in zip(patterns,spans,strict=True)}
        return [[vote for p in row for vote in votes[p]] for row in rows]

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()


class PositionalWorker:
    def __init__(self, descriptor, executable):
        self.desc = descriptor
        self.name, self.version = descriptor.name, descriptor.version
        t = clock()
        self.proc = subprocess.Popen([executable,'positional-worker',descriptor.index,descriptor.text],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            ready_line = self.proc.stdout.readline()
            self.ready_bytes = len(ready_line)
            self.ready = json.loads(ready_line)
            require(self.ready['ready'] == 1, 'protocol version')
        except BaseException:
            self.close(); raise
        self.start_ns = clock()-t

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()
        self.proc.stdout.close()

    def request(self, patterns):
        require(1 <= len(patterns) <= 16 and all(isinstance(p, bytes) and 1 <= len(p) <= 256 for p in patterns), 'frame capacity')
        t = clock()
        data = bytes([len(patterns)]) + b''.join(struct.pack('<H',len(p))+p for p in patterns)
        try:
            self.proc.stdin.write(struct.pack('<I',len(data))+data); self.proc.stdin.flush()
            line = self.proc.stdout.readline(); trailer = self.proc.stdout.readline(); wire_ns = clock()-t
            require(line.endswith(b'\n') and trailer.endswith(b'\n'), 'unterminated positional response')
            record = json.loads(line)
            fence = json.loads(trailer)
            require(isinstance(fence, dict) and fence.keys() == {'send_ns'} and
                    type(fence['send_ns']) is int and fence['send_ns'] >= 0, 'invalid timing trailer')
            record['send_ns'] = fence['send_ns']
            record.update(wire_ns=wire_ns, tx=4+len(data), rx=len(line)+len(trailer))
            addresses=[]
            for row,p in zip(record['rows'],patterns,strict=True):
                raw,hits = row['raw'],row['hits']
                require(row['status'] == ('complete' if raw <= 8 else 'overcap') and
                        len(row['lf']) == (raw if raw <= 8 else 0) and
                        len({h[0] for h in hits}) == len(hits) <= len(row['lf']), 'row completeness')
                spans=[]
                for pos,d,j in hits:
                    starts=self.desc.starts
                    require(0<=d<len(starts) and j>=0 and pos==int(starts[d])+j and
                            pos+len(p)<=(starts[d+1] if d+1<len(starts) else self.desc.size),'source span')
                    spans.append(Span(self.name,self.version,d,j,j+len(p)))
                addresses.append(spans)
            record['validated_ns']=clock()-t
            return addresses,record
        except BaseException:
            self.proc.kill(); self.close(); raise
