"""Native session binding witness. No parser, oracle offsets or raw prefix in policy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from .xsa_binding_fixture import fixture, lookup, K


class Session:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe, 'session', '8'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def rpc(self, cmd):
        self.p.stdin.write((cmd+'\n').encode()); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line: raise RuntimeError('XSA session closed: '+cmd[:80])
        return json.loads(line)

    def close(self):
        self.p.stdin.close()
        if self.p.wait(timeout=10): raise RuntimeError('XSA session failed')


def program(rpc, query, hi):
    """Only public query bytes, history horizon and generic index reads are available.

    This *handwritten* program knows the record grammar. No join helper on server.
    Scope a named header from its end through the next structural peer. Payloads
    can equal 'G' or 'R'; a tag there is followed by padding, not a hex digit.
    """
    trace = []
    def read(cmd):
        ans = rpc(cmd); trace.append(dict(cmd=cmd,reply=ans)); return ans
    def locate(key, lo, end):
        return read(f'locate {key.hex()} {lo} {end} 4096')['positions']
    def byte(p): return read(f'byte {p}')['byte']
    def unique(key, lo, end):
        hits = locate(key, lo, end)
        if len(hits)!=1: raise ValueError(f'expected one hit, got {len(hits)}')
        return hits[0]
    def scope(key, lo, end):
        begin = unique(key,lo,end)+len(key)
        for pos in locate(key[:1],begin,end):
            if pos+1<end and byte(pos+1) in b'0123456789abcdef': return begin,pos
        return begin,end
    if len(query)!=17 or query[0]!=ord('?') or query[-1]!=ord('='): raise ValueError('query format')
    g,r,f = query[1:6],query[6:11],query[11:16]
    span = scope(g,0,hi)
    span = scope(r,*span)
    address = unique(f,*span)+len(f)+K
    if address>=span[1]: raise ValueError('derived address escaped selected row')
    return byte(address),trace


def check(exe,out):
    first=[]; totals={}; targets=0; proof=hashlib.sha256()
    for seed in range(64):
        for prefix,y in fixture(seed):
            if lookup(prefix)!=y: raise ValueError('independent oracle mismatch')
            s=Session(exe); q=prefix[-17:];hi=len(prefix)-len(q)
            # Partial seal inside requested field anchor tests a matching span
            # crossing the base/tail seam. The split is a storage choice only.
            cut=prefix.index(q[11:16])+2
            try:
                for mode in ('live','partial','sealed'):
                    if mode=='live': s.rpc('append '+prefix.hex())
                    elif mode=='partial':
                        s.close();s=Session(exe);s.rpc('append '+prefix[:cut].hex());s.rpc('seal');s.rpc('append '+prefix[cut:].hex())
                    else: s.rpc('seal')
                    got,t=program(s.rpc,q,hi)
                    if got!=y: raise ValueError((seed,mode,got,y))
                    if mode=='live':reference=[(x['cmd'],x['reply'].get('positions',x['reply'].get('byte'))) for x in t]
                    elif reference!=[(x['cmd'],x['reply'].get('positions',x['reply'].get('byte'))) for x in t]: raise ValueError('physical layout changed operation trace')
                    if seed==0 and mode=='sealed':first.append(dict(query=q.decode(),answer_hex=f'{got:02x}',operations=t))
                    c=totals.setdefault(mode,Counter());c['targets']+=1;c['calls']+=len(t)
                    for x in t:
                        a=x['reply'];c['sealed_candidate_rows']+=a.get('sealed_rows',0);c['tail_starts_examined']+=a.get('tail_starts',0);c['positions_returned']+=len(a.get('positions',[]))
                proof.update(prefix);proof.update(bytes([y]));targets+=1
            finally: s.close()
    receipt=dict(engine_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),targets=targets,transcript_sha256=proof.hexdigest(),counts=totals,first_pair=first)
    out.write_text(json.dumps(receipt,indent=2)+'\n');print(targets,totals,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('exe',type=Path);p.add_argument('out',type=Path);a=p.parse_args();check(a.exe.resolve(),a.out)
