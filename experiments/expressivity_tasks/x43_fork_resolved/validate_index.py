"""Independent m1 Huffman/gamma decoder and FM repeated-context enumeration.

Validation uses O(n) arrays on this tiny fixture, separate from the bounded
xsa builder. No direct-SA labels or lengths are read until comparison.
"""
import json
import struct
import zlib
import numpy as np
from index_pipeline import ROOT, OUT, XSA, run, digest

class Bits:
    def __init__(self,b): self.b=b; self.p=0
    def bit(self):
        v=(self.b[self.p//8]>>(self.p%8))&1; self.p+=1; return v

def decode(path):
    b=path.read_bytes(); assert b[:8]==b'SXI2\x06\x00\x00\x00'
    n,_,runs=struct.unpack_from('<QQQ',b,8)
    count,hsize=struct.unpack_from('<II',b,32)
    header=bytearray(b[:hsize]); crc=struct.unpack_from('<I',header,56)[0]; header[56:60]=b'\0'*4
    assert zlib.crc32(header)==crc
    members={}
    for i in range(count):
        mid,codec,off,size,items,crc,_=struct.unpack_from('<IIQQQII',b,64+40*i)
        data=b[off:off+size]; assert zlib.crc32(data)==crc
        members[mid]=(codec,data)
    codec,m=members[1]; assert codec==114
    c=np.frombuffer(m[:2048],dtype='<u8').astype(np.int64)
    codes={}; code=prev=0
    for width,sym in sorted((w,s) for s,w in enumerate(m[2048:2304]) if w):
        code <<= width-prev; codes[width,code]=sym; code+=1; prev=width
    blocks,width,total,payload=struct.unpack_from('<QQQQ',m,2304); assert total==runs
    heads=[]; lens=[]
    for block in range(blocks):
        off,hbits,lbits=struct.unpack_from('<QII',m,2336+16*block)
        start=payload+off; h=Bits(m[start:start+(hbits+7)//8]); start+=(hbits+7)//8
        l=Bits(m[start:start+(lbits+7)//8])
        for _ in range(min(width,runs-len(heads))):
            w=v=0
            while (w,v) not in codes: v=2*v+h.bit(); w+=1
            heads.append(codes[w,v])
            zeros=0
            while l.bit()==0: zeros+=1
            v=1
            for _ in range(zeros): v=2*v+l.bit()
            lens.append(v)
        assert h.p==hbits and l.p==lbits
    bwt=np.repeat(np.array(heads,dtype=np.uint8),lens); assert len(bwt)==n
    sigma=np.frombuffer(members[7][1],dtype=np.uint8) if 7 in members else np.arange(256,dtype=np.uint8)
    return bwt,c,sigma,np.array(lens)

def main():
    container=OUT/'build/corpus.sxi2'; corpus=(OUT/'corpus/corpus.bin').read_bytes(); raw=(ROOT/'input.txt').read_bytes()
    binding=json.loads((OUT/'binding.json').read_text()); offset=binding['body_offset']; size=len(raw)
    bwt,c,sigma,runlens=decode(container); n=len(bwt)
    occ={int(ch):np.flatnonzero(bwt==ch) for ch in np.unique(bwt)}
    lf=np.empty(n,dtype=np.int64)
    for ch,rows in occ.items(): lf[rows]=c[ch]+np.arange(len(rows))
    # END is the unique minimum symbol; recover coordinates by one LF cycle.
    sa=np.full(n,-1,dtype=np.int64); row=0
    for step in range(n):
        assert sa[row]==-1
        sa[row]=(n-1-step)%n; row=lf[row]
    assert row==0 and np.all(sa>=0)
    normalized=np.append(sigma[np.frombuffer(corpus,dtype=np.uint8)],0)
    assert np.array_equal(bwt,normalized[(sa-1)%n])
    inverse=np.empty(n,dtype=np.int64); inverse[sa]=np.arange(n)
    # Grow every repeated left context by one byte with backward search.
    # Boundary positions removed from intervals enforce precisely part1c's
    # finite target convention, excluding title, separator and virtual END.
    active=np.arange(size,dtype=np.int64); lo=np.zeros(size,dtype=np.int64); hi=np.full(size,n,dtype=np.int64)
    lengths=np.zeros(size,dtype=np.int64); fork=np.zeros(size,dtype=bool)
    previous_lo=lo.copy(); previous_hi=hi.copy(); depth=0; cache={}
    def finalize(ids,left,right,d):
        for i,l,r in zip(ids,left,right):
            key=(int(l),int(r),d)
            if key not in cache:
                targets=sa[l:r]+d-offset
                targets=targets[(targets>=d)&(targets<size)]
                assert len(targets)>=2
                cache[key]=len(set(raw[int(j)] for j in targets))>1
            lengths[i]=d; fork[i]=cache[key]
    while len(active):
        depth+=1
        bounded=active>=depth
        finalize(active[~bounded],previous_lo[~bounded],previous_hi[~bounded],depth-1)
        active=active[bounded]; previous_lo=previous_lo[bounded]; previous_hi=previous_hi[bounded]
        if not len(active): break
        chars=sigma[np.frombuffer(raw,dtype=np.uint8)[active-depth]]
        lo=np.empty(len(active),dtype=np.int64); hi=lo.copy()
        for ch in np.unique(chars):
            mask=chars==ch; rows=occ[int(ch)]
            lo[mask]=c[ch]+np.searchsorted(rows,previous_lo[mask]); hi[mask]=c[ch]+np.searchsorted(rows,previous_hi[mask])
        invalid=np.sort(np.concatenate((inverse[:offset],inverse[offset+size-depth:])))
        counts=hi-lo-(np.searchsorted(invalid,hi)-np.searchsorted(invalid,lo))
        repeated=counts>=2
        finalize(active[~repeated],previous_lo[~repeated],previous_hi[~repeated],depth-1)
        active=active[repeated]; previous_lo=lo[repeated]; previous_hi=hi[repeated]
        if depth%50==0: print('depth',depth,'active',len(active),flush=True)
    np.savez_compressed(OUT/'labels.npz',fork=fork,context_length=lengths)
    with np.load(ROOT/'receipts/part1c/labels.npz') as direct:
        report=dict(positions=size,fork=int(fork.sum()),forced=int((~fork).sum()),label_agreements=int((fork==direct['fork']).sum()),label_disagreements=int((fork!=direct['fork']).sum()),length_disagreements=int((lengths!=direct['context_length']).sum()),maximum_context=int(lengths.max()),container_crc_verified=True,lf_text_verified=True,method='m1 decode, LF coordinates, vectorized FM backward extension; finite body targets only',container_sha256=digest(container),labels_sha256=digest(OUT/'labels.npz'))
    (OUT/'cross_validation.json').write_text(json.dumps(report,indent=2)+'\n'); print(report,flush=True)
    patterns=[raw[p:p+24] for p in [0,1,100,1000,10000,size//2,size-24]]+[raw[-10:],raw[:1],b'\x00',b'\xff',b'\x01'*24,b'x'*65536]
    (OUT/'patterns.bin').write_bytes(b'\x1e'.join(patterns)+b'\x1e')
    p=run('query',[XSA,'query',container,'--patterns',OUT/'patterns.bin','-j','1','--output','jsonl','--verify']); assert p.returncode==0
    answers=[json.loads(line) for line in p.stdout.splitlines()]
    mismatches=[]; checks=[]
    for pat,answer in zip(patterns,answers):
        expected=[]; start=0
        while True:
            pos=corpus.find(pat,start)
            if pos<0: break
            expected.append(pos); start=pos+1
        match=sorted(answer['offsets'])==expected and answer['count']==len(expected)
        checks.append(dict(pattern_id=answer['pattern_id'],pattern_hex=pat.hex() if len(pat)<100 else None,pattern_bytes=len(pat),expected_count=len(expected),actual_count=answer['count'],match=match,expected_offsets=expected,actual_offsets=answer['offsets']))
        if not match: mismatches.append(checks[-1])
    edges={}
    for name,data in [('empty_file',b''),('empty_record',b'a\x1e\x1eb\x1e'),('too_long',b'x'*65537+b'\x1e')]:
        file=OUT/(name+'.bin'); file.write_bytes(data)
        p=run(name,[XSA,'query',container,'--patterns',file,'-j','1']); assert p.returncode!=0; edges[name]='refused as documented'
    report=dict(status='failed' if mismatches else 'passed',mismatches=mismatches,checks=checks,patterns=len(patterns),oracle='overlapping exact byte scan of corpus; offsets include 16-byte title prefix',edges=edges)
    (OUT/'query_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(1 if mismatches else 0)

if __name__=='__main__': main()
