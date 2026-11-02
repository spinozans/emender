"""Independent receipt consistency and direct Wikipedia-byte checks; no training."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14'

def read(name):
    p=OUT/name
    return p.read_bytes() if p.exists() else gzip.decompress((OUT/(name+'.gz')).read_bytes())

def counts(name):
    rows=read(name+'.stdout').splitlines()
    assert [int(r.split(b'\t',1)[0]) for r in rows]==list(range(len(rows)))
    return [int(r.rsplit(b'\t',1)[1]) for r in rows]

def main():
    manifest=OUT/'MANIFEST.json'
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text())['sha256'].items():
            path=ROOT.parents[2]/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
    summary=json.loads((OUT/'coverage.json').read_text())
    samples=json.loads(read('samples.json')); n=len(samples)
    maxctx=json.loads((OUT/'max_context.json').read_text()); assert len(maxctx)==n
    prov=json.loads((OUT/'provenance.json').read_text())
    assert hashlib.sha256(Path('/mnt/nvme2n1/erikg/wiki/MANIFEST.json').read_bytes()).hexdigest()==prov['wiki_manifest_sha256']
    support={k:counts('support_'+str(k)) for k in [16,64,256,1024]}
    for k,cs in support.items():
        assert summary['support_rates'][str(k)]==sum(c>0 for c in cs)/n
    for i in range(n):
        assert all(support[a][i]>=support[b][i] for a,b in [(16,64),(64,256),(256,1024)])
        for k,c in support.items(): assert (c[i]>0)==(maxctx[i]>=k)
    # Replay binary-search evidence independently and certify exact maximality.
    known=[{} for _ in samples]
    for k,cs in support.items():
        for i,c in enumerate(cs): known[i][k]=c
    lo=[0]*n; hi=[1025]*n
    for k,cs in support.items():
        for i,c in enumerate(cs):
            if c: lo[i]=max(lo[i],k)
            else: hi[i]=min(hi[i],k)
    round_id=0
    while any(h-l>1 for l,h in zip(lo,hi)):
        name=f'backoff_{round_id}'; round_id+=1
        active=[i for i in range(n) if hi[i]-lo[i]>1]; pats=read(name+'.patterns').split(b'\x1e')[:-1]; cs=counts(name)
        assert len(active)==len(pats)==len(cs)
        for i,p,c in zip(active,pats,cs):
            m=(lo[i]+hi[i])//2; assert p==bytes.fromhex(samples[i]['left'])[-m:]
            known[i][m]=c
            if c: lo[i]=m
            else: hi[i]=m
    assert lo==maxctx and all(h-l==1 for l,h in zip(lo,hi))
    observed=counts('observed_next'); feat=json.loads(read('features.json')); alphabet=feat['alphabet']
    assert [row['sample'] for row in feat['rows']]==list(range(n))
    raw_branches=counts('branches'); active=[i for i,k in enumerate(maxctx) if k>0]
    assert len(raw_branches)==len(active)*255
    for j,i in enumerate(active):
        raw_row=raw_branches[j*255:(j+1)*255]; feature=feat['rows'][i]
        if feature.get('valid',True):
            assert feature['counts']==raw_row and all(0<=c<=100093934467 for c in raw_row)
        else:
            assert any(c>100093934467 for c in raw_row)
    assert [r['sample'] for r in feat['rows'] if not r.get('valid',True)]==[2769,9057]
    for row in feat['rows']:
        i=row['sample']; k=maxctx[i]
        if not row.get('valid',True):
            assert i in {2769,9057} and row['counts'] is None and row['entropy'] is None and row['support_size'] is None and row['coverage'] is None
            continue
        if k==0:
            assert not any(row['counts']) and row['support_size']==0 and row['entropy']==0 and not row['coverage']
            continue
        assert known[i][k]>0
        assert sum(row['counts'])<=known[i][k]
        first=bytes.fromhex(samples[i]['right'])[0]
        assert row['counts'][alphabet.index(first)]==observed[i]
        assert row['support_size']==sum(row['counts'])
        assert row['coverage']==bool(row['support_size'])
        total=row['support_size']
        entropy=-sum((v/total)*math.log(v/total) for v in row['counts'] if v) if total else 0
        assert abs(entropy-row['entropy'])<1e-12
    # Exact offsets checked against immutable corpus bytes, rather than trusting locate output.
    exact=counts('verbatim_count'); spans=read('verbatim_count.patterns').split(b'\x1e')[:-1]
    located=[json.loads(r) for r in read('verbatim_locate.stdout').decode().splitlines()]
    assert summary['verbatim_256_rate']==sum(c>0 for c in exact)/len(exact)
    assert [r['pattern_id'] for r in located]==list(range(len(exact)))
    checked=0
    with open('/mnt/nvme2n1/erikg/wiki/corpus.bin','rb') as f:
        for i,row in enumerate(located):
            offsets=row['offsets']; assert bool(offsets)==bool(exact[i])
            for offset in offsets:
                f.seek(offset); assert f.read(len(spans[i]))==spans[i]; checked+=1
    long_rows=[json.loads(r) for r in read('long_context_locate.stdout').decode().splitlines()]
    long_patterns=read('long_context_locate.patterns').split(b'\x1e')[:-1]
    refs=json.loads((OUT/'long_context_refs.json').read_text())
    assert len(long_rows)==len(long_patterns)==len(refs)==68
    long_checked=0
    with open('/mnt/nvme2n1/erikg/wiki/corpus.bin','rb') as f:
        for i,(row,pattern,ref) in enumerate(zip(long_rows,long_patterns,refs)):
            assert row['pattern_id']==i and len(row['offsets'])==1
            assert pattern==bytes.fromhex(samples[ref['sample']]['left'])[-ref['length']:]
            f.seek(row['offsets'][0]); assert f.read(len(pattern))==pattern; long_checked+=1
    # Independent library decoder validates sampled source bytes, including BPE byte joins.
    from tokenizers import Tokenizer
    pile=ROOT.parent/'x42_pile_scale/data/pile24'
    manifest=json.loads((pile/'manifest.json').read_text())
    assert hashlib.sha256((pile/'manifest.json').read_bytes()).hexdigest()==prov['pile_manifest_sha256']
    assert hashlib.sha256((pile/'tokenizer.json').read_bytes()).hexdigest()==manifest['tokenizer_sha256']
    assert (pile/'train.bin').stat().st_size==2*manifest['tokens']['train']
    assert hashlib.sha256((Path.home()/'sxgc/xsa/target/release/xsa').read_bytes()).hexdigest()==prov['xsa_sha256']
    decoder=Tokenizer.from_file(str(pile/'tokenizer.json'))
    stream=np.memmap(pile/'train.bin',dtype='<u2',mode='r')
    for sample in samples[:100]:
        pos=sample['position']; right_ids=list(map(int,stream[pos:pos+256]))
        if 50256 in right_ids: right_ids=right_ids[:right_ids.index(50256)]
        decoded=decoder.decode(right_ids)
        # Compare a valid UTF-8 prefix only: byte 256 can bisect a Unicode scalar.
        raw=bytes.fromhex(sample['right'])
        prefix=raw.decode('utf8',errors='replace').rstrip('\ufffd')
        assert decoded.startswith(prefix), (pos,repr(prefix[:30]),repr(decoded[:30]))
    result=dict(status='passed',sample_count=n,maximality_checks=n,branch_observed_checks=sum(r.get('valid',True) and r['context_bytes']>0 for r in feat['rows']),rejected_features=[r['sample'] for r in feat['rows'] if not r.get('valid',True)],query_health='failed: 2 UINT64_MAX branch counts; training blocked',direct_locate_checks=checked,long_context_direct_checks=long_checked,decoder_checks=100,manual_notes='No GPU, no training, Wikipedia assets read only. CLI -j 1 does not constrain the default Rayon global pool; timings are host measurements, not single-core claims.')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
