"""CPU-only, deterministic Pile token-boundary / Wikipedia batch-query measurement."""
import argparse, hashlib, json, math, random, subprocess, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PILE = ROOT.parent / 'x42_pile_scale/data/pile24'
INDEX = Path('/mnt/nvme2n1/erikg/wf-finish-as-100gb-20261102T044713Z/corpus.sxi2')
XSA = Path.home() / 'sxgc/xsa/target/release/xsa'
OUT = ROOT / 'receipts/c14'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def query(patterns, name, locate=False):
    # CLI record separator cannot encode byte 30; never silently truncate it.
    assert all(p and b'\x1e' not in p for p in patterns)
    path = OUT / (name + '.patterns')
    path.write_bytes(b'\x1e'.join(patterns) + b'\x1e')
    cmd = [str(XSA), 'query', str(INDEX), '--patterns', str(path), '-j', '1']
    cmd += ['--limit', '1', '--output', 'jsonl', '--docs', '/mnt/nvme2n1/erikg/wiki'] if locate else ['--count']
    start = time.monotonic()
    with (OUT / (name + '.stdout')).open('wb') as out, (OUT / (name + '.stderr')).open('wb') as err:
        subprocess.run(cmd, stdout=out, stderr=err, check=True)
    elapsed = time.monotonic() - start
    raw = (OUT / (name + '.stdout')).read_bytes().splitlines()
    if locate:
        result = [json.loads(r) for r in raw]
    else:
        result = [int(r.rsplit(b'\t', 1)[1]) for r in raw]
    assert len(result) == len(patterns), (name, len(result), len(patterns))
    timings.append(dict(name=name, patterns=len(patterns), seconds=elapsed, command=cmd))
    (OUT / 'timings.json').write_text(json.dumps(timings, indent=2))
    if not locate and any(c<0 or c>100093934467 for c in result):
        raise ValueError(f'{name}: impossible interval width; raw query receipts retained; extraction rejected')
    return result

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', type=int, default=10000); args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    vocab = json.loads((PILE/'tokenizer.json').read_text())['model']['vocab']
    bs = list(range(33,127))+list(range(161,173))+list(range(174,256)); cs = bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    inverse = {chr(c): b for b,c in zip(bs,cs)}
    tokens = {v: bytes(inverse[c] for c in k) for k,v in vocab.items() if v != 50256}
    stream = np.memmap(PILE/'train.bin', dtype='<u2', mode='r')
    rng=random.Random(1401); samples=[]; rejected=0
    while len(samples)<args.samples:
        pos=rng.randrange(1024,len(stream)-256)
        ids=list(map(int,stream[pos-1024:pos+256])); before=ids[:1024]
        if 50256 in before: before=before[len(before)-1-before[::-1].index(50256)+1:]
        after=ids[1024:]; after=after[:after.index(50256)] if 50256 in after else after
        left=b''.join(tokens[i] for i in before); right=b''.join(tokens[i] for i in after)
        if len(left)<1024 or len(right)<256 or b'\x1e' in left[-1024:]+right[:256]: rejected+=1; continue
        samples.append(dict(position=pos,left=left[-1024:].hex(),right=right[:256].hex()))
    provenance=dict(seed=1401, samples=args.samples, rejected=rejected, sampling='uniform token boundaries, reject document boundary within 1024 left / 256 right bytes and CLI separator byte 30', pile_manifest_sha256=sha(PILE/'manifest.json'), wiki_manifest_sha256=sha(Path('/mnt/nvme2n1/erikg/wiki/MANIFEST.json')), xsa_sha256=sha(XSA), sxgc_head=subprocess.check_output(['git','-C',str(Path.home()/'sxgc'),'rev-parse','HEAD']).decode().strip(), index_bytes=INDEX.stat().st_size)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)); (OUT/'samples.json').write_text(json.dumps(samples))
    left=[bytes.fromhex(s['left']) for s in samples]; right=[bytes.fromhex(s['right']) for s in samples]
    counts={}
    for k in [16,64,256,1024]: counts[k]=query([s[-k:] for s in left],f'support_{k}')
    lo=[0]*len(left); hi=[1025]*len(left)
    for k in [16,64,256,1024]:
        for i,c in enumerate(counts[k]):
            if c: lo[i]=max(lo[i],k)
            else: hi[i]=min(hi[i],k)
    round_id=0
    while any(h-l>1 for l,h in zip(lo,hi)):
        active=[i for i in range(len(left)) if hi[i]-lo[i]>1]; mids=[(lo[i]+hi[i])//2 for i in active]
        cs=query([left[i][-m:] for i,m in zip(active,mids)],f'backoff_{round_id}')
        for i,m,c in zip(active,mids,cs):
            if c: lo[i]=m
            else: hi[i]=m
        round_id+=1
    (OUT/'max_context.json').write_text(json.dumps(lo))
    observed=query([s[-max(k,1):]+r[:1] for s,r,k in zip(left,right,lo)],'observed_next')
    # Enumerate right extensions, not FM backward extensions: correct causal branch orientation.
    active=[i for i,k in enumerate(lo) if k>0]; alphabet=[b for b in range(256) if b!=30]
    cs=query([left[i][-lo[i]:]+bytes([b]) for i in active for b in alphabet],'branches')
    features=[]
    for j,i in enumerate(active):
        row=cs[j*255:(j+1)*255]; total=sum(row); entropy=-sum((v/total)*math.log(v/total) for v in row if v) if total else 0
        features.append(dict(sample=i,context_bytes=lo[i],counts=row,support_size=total,entropy=entropy,coverage=bool(total)))
    features.extend(dict(sample=i,context_bytes=0,counts=[0]*255,support_size=0,entropy=0.,coverage=False) for i,k in enumerate(lo) if k==0)
    features.sort(key=lambda row: row['sample'])
    (OUT/'features.json').write_text(json.dumps(dict(alphabet=alphabet,rows=features)))
    spans=[left[i][-128:]+right[i][:128] for i in range(min(1000,len(left)))]
    exact=query(spans,'verbatim_count'); located=query(spans,'verbatim_locate',True)
    summary=dict(provenance=provenance,support_rates={k:sum(c>0 for c in v)/len(v) for k,v in counts.items()},max_context_quantiles={str(q):float(np.quantile(lo,q)) for q in [0,.1,.25,.5,.75,.9,.95,.99,1]},right_censored_1024=sum(k==1024 for k in lo),branch_agreement_supported=sum(observed[i]>0 for i in active)/len(active),verbatim_256_rate=sum(c>0 for c in exact)/len(exact),located_patterns=sum(bool(r.get('offsets')) for r in located),timings=timings,branch_alphabet_excluded=[30])
    (OUT/'coverage.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__':
    timings=[]; main()
