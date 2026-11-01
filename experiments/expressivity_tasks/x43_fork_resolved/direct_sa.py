"""Finite-byte repeated-left-context decomposition; see DERIVATION_PART1C.md."""
import hashlib
import json
from pathlib import Path
import numpy as np
from decompose import summarize

ROOT = Path(__file__).parent

def suffix_array(a):
    n = len(a)
    sa = np.arange(n)
    rank = np.asarray(a, dtype=np.int64)
    k = 1
    while k < n:
        second = np.full(n, -1, dtype=np.int64)
        second[:-k] = rank[k:]
        sa = sa[np.argsort(second[sa], kind='stable')]
        sa = sa[np.argsort(rank[sa], kind='stable')]
        change = np.ones(n, dtype=bool)
        change[1:] = (rank[sa[1:]] != rank[sa[:-1]]) | (second[sa[1:]] != second[sa[:-1]])
        new = np.empty(n, dtype=np.int64)
        new[sa] = np.cumsum(change)-1
        rank = new
        if rank.max() == n-1:
            break
        k *= 2
    return sa

def classify(raw):
    n = len(raw)
    if n < 2:
        raise ValueError('need at least two positions')
    # Empty suffix as unique -1 symbol; drop full reverse suffix (target n).
    a = np.frombuffer(raw[::-1], dtype=np.uint8).astype(np.int64)
    a = np.append(a, -1)
    full = suffix_array(a)
    inv = np.empty(n+1, dtype=np.int64); inv[full] = np.arange(n+1)
    lc = np.zeros(n+1, dtype=np.int64)
    h = 0
    for p in range(n+1):
        r = inv[p]
        if r == 0:
            h = 0
            continue
        q = full[r-1]
        while p+h < n and q+h < n and a[p+h] == a[q+h]:
            h += 1
        lc[r] = h
        h = max(h-1, 0)
    r = int(inv[0])
    # Removing a suffix joins its two neighbors with min of their LCPs.
    if r+1 < n+1:
        lc[r+1] = min(lc[r], lc[r+1])
    sa = np.delete(full, r)
    lc = np.delete(lc, r)
    left = np.zeros(n, dtype=np.int64); right = np.full(n, n, dtype=np.int64)
    stack = []
    for e in range(1,n):
        while stack and lc[stack[-1]] >= lc[e]:
            stack.pop()
        left[e] = stack[-1] if stack else 0
        stack.append(e)
    stack = []
    for e in range(n-1,0,-1):
        while stack and lc[stack[-1]] >= lc[e]:
            stack.pop()
        right[e] = stack[-1] if stack else n
        stack.append(e)
    pos = n-sa
    lengths = np.zeros(n,dtype=np.int64)
    intervals = np.empty((n,2),dtype=np.int64)
    for r,i in enumerate(pos):
        e = r if r == n-1 or lc[r] >= lc[r+1] else r+1
        lengths[i] = lc[e]
        intervals[i] = (left[e],right[e]) if lc[e] else (0,n)
    return pos, lengths, intervals

def brute(raw):
    out=[]
    for i in range(len(raw)):
        for l in range(i,-1,-1):
            occ=[j for j in range(l,len(raw)) if raw[j-l:j] == raw[i-l:i]]
            if len(occ)>=2:
                out.append((l,occ)); break
    return out

def tests():
    examples=[b'abac',b'aaaa',b'abab',b'abxabz',b'banana',b'abc',bytes([0,255,0,1])]
    rng=np.random.default_rng(43)
    examples += [bytes(rng.integers(0,4,size=n,dtype=np.uint8)) for n in range(2,65) for _ in range(8)]
    for raw in examples:
        pos,lens,intervals=classify(raw)
        for i,(l,occ) in enumerate(brute(raw)):
            lo,hi=intervals[i]
            assert lens[i]==l and sorted(pos[lo:hi])==occ,(raw,i,l,occ)
    # Explicit known fork/forced examples.
    raw=b'abac'; pos,lens,iv=classify(raw)
    labels=[len(set(raw[j] for j in pos[lo:hi]))>1 for lo,hi in iv]
    assert labels == [True,True,True,True]
    pos, lens, iv = classify(b'aaaa')
    assert all(len(set(b'aaaa'[j] for j in pos[lo:hi])) == 1 for lo, hi in iv)
    assert lens.tolist() == [0, 1, 2, 2]
    return {'status':'passed','examples':len(examples),'checks':['SA intervals and maximal lengths equal independent exhaustive search','abac known labels: all fork','overlaps, empty contexts, binary bytes, finite boundaries']}

def main():
    out=ROOT/'receipts/part1c'; out.mkdir(exist_ok=True)
    test=tests(); (out/'tests.json').write_text(json.dumps(test,indent=2)+'\n')
    raw=(ROOT/'input.txt').read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    assert digest=='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    manifest=json.loads((ROOT/'receipts/artifact_manifest.json').read_text())
    for name,meta in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['sha256'],name
    print('tests and all retained hashes passed',flush=True)
    pos,lens,iv=classify(raw); print('SA/LCP complete',flush=True)
    n=len(raw); split=int(.9*n); data=np.frombuffer(raw,dtype=np.uint8)
    fork=np.zeros(n,dtype=bool); cache={}; hist={}; tops=[]
    unigram=np.bincount(data[:split],minlength=256)/split
    oracle_nll=np.full(n,np.nan); oracle_correct=np.zeros(n,dtype=bool)
    covered=np.zeros(n,dtype=bool); zero=np.zeros(n,dtype=bool)
    for i,(lo,hi) in enumerate(iv):
        key=(int(lo),int(hi))
        if key not in cache:
            occ=pos[lo:hi]; counts=np.bincount(data[occ],minlength=256)
            forked=np.count_nonzero(counts)>1
            train=occ[occ<split]; tc=np.bincount(data[train],minlength=256)
            cache[key]=(forked,tc)
            if forked:
                p=counts[counts>0]/len(occ)
                tops.append(dict(context_length=int(lens[i]),context_hex=raw[i-lens[i]:i].hex(),occurrences=len(occ),entropy_bits=float(-(p*np.log2(p)).sum()),branches={str(j):int(c) for j,c in enumerate(counts) if c}))
        fork[i],tc=cache[key]
        if tc.sum():
            covered[i]=True; zero[i]=tc[data[i]]==0
            oracle_nll[i]=-np.log(tc[data[i]]/tc.sum()) if tc[data[i]] else np.inf
            oracle_correct[i]=tc.argmax()==data[i]
    print('counts complete',flush=True)
    models=[]
    for arm in ('emender','transformer'):
        with np.load(ROOT/f'{arm}_per_position.npz') as ev:
            positions=ev['position']; assert np.array_equal(positions,np.arange(split+1,n))
            models.append(dict(arm=arm,**summarize(ev['nll'],ev['correct'],fork[positions])))
    targets=np.arange(split+1,n); baselines={}
    for name,mask in [('forced',~fork[targets]),('fork',fork[targets])]:
        t=targets[mask]; c=t[covered[t]]; finite=c[~zero[c]]
        baselines[name]=dict(positions=len(t),unigram_mean_nll=float(-np.log(unigram[data[t]]).mean()),unigram_accuracy=float((data[t]==unigram.argmax()).mean()),oracle_covered=len(c),oracle_coverage=float(len(c)/len(t)),oracle_zero_probability_targets=int(zero[c].sum()),oracle_mean_nll='infinity' if zero[c].any() else float(oracle_nll[c].mean()),oracle_accuracy=float(oracle_correct[c].mean()),oracle_positive_support_mean_nll=float(oracle_nll[finite].mean()))
    unique,counts=np.unique(lens[fork],return_counts=True)
    report=dict(corpus_sha256=digest,corpus_bytes=n,retained_artifacts_hash_verified=True,chi_position=int(fork.sum()),chi_position_over_n=float(fork.mean()),chi_is_minimum_suffixient_cardinality=False,fork_context_length_histogram={str(k):int(v) for k,v in zip(unique,counts)},distinct_fork_contexts=len(tops),top_forks_by_branch_entropy=sorted(tops,key=lambda x:-x['entropy_bits'])[:30],forced_byte_distribution={str(k):int(v) for k,v in enumerate(np.bincount(data[~fork],minlength=256)) if v},models=models,baselines=baselines)
    np.savez_compressed(out/'labels.npz',fork=fork,context_length=lens)
    report['labels_sha256']=hashlib.sha256((out/'labels.npz').read_bytes()).hexdigest()
    (out/'decomposition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(density=report['chi_position_over_n'],models=models,baselines=baselines)),flush=True)

if __name__=='__main__':
    main()
