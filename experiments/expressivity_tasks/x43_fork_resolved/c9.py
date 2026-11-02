"""Registered C9 direct-SA generation; never invokes xsa."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .run import Control
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from .direct_sa import suffix_array
OUT = ROOT/'receipts/c9'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

class Index:
    def __init__(self, raw):
        self.raw = raw
        self.sa = suffix_array(np.frombuffer(raw, dtype=np.uint8))
    def interval(self, context):
        if not context:
            return 0, len(self.sa)
        n = len(context)
        def bound(upper):
            lo, hi = 0, len(self.sa)
            while lo < hi:
                mid = (lo+hi)//2
                at = int(self.sa[mid])
                value = self.raw[at:at+n]
                if value < context or (upper and value == context):
                    lo = mid+1
                else:
                    hi = mid
            return lo
        return bound(False), bound(True)
    def count(self, context):
        lo, hi = self.interval(context)
        ends = self.sa[lo:hi]+len(context)
        return ends[ends < len(self.raw)]
    def distribution(self, history):
        context = bytes(history[-128:])
        full = self.count(context)
        lo, hi = 0, len(context)
        while lo < hi:
            mid = (lo+hi+1)//2
            if len(self.count(context[-mid:])) >= 2:
                lo = mid
            else:
                hi = mid-1
        ends = self.count(context[-lo:] if lo else b'')
        counts = np.bincount(np.frombuffer(self.raw, dtype=np.uint8)[ends], minlength=256)
        return counts/counts.sum(), lo, len(full), len(context)

def tests():
    checks = 0
    for raw in (b'abac', b'aaaa', b'banana', bytes([0,255,30,0,255])):
        idx = Index(raw)
        for context in (b'', b'x') + tuple(raw[a:b] for a in range(len(raw)) for b in range(a+1,len(raw)+1)):
            actual = sorted(idx.count(context).tolist())
            expected = [i for i in range(len(context),len(raw)) if raw[i-len(context):i] == context]
            assert actual == expected, (raw, context, actual, expected)
            checks += 1
        for i in range(1, len(raw)+1):
            q,l,_,_ = idx.distribution(raw[:i])
            assert np.isclose(q.sum(), 1) and (q >= 0).all()
            if l:
                assert len(idx.count(raw[:i][-l:])) >= 2
    return dict(status='passed', exact_interval_checks=checks, includes=['finite ends','overlap','empty support','NUL/255/30','normalized repeated-suffix distributions'])

def runs(flags):
    lengths=[]; current=0
    for x in flags:
        if x: current+=1
        elif current: lengths.append(current); current=0
    if current: lengths.append(current)
    return lengths

def novelty(samples, reference):
    rows=[]
    for n in range(4,33):
        grams=[{s[i:i+n] for i in range(len(s)-n+1)} for s in samples]
        wanted=set().union(*grams)
        found=set()
        for i in range(len(reference)-n+1):
            g=reference[i:i+n]
            if g in wanted: found.add(g)
        per=[]
        for s in samples:
            total=max(0,len(s)-n+1)
            overlap=sum(s[i:i+n] in found for i in range(total))
            per.append(dict(total=total, overlapping=overlap, novel_rate=1-overlap/total))
        total=sum(x['total'] for x in per); overlapping=sum(x['overlapping'] for x in per)
        rows.append(dict(n=n,total=total,overlapping=overlapping,novel_rate=1-overlapping/total,per_prompt=per))
    return rows

def longest_copy(sample, index):
    longest=0
    for start in range(len(sample)):
        lo, hi = 0, len(sample)-start
        while lo < hi:
            mid=(lo+hi+1)//2
            a,b=index.interval(sample[start:start+mid])
            if b>a: lo=mid
            else: hi=mid-1
        longest=max(longest,lo)
    return longest

def main():
    OUT.mkdir(exist_ok=True)
    write('tests.json',tests())
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/'receipts/artifact_manifest.json').read_text())
    assert sha(ROOT/'transformer.pt') == manifest['transformer.pt']['sha256']
    raw=(ROOT/'input.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    train=raw[:int(.9*len(raw))]
    calibration=json.loads((ROOT/'receipts/part2/fusion.json').read_text())['models']['transformer']['scopes']['train']
    model=Control(2000000).cuda()
    model.load_state_dict(torch.load(ROOT/'transformer.pt',map_location='cuda',weights_only=True))
    model.eval()
    index=Index(train)
    prompts=[raw[offset:offset+64] for offset in (0,10000,20000,30000)]
    write('binding.json',dict(protocol_sha256=sha(ROOT/'PREREGISTRATION_C9.md'),code_sha256=sha(__file__),
        checkpoint_sha256=sha(ROOT/'transformer.pt'), corpus_sha256=sha(ROOT/'input.txt'),
        train_bytes=len(train),train_sha256=hashlib.sha256(train).hexdigest(),
        fusion_sha256=sha(ROOT/'receipts/part2/fusion.json'),calibration=calibration,
        prompts_hex=[p.hex() for p in prompts],physical_gpu=os.environ['GPU_LEASE_HELD'],
        precision='float32',independent_judge=False))
    samples=[]; results=[]; start=time.monotonic()
    for mode in ('neural_only','walk_fork'):
        for j,prompt in enumerate(prompts):
            rng=np.random.default_rng(4309+j)
            history=bytearray(prompt); steps=[]
            with torch.no_grad():
                for step in range(1024):
                    q,length,full,total_length=index.distribution(history)
                    logits=model(torch.tensor(list(history[-128:]),device='cuda')[None])[0,-1]
                    logp=logits.log_softmax(-1).double().cpu().numpy(); p=np.exp(logp)
                    support=int(np.count_nonzero(q)); forced=mode=='walk_fork' and support==1
                    g=0.
                    if mode=='walk_fork' and support>1:
                        feature=np.array([p.max(),np.log1p(length),np.log1p(support)])
                        feature=(feature-np.array(calibration['feature_center']))/np.array(calibration['feature_scale'])
                        z=np.r_[1.,feature]@np.array(calibration['learned_coefficients'])
                        g=float(1/(1+np.exp(-np.clip(z,-35,35))))
                    distribution=q if forced else (1-g)*p+g*q
                    distribution=distribution/distribution.sum()
                    byte=int(rng.choice(256,p=distribution))
                    steps.append(dict(step=step,byte=byte,nll=float(-logp[byte]),support=support,
                        active_length=length,full_context_length=total_length,full_occurrences=full,
                        forced=forced,index_weight=g,departure=bool(q[byte]==0),
                        empty_full_context=full==0,backoff=length<total_length))
                    history.append(byte)
            generated=bytes(history[len(prompt):]); empty=[s['empty_full_context'] for s in steps]
            reentries=sum(empty[k-1] and not empty[k] for k in range(1,len(empty)))
            row=dict(mode=mode,prompt=j,bytes=len(generated),forced_steps=sum(s['forced'] for s in steps),
                fork_steps=sum(s['support']>1 for s in steps),departures=sum(s['departure'] for s in steps),
                full_context_empty_steps=sum(empty),full_context_empty_runs=runs(empty),
                full_context_reentries=reentries,terminal_excursion=bool(empty[-1]),
                backoff_steps=sum(s['backoff'] for s in steps),
                suffix_backoff_successes=sum(s['empty_full_context'] and s['support']>0 for s in steps),
                active_suffix_length_histogram={str(k):sum(s['active_length']==k for s in steps) for k in sorted({s['active_length'] for s in steps})},
                self_nll=float(np.mean([s['nll'] for s in steps])),
                longest_train_copy=longest_copy(generated,index),sha256=hashlib.sha256(generated).hexdigest())
            results.append(row); samples.append(dict(mode=mode,prompt=j,hex=generated.hex(),text=generated.decode('utf8',errors='replace')))
            write(f'{mode}_{j}_steps.json',steps)
            print(json.dumps(row),flush=True)
    write('samples.json',samples)
    novelty_rows={}
    for mode in ('neural_only','walk_fork'):
        ss=[bytes.fromhex(s['hex']) for s in samples if s['mode']==mode]
        novelty_rows[mode]=dict(train=novelty(ss,train),whole_input=novelty(ss,raw))
    write('results.json',dict(status='measured',results=results,novelty=novelty_rows,
        samples_sha256=sha(OUT/'samples.json'),elapsed_seconds=time.monotonic()-start,
        limitations=['one retained dense head and four prompts','self NLL is not independent quality',
                      'forced singleton walks have no neural escape','C3 calibration geometry transferred to causal repeated suffixes']))

if __name__ == '__main__':
    main()
