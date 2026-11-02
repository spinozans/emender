"""Inspect low-complexity overlap and locate all sampled long supported contexts."""
import json
from pathlib import Path
import c14_coverage as bridge

out=bridge.OUT
samples=json.loads((out/'samples.json').read_text())
strata={}; patterns=[]; refs=[]
for k in [16,64,256,1024]:
    counts=[int(r.rsplit(b'\t',1)[1]) for r in (out/f'support_{k}.stdout').read_bytes().splitlines()]
    nonwhite=[bool(bytes.fromhex(s['left'])[-k:].strip()) for s in samples]
    strata[str(k)]=dict(supported=sum(c>0 for c in counts),nonwhitespace_supported=sum(c>0 and w for c,w in zip(counts,nonwhite)),nonwhitespace_denominator=sum(nonwhite),whitespace_supported=sum(c>0 and not w for c,w in zip(counts,nonwhite)))
    if k>=256:
        for i,c in enumerate(counts):
            if c:
                patterns.append(bytes.fromhex(samples[i]['left'])[-k:])
                refs.append(dict(sample=i,length=k,count=c,nonwhitespace=nonwhite[i],distinct_bytes=len(set(patterns[-1]))))
exact=[int(r.rsplit(b'\t',1)[1]) for r in (out/'verbatim_count.stdout').read_bytes().splitlines()]
spans=(out/'verbatim_count.patterns').read_bytes().split(b'\x1e')[:-1]
strata['centered_256']=dict(denominator=len(exact),supported=sum(c>0 for c in exact),nonwhitespace_supported=sum(c>0 and bool(p.strip()) for c,p in zip(exact,spans)),whitespace_supported=sum(c>0 and not p.strip() for c,p in zip(exact,spans)))
(out/'content_strata.json').write_text(json.dumps(strata,indent=2))
(out/'long_context_refs.json').write_text(json.dumps(refs,indent=2))
bridge.timings=json.loads((out/'timings.json').read_text())
bridge.query(patterns,'long_context_locate',True)
summary=json.loads((out/'coverage.json').read_text()); summary['timings']=bridge.timings
summary['long_context_located_patterns']=len(refs)
(out/'coverage.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(strata,indent=2))
