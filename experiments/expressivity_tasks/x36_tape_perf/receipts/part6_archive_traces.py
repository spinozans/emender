"""Losslessly compress part-6 Chrome traces and verify receipt attribution."""
import gzip
import hashlib
import json
from pathlib import Path
import runpy
R=Path(__file__).resolve().parent
summarize=runpy.run_path(str(R.parent/'profile_summary.py'))['summarize']
manifest=[]
for source in sorted(R.glob('part6*_trace.json')):
    raw=source.read_bytes()
    target=source.with_suffix(source.suffix+'.gz')
    packed=gzip.compress(raw,mtime=0)
    target.write_bytes(packed)
    assert gzip.decompress(target.read_bytes())==raw
    manifest.append(dict(path=target.name,uncompressed_bytes=len(raw),compressed_bytes=len(packed),
                         uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                         compressed_sha256=hashlib.sha256(packed).hexdigest()))
    source.unlink()
checks=[]
for receipt in sorted(R.glob('part6*.json')):
    if receipt.name.endswith('_trace.json'):continue
    payload=json.loads(receipt.read_text())
    records=payload if isinstance(payload,list) else payload.get('results',[]) if isinstance(payload,dict) else []
    changed=False
    for record in records:
        name=record.get('profile_trace')
        if not name:continue
        trace=R/(name if name.endswith('.gz') else name+'.gz')
        if not trace.exists():continue
        assert summarize(trace)==record['profile'],(receipt.name,trace.name)
        record['profile_trace']=trace.name
        changed=True
        checks.append(dict(receipt=receipt.name,trace=trace.name,summary_equivalence=True))
    if changed:receipt.write_text(json.dumps(payload,indent=2)+'\n')
(R/'part6_trace_archive.json').write_text(json.dumps(dict(traces=manifest,checks=checks),indent=2)+'\n')
print(f'{len(manifest)} traces compressed; {len(checks)} profile summaries verified unchanged')
