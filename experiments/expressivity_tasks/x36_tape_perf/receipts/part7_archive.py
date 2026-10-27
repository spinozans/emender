"""Losslessly archive part-7 traces and verify their summaries."""
import gzip
import hashlib
import json
from pathlib import Path
from experiments.expressivity_tasks.x36_tape_perf.profile_summary import summarize

root=Path(__file__).resolve().parent
receipts=[]
for name in ('part7_profile.json','part7_fresh_t512.json','part7_fresh_t4096.json'):
    path=root/name
    payload=json.loads(path.read_text())
    records=payload['results'] if isinstance(payload,dict) else payload
    for record in records:
        if 'profile_trace' not in record:continue
        source=root/record['profile_trace']
        raw=source.read_bytes()
        target=source.with_suffix(source.suffix+'.gz')
        target.write_bytes(gzip.compress(raw,mtime=0))
        assert gzip.decompress(target.read_bytes())==raw
        assert summarize(target)==record['profile']
        receipts.append(dict(path=target.name,raw_sha256=hashlib.sha256(raw).hexdigest(),
                             gzip_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             raw_bytes=len(raw),compressed_bytes=target.stat().st_size,
                             lossless=True,profile_summary_equal=True))
        record['profile_trace']=target.name
        source.unlink()
    path.write_text(json.dumps(payload,indent=2)+'\n')
(root/'part7_trace_archives.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps(receipts,indent=2))
