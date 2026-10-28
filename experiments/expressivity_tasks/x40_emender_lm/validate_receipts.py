"""Verify lossless profile archives and their derived summaries, without GPU work."""
import gzip
import hashlib
import json
from pathlib import Path
from ..x36_tape_perf.profile_summary import summarize
from .performance import write


def main():
    root=Path(__file__).resolve().parent/'receipts';rows=[]
    for path in sorted(root.glob('part3*/trace.json.gz')):
        compressed=path.read_bytes();raw=gzip.decompress(compressed)
        assert gzip.decompress(gzip.compress(raw))==raw
        stored=json.loads((path.parent/'profile.json').read_text())
        derived=summarize(path)
        for key,value in derived.items():assert value==stored[key],(str(path),key)
        rows.append(dict(path=str(path.relative_to(root)),status='passed',
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            decompressed_sha256=hashlib.sha256(raw).hexdigest(),events=len(json.loads(raw)['traceEvents'])))
    write(root/'part3_integrity.json',dict(status='passed',archives=rows))
    print(f'{len(rows)} lossless profile archives match their raw-derived summaries')

if __name__=='__main__':main()
