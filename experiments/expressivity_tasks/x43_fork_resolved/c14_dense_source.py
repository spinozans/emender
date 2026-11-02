"""Fail-closed full-source materialization for the registered C14 dense arm."""
import json
import hashlib
import time
from pathlib import Path
from .c14_documents import raw_documents, sha
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT/'receipts/c14_dense'

def main():
    OUT.mkdir(exist_ok=True)
    binding = json.loads((ROOT/'receipts/c14_byte_source.json').read_text())
    source = Path(binding['raw_source'])
    index = REPO/binding['documents_index']
    start = time.monotonic()
    assert sha(index) == binding['documents_index_sha256']
    assert sha(source) == binding['raw_source_sha256']
    dest = Path('/mnt/nvme2n1/erikg/c14_dense_bytes')
    dest.mkdir(exist_ok=True)
    matched = scanned = size = excluded = 0
    with index.open() as receipts, (dest/'bytes.partial').open('wb') as corpus, (dest/'offsets.partial.jsonl').open('w') as offsets:
        wanted = json.loads(next(receipts))
        for raw in raw_documents(source):
            scanned += 1
            if hashlib.sha256(raw).hexdigest() != wanted['sha256']:
                continue
            matched += 1
            if wanted['truncated']:
                excluded += 1
            else:
                offsets.write(json.dumps(dict(**wanted, byte_offset=size, byte_length=len(raw)))+'\n')
                corpus.write(raw)
                size += len(raw)
            if matched % 10000 == 0:
                print(json.dumps(dict(matched=matched, scanned=scanned, bytes=size)), flush=True)
            line = next(receipts, None)
            if line is None:
                wanted = None
                break
            wanted = json.loads(line)
    result = dict(status='passed' if wanted is None else 'failed', matched_documents=matched,
                  scanned_raw_documents=scanned, materialized_bytes=size, excluded_token_truncated=excluded,
                  next_missing_receipt=wanted, source_binding=binding,
                  elapsed_seconds=time.monotonic()-start, training_started=False,
                  partial_corpus=str(dest/'bytes.partial'),
                  reason='complete source matched' if wanted is None else 'bound raw shard exhausted before all selected documents matched; partial corpus forbidden for training')
    if wanted is None:
        (dest/'bytes.partial').rename(dest/'bytes.bin')
        (dest/'offsets.partial.jsonl').rename(dest/'offsets.jsonl')
        result.update(corpus_sha256=sha(dest/'bytes.bin'), offsets_sha256=sha(dest/'offsets.jsonl'))
    (OUT/'source_preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return 0 if wanted is None else 1
if __name__ == '__main__':
    raise SystemExit(main())
