"""Bounded, document-deduplicated GPT-2 token packing and lazy matched sampler."""
import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import time
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(records, tokenizer, output, max_bytes, provenance, valid_fraction=.01):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not 0 < valid_fraction < 1 or max_bytes < 4096:
        raise ValueError('invalid split/budget')
    if shutil.disk_usage(output).free < max_bytes + (10 << 30):
        raise RuntimeError('disk guard: reserve 10 GiB beyond requested output')
    db = sqlite3.connect(output / 'dedup.sqlite')
    db.execute('create table seen (hash blob primary key)')
    counts = dict(train=0, valid=0)
    caps = dict(valid=int(max_bytes*valid_fraction)//2, train=int(max_bytes*(1-valid_fraction))//2)
    files = {k: open(output / (k+'.bin'), 'wb') for k in counts}
    duplicates = documents = scanned = 0
    split_documents = dict(train=0, valid=0)
    boundaries = dict(train=None, valid=None)
    started = time.time()
    def progress(status):
        payload = dict(status=status, pid=__import__('os').getpid(), started=started,
                       updated=time.time(), source_documents_scanned=scanned,
                       documents=documents, tokens=counts, target_tokens=caps,
                       split_documents=split_documents, boundaries=boundaries)
        temporary = output/'progress.tmp'
        temporary.write_text(json.dumps(payload, indent=2)+'\n')
        temporary.replace(output/'progress.json')
    progress('running')
    index = open(output/'documents.jsonl', 'w')
    try:
        for source_index, record in enumerate(records):
            scanned = source_index + 1
            # Bound pathological documents before tokenization or SQLite growth.
            text = record['text'][:1_000_000]
            key = hashlib.sha256(text.encode()).digest()
            if db.execute('insert or ignore into seen values (?)', (key,)).rowcount == 0:
                duplicates += 1
                continue
            split = 'valid' if int.from_bytes(key[:8], 'big') / 2**64 < valid_fraction else 'train'
            if counts[split] >= caps[split]:
                continue
            ids = tokenizer.encode(text).ids + [50256]
            original_tokens = len(ids)
            ids = ids[:caps[split]-counts[split]]
            if ids and max(ids) >= 50257:
                raise ValueError('not GPT-2 vocabulary')
            np.asarray(ids, dtype='<u2').tofile(files[split])
            index.write(json.dumps(dict(source_index=source_index, sha256=key.hex(),
                             split=split, token_start=counts[split], tokens=len(ids),
                             truncated=len(ids)<original_tokens))+'\n')
            if boundaries[split] is None:
                boundaries[split] = dict(first_source_index=source_index)
            boundaries[split]['last_source_index'] = source_index
            split_documents[split] += 1
            counts[split] += len(ids)
            documents += 1
            if documents % 1000 == 0:
                db.commit()
                index.flush()
                progress('running')
                if db.execute('pragma page_count').fetchone()[0]*4096 > (2 << 30):
                    raise RuntimeError('dedup metadata exceeds 2 GiB guard')
                if shutil.disk_usage(output).free < (10 << 30):
                    raise RuntimeError('disk guard: free space below 10 GiB')
                if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > (90 << 30):
                    raise RuntimeError('package build exceeds 90 GiB guard')
            if all(counts[k] >= caps[k] for k in caps):
                break
    except BaseException:
        progress('failed')
        raise
    finally:
        index.close()
        for f in files.values():
            f.close()
        db.commit()
        db.close()
    tokenizer.save(str(output / 'tokenizer.json'))
    receipt = dict(provenance=provenance, tokenizer='openai-community/gpt2',
                   tokenizer_sha256=digest(output/'tokenizer.json'), eod=50256,
                   dtype='<u2', tokens=counts, documents=documents, duplicates=duplicates,
                   dedup='exact SHA256 after 1M-character cap; document-hash split before packing; no fuzzy dedup',
                   split_sha256={k:digest(output/(k+'.bin')) for k in counts},
                   complete=all(counts[k] >= caps[k] for k in caps), max_bytes=max_bytes)
    receipt.update(source_documents_scanned=scanned, split_documents=split_documents,
                   source_interval=dict(first_source_index=0, last_source_index=scanned-1),
                   split_boundaries=boundaries,
                   stored_order='source document order; no shuffle before packing; exact duplicates removed; hash split; 1M-character cap; final split document may be truncated',
                   document_index_sha256=digest(output/'documents.jsonl'))
    (output/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    progress('complete' if receipt['complete'] else 'incomplete')
    return receipt


class Stream:
    """Affine permutation of packed windows, independent of arm and batch size."""
    def __init__(self, path, context, seed=3401):
        self.data = np.memmap(path, dtype='<u2', mode='r')
        self.context = context
        self.n = (len(self.data)-1)//context
        if self.n < 1:
            raise ValueError('split too small')
        self.a = seed*2+1
        while math.gcd(self.a, self.n) != 1:
            self.a += 2
        self.b = seed % self.n

    def window(self, index):
        start = ((self.a*(index % self.n)+self.b) % self.n)*self.context
        return np.array(self.data[start:start+self.context+1], dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--gib', type=float, default=24)
    p.add_argument('--local-jsonl')
    a = p.parse_args()
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tokenizer_revision = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
    tok = Tokenizer.from_file(hf_hub_download('gpt2','tokenizer.json',revision=tokenizer_revision))
    if a.local_jsonl:
        records = (json.loads(line) for line in open(a.local_jsonl))
        provenance = dict(source=a.local_jsonl, sha256=digest(a.local_jsonl), substitution=True)
    else:
        from datasets import load_dataset
        revision = '3be90335b66f24456a5d6659d9c8d208c0357119'
        records = load_dataset('monology/pile-uncopyrighted', revision=revision, split='train', streaming=True)
        provenance = dict(source='monology/pile-uncopyrighted', revision=revision, split='train', streaming=True,
                          substitution='uncopyrighted Pile subset; not full original Pile')
    provenance['tokenizer_revision'] = tokenizer_revision
    print(json.dumps(build(records, tok, a.output, int(a.gib*(1 << 30)), provenance), indent=2))

if __name__ == '__main__':
    main()
