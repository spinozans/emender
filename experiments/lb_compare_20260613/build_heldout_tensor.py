#!/usr/bin/env python3
"""Build a FIXED, disjoint held-out token tensor from the Pile tail for lb-compare.

ONE byte-for-byte identical held-out slice scored by every CMA-best model, so
held-out BPB/CE is apples-to-apples. Slice is taken from the FAR TAIL of
pile.txt (offsets >= TAIL_START), disjoint from the seed42 training stream's
typical sampling at the 15-min budget (a few hundred MB out of 1.3 TB). p50k_base
tokenizer (the search tokenizer); records exact UTF-8 bytes of the scored tokens
so BPB = (CE_nats/ln2)/bytes_per_token is exact and tokenizer-comparable.

REAL data — extracted directly from the real Pile corpus. No fabrication.
"""
import os, sys, mmap, json, math
import numpy as np
import torch
import tiktoken

DATA = '/home/erikg/elman/data/pile.txt'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heldout_p50k_2048.pt')
TOK = 'p50k_base'
CHUNK = 2048            # context; we store CHUNK+1 for the target shift
N_CHUNKS = 64           # held-out chunks (64 * 2048 = 131072 scored tokens)
SEED = 7777
TAIL_FRACTION = 0.90    # sample only from the last 10% of the file (disjoint tail)

enc = tiktoken.get_encoding(TOK)
f = open(DATA, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
size = len(mm)
tail_start = int(size * TAIL_FRACTION)
rng = np.random.RandomState(SEED)
need_bytes = (CHUNK + 1) * 8   # safety bytes/token
max_start = size - need_bytes - 1

chunks = []
attempts = 0
while len(chunks) < N_CHUNKS:
    attempts += 1
    if attempts > N_CHUNKS * 50:
        raise RuntimeError('too many failed extraction attempts')
    pos = rng.randint(tail_start, max_start)
    raw = bytes(mm[pos:pos + need_bytes])
    try:
        s = raw.decode('utf-8', errors='replace')
        toks = enc.encode(s, disallowed_special=())
    except Exception:
        continue
    if len(toks) < CHUNK + 2:
        continue
    toks = toks[1:CHUNK + 2]          # drop first (mis-aligned), keep CHUNK+1
    if len(toks) < CHUNK + 1:
        continue
    chunks.append(toks[:CHUNK + 1])

t = torch.tensor(chunks, dtype=torch.long)   # [N_CHUNKS, CHUNK+1]
# Exact UTF-8 bytes of the SCORED tokens (positions 1..CHUNK of each chunk, the
# targets), matching the CE denominator (we score CHUNK tokens per chunk).
scored = t[:, 1:]                              # [N, CHUNK]
total_scored = scored.numel()
total_bytes = 0
for row in scored.tolist():
    total_bytes += len(enc.decode(row).encode('utf-8'))
bpt = total_bytes / total_scored
payload = {
    'chunks': t, 'chunk_size': CHUNK, 'n_chunks': N_CHUNKS,
    'tokenizer': TOK, 'seed': SEED, 'tail_fraction': TAIL_FRACTION,
    'scored_tokens': total_scored, 'total_utf8_bytes': total_bytes,
    'bytes_per_token': bpt, 'data': DATA, 'file_size': size,
    'tail_start_byte': tail_start,
}
torch.save(payload, OUT)
print(json.dumps({k: v for k, v in payload.items() if k != 'chunks'}, indent=2))
print('SAVED', OUT, 'shape', tuple(t.shape))
