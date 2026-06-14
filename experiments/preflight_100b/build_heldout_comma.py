import os, mmap, math, torch, numpy as np, tiktoken
DATA='/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt'
OUT='/home/erikg/ndm/.wg-worktrees/agent-1433/experiments/preflight_100b/heldout_comma_p50k_2048.pt'
TOK='p50k_base'; CHUNK=2048; N_CHUNKS=32; SEED=7777; TAIL=0.90
enc=tiktoken.get_encoding(TOK)
f=open(DATA,'rb'); mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); size=len(mm)
tail_start=int(size*TAIL); need=(CHUNK+1)*8; max_start=size-need-1
rng=np.random.RandomState(SEED); chunks=[]; att=0
while len(chunks)<N_CHUNKS:
    att+=1
    if att>N_CHUNKS*50: raise RuntimeError('too many fails')
    pos=rng.randint(tail_start,max_start); raw=bytes(mm[pos:pos+need])
    try:
        s=raw.decode('utf-8',errors='replace'); toks=enc.encode(s,disallowed_special=())
    except Exception: continue
    if len(toks)<CHUNK+2: continue
    toks=toks[1:CHUNK+2]
    if len(toks)<CHUNK+1: continue
    chunks.append(toks[:CHUNK+1])
t=torch.tensor(chunks,dtype=torch.long); scored=t[:,1:]; tot=scored.numel(); tb=0
for row in scored.tolist():
    tb+=len(enc.decode(row).encode('utf-8'))
bpt=tb/tot
torch.save({'chunks':t,'bytes_per_token':bpt,'tokenizer':TOK,'n_chunks':N_CHUNKS,'chunk':CHUNK},OUT)
print(f'SAVED {OUT} chunks={t.shape} bytes_per_token={bpt:.4f} scored_tokens={tot}')
