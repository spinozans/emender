"""Download, tokenize and pack TinyStories without validation leakage."""
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    RUNS.mkdir(exist_ok=True)
    provenance = {'source': 'TinyStories', 'fallback': False, 'train_token_cap':200_000_000, 'files': {}}
    for split in ('train', 'valid'):
        path = RUNS/f'{split}.txt'
        url = f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt'
        if not path.exists():
            tmp = path.with_suffix('.download')
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        provenance['files'][split] = dict(url=url, sha256=sha(path))
    tokpath = RUNS/'tokenizer.json'
    if not tokpath.exists():
        tok = Tokenizer(models.BPE(unk_token='[UNK]'))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=32768, special_tokens=['[PAD]', '[UNK]', '<|endoftext|>'],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train([str(RUNS/'train.txt')], trainer)
        tok.save(str(tokpath))
    tok = Tokenizer.from_file(str(tokpath))
    for split in ('train', 'valid'):
        out = RUNS/f'{split}.bin'
        if not out.exists():
            tmp = out.with_suffix('.packing')
            with open(RUNS/f'{split}.txt') as src, open(tmp, 'wb') as dst:
                # bounded memory, preserve all text and explicit EOD tokens
                lines = []
                written=0
                cap=200_000_000 if split=='train' else 2**63
                for line in src:
                    lines.append(line)
                    if len(lines) == 256:
                        ids=np.asarray(tok.encode(''.join(lines)).ids,dtype=np.uint16)[:cap-written]
                        ids.tofile(dst);written+=len(ids)
                        lines.clear()
                        if written>=cap:
                            break
                if lines:
                    np.asarray(tok.encode(''.join(lines)).ids, dtype=np.uint16)[:cap-written].tofile(dst)
            tmp.rename(out)
        provenance['files'][split]['tokens'] = out.stat().st_size//2
        provenance['files'][split]['packed_sha256'] = sha(out)
    provenance['tokenizer_sha256'] = sha(tokpath)
    provenance['vocab'] = tok.get_vocab_size()
    (ROOT/'receipts').mkdir(exist_ok=True)
    (ROOT/'receipts/data.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


def batch(split, step, batch_size=8, context=512, seed=3401):
    data = np.memmap(RUNS/f'{split}.bin', mode='r', dtype=np.uint16)
    blocks = (len(data)-1)//context
    if blocks < batch_size:
        raise ValueError('corpus too small')
    # Affine permutation with a coprime stride: deterministic, no giant index array.
    stride = 104729
    import math
    while math.gcd(stride, blocks) != 1:
        stride += 2
    indices = ((np.arange(batch_size)+step*batch_size)*stride+seed) % blocks
    x = np.stack([data[i*context:i*context+context+1] for i in indices]).astype(np.int64)
    import torch
    return torch.from_numpy(x[:, :-1].copy()), torch.from_numpy(x[:, 1:].copy())
