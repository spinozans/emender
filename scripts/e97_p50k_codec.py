#!/usr/bin/env python3
"""Shared p50k_base tokenizer codec for the E97 llama.cpp port.

Single source of truth for: the GPT-2 byte<->unicode table, the tiktoken
p50k_base vocab (50281 ids incl. the 24 codex whitespace-run extras), and the
BPE merge list in id order. Used by both the GGUF converter and the
round-trip proof so they cannot drift.

Extras reconstruction (ids 50257..50280, space runs of length 2..25): the
merge pair is NOT the min-lex split — it is the pair greedy BPE produces when
merging the token's own pre-image with all ranks >= the token's id excluded
(the merge that CREATED T in training). Verified byte-exact against tiktoken
by prove_e97_tokenizer_roundtrip.py.
"""

from __future__ import annotations

from pathlib import Path

import tiktoken

VOCAB = 50281
EOT_ID = 50256
N_BASE_MERGES = 50000
N_EXTRAS = 24

# tiktoken p50k_base pattern (= r50k pat str; verified against tiktoken_ext.openai_public)
R50K_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def base_merges() -> list[str]:
    """The 50000 GPT-2 base merges (ids 256..50255), from the canonical
    vocab.bpe (cached at /tmp/gpt2_vocab.bpe, fetched from
    openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe)."""
    path = Path("/tmp/gpt2_vocab.bpe")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#version"), "vocab.bpe header missing"
    merges = [line.strip() for line in lines[1:] if line.strip()]
    assert len(merges) == N_BASE_MERGES, f"expected {N_BASE_MERGES} base merges, got {len(merges)}"
    return merges


def vocab_bytes(enc=None) -> dict[int, bytes]:
    if enc is None:
        enc = tiktoken.get_encoding("p50k_base")
    assert enc.n_vocab == VOCAB
    out = {i: enc.decode_bytes([i]) for i in range(VOCAB)}
    out[EOT_ID] = b"endoftext"
    return out


def tokens_and_merges(enc=None) -> tuple[list[str], list[str], dict[int, bytes]]:
    """(tokens, merges, vocab_bytes) in llama.cpp gpt2 string form.

    tokens[i] = byte-encoder string of id i; tokens[EOT_ID] = 'endoftext'.
    merges[i] corresponds to id 256+i for the base, and the 24 extras occupy
    ids 50257..50280 = merge indices 50001..50024.
    """
    if enc is None:
        enc = tiktoken.get_encoding("p50k_base")
    table = gpt2_bytes_to_unicode()
    vb = vocab_bytes(enc)
    tokens = ["".join(table[b] for b in vb[i]) for i in range(VOCAB)]
    tokens[EOT_ID] = "endoftext"

    enc_to_id = {tokens[i]: i for i in range(VOCAB) if i != EOT_ID}
    merges = base_merges()
    for j, line in enumerate(merges):
        a, b = line.split(" ")
        assert enc_to_id[a] < 256 + j and enc_to_id[b] < 256 + j, f"base merge {j} references future token"

    for tid in range(EOT_ID + 1, VOCAB):
        word = tokens[tid]
        # simulation: greedy-merge the pre-image with ranks >= tid excluded
        parts = list(word)
        reduced = {pair: r for pair, r in _iter_ranks(merges) if r < tid}
        while True:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = reduced.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None or len(parts) == 2:
                break
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        assert len(parts) == 2, f"extra {tid} ({word!r}) did not reduce to a pair: {parts}"
        merges.append(f"{parts[0]} {parts[1]}")
        enc_to_id[word] = tid
    assert len(merges) == N_BASE_MERGES + N_EXTRAS
    return tokens, merges, vb


def _iter_ranks(merges: list[str]):
    for j, line in enumerate(merges):
        a, b = line.split(" ")
        yield (a, b), 256 + j
