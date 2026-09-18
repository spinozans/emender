#!/usr/bin/env python3
"""Tokenizer round-trip proof for the E97 llama.cpp port (Stage P1, CPU-only).

Establishes, byte-exactly, that a pure-python BPE built from the shared codec
(scripts/e97_p50k_codec.py: tiktoken-derived p50k_base vocab of 50281 ids
incl. the 24 codex whitespace-run extras, GPT-2 base merges + simulation-
reconstructed extra merges, and the r50k pre-tokenization regex) produces
IDENTICAL token id sequences to tiktoken p50k_base over:

  1. code-shaped corpus strings,
  2. edge cases: emoji, CJK, tabs, all 256 raw byte values, long whitespace
     runs (the codex extras' home turf), contractions, ZWJ sequences,
  3. 1000 random UTF-8 strings.

It also round-trips decode: decode(encode(x)) == x for every input.

This is the evidence that the GGUF tokenizer payload (tokens + merges) is
faithful, and that llama.cpp's gpt2 BPE path can be driven with the p50k
rank set (p50k_base's pre-tokenizer regex IS the r50k pattern — verified
against tiktoken's own pat_str).

Usage: .venv/bin/python scripts/prove_e97_tokenizer_roundtrip.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import regex as re
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e97_p50k_codec import R50K_PAT, VOCAB, EOT_ID, tokens_and_merges, vocab_bytes


def build_bpe(enc):
    """Return (encode_fn, decode_fn) driving a pure-python BPE over the
    shared codec's vocab/merges."""
    table = None
    from e97_p50k_codec import gpt2_bytes_to_unicode
    table = gpt2_bytes_to_unicode()
    vb = vocab_bytes(enc)
    tokens, merges, _ = tokens_and_merges(enc)
    enc_to_id = {tokens[i]: i for i in range(VOCAB) if i != EOT_ID}

    merge_ranks: dict[tuple[str, str], int] = {}
    for j, line in enumerate(merges):
        a, b = line.split(" ")
        merge_ranks[(a, b)] = 256 + j

    pat = re.compile(R50K_PAT)

    def bpe(word: str) -> list[str]:
        parts = list(word)
        while len(parts) > 1:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = merge_ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return parts

    def encode(text: str) -> list[int]:
        ids = []
        for chunk in pat.findall(text):
            # map raw bytes through the GPT-2 byte encoder BEFORE merge matching
            word = "".join(table[b] for b in chunk.encode("utf-8"))
            for piece in bpe(word):
                ids.append(enc_to_id[piece])
        return ids

    def decode(ids: list[int]) -> bytes:
        out = bytearray()
        for i in ids:
            if i == EOT_ID:
                continue  # specials carry no bytes
            out.extend(vb[i])
        return bytes(out)

    return encode, decode

def random_utf8_strings(n: int, rng: random.Random) -> list[str]:
    out = []
    for _ in range(n):
        length = rng.randint(1, 120)
        chars = []
        for _ in range(length):
            kind = rng.random()
            if kind < 0.35:
                chars.append(chr(rng.randint(0x20, 0x7E)))
            elif kind < 0.55:
                chars.append(chr(rng.randint(0x4E00, 0x9FFF)))  # CJK
            elif kind < 0.7:
                chars.append(chr(rng.randint(0x1F300, 0x1F9FF)))  # emoji
            elif kind < 0.85:
                chars.append(rng.choice(["\t", "\n", " ", "  ", "   ", "'", "’"]))
            else:
                chars.append(chr(rng.randint(0xA0, 0x2FF)))
        out.append("".join(chars))
    return out


CODE_CORPUS = [
    "def merge_e97_task_vector(src: Path, dst: Path) -> None:\n    with open(src) as f:\n        data = json.load(f)\n    return {k: v for k, v in data.items() if v}",
    "class E97SplitEditLayer(nn.Module):\n    def forward(self, x, state=None):\n        qkv = self.qkv_proj(x)\n        return state, qkv",
    "for i in range(18):\n    h = h + mixer(rms_norm(h))\n    h = h + mlp(rms_norm_2(h))\nlogits = lm_head(rms_norm(h))",
    "SELECT id, sha256 FROM packs WHERE training_eligible = 1 AND optimizer_updates_authorized = 32;",
    "$ git rev-parse HEAD && find . -name '*.py' | xargs sha256sum > source.sha256",
    "curl -s https://huggingface.co/spinozans/emender-e97-4b-pi-instruction-checkpoints/raw/main/LATEST.json",
    "    indented\ttabs\t\tand   spaced    text\r\nwith CRLF endings\r\n",
    "{\"active_path\": \"/mnt/nvme2n1/erikg/e97_systematic_posttraining\", \"nonce\": 16049, \"key\": 743062}",
]

EDGE_CASES = [
    ("eot-in-text", "uses the \u2400\u240a stand-in pair for the special token id 50256 inline"),
    ("contractions", "I'm you're they've we'll it's don't can't he'd she'd"),
    ("tabs-runs", "a\t\t\tb\n\n\nc                    d                         e"),
    ("emoji-zwj", "family: 👨\u200d👩\u200d👧\u200d👦 flag: 🏴\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F"),
    ("cjk", "状态空间模型与序列递归结构的训练与推理 4B参数"),
    ("mixed-script", "Ω-loop über naïve café —北京的冬天格外冷"),
    ("all-256-bytes", None),  # filled below (raw bytes via latin-1 decode)
    ("long-space-run", "x" + " " * 25 + "y" + " " * 40 + "z"),
    ("long-tab-run", "a" + "\t" * 30 + "b" + "\t" * 7),
    ("newline-flood", "\n" * 100),
    ("zero-width", "invis​ible‌chars‍here"),
    ("numerics", "3.14159265358979 0xdeadbeef 1e-5 4,045,972,080"),
]


def main() -> int:
    enc = tiktoken.get_encoding("p50k_base")
    encode, decode = build_bpe(enc)
    vb = vocab_bytes(enc)
    from e97_p50k_codec import tokens_and_merges
    _, merges, _ = tokens_and_merges(enc)
    print(f"vocab: {VOCAB} ids | merges: {len(merges)} | pattern: {R50K_PAT}")

    # sanity: the reconstructed extras are exactly the long whitespace runs
    extras = {i: vb[i] for i in range(50257, VOCAB)}
    print("codex extras (ids 50257..50280):")
    for i, bs in extras.items():
        assert bs == bs[:1] * len(bs) and bs[0:1] in (b" ", b"\t", b"\n"), f"unexpected extra {i}: {bs!r}"
        print(f"  {i}: {bs[:1]!r} x{len(bs)}")

    cases: list[tuple[str, str]] = [(f"code-{i}", s) for i, s in enumerate(CODE_CORPUS)]
    for name, s in EDGE_CASES:
        if s is None:
            s = bytes(range(256)).decode("latin-1")
        cases.append((name, s))
    rng = random.Random(20260821)
    cases += [(f"rand-{i}", s) for i, s in enumerate(random_utf8_strings(1000, rng))]

    enc_mismatch, dec_mismatch, empty_fail = 0, 0, 0
    total_ids = 0
    for name, text in cases:
        ref = enc.encode(text, allowed_special=set())
        got = encode(text)
        total_ids += len(ref)
        if ref != got:
            enc_mismatch += 1
            first = next((j for j, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
            print(f"ENCODE MISMATCH [{name}] at {first}: ref={ref[first - 2:first + 3]} got={got[first - 2:first + 3]}")
            continue
        if decode(got) != text.encode("utf-8"):
            dec_mismatch += 1
            print(f"DECODE MISMATCH [{name}]")
        if not got:
            empty_fail += 1
    n = len(cases)
    print(f"encode:  {n - enc_mismatch}/{n} inputs byte-exact vs tiktoken ({total_ids} tokens)")
    print(f"decode:  {n - dec_mismatch}/{n} round-trips exactly")
    print(f"empty:   {empty_fail} inputs produced no tokens")
    verdict = "PASS" if (enc_mismatch == 0 and dec_mismatch == 0 and empty_fail == 0) else "FAIL"
    print(f"TOKENIZER_ROUNDTRIP_{verdict}")
    print(
        "llama.cpp wiring note: p50k_base uses the r50k (GPT-2) pre-tokenizer regex;"
        " the GGUF gpt2 tokenizer path needs NO regex override, only the p50k rank"
        " set (vocab 50281, merges 50024, eot 50256)."
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
