#!/usr/bin/env python3
"""Estimate corpus bytes per tokenizer token from random context windows.

The 1.3B racers report loss in nats/token because they train with a BPE
tokenizer.  For byte-level comparisons and plots, convert by

    bits_per_byte = (nats_per_token / ln(2)) / bytes_per_token

This script samples random token windows from an mmap'd corpus and estimates the
bytes/token conversion factor for the exact tokenizer/context setup.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate bytes/token and nats/token -> bits/byte conversion."
    )
    parser.add_argument(
        "--data",
        default="data/pile.txt",
        help=(
            "Raw text corpus path. The canonical run uses the training corpus "
            "(e.g. ~/elman/data/pile.txt); override to point at a local copy."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default="p50k_base",
        help="tiktoken encoding name, or 'byte' for byte-level identity",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=2048,
        help="Token context length to sample, matching training chunk size",
    )
    parser.add_argument("--samples", type=int, default=2000, help="Random windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bytes-per-token-safety",
        type=int,
        default=6,
        help="Raw bytes to read per requested token; matches TokenizedStreamDataset default",
    )
    parser.add_argument(
        "--drop-first-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop first token after random byte offset, matching TokenizedStreamDataset",
    )
    parser.add_argument(
        "--snap-delimiter",
        action="store_true",
        help="Snap start to after next ASCII 0x1e delimiter within --snap-search-bytes",
    )
    parser.add_argument("--snap-search-bytes", type=int, default=4096)
    parser.add_argument(
        "--nats-per-token",
        type=float,
        nargs="*",
        default=[],
        help="Optional losses to convert to bits/byte using the estimated mean",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output")
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def load_encoder(name: str):
    if name == "byte":
        return None
    import tiktoken

    return tiktoken.get_encoding(name)


def token_bytes(enc, token: int) -> int:
    if enc is None:
        return 1
    return len(enc.decode_single_token_bytes(token))


def sample_token_window(mm: mmap.mmap, enc, args: argparse.Namespace, rng: random.Random):
    if enc is None:
        max_start = len(mm) - args.chunk_tokens
        if max_start <= 0:
            raise ValueError("Corpus is smaller than requested chunk")
        start = rng.randrange(max_start)
        toks = list(mm[start : start + args.chunk_tokens])
        return toks, args.chunk_tokens

    read_bytes = args.chunk_tokens * args.bytes_per_token_safety
    max_start = len(mm) - read_bytes - 1
    if max_start <= 0:
        raise ValueError("Corpus is smaller than requested sample window")

    for _ in range(100):
        start = rng.randrange(max_start)
        if args.snap_delimiter:
            snap = mm.find(b"\x1e", start, min(start + args.snap_search_bytes, len(mm)))
            if snap >= 0:
                start = min(snap + 1, max_start)

        raw = bytes(mm[start : start + read_bytes])
        text = raw.decode("utf-8", errors="replace")
        toks = enc.encode(text, disallowed_special=())
        if args.drop_first_token:
            toks = toks[1:]
        if len(toks) >= args.chunk_tokens:
            toks = toks[: args.chunk_tokens]
            represented_bytes = sum(token_bytes(enc, t) for t in toks)
            return toks, represented_bytes

    raise RuntimeError(
        "Failed to produce enough tokens from sampled byte windows; "
        "increase --bytes-per-token-safety"
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    enc = load_encoder(args.tokenizer)
    rng = random.Random(args.seed)

    bytes_per_token = []
    bytes_per_window = []

    with data_path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for _ in range(args.samples):
            toks, represented_bytes = sample_token_window(mm, enc, args, rng)
            if len(toks) != args.chunk_tokens:
                raise AssertionError("internal error: wrong token window length")
            bpt = represented_bytes / args.chunk_tokens
            bytes_per_token.append(bpt)
            bytes_per_window.append(represented_bytes)
        corpus_bytes = len(mm)

    mean_bpt = statistics.fmean(bytes_per_token)
    stdev_bpt = statistics.stdev(bytes_per_token) if len(bytes_per_token) > 1 else 0.0
    sem_bpt = stdev_bpt / math.sqrt(len(bytes_per_token))
    bits_per_byte_per_nat = 1.0 / (math.log(2.0) * mean_bpt)

    result = {
        "data": str(data_path),
        "corpus_bytes": corpus_bytes,
        "tokenizer": args.tokenizer,
        "chunk_tokens": args.chunk_tokens,
        "samples": args.samples,
        "seed": args.seed,
        "drop_first_token": args.drop_first_token,
        "snap_delimiter": args.snap_delimiter,
        "mean_bytes_per_token": mean_bpt,
        "stdev_bytes_per_token": stdev_bpt,
        "sem_bytes_per_token": sem_bpt,
        "median_bytes_per_token": percentile(bytes_per_token, 50),
        "p05_bytes_per_token": percentile(bytes_per_token, 5),
        "p95_bytes_per_token": percentile(bytes_per_token, 95),
        "mean_bytes_per_window": statistics.fmean(bytes_per_window),
        "bits_per_byte_per_nat_per_token": bits_per_byte_per_nat,
        "conversions": [
            {
                "nats_per_token": loss,
                "bits_per_token": loss / math.log(2.0),
                "bits_per_byte": loss * bits_per_byte_per_nat,
            }
            for loss in args.nats_per_token
        ],
    }

    print(f"data: {result['data']}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"chunk_tokens: {args.chunk_tokens}")
    print(f"samples: {args.samples}")
    print(f"mean bytes/token: {mean_bpt:.6f}")
    print(f"std bytes/token:  {stdev_bpt:.6f}")
    print(f"sem bytes/token:  {sem_bpt:.6f}")
    print(
        "p05/median/p95:   "
        f"{result['p05_bytes_per_token']:.6f} / "
        f"{result['median_bytes_per_token']:.6f} / "
        f"{result['p95_bytes_per_token']:.6f}"
    )
    print(f"mean bytes/2k-token window: {result['mean_bytes_per_window']:.1f}")
    print()
    print("conversion:")
    print("  bits_per_byte = nats_per_token / ln(2) / bytes_per_token")
    print(f"  bits_per_byte = nats_per_token * {bits_per_byte_per_nat:.6f}")
    for conv in result["conversions"]:
        print(
            "  "
            f"{conv['nats_per_token']:.6f} nats/token -> "
            f"{conv['bits_per_token']:.6f} bits/token -> "
            f"{conv['bits_per_byte']:.6f} bits/byte"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
