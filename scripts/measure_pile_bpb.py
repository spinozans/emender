#!/usr/bin/env python3
"""Measure tokenizer-invariant bits-per-byte (BPB) on a held-out Pile slice.

ONE identical pipeline for every model. Each model uses its OWN tokenizer; the
byte denominator is the shared underlying UTF-8 text, so BPB is comparable
across tokenizers:

    bpb = (sum of per-token NLL in nats / ln 2) / (total UTF-8 bytes of text)

Method (every model, identical):
  * Extract ONE held-out text slice from the Pile corpus, cached to disk, so
    every model sees byte-for-byte the same text (sha256 recorded).
  * Tokenize the whole slice with the model's own tokenizer.
  * Sliding-window NLL at a fixed context (default 2048, matching E88's
    training context). With stride < context, every token is scored exactly
    once with up to (context-1) tokens of real left context (the standard
    HuggingFace fixed-length-model perplexity recipe). Only the single very
    first token of the sequence is unscored.
  * Sum NLL in nats over scored tokens; bpb = (nll/ln2) / bytes.

GPU 0 ONLY. fp16/bf16 inference. REAL MEASUREMENT — no fabricated numbers.
Any model that fails to load/run is reported with its exact exception.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

# Hard-pin to GPU 0 before importing torch so we can never touch the training
# GPUs (1-7), regardless of how this process is launched.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

LN2 = math.log(2.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_slice(corpus: str, offset: int, n_bytes: int, cache: Path) -> dict:
    """Extract a whole-line UTF-8 slice from `corpus` and cache it.

    Starts at the first newline at/after `offset` (clean line boundary) and
    ends at the last newline within the read window, so both edges fall on
    newline (single-byte ASCII) boundaries -> no split multibyte codepoints and
    decoded_text.encode('utf-8') reproduces the raw byte slice exactly.
    """
    if cache.exists():
        raw = cache.read_bytes()
        log(f"slice cache hit: {cache} ({len(raw)} bytes)")
    else:
        with open(corpus, "rb") as f:
            f.seek(offset)
            # advance to next line boundary so we begin on a fresh document line
            head = f.read(1 << 20)
            nl = head.find(b"\n")
            start = offset + (nl + 1 if nl >= 0 else 0)
            f.seek(start)
            raw = f.read(n_bytes)
        # trim trailing partial line so the slice ends on a newline boundary
        last_nl = raw.rfind(b"\n")
        if last_nl >= 0:
            raw = raw[: last_nl + 1]
        cache.write_bytes(raw)
        log(f"slice extracted -> {cache} ({len(raw)} bytes)")

    text = raw.decode("utf-8")
    # round-trip invariant: the byte denominator is exactly these bytes
    enc = text.encode("utf-8")
    assert enc == raw, "utf-8 round-trip mismatch — byte denominator unsafe"
    sha = hashlib.sha256(raw).hexdigest()
    return {
        "text": text,
        "bytes": len(raw),
        "sha256": sha,
        "offset": offset,
        "requested_bytes": n_bytes,
    }


@torch.no_grad()
def measure_model(model_id: str, text: str, context: int, stride: int,
                  dtype: torch.dtype) -> dict:
    """Return measured stats for one model on the shared text, or an error dict."""
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())

    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    seq_len = ids.size(1)

    nll_nats = 0.0
    n_scored = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + context, seq_len)
        trg_len = end - prev_end  # tokens newly scored in this window
        window = ids[:, begin:end]
        target = window.clone()
        target[:, :-trg_len] = -100  # mask the context (already-scored) tokens
        out = model(window, labels=target)
        # HF returns MEAN NLL (nats) over the (trg_len-1 or trg_len) scored
        # positions; recover the SUM. Number of scored positions = count of
        # label entries != -100 after the internal shift.
        shift_labels = target[:, 1:]
        n_pos = int((shift_labels != -100).sum().item())
        if n_pos > 0:
            nll_nats += float(out.loss.item()) * n_pos
            n_scored += n_pos
        prev_end = end
        if end == seq_len:
            break

    dt = time.time() - t0
    del model
    torch.cuda.empty_cache()
    bytes_eval = len(text.encode("utf-8"))
    bpb = (nll_nats / LN2) / bytes_eval
    return {
        "model_id": model_id,
        "params": n_params,
        "params_billions": round(n_params / 1e9, 3),
        "tokens_scored": n_scored,
        "seq_len": seq_len,
        "bytes": bytes_eval,
        "bytes_per_token": round(bytes_eval / seq_len, 3),
        "nll_nats_sum": nll_nats,
        "bpb": bpb,
        "ppl_token": math.exp(nll_nats / n_scored) if n_scored else None,
        "context": context,
        "stride": stride,
        "dtype": str(dtype),
        "seconds": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/mnt/nvme2n1/erikg/pile.txt")
    ap.add_argument("--offset", type=int, default=1_000_000_000_000,
                    help="byte offset into corpus for the held-out slice")
    ap.add_argument("--bytes", type=int, default=10_000_000,
                    help="approx bytes to read for the slice")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--cache", default="scripts/.pile_heldout_slice.txt")
    ap.add_argument("--out", default="scripts/.pile_bpb_results.json")
    ap.add_argument("--models", nargs="+", default=[
        "EleutherAI/pythia-1.4b",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/pythia-1b",
        "facebook/opt-1.3b",
    ])
    args = ap.parse_args()

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "must run on GPU 0 only"
    dtype = getattr(torch, args.dtype)
    log(f"device: {torch.cuda.get_device_name(0)} | dtype={args.dtype} "
        f"| context={args.context} stride={args.stride}")

    sl = extract_slice(args.corpus, args.offset, args.bytes, Path(args.cache))
    log(f"slice: {sl['bytes']} bytes, sha256={sl['sha256'][:16]}..., "
        f"offset={sl['offset']}")

    results = []
    for mid in args.models:
        try:
            log(f"=== measuring {mid} ===")
            r = measure_model(mid, sl["text"], args.context, args.stride, dtype)
            log(f"{mid}: bpb={r['bpb']:.4f} ppl={r['ppl_token']:.2f} "
                f"tokens={r['tokens_scored']} ({r['seconds']}s)")
            results.append(r)
        except Exception as e:  # report exact error, keep going
            import traceback
            err = f"{type(e).__name__}: {e}"
            log(f"FAILED {mid}: {err}")
            results.append({"model_id": mid, "error": err,
                            "traceback": traceback.format_exc()})

    payload = {
        "slice": {k: v for k, v in sl.items() if k != "text"},
        "config": {"context": args.context, "stride": args.stride,
                   "dtype": args.dtype, "corpus": args.corpus,
                   "formula": "bpb = (sum NLL nats / ln2) / total_utf8_bytes"},
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
