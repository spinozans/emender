#!/usr/bin/env python3
"""Tokenizer-invariant BPB on the held-out comma-pile slice — the second
(contamination-free) distribution.

Reuses the EXACT scoring code from `measure_pile_bpb.measure_model` so the comma
panel is produced by the identical pipeline as the Pile panel. Each model uses its
own tokenizer; the denominator is the shared UTF-8 byte count of the comma slice:

    bpb = (sum per-token NLL nats / ln2) / total_utf8_bytes

Per-model context/stride MIRRORS the Pile eval exactly:
  * GPT-NeoX-tokenizer transformers (pythia/gpt-neo): ctx 2048, stride 1024
  * gpt2-xl / opt-1.3b: ctx 1024, stride 512 (gpt2-xl max position = 1024)

GPU 0 ONLY. REAL MEASUREMENT — failures recorded with the exact exception.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from measure_pile_bpb import measure_model, log  # noqa: E402

HERE = Path(__file__).resolve().parent
SLICE = HERE / ".comma_slice.txt"
OUT = HERE / ".comma_bpb_results.json"

# (model_id, context, stride) — mirrors the Pile eval per-model protocol
MODELS = [
    ("EleutherAI/pythia-1.4b", 2048, 1024),
    ("EleutherAI/gpt-neo-1.3B", 2048, 1024),
    ("EleutherAI/pythia-1b", 2048, 1024),
    ("facebook/opt-1.3b", 1024, 512),
    ("gpt2-xl", 1024, 512),
]


def main() -> int:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "must run on GPU 0 only"
    dtype = torch.float16
    raw = SLICE.read_bytes()
    text = raw.decode("utf-8")
    import hashlib
    sha = hashlib.sha256(raw).hexdigest()
    log(f"device: {torch.cuda.get_device_name(0)} | dtype=float16 | "
        f"comma slice {len(raw)} bytes sha={sha[:16]}...")

    results = []
    for mid, ctx, stride in MODELS:
        try:
            log(f"=== measuring {mid} (ctx={ctx} stride={stride}) ===")
            r = measure_model(mid, text, ctx, stride, dtype)
            log(f"{mid}: bpb={r['bpb']:.4f} ppl={r['ppl_token']:.2f} "
                f"tokens={r['tokens_scored']} ({r['seconds']}s)")
            results.append(r)
        except Exception as e:
            import traceback
            err = f"{type(e).__name__}: {e}"
            log(f"FAILED {mid}: {err}")
            results.append({"model_id": mid, "error": err,
                            "traceback": traceback.format_exc()})

    payload = {
        "slice": {"source": "comma-pile main-mix v0.1", "bytes": len(raw),
                  "sha256": sha, "cache": str(SLICE)},
        "config": {"dtype": "float16",
                   "formula": "bpb = (sum NLL nats / ln2) / total_utf8_bytes",
                   "per_model_context": {m: [c, s] for m, c, s in MODELS}},
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
