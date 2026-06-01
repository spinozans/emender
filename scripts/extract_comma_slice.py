#!/usr/bin/env python3
"""Extract a held-out, document-aligned slice from the comma-pile main-mix.

Source: the Common Pile v0.1 distribution-matched main-mix, a single ~1 TB text
file whose documents are separated by the RECORD SEPARATOR byte 0x1E (RS).

We pick a RANDOM DEEP byte offset (well past the start, where the racer's <1-epoch
stream is least likely to have consumed), advance to the next 0x1E so we begin on a
fresh document boundary, read ~10 MB, then trim back to the last 0x1E so the slice
contains only COMPLETE documents. 0x1E is a single ASCII control byte (< 0x80) and
can never be part of a multibyte UTF-8 sequence, so trimming on it is byte-safe and
the decoded text re-encodes to exactly the raw slice (round-trip asserted).

Writes paper/review/comma_slice.json (provenance) and a cached copy of the exact
bytes so the neural eval and the classical compressors all score byte-identical
input.

NO fabrication, NO mock data: the offset is drawn from os.urandom, the bytes are
read from the real corpus, and every recorded field is computed from those bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

CORPUS = "/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt"
RS = 0x1E  # document delimiter (RECORD SEPARATOR)
READ_BYTES = 10_000_000  # ~10 MB target slice
SEARCH_WINDOW = 8_000_000  # max bytes to scan for the leading doc boundary
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = HERE / ".comma_slice.txt"
OUT_JSON = REPO / "paper" / "review" / "comma_slice.json"


def pick_deep_offset(total: int) -> int:
    """Random offset in the deep [0.50, 0.95) region of the corpus."""
    r = struct.unpack("<Q", os.urandom(8))[0] / 2**64  # uniform [0,1)
    frac = 0.50 + r * 0.45
    return int(frac * total)


def main() -> int:
    total = os.path.getsize(CORPUS)
    requested_offset = pick_deep_offset(total)

    with open(CORPUS, "rb") as f:
        # advance to the next document boundary (first RS at/after the offset)
        f.seek(requested_offset)
        head = f.read(SEARCH_WINDOW)
        rs = head.find(bytes([RS]))
        if rs < 0:
            raise RuntimeError(
                f"no 0x1E within {SEARCH_WINDOW} bytes of offset {requested_offset}"
            )
        start = requested_offset + rs + 1  # first byte of the next whole document
        f.seek(start)
        raw = f.read(READ_BYTES)

    # trim trailing partial document: keep up to and including the last RS so the
    # slice ends exactly on a document boundary (only complete documents scored)
    last_rs = raw.rfind(bytes([RS]))
    if last_rs < 0:
        raise RuntimeError("no 0x1E in the 10 MB window — document larger than slice")
    raw = raw[: last_rs + 1]

    text = raw.decode("utf-8")
    assert text.encode("utf-8") == raw, "utf-8 round-trip mismatch — denominator unsafe"

    sha = hashlib.sha256(raw).hexdigest()
    num_documents = raw.count(bytes([RS]))  # = count of trailing RS delimiters
    CACHE.write_bytes(raw)

    spec = {
        "description": (
            "Held-out, document-aligned comma-pile (Common Pile v0.1 main-mix) byte "
            "slice for tokenizer-invariant BPB; the contamination-free second "
            "distribution. Byte-identical input feeds the neural eval and the "
            "classical compressors. Documents separated by 0x1E (RECORD SEPARATOR)."
        ),
        "source_path": CORPUS,
        "requested_byte_offset": requested_offset,
        "byte_offset": start,
        "byte_length": len(raw),
        "total_bytes": total,
        "sha256": sha,
        "num_documents": num_documents,
        "offset_fraction_of_corpus": round(start / total, 6),
        "delimiter_hex": "0x1E",
        "cached_copy_abspath": str(CACHE),
        "extraction": (
            "random deep offset in [0.50,0.95)*total (os.urandom); advance to next "
            "0x1E (+1) for a fresh document start; read 10_000_000 bytes; trim to last "
            "0x1E so only complete documents remain."
        ),
        "context_for_neural_eval": 2048,
    }
    OUT_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in spec.items() if k != "description"}, indent=2))
    print(f"\nwrote {OUT_JSON}\ncached {CACHE} ({len(raw)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
