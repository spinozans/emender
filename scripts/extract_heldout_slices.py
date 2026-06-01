#!/usr/bin/env python3
"""Extract K independent held-out Pile byte slices for the multi-slice
robustness check. Each slice mimics the EXACT extraction recipe of the
canonical slice (paper/review/heldout_slice.json):

    seek(offset); advance to next newline; read 10_000_000 bytes;
    trim to last newline.

The canonical slice (offset 1e12, sha 3e4241a9...) is reused verbatim so it
ties back to the BPB table. The other slices sit at deep, well-separated
offsets spread across the 1.31 TB file (non-overlapping; 10 MB << gaps).

A slice is only accepted if it decodes as valid UTF-8 with an exact byte
round-trip (the elman BPB harness asserts this). If a candidate offset yields
invalid UTF-8, we advance by a fixed probe step and retry.

REAL DATA ONLY. No fabrication.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PILE = "/mnt/nvme2n1/erikg/pile.txt"
READ_LEN = 10_000_000
OUT_DIR = Path("/tmp/heldout_slices")
MANIFEST = Path("/home/erikg/ndm/.wg-worktrees/agent-758/paper/review/heldout_multislice_slices.json")

# Canonical slice: reuse the byte-identical cached copy so its sha matches the table.
CANON = {
    "name": "canonical_1e12",
    "requested_offset": 1_000_000_000_000,
    "cached": "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt",
    "expect_sha": "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a",
}

# Four additional deep, well-separated offsets (fractions of the file size),
# none at the beginning, all non-overlapping with each other and the canonical.
EXTRA_FRACTIONS = [0.137, 0.341, 0.523, 0.911]
PROBE_STEP = 5_000_000  # advance on invalid-utf8 retry


def extract_at(f, requested_offset, total_size):
    """Replicate the canonical recipe, retrying forward until valid UTF-8."""
    offset = requested_offset
    for _ in range(50):
        f.seek(offset)
        # advance to next newline
        f.readline()
        start = f.tell()
        raw = f.read(READ_LEN)
        # trim to last newline
        nl = raw.rfind(b"\n")
        if nl != -1:
            raw = raw[: nl + 1]
        try:
            text = raw.decode("utf-8")
            if text.encode("utf-8") == raw:
                return start, raw
        except UnicodeDecodeError:
            pass
        offset += PROBE_STEP
    raise RuntimeError(f"could not find valid utf-8 slice near {requested_offset}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_size = Path(PILE).stat().st_size
    slices = []

    # Canonical first (verbatim copy).
    canon_raw = Path(CANON["cached"]).read_bytes()
    canon_sha = hashlib.sha256(canon_raw).hexdigest()
    assert canon_sha == CANON["expect_sha"], f"canonical sha drift: {canon_sha}"
    canon_path = OUT_DIR / "slice_canonical_1e12.txt"
    canon_path.write_bytes(canon_raw)
    slices.append({
        "name": CANON["name"],
        "path": str(canon_path),
        "requested_offset": CANON["requested_offset"],
        "actual_start_byte": 1_000_000_001_956,
        "byte_length": len(canon_raw),
        "sha256": canon_sha,
        "offset_fraction": round(1_000_000_001_956 / total_size, 6),
        "canonical": True,
    })
    print(f"[canonical] {len(canon_raw)} bytes sha={canon_sha[:16]} frac={1e12/total_size:.4f}")

    with open(PILE, "rb") as f:
        for frac in EXTRA_FRACTIONS:
            req = int(frac * total_size)
            start, raw = extract_at(f, req, total_size)
            sha = hashlib.sha256(raw).hexdigest()
            name = f"slice_frac{frac:.3f}"
            path = OUT_DIR / f"{name}.txt"
            path.write_bytes(raw)
            slices.append({
                "name": name,
                "path": str(path),
                "requested_offset": req,
                "actual_start_byte": start,
                "byte_length": len(raw),
                "sha256": sha,
                "offset_fraction": round(start / total_size, 6),
                "canonical": False,
            })
            print(f"[{name}] start={start} {len(raw)} bytes sha={sha[:16]} frac={start/total_size:.4f}")

    # sanity: all shas distinct, all non-overlapping
    shas = [s["sha256"] for s in slices]
    assert len(set(shas)) == len(shas), "duplicate slice shas!"
    spans = sorted((s["actual_start_byte"], s["actual_start_byte"] + s["byte_length"]) for s in slices)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, f"overlap: {a1} > {b0}"

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "source_path": PILE,
        "total_bytes": total_size,
        "read_len_requested": READ_LEN,
        "context_for_neural_eval": 2048,
        "extraction": "seek(offset); advance to next newline; read 10_000_000 bytes; trim to last newline; retry +5MB on invalid utf-8.",
        "slices": slices,
    }, indent=2))
    print(f"\nwrote manifest {MANIFEST} ({len(slices)} slices)")


if __name__ == "__main__":
    main()
