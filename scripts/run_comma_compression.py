#!/usr/bin/env python3
"""Classical compression baselines on the byte-identical comma-pile held-out slice.

Reads paper/review/comma_slice.json, re-extracts EXACTLY those bytes from the
source corpus, verifies the sha256 against the descriptor (abort on mismatch), then
compresses the single stream with each tool/level and records compressed size +
compression bpb = compressed_bytes*8 / original_bytes.

Same tool matrix and protocol as the Pile compression bench
(scripts/../paper/review/run_compression_bench.py). CPU-only. No fabrication.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLICE_JSON = HERE.parent / "paper" / "review" / "comma_slice.json"
OUT_JSON = HERE / ".comma_compression_results.json"


def extract_bytes(source_path, byte_offset, byte_length):
    with open(source_path, "rb") as fh:
        fh.seek(byte_offset)
        data = fh.read(byte_length)
    if len(data) != byte_length:
        raise RuntimeError(f"read {len(data)} != expected {byte_length}")
    return data


def tool_version(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        return (out.stdout + out.stderr).strip().splitlines()[0]
    except Exception as e:
        return f"(version unavailable: {e})"


def compress(argv, data):
    t0 = time.time()
    try:
        proc = subprocess.run(argv, input=data, capture_output=True)
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return None, elapsed, f"exit {proc.returncode}: {proc.stderr.decode('utf-8','replace').strip()}"
    return len(proc.stdout), elapsed, None


def main():
    spec = json.loads(SLICE_JSON.read_text())
    data = extract_bytes(spec["source_path"], int(spec["byte_offset"]),
                         int(spec["byte_length"]))
    actual_sha = hashlib.sha256(data).hexdigest()
    print(f"sha expected={spec['sha256']}\nsha actual  ={actual_sha}", file=sys.stderr)
    if actual_sha != spec["sha256"]:
        raise SystemExit("SHA256 MISMATCH — refusing to proceed (byte-identical required)")
    original = len(data)
    print(f"sha OK; original={original} bytes", file=sys.stderr)

    runs = [
        ("gzip", "-9", ["gzip", "-9", "-c"]),
        ("bzip2", "-9", ["bzip2", "-9", "-c"]),
        ("xz", "-9", ["xz", "-9", "-c", "-T", "1"]),
        ("xz", "-9e", ["xz", "-9e", "-c", "-T", "1"]),
        ("zstd", "-19", ["zstd", "-19", "-c", "-T0"]),
        ("zstd", "--ultra -22", ["zstd", "--ultra", "-22", "-c", "-T0"]),
    ]
    results = []
    for tool, level, argv in runs:
        print(f"running {tool} {level} ...", file=sys.stderr)
        size, elapsed, err = compress(argv, data)
        bpb = (size * 8) / original if size is not None else None
        results.append({"tool": tool, "level": level, "argv": argv,
                        "compressed_bytes": size, "bpb": bpb,
                        "elapsed_s": elapsed, "error": err})
        if err:
            print(f"  FAILED: {err}", file=sys.stderr)
        else:
            print(f"  {size} bytes, bpb={bpb:.6f}, {elapsed:.1f}s", file=sys.stderr)

    versions = {
        "gzip": tool_version(["gzip", "--version"]),
        "bzip2": tool_version(["bzip2", "--help"]),
        "xz": tool_version(["xz", "--version"]),
        "zstd": tool_version(["zstd", "--version"]),
    }
    OUT_JSON.write_text(json.dumps(
        {"original_bytes": original, "sha256": actual_sha,
         "results": results, "versions": versions}, indent=2))
    print(f"wrote {OUT_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
