#!/usr/bin/env python3
"""Classical compression-ratio baselines on a byte-identical Pile held-out slice.

Reads paper/review/heldout_slice.json, extracts EXACTLY those bytes from the
source, verifies the sha256, then compresses the single stream with each
tool/level and records compressed size + compression bpb.

CPU-only. No fabrication: every bpb value comes from a real subprocess run;
failures are recorded with the exact error.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SLICE_JSON = os.path.join(HERE, "heldout_slice.json")
OUT_MD = os.path.join(HERE, "COMPRESSION_BPB.md")


def read_slice_spec():
    with open(SLICE_JSON) as fh:
        return json.load(fh)


def extract_bytes(source_path, byte_offset, byte_length):
    """Seek to byte_offset and read exactly byte_length bytes."""
    with open(source_path, "rb") as fh:
        fh.seek(byte_offset)
        data = fh.read(byte_length)
    if len(data) != byte_length:
        raise RuntimeError(
            f"Read {len(data)} bytes, expected {byte_length} "
            f"(offset={byte_offset}, source={source_path})"
        )
    return data


def tool_version(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        return (out.stdout + out.stderr).strip().splitlines()[0]
    except Exception as e:  # pragma: no cover
        return f"(version unavailable: {e})"


def compress(name, argv, data):
    """Run `argv` reading stdin, writing compressed stream to stdout.

    Returns (compressed_size_bytes, elapsed_s, error_or_None).
    """
    t0 = time.time()
    try:
        proc = subprocess.run(argv, input=data, capture_output=True)
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        return None, elapsed, f"exit {proc.returncode}: {err}"
    return len(proc.stdout), elapsed, None


def main():
    spec = read_slice_spec()
    source_path = spec["source_path"]
    byte_offset = int(spec["byte_offset"])
    byte_length = int(spec["byte_length"])
    expected_sha = spec["sha256"]

    print(f"slice: source={source_path} offset={byte_offset} "
          f"length={byte_length}", file=sys.stderr)
    data = extract_bytes(source_path, byte_offset, byte_length)
    actual_sha = hashlib.sha256(data).hexdigest()
    print(f"sha256 expected={expected_sha}", file=sys.stderr)
    print(f"sha256 actual  ={actual_sha}", file=sys.stderr)
    if actual_sha != expected_sha:
        raise SystemExit(
            f"SHA256 MISMATCH: extracted slice does not match heldout_slice.json.\n"
            f"  expected={expected_sha}\n  actual  ={actual_sha}\n"
            f"Refusing to proceed — byte-identical input is required."
        )
    original_size = len(data)
    print(f"sha256 OK; original_size={original_size} bytes", file=sys.stderr)

    # tool/level matrix — order matters for the output table
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
        size, elapsed, err = compress(tool, argv, data)
        bpb = (size * 8) / original_size if size is not None else None
        results.append({
            "tool": tool, "level": level, "argv": argv,
            "compressed_bytes": size, "bpb": bpb,
            "elapsed_s": elapsed, "error": err,
        })
        if err:
            print(f"  FAILED: {err}", file=sys.stderr)
        else:
            print(f"  {size} bytes, bpb={bpb:.6f}, {elapsed:.1f}s",
                  file=sys.stderr)

    versions = {
        "gzip": tool_version(["gzip", "--version"]),
        "bzip2": tool_version(["bzip2", "--help"]),
        "xz": tool_version(["xz", "--version"]),
        "zstd": tool_version(["zstd", "--version"]),
    }

    write_markdown(spec, original_size, actual_sha, results, versions)
    print(f"wrote {OUT_MD}", file=sys.stderr)


def write_markdown(spec, original_size, sha, results, versions):
    ratio_base = original_size
    lines = []
    lines.append("# Classical Compression-Ratio Baselines on The Pile Held-Out Slice")
    lines.append("")
    lines.append("Compression as a language-modeling anchor: the bits-per-byte (bpb) a")
    lines.append("general-purpose compressor achieves on the *exact same* held-out Pile")
    lines.append("bytes used by the neural eval (`pile-bpb-measure`). These are CPU-only,")
    lines.append("model-free lower-effort baselines — a neural LM that beats them on bpb is")
    lines.append("genuinely modeling structure the classical coders miss.")
    lines.append("")
    lines.append("## Result table")
    lines.append("")
    lines.append(f"Original (uncompressed) slice size: **{original_size:,} bytes** "
                 f"({original_size} bytes).")
    lines.append("")
    lines.append("bpb = (compressed_size_bytes × 8) / original_size_bytes. "
                 "Single stream, whole slice compressed at once.")
    lines.append("")
    lines.append("| Tool | Level | Compressed bytes | Ratio | Compression bpb |")
    lines.append("|------|-------|------------------:|------:|----------------:|")
    for r in results:
        if r["error"]:
            lines.append(f"| {r['tool']} | {r['level']} | FAILED | — | "
                         f"FAILED — {r['error']} |")
        else:
            ratio = ratio_base / r["compressed_bytes"]
            lines.append(
                f"| {r['tool']} | {r['level']} | {r['compressed_bytes']:,} | "
                f"{ratio:.3f}× | {r['bpb']:.4f} |"
            )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- **Single-stream**: the entire slice is fed to each compressor as one")
    lines.append("  stdin stream (no chunking, no archive container, no tar). Compressed")
    lines.append("  size is the exact byte length of the tool's stdout.")
    lines.append("- **CPU-only, no GPU.** Runs in parallel with the neural eval.")
    lines.append("- Exact invocations (stdin → stdout):")
    lines.append("")
    for r in results:
        lines.append(f"  - {r['tool']} {r['level']}: `{' '.join(r['argv'])}`")
    lines.append("")
    lines.append("Per-run wall-clock (informational, not a benchmark of speed):")
    lines.append("")
    for r in results:
        if r["error"] is None and r["elapsed_s"] is not None:
            lines.append(f"  - {r['tool']} {r['level']}: {r['elapsed_s']:.1f}s")
    lines.append("")
    lines.append("## Tool versions")
    lines.append("")
    for k in ("gzip", "bzip2", "xz", "zstd"):
        lines.append(f"- **{k}**: {versions[k]}")
    lines.append("")
    lines.append("## Same-slice confirmation (byte-identical to neural eval)")
    lines.append("")
    lines.append("The bytes compressed here are byte-identical to those evaluated by the")
    lines.append("neural BPB measurement. Provenance from `paper/review/heldout_slice.json`:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(spec, indent=2))
    lines.append("```")
    lines.append("")
    lines.append(f"- **source_path**: `{spec.get('source_path')}`")
    lines.append(f"- **byte_offset**: {spec.get('byte_offset')}")
    lines.append(f"- **byte_length**: {spec.get('byte_length')}")
    lines.append(f"- **sha256 (verified by re-extraction & re-hash)**: `{sha}`")
    lines.append("")
    lines.append("The sha256 above was recomputed from the bytes actually read from the")
    lines.append("source and matched the value in `heldout_slice.json` before any")
    lines.append("compression was run; a mismatch aborts the script.")
    lines.append("")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
