# Classical Compression-Ratio Baselines on The Pile Held-Out Slice

Compression as a language-modeling anchor: the bits-per-byte (bpb) a
general-purpose compressor achieves on the *exact same* held-out Pile
bytes used by the neural eval (`pile-bpb-measure`). These are CPU-only,
model-free lower-effort baselines — a neural LM that beats them on bpb is
genuinely modeling structure the classical coders miss.

## Result table

Original (uncompressed) slice size: **9,999,511 bytes** (9999511 bytes).

bpb = (compressed_size_bytes × 8) / original_size_bytes. Single stream, whole slice compressed at once.

| Tool | Level | Compressed bytes | Ratio | Compression bpb |
|------|-------|------------------:|------:|----------------:|
| gzip | -9 | 3,503,416 | 2.854× | 2.8029 |
| bzip2 | -9 | 2,960,226 | 3.378× | 2.3683 |
| xz | -9 | 2,737,100 | 3.653× | 2.1898 |
| xz | -9e | 2,737,672 | 3.653× | 2.1902 |
| zstd | -19 | 2,812,348 | 3.556× | 2.2500 |
| zstd | --ultra -22 | 2,811,634 | 3.556× | 2.2494 |

Best classical bpb on this slice: **xz -9 at 2.1898**. Note `xz -9e` came out
*marginally larger* than `xz -9` (2,737,672 vs 2,737,100 bytes) — extreme mode
does not always help and was very slightly worse here; both values are reported
exactly as measured, not adjusted. `zstd -22` likewise gave essentially no gain
over `zstd -19` on this 10 MB stream.

### Interpretation (anchor only — not a like-for-like comparison)

These classical coders sit at **~2.19–2.80 bpb**, well above the neural Pile
bpb numbers (E88 train-loss 0.974; the open Pile-trained ~1.3 B transformers
measured by `pile-bpb-measure` land below ~1.0 on this same slice). That gap is
the expected and intended message of the "LM-as-compression" framing: a learned
model captures far more of the text's statistical structure than a general
dictionary/entropy coder. The classical numbers are a model-free floor, not a
competitor — they bound how much of the bpb is "free" from generic redundancy
(repeats, n-gram entropy) versus genuine language modeling. They are *not*
directly comparable to the neural figures (single fixed-context forward NLL with
each model's own tokenizer vs. a streaming general compressor), so read them as
a sanity anchor on the same bytes, not a leaderboard entry.

## Method

- **Single-stream**: the entire slice is fed to each compressor as one
  stdin stream (no chunking, no archive container, no tar). Compressed
  size is the exact byte length of the tool's stdout.
- **CPU-only, no GPU.** Runs in parallel with the neural eval.
- Exact invocations (stdin → stdout):

  - gzip -9: `gzip -9 -c`
  - bzip2 -9: `bzip2 -9 -c`
  - xz -9: `xz -9 -c -T 1`
  - xz -9e: `xz -9e -c -T 1`
  - zstd -19: `zstd -19 -c -T0`
  - zstd --ultra -22: `zstd --ultra -22 -c -T0`

Per-run wall-clock (informational, not a benchmark of speed):

  - gzip -9: 0.7s
  - bzip2 -9: 0.7s
  - xz -9: 5.4s
  - xz -9e: 4.5s
  - zstd -19: 4.6s
  - zstd --ultra -22: 5.6s

## Tool versions

- **gzip**: gzip 1.12
- **bzip2**: bzip2, a block-sorting file compressor.  Version 1.0.8, 13-Jul-2019.
- **xz**: xz (XZ Utils) 5.4.5
- **zstd**: *** Zstandard CLI (64-bit) v1.5.5, by Yann Collet ***

## Same-slice confirmation (byte-identical to neural eval)

The bytes compressed here are byte-identical to those evaluated by the
neural BPB measurement. Provenance from `paper/review/heldout_slice.json`:

```json
{
  "source_path": "/mnt/nvme2n1/erikg/pile.txt",
  "requested_byte_offset": 1000000000000,
  "actual_start_byte": 1000000001956,
  "byte_offset": 1000000001956,
  "byte_length": 9999511,
  "total_bytes": 9999511,
  "sha256": "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a"
}
```

- **source_path**: `/mnt/nvme2n1/erikg/pile.txt`
- **byte_offset**: 1000000001956
- **byte_length**: 9999511
- **sha256 (verified by re-extraction & re-hash)**: `3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a`

The sha256 above was recomputed from the bytes actually read from the
source and matched the value in `heldout_slice.json` before any
compression was run; a mismatch aborts the script.

### Provenance note (how byte-identity was guaranteed)

The `pile-bpb-measure` task did **not** write `paper/review/heldout_slice.json`
(the agreed handoff file); it instead cached the raw slice to its own worktree at
`scripts/.pile_heldout_slice.txt` and recorded the slice in `scripts/.bpb_run.log`
as `9999511 bytes, sha256=3e4241a946e76c31..., offset=1000000000000`. Rather than
substitute a different slice, this task **independently re-extracted the
byte-identical slice from the source corpus** using the exact algorithm in the
neural eval's `scripts/measure_pile_bpb.py` (`extract_slice`): seek to byte
`1_000_000_000_000`, advance to the next `\n` (clean line boundary → actual start
byte `1000000001956`), read 10,000,000 bytes, then trim the trailing partial line
back to the last `\n`. This reproduced **exactly 9,999,511 bytes** hashing to
`3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a` — matching
both the neural eval's on-disk cache file and the sha recorded in its run log.
`heldout_slice.json` written by this task therefore records the verified
provenance of that same slice. (The `ddafac3e…` sha that appeared in the
neural eval's WG progress message is a transcription slip; the authoritative
`.bpb_run.log` and the cache file both read `3e4241a9…`, which is what was
scored and what was compressed here.)

`main.typ` was NOT modified by this task.

