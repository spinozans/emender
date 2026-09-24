#!/usr/bin/env python3
"""Build the E97 mainmix anchor draws authority (cohort B).

Purpose: the second anchor cohort comes from the packed mainmix stream itself —
the exact stream the base model consumed — as the general-distribution anchor
(all sources mixed, uncurated). Random 16MB-aligned windows of
``commapile_mainmix_v0.1_1tb.txt`` are scanned across the 0x1e record separator
exactly as pre-training consumed the stream: window-edge fragments are kept
(scan-across), documents keep stream order within each window, and the bytes
are decoded with errors='replace' like the pre-training loader.

Each delimited document becomes one document-causal record in the established
masked-SFT authority format (identical to the commapile document-causal and
long-document-anchor-library authorities): ``encode_ordinary(text)`` plus EOT,
first record token unsupervised. In boundary-aware packs the record boundary is
the context reset that replaces the consumed 0x1e delimiter.

The only curation: frozen probe-panel document texts are excluded (so the NLL
retention panel never trains on its own probe documents). No length filter, no
deduplication — epoch-resampled duplicates inside the stream are part of the
pre-training distribution and are retained.

CPU-only by construction: no CUDA import, no GPU state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct

import tiktoken

try:
    from scripts.build_e97_commapile_document_sft import (
        DATASET_ID, DATASET_REVISION, TOKENIZER,
    )
    from scripts.build_e97_tulu3_sft import TOKENIZER_CACHE_KEY, TOKENIZER_SHA256
except ModuleNotFoundError:  # pragma: no cover - script-path invocation
    from build_e97_commapile_document_sft import (
        DATASET_ID, DATASET_REVISION, TOKENIZER,
    )
    from build_e97_tulu3_sft import TOKENIZER_CACHE_KEY, TOKENIZER_SHA256

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256

SCHEMA = AUTHORITY_SCHEMA
EOT_TOKEN = 50256
WINDOW_BYTES = 16 * 1024 * 1024  # the mainmix buffer size: windows are aligned to it
MAX_RECORD_TOKENS = 65_537  # = 64K pack sequence_tokens
SEGMENT_TOKENS = MAX_RECORD_TOKENS
PROBE_SCHEMA = "emender-e97-document-nll-probe-panel-v1"


def _stable_hash(*parts: object) -> bytes:
    payload = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).digest()


def _verify_tokenizer_cache() -> None:
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_root:
        raise SystemExit("TIKTOKEN_CACHE_DIR must bind the verified p50k cache")
    cache_object = Path(cache_root) / TOKENIZER_CACHE_KEY
    if not cache_object.is_file() or sha256(cache_object) != TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache object is missing or corrupt")


def _probe_text_sha256s(panel_manifest: Path) -> set[str]:
    panel = json.loads(panel_manifest.read_text())
    if panel.get("schema") != PROBE_SCHEMA:
        raise SystemExit("unsupported probe panel manifest")
    return {row["text_sha256"] for row in panel["documents"]}


def _window_indices(count: int, limit: int, seed: int) -> list[int]:
    """Deterministic seeded draw of distinct window indices, hash-sorted."""
    if count > limit:
        raise SystemExit(f"window count {count} exceeds {limit} aligned windows")
    picked: set[int] = set()
    state = f"e97-mainmix-anchor-draws-v1\0{seed}".encode()
    while len(picked) < count:
        digest = hashlib.sha256(state).digest()
        picked.add(int.from_bytes(digest[:8], "little") % limit)
        state = digest
    return sorted(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--stream-sha256", required=True,
                        help="expected SHA-256 of the packed mainmix stream")
    parser.add_argument("--probe-panel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--windows", type=int, required=True,
                        help="number of 16MB-aligned windows to draw")
    parser.add_argument("--seed", type=int, default=974_131)
    args = parser.parse_args()
    _verify_tokenizer_cache()
    encoding = tiktoken.get_encoding(TOKENIZER)

    stream_sha = sha256(args.stream)
    if stream_sha != args.stream_sha256:
        raise SystemExit("mainmix stream SHA-256 mismatch")
    probe_sha256s = _probe_text_sha256s(args.probe_panel)

    stream_size = args.stream.stat().st_size
    window_limit = stream_size // WINDOW_BYTES
    window_indices = _window_indices(args.windows, window_limit, args.seed)

    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    counts: Counter = Counter()
    window_receipts = []
    output_offset = 0
    with open(args.stream, "rb") as stream_raw:
        mapped = mmap.mmap(stream_raw.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            with (outputs["tokens"].open("wb", buffering=4 << 20) as token_out,
                  outputs["mask"].open("wb", buffering=4 << 20) as mask_out,
                  outputs["index"].open("wb", buffering=1 << 20) as index_out,
                  outputs["metadata"].open("w", buffering=1 << 20) as metadata_out):
                for window_ordinal, window_index in enumerate(window_indices, 1):
                    start = window_index * WINDOW_BYTES
                    window = bytes(mapped[start:start + WINDOW_BYTES])
                    if not window:
                        raise RuntimeError(f"empty window {window_index}")
                    fragments = window.split(b"\x1e")
                    window_counts: Counter = Counter()
                    window_tokens = 0
                    for record_ordinal, fragment in enumerate(fragments):
                        text = fragment.decode("utf-8", errors="replace")
                        if not text:
                            window_counts["empty_records"] += 1
                            continue
                        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
                        if text_sha256 in probe_sha256s:
                            window_counts["excluded_probe_panel_text"] += 1
                            continue
                        token_values = encoding.encode_ordinary(text)
                        token_values.append(EOT_TOKEN)
                        segments = max(1, -(-len(token_values) // SEGMENT_TOKENS))
                        for segment in range(segments):
                            chunk = token_values[segment * SEGMENT_TOKENS:
                                                 (segment + 1) * SEGMENT_TOKENS]
                            token_count = len(chunk)
                            identity = (f"mainmix:window{window_index}"
                                        f":record{record_ordinal}")
                            if segments > 1:
                                identity += f":segment{segment + 1}of{segments}"
                            effective_targets = token_count - 1
                            token_out.write(struct.pack(
                                f"<{token_count}I", *chunk))
                            # First record token unsupervised: the record
                            # boundary is the reset that replaces the consumed
                            # 0x1e delimiter (document-causal convention).
                            mask_out.write(
                                b"\x00" + bytes([1]) * effective_targets)
                            split = int.from_bytes(_stable_hash(
                                "mainmix-anchor-draws-v1", identity),
                                "little") % 100 == 0
                            index_out.write(RECORD_INDEX.pack(
                                output_offset, token_count, effective_targets,
                                int(split)))
                            metadata_out.write(json.dumps({
                                "id": identity,
                                "source": "commapile:mainmix_draws",
                                "source_name": "mainmix_draws",
                                "window_index": window_index,
                                "window_start": start,
                                "record_ordinal": record_ordinal,
                                "window_edge_fragment": (
                                    record_ordinal == 0
                                    or record_ordinal == len(fragments) - 1),
                                "text_sha256": text_sha256,
                                "segment": segment + 1 if segments > 1 else None,
                                "segments": segments if segments > 1 else None,
                                "split": int(split),
                                "tokens": token_count,
                                "causal_targets": effective_targets,
                            }, sort_keys=True) + "\n")
                            output_offset += token_count
                            window_tokens += token_count
                            window_counts["records"] += 1
                            window_counts["segmented_records"] += int(segments > 1)
                            window_counts["tokens"] += token_count
                            window_counts["causal_targets"] += effective_targets
                            counts["records"] += 1
                            counts["segmented_records"] += int(segments > 1)
                            counts["tokens"] += token_count
                            counts["causal_targets"] += effective_targets
                            counts["validation_records" if split
                                   else "train_records"] += 1
                    window_receipts.append({
                        "window_index": window_index,
                        "window_start": start,
                        **{key: value for key, value in sorted(
                            window_counts.items()) if key != "segmented_records"},
                    })
        finally:
            mapped.close()

    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "training_eligible": True,
        "purpose": ("mainmix anchor draws: random 16MB-aligned windows of the "
                    "packed mainmix stream scanned across 0x1e delimiters — "
                    "the general-distribution anchor cohort, uncurated except "
                    "probe-panel exclusion"),
        "cohort": "e97-long-document-anchor-mainmix-draws-v1",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "stream_path": str(args.stream.resolve()),
        "stream_sha256": stream_sha,
        "stream_bytes": stream_size,
        "window_bytes": WINDOW_BYTES,
        "window_alignment": "16MiB (the mainmix output buffer size)",
        "windows_requested": args.windows,
        "window_seed": args.seed,
        "window_selection": ("deterministic seeded hash draw of distinct "
                             "aligned window indices, processed in ascending "
                             "byte order (scan-across)"),
        "probe_panel_manifest_sha256": sha256(args.probe_panel),
        "probe_exclusion": ("probe-panel document texts are excluded so the "
                            "frozen NLL panel measures drift without "
                            "train-on-probe contamination; this is the only "
                            "curation applied"),
        "edge_semantics": ("window-edge documents are partial (scan-across "
                           "truncation) exactly as pre-training windows cut the "
                           "stream; decode uses errors='replace' like the "
                           "pre-training loader"),
        "tokenizer": TOKENIZER,
        "tokenizer_cache_sha256": TOKENIZER_SHA256,
        "max_record_tokens": MAX_RECORD_TOKENS,
        "segmentation": ("documents longer than 65537 tokens are segmented only "
                         "at the 64K pack boundary: consecutive records in "
                         "reading order; the final segment carries EOT"),
        "splits": ("1% deterministic hash validation split "
                   "(mainmix-anchor-draws-v1 salad)"),
        "counts": {
            **dict(counts),
            "assistant_target_tokens": int(counts["causal_targets"]),
        },
        "windows": window_receipts,
        "outputs": {name: {"path": path.name, "bytes": path.stat().st_size,
                           "sha256": sha256(path)}
                    for name, path in outputs.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest},
                     sort_keys=True))


if __name__ == "__main__":
    main()
