#!/usr/bin/env python3
"""Render the E97 long-document library anchor as window-cut fragments (v2).

OPERATOR DESIGN RULING (supersedes the whole-books render): the anchor data is
DOCUMENT CONCATENATION read as arbitrary-entry window cuts, NOT
document-aligned whole books. Whole-book alignment overtrains document
beginings and never exercises mid-document state entry — a competence the
pile's raw window cuts built during pre-training and one the state-forking
deployment demos depend on. The machinery already supports this:
``ndm/data/masked_sft_dataset.py`` gives every record a token-aligned
``reset_before`` mask (clean recurrent-state reset at each record start), so
a mid-book window-cut fragment is just a record and gets fresh-state
arbitrary entry by construction.

Render rule (mirrors the mainmix-draws authority, the cohort the operator
cites as already matching the design):

1. Stream: the sha-bound library authority's TRAIN-split records (whole
   books) concatenated in authority record order. Every source record already
   ends with the EOT record-separator token (50256), so concatenation yields
   the document-concatenation stream with the record separator between
   documents — the anchor tokens are reused verbatim, NO re-tokenization, and
   the source authority's holdout/probe exclusions are inherited unchanged.
2. Window cuts: 64K-token windows (65,536 tokens; the boundary-aware pack
   sequence is 65,537 so every window packs whole) entered at arbitrary stream
   positions — a seeded phase offset selects the cut phase, then a stride
   tiling covers the stream exactly once. The stream head before the phase
   and the stream tail past the last full window are kept as partial edge
   fragments (records), exactly as pre-training windows cut the stream.
3. Each window is ONE masked-SFT record: first token unsupervised (the
   record boundary is the reset), all in-window tokens supervised — including
   the in-window document separators, so state flows across document
   boundaries inside a window exactly as the raw stream was read; the hidden
   state resets only at the fragment (record) start. 1% deterministic hash
   validation split, same convention as the mainmix authority.

CPU-only by construction: no CUDA import, no GPU state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256

EOT_TOKEN = 50256
WINDOW_TOKENS = 65_536          # 64K windows; pack sequence is 65,537
SPLIT_SALT = "e97-library-fragment-render-v1"


def _stable_hash(*parts: object) -> bytes:
    payload = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).digest()


def _phase(seed: int, window_tokens: int) -> int:
    digest = hashlib.sha256(f"{SPLIT_SALT}\0phase\0{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % window_tokens


def _split_flag(window_ordinal: int) -> int:
    return int(int.from_bytes(
        _stable_hash(SPLIT_SALT, f"window{window_ordinal}"), "little") % 100 == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="the whole-books library authority root")
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-tokens", type=int, default=WINDOW_TOKENS)
    parser.add_argument("--seed", type=int, default=974_206)
    args = parser.parse_args()
    if not 0 < args.window_tokens <= 65_537:
        raise SystemExit("window size must fit the 65,537-token pack sequence")

    # --- source authority identity (fail-closed) ---
    source_manifest_path = args.source_root / "manifest.json"
    if sha256(source_manifest_path) != args.source_manifest_sha256:
        raise SystemExit("source library authority manifest identity mismatch")
    source_manifest = json.loads(source_manifest_path.read_text())
    if (source_manifest.get("schema") != AUTHORITY_SCHEMA
            or source_manifest.get("status") != "complete"
            or source_manifest.get("training_eligible") is not True):
        raise SystemExit("source is not a complete training-eligible authority")
    outputs = source_manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"tokens", "mask", "index", "metadata"}:
        raise SystemExit("source authority outputs are invalid")
    payloads = {}
    for name, spec in outputs.items():
        path = args.source_root / spec["path"]
        if (path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]):
            raise SystemExit(f"source payload identity mismatch: {name}")
        payloads[name] = path

    rows = [json.loads(line) for line in payloads["metadata"].open()]
    index_bytes = payloads["index"].read_bytes()
    if len(index_bytes) != RECORD_INDEX.size * len(rows):
        raise SystemExit("source index shape mismatch")
    tokens_mm = np.memmap(payloads["tokens"], dtype="<u4", mode="r")

    # --- build the concatenated TRAIN stream in authority record order ---
    # Document structure: every source record ends with the EOT record-separator
    # token (50256) EXCEPT the non-final segments of the 15 >65,537-token
    # documents segmented at the 64K pack boundary ("the final segment carries
    # EOT") — those records continue their document without a separator, so a
    # DOCUMENT is a maximal run of records joined without an intervening EOT.
    train_records = []           # (stream_start, token_count, source_name, record_id)
    stream = np.empty(source_manifest["counts"]["tokens"], dtype="<u4")
    cursor = 0
    eot_terminated = []          # per train record: does the record end with EOT?
    for i, row in enumerate(rows):
        offset, n, want, split = RECORD_INDEX.unpack_from(index_bytes, i * RECORD_INDEX.size)
        if split:
            continue  # the source authority's 1% validation records stay out
        stream[cursor:cursor + n] = tokens_mm[offset:offset + n]
        terminated = bool(tokens_mm[offset + n - 1] == EOT_TOKEN)
        eot_terminated.append(terminated)
        train_records.append((cursor, n, row["source_name"], row["id"]))
        cursor += n
    stream_length = cursor
    stream = stream[:stream_length]
    del tokens_mm
    record_starts = np.array([r[0] for r in train_records], dtype=np.int64)

    # document starts: the stream head plus every position right after an
    # EOT-terminated record (the separator ends the PREVIOUS document)
    doc_start_positions = [0]
    for (start, n, _name, _rid), terminated in zip(train_records, eot_terminated):
        if terminated and start + n < stream_length:
            doc_start_positions.append(start + n)
    doc_starts = np.array(doc_start_positions, dtype=np.int64)

    # sanity: every in-stream EOT is a record terminator, and every EOT-
    # terminated record's last token is EOT (encode_ordinary cannot emit 50256)
    if int(np.count_nonzero(stream == EOT_TOKEN)) != sum(eot_terminated):
        raise SystemExit("stream separator accounting mismatch (stray EOT token)")

    # --- window cuts: seeded phase + stride tiling; edge fragments kept ---
    phase = _phase(args.seed, args.window_tokens)
    windows = []                 # (start, stop)
    if phase > 0:
        windows.append((0, phase))          # stream-head edge fragment
    start = phase
    while start < stream_length:
        windows.append((start, min(start + args.window_tokens, stream_length)))
        start += args.window_tokens
    if not windows:
        raise SystemExit("empty render: no windows over the stream")

    # --- emit one record per window ---
    args.output_root.mkdir(parents=True, exist_ok=False)
    out_tokens = args.output_root / "tokens.uint32.bin"
    out_mask = args.output_root / "assistant_mask.uint8.bin"
    out_index = args.output_root / "records.idx"
    out_metadata = args.output_root / "records.jsonl"

    counts: Counter = Counter()
    source_windows: Counter = Counter()
    entry_mid_document = 0
    entry_buckets: Counter = Counter()
    separator_total = 0
    documents_spanned_total = 0
    full_windows = 0
    output_offset = 0
    receipts = []
    with (out_tokens.open("wb", buffering=4 << 20) as token_out,
          out_mask.open("wb", buffering=4 << 20) as mask_out,
          out_index.open("wb", buffering=1 << 20) as index_out,
          out_metadata.open("w", buffering=1 << 20) as metadata_out):
        for ordinal, (start, stop) in enumerate(windows):
            chunk = stream[start:stop]
            n = int(chunk.size)
            targets = n - 1
            # the record containing the window start (bisect right on starts)
            container = int(np.searchsorted(record_starts, start, side="right") - 1)
            # the DOCUMENT containing the window start: a segmented document's
            # segments join without a separator, so look back to the document start
            doc_index = int(np.searchsorted(doc_starts, start, side="right") - 1)
            doc_start = int(doc_starts[doc_index])
            entry_offset = int(start - doc_start)
            mid_document = bool(entry_offset > 0)
            # in-window document separators: EOT terminators inside the window
            # (encode_ordinary cannot produce 50256, so every EOT is a separator,
            # except the stream-final EOT, which only terminates the last document)
            separators = int(np.count_nonzero(chunk == EOT_TOKEN))
            if stop == stream_length and separators > 0:
                separators -= 1
            documents_spanned = separators + 1
            primary_source = train_records[container][2]
            sources = sorted({train_records[j][2] for j in range(
                int(np.searchsorted(record_starts, start, side="right")),
                int(np.searchsorted(record_starts, stop, side="right")))})
            identity = f"library-fragments:window{ordinal}:stream{start}"
            split = _split_flag(ordinal)
            token_out.write(chunk.astype("<u4", copy=False).tobytes())
            mask_out.write(b"\x00" + bytes([1]) * targets)
            index_out.write(RECORD_INDEX.pack(output_offset, n, targets, split))
            metadata_out.write(json.dumps({
                "id": identity,
                "source": "e97-long-document-anchor-library-fragments-v2",
                "source_name": primary_source,
                "sources_spanned": sources,
                "window_ordinal": ordinal,
                "stream_start": start,
                "stream_stop": stop,
                "window_edge_fragment": bool(n < args.window_tokens),
                "entry_offset_in_document": entry_offset,
                "mid_document_entry": mid_document,
                "document_separators_in_window": separators,
                "documents_spanned": documents_spanned,
                "source_record_id": train_records[container][3],
                "split": split,
                "tokens": n,
                "causal_targets": targets,
            }, sort_keys=True) + "\n")
            output_offset += n
            counts["records"] += 1
            counts["tokens"] += n
            counts["causal_targets"] += targets
            counts["validation_records" if split else "train_records"] += 1
            counts["full_windows"] += int(n == args.window_tokens)
            source_windows[primary_source] += 1
            entry_mid_document += int(mid_document)
            entry_buckets[min(entry_offset // 8192, 7)] += 1
            separator_total += separators
            documents_spanned_total += documents_spanned
            if len(receipts) < 3 or n < args.window_tokens:
                receipts.append({
                    "id": identity, "stream_start": start, "tokens": n,
                    "entry_offset_in_document": entry_offset,
                    "mid_document_entry": mid_document,
                    "document_separators_in_window": separators,
                    "documents_spanned": documents_spanned,
                })

    # --- read-back verification: re-derive every record from the payloads ---
    verify_rows = [json.loads(line) for line in out_metadata.open()]
    verify_index = out_index.read_bytes()
    tokens_back = np.memmap(out_tokens, dtype="<u4", mode="r")
    if len(verify_index) != RECORD_INDEX.size * len(verify_rows):
        raise SystemExit("emitted index shape mismatch")
    offset_expected = 0
    for i, row in enumerate(verify_rows):
        off, n, want, split = RECORD_INDEX.unpack_from(verify_index, i * RECORD_INDEX.size)
        if (off != offset_expected or n != row["tokens"] or want != row["causal_targets"]
                or split != row["split"] or want != n - 1
                or not np.array_equal(tokens_back[off:off + n], stream[row["stream_start"]:row["stream_stop"]])):
            raise SystemExit(f"emitted record verification failed: {row['id']}")
        offset_expected += n
    if offset_expected != int(counts["tokens"]):
        raise SystemExit("emitted token accounting mismatch")
    del tokens_back

    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "training_eligible": True,
        "purpose": ("library anchor v2 render per the operator design ruling: "
                    "document concatenation read as arbitrary-entry window cuts, "
                    "NOT document-aligned whole books — 64K windows entered at "
                    "arbitrary stream positions with hidden-state reset at every "
                    "record (fragment) start; in-window document separators are "
                    "supervised tokens so state flows across document boundaries "
                    "inside a window exactly as the raw pre-training stream was read"),
        "cohort": "e97-long-document-anchor-library-fragments-v2",
        "render_rule": ("seeded-phase stride tiling of the concatenated TRAIN-split "
                        "source records in authority record order; stream-head and "
                        "stream-tail partial windows kept as edge-fragment records"),
        "source_authority": str(args.source_root.resolve()),
        "source_authority_manifest_sha256": args.source_manifest_sha256,
        "source_authority_cohort": source_manifest.get("cohort"),
        "exclusions_inherited": ("the source authority's holdout (identity + shingle) and "
                                 "frozen probe-panel exclusions are inherited verbatim: "
                                 "this render reuses its token payloads and adds no text; "
                                 "the source 1% validation records stay out of the stream"),
        "stream_construction": ("source TRAIN records concatenated in record order; each "
                               "record already ends with the EOT record-separator token "
                               "(50256), so documents are joined by the record separator "
                               "with no re-tokenization"),
        "window_tokens": args.window_tokens,
        "window_phase_seed": args.seed,
        "window_phase": phase,
        "reset_semantics": ("one record per window; reset_before at the fragment start "
                            "clears the recurrent state (ndm.data.masked_sft_dataset."
                            "pack_at_with_boundaries); in-window EOT separators do NOT "
                            "reset — the state carries across document boundaries "
                            "inside the window (document concatenation read)"),
        "split_convention": ("1% deterministic hash validation split "
                             "(library-fragment-render-v1 salad)"),
        "counts": {
            **{k: int(v) for k, v in counts.items()},
            "assistant_target_tokens": int(counts["causal_targets"]),
            "mid_document_entry_windows": int(entry_mid_document),
            "document_aligned_entry_windows": int(counts["records"] - entry_mid_document),
            "total_document_separators_in_stream": sum(eot_terminated) - int(eot_terminated[-1])
                if len(eot_terminated) > 1 else 0,
            "total_documents_in_stream": int(len(doc_starts)),
            "in_window_document_separators": int(separator_total),
            "total_documents_spanned": int(documents_spanned_total),
        },
        "entry_position_histogram_tokens": {
            f"bucket_{k * 8192}_{(k + 1) * 8192}": int(v) for k, v in sorted(entry_buckets.items())
        },
        "per_source_windows": dict(sorted(source_windows.items())),
        "window_receipts": receipts,
        "outputs": {
            "tokens": {"path": out_tokens.name, "bytes": out_tokens.stat().st_size,
                       "sha256": sha256(out_tokens)},
            "mask": {"path": out_mask.name, "bytes": out_mask.stat().st_size,
                     "sha256": sha256(out_mask)},
            "index": {"path": out_index.name, "bytes": out_index.stat().st_size,
                      "sha256": sha256(out_index)},
            "metadata": {"path": out_metadata.name, "bytes": out_metadata.stat().st_size,
                          "sha256": sha256(out_metadata)},
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest_sha256": sha256(manifest_path),
        "records": manifest["counts"]["records"],
        "tokens": manifest["counts"]["tokens"],
        "causal_targets": manifest["counts"]["causal_targets"],
        "mid_document_entry_windows": manifest["counts"]["mid_document_entry_windows"],
        "phase": phase,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
