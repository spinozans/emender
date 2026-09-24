#!/usr/bin/env python3
"""Build the E97 long-document anchor library (cohort A) and its frozen
document-NLL probe panel.

Purpose: the operator's directive that post-training must anchor on the base
pre-training distribution — very long text of all types, especially old books
— because this sequential-recurrent architecture's state dynamics were formed
on exactly that distribution. This builder scans the pinned CommaPile source
shards for the book/long-form sources, ranks documents by length, selects the
long tail under per-source token quotas, and emits a document-causal masked-SFT
authority in the established format (``ndm.data.masked_sft_dataset`` schema,
identical to ``build_e97_commapile_document_sft.py`` output) so the standard
pack builders/validators and E2 prep machinery can consume it unchanged.

Subcommands:
  index        pass 1: stream every shard of the selected sources and write a
               per-source candidate length index (books: every document over
               --book-min-chars; long-form/probe sources: the top-K longest).
  select       rank candidates and emit the selection manifest (per-source
               counts, quotas and length histograms).
  freeze-probe pick the frozen 8-document NLL probe panel (hash-ranked choice
               among the top-1000 longest documents per probe source) and emit
               token-id panels for the CPU GGML runner plus a manifest.
  build        pass 2: extract the selected documents, apply the same holdout
               identity+shingle exclusion as the commapile document-causal
               builder, exclude duplicate texts and probe-panel texts,
               tokenize whole documents, and write the authority. Documents
               longer than max-record-tokens are segmented ONLY at the 64K
               pack boundary (consecutive segment records in reading order;
               the final segment carries the EOT).

CPU-only by construction: no CUDA import, no GPU state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import struct

import tiktoken

try:  # Support both ``python -m`` and the documented script-path invocation.
    from scripts.build_e97_commapile_document_sft import (
        DATASET_ID, DATASET_REVISION, TOKENIZER, _git, _holdout_signatures,
        _lfs_identity, _overlap_reason,
    )
    from scripts.build_e97_tulu3_sft import TOKENIZER_CACHE_KEY, TOKENIZER_SHA256
except ModuleNotFoundError:  # pragma: no cover - exercised by subprocess CLIs
    from build_e97_commapile_document_sft import (
        DATASET_ID, DATASET_REVISION, TOKENIZER, _git, _holdout_signatures,
        _lfs_identity, _overlap_reason,
    )
    from build_e97_tulu3_sft import TOKENIZER_CACHE_KEY, TOKENIZER_SHA256

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256

SCHEMA = AUTHORITY_SCHEMA
EOT_TOKEN = 50256
MAX_RECORD_TOKENS = 65_537  # = 64K pack sequence_tokens; a record fills one pack
SEGMENT_TOKENS = MAX_RECORD_TOKENS
PROBE_SCHEMA = "emender-e97-document-nll-probe-panel-v1"
PROBE_SALAD = "e97-document-nll-probe-v1"

BOOK_SOURCES = (
    "project_gutenberg",
    "pre_1929_books",
    "doab",
    "biodiversity_heritage_library",
    "library_of_congress",
    "pressbooks",
    "public_domain_review",
)
LONGFORM_SOURCES = (
    "arxiv_papers",
    "caselaw_access_project",
    "peS2o",
    "ubuntu_irc",
)
PROBE_SOURCES = BOOK_SOURCES[:2] + (
    "arxiv_papers",
    "caselaw_access_project",
    "ubuntu_irc",
    "libretexts",
    "news",
    "stackexchange",
)
# Cohort A token quotas (documents are taken longest-first per source until the
# quota is met). Books carry ~2/3 of the mass per the operator's emphasis.
DEFAULT_QUOTAS = {
    "project_gutenberg": 8_000_000,
    "pre_1929_books": 12_000_000,
    "doab": 6_000_000,
    "biodiversity_heritage_library": 2_000_000,
    "library_of_congress": 8_000_000,
    "pressbooks": 2_000_000,
    "public_domain_review": 300_000,
    "arxiv_papers": 6_000_000,
    "caselaw_access_project": 5_000_000,
    "peS2o": 6_000_000,
    "ubuntu_irc": 4_000_000,
}
HISTOGRAM_BUCKETS = (
    20_000, 30_000, 40_000, 60_000, 80_000, 100_000, 120_000, 128_001,
)


def _stable_hash(*parts: object) -> bytes:
    payload = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).digest()


def _shards(root: Path, source: str) -> list[Path]:
    shards = sorted((root / source).glob("*.jsonl.gz"))
    if not shards:
        raise RuntimeError(f"source contains no jsonl.gz shards: {source}")
    return shards


def _verify_checkout(root: Path) -> None:
    if _git(root, "rev-parse", "HEAD") != DATASET_REVISION:
        raise SystemExit("CommaPile checkout revision mismatch")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("CommaPile checkout has modified tracked files")


def _verify_tokenizer_cache() -> None:
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_root:
        raise SystemExit("TIKTOKEN_CACHE_DIR must bind the verified p50k cache")
    cache_object = Path(cache_root) / TOKENIZER_CACHE_KEY
    if not cache_object.is_file() or sha256(cache_object) != TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache object is missing or corrupt")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def _index_shard(job: tuple) -> tuple:
    """Scan one shard; books keep every document over the floor, long-form
    sources keep the top-K longest via a bounded min-heap."""
    source, shard_number, shard_path, book, keep, book_min_chars = job
    import gzip as _gzip
    kept = []
    heap: list = []
    scanned = 0
    errors = 0
    with _gzip.open(shard_path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                continue
            chars = len(text)
            if book:
                if chars > book_min_chars:
                    kept.append((chars, shard_number, line_number))
            elif keep > 0:
                entry = (chars, shard_number, line_number)
                if len(heap) < keep:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    if not book:
        kept = sorted(heap, reverse=True)
    return source, shard_number, scanned, errors, kept


def _cmd_index(args: argparse.Namespace) -> None:
    _verify_checkout(args.input_root)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp

    sources = sorted(set(args.source))
    jobs = []
    for source in sources:
        book = source in BOOK_SOURCES
        keep = 0 if book else args.longform_keep
        for shard_number, shard in enumerate(_shards(args.input_root, source)):
            jobs.append((source, shard_number, str(shard), book, keep,
                         args.book_min_chars))

    with mp.Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_index_shard, jobs, chunksize=1))
    per_source: dict[str, dict[int, list]] = {}
    stats: dict[str, dict] = {}
    for source, shard_number, scanned, errors, kept in results:
        per_source.setdefault(source, {})[shard_number] = kept
        entry = stats.setdefault(source, {
            "shards": 0, "scanned_records": 0, "json_errors": 0,
            "candidates": 0, "candidate_chars": 0, "candidate_min_chars": None,
            "candidate_max_chars": 0,
        })
        entry["shards"] += 1
        entry["scanned_records"] += scanned
        entry["json_errors"] += errors
        entry["candidates"] += len(kept)
        entry["candidate_chars"] += sum(c for c, _, _ in kept)
        mins = [c for c, _, _ in kept]
        if mins:
            current = entry["candidate_min_chars"]
            entry["candidate_min_chars"] = min(current, min(mins)) if current else min(mins)
            entry["candidate_max_chars"] = max(entry["candidate_max_chars"], max(mins))
    for source in sources:
        path = args.index_dir / f"{source}.candidates.jsonl"
        merged = []
        for shard_number in sorted(per_source.get(source, {})):
            merged.extend(per_source[source][shard_number])
        merged.sort(key=lambda item: (-item[0], item[1], item[2]))
        with path.open("w") as out:
            for chars, shard_number, line_number in merged:
                out.write(json.dumps({
                    "source": source,
                    "shard": shard_number,
                    "line": line_number,
                    "chars": chars,
                }, sort_keys=True) + "\n")
        stats[source]["source"] = source
    (args.index_dir / "index-stats.json").write_text(
        json.dumps({"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                    "book_min_chars": args.book_min_chars,
                    "longform_keep": args.longform_keep, "sources": stats},
                   indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index_dir": str(args.index_dir), "sources": stats},
                     sort_keys=True))


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def _histogram(chars_list: list[int]) -> dict[str, int]:
    buckets = {f"<{limit}": 0 for limit in HISTOGRAM_BUCKETS}
    buckets[f">={HISTOGRAM_BUCKETS[-2]}"] = 0
    for chars in chars_list:
        placed = False
        for limit in HISTOGRAM_BUCKETS:
            if chars < limit:
                buckets[f"<{limit}"] += 1
                placed = True
                break
        if not placed:
            buckets[f">={HISTOGRAM_BUCKETS[-2]}"] += 1
    return {key: value for key, value in buckets.items() if value}


def _cmd_select(args: argparse.Namespace) -> None:
    quotas = dict(DEFAULT_QUOTAS)
    if args.quota:
        for item in args.quota:
            source, _, value = item.partition("=")
            if source not in DEFAULT_QUOTAS:
                raise SystemExit(f"unknown quota source: {source}")
            quotas[source] = int(value)
    manifest = {
        "schema": "emender-e97-long-document-anchor-selection-v1",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "book_sources": list(BOOK_SOURCES),
        "longform_sources": list(LONGFORM_SOURCES),
        "book_min_chars": args.book_min_chars,
        "selection_rule": ("per source, candidates ranked by document length "
                          "descending (then shard, line); taken in rank order "
                          "until the source token quota is met"),
        "est_chars_per_token": args.chars_per_token,
        "quota_tokens": quotas,
        "est_quotas_note": ("token quotas are exact at build time; this "
                            "manifest estimates candidate ceilings with a "
                            f"1+{args.margin:.2f} margin"),
        "sources": {},
    }
    for source in (*BOOK_SOURCES, *LONGFORM_SOURCES):
        path = args.index_dir / f"{source}.candidates.jsonl"
        candidates = [json.loads(line) for line in path.open()]
        chars = [item["chars"] for item in candidates]
        budget_chars = quotas[source] * args.chars_per_token * (1 + args.margin)
        selected = []
        running = 0
        for item in candidates:
            if running >= budget_chars:
                break
            selected.append(item)
            running += item["chars"]
        manifest["sources"][source] = {
            "quota_tokens": quotas[source],
            "candidates_indexed": len(candidates),
            "candidate_ceiling": len(selected),
            "candidate_ceiling_chars": running,
            "candidate_ceiling_est_tokens": round(running / args.chars_per_token),
            "min_chars": min(chars) if chars else None,
            "max_chars": max(chars) if chars else None,
            "median_chars": sorted(chars)[len(chars) // 2] if chars else None,
            "histogram_chars": _histogram(chars[:args.histogram_sample]),
        }
    manifest_path = args.output
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selection_manifest": str(manifest_path),
                      "manifest_sha256": sha256(manifest_path)}, sort_keys=True))


# ---------------------------------------------------------------------------
# freeze-probe
# ---------------------------------------------------------------------------

def _read_record(root: Path, source: str, shard: int, line: int) -> dict:
    shards = _shards(root, source)
    if shard >= len(shards):
        raise RuntimeError(f"shard index {shard} out of range for {source}")
    with gzip.open(shards[shard], "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, row in _iter_rows_shard(stream):
            if line_number == line:
                return row
    raise RuntimeError(f"probe record not found: {source}#{shard}:{line}")


def _iter_rows(shard_path: Path):
    """Stream (line_number, row) pairs from one shard. JSON decode failures
    yield an ``{'_error': ...}`` row so the build can count and exclude the
    record (pass-1 indexing skipped those lines, so hits should not occur)."""
    with gzip.open(shard_path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {"_error": str(exc)}
                continue
            yield line_number, row


def _iter_rows_shard(stream):
    for line_number, line in enumerate(stream, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        yield line_number, row


def _cmd_freeze_probe(args: argparse.Namespace) -> None:
    _verify_checkout(args.input_root)
    _verify_tokenizer_cache()
    encoding = tiktoken.get_encoding(TOKENIZER)
    args.panel_root.mkdir(parents=True, exist_ok=True)
    panel_rows = []
    for source in PROBE_SOURCES:
        path = args.index_dir / f"{source}.candidates.jsonl"
        candidates = [json.loads(line) for line in path.open()][:args.probe_top]
        if not candidates:
            raise SystemExit(f"probe source has no candidates: {source}")
        ranked = sorted(
            candidates,
            key=lambda item: _stable_hash(PROBE_SALAD, source, item["shard"], item["line"]),
        )
        chosen = ranked[0]
        row = _read_record(args.input_root, source, chosen["shard"], chosen["line"])
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise SystemExit(f"probe record missing text: {source}")
        token_values = encoding.encode_ordinary(text)
        kept = token_values[: args.max_scored_tokens + 1]
        panel_rows.append({
            "id": f"doc-nll:{source}",
            "source": source,
            "shard": chosen["shard"],
            "line": chosen["line"],
            "chars": len(text),
            "full_tokens": len(token_values),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "panel_tokens": len(kept),
            "scored_tokens": len(kept) - 1,
            "truncated": len(token_values) > len(kept),
            "rule": f"hash-ranked choice among the top {args.probe_top} longest "
                    f"documents of {source} (salad {PROBE_SALAD})",
        })
        with (args.panel_root / "probe-panel.token-ids.jsonl").open("a") as out:
            out.write(json.dumps({"id": panel_rows[-1]["id"],
                                  "token_ids": kept}) + "\n")
    panel = {
        "schema": PROBE_SCHEMA,
        "purpose": ("frozen teacher-forced document-NLL retention probe: fixed "
                    "long documents, causal NLL per document, CPU GGML runner"),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "max_scored_tokens_per_doc": args.max_scored_tokens,
        "selection_rule": ("hash-ranked choice among the top-1000 longest "
                           "documents per source; two book eras via "
                           "project_gutenberg and pre_1929_books"),
        "documents": panel_rows,
    }
    manifest_path = args.panel_root / "probe-panel-manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"probe_panel": str(manifest_path),
                      "manifest_sha256": sha256(manifest_path)}, sort_keys=True))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _probe_text_sha256s(panel_manifest: Path) -> set[str]:
    panel = json.loads(panel_manifest.read_text())
    if panel.get("schema") != PROBE_SCHEMA:
        raise SystemExit("unsupported probe panel manifest")
    return {row["text_sha256"] for row in panel["documents"]}


def _cmd_build(args: argparse.Namespace) -> None:
    _verify_checkout(args.input_root)
    _verify_tokenizer_cache()
    encoding = tiktoken.get_encoding(TOKENIZER)
    selection = json.loads(args.selection.read_text())
    if selection.get("schema") != "emender-e97-long-document-anchor-selection-v1":
        raise SystemExit("unsupported selection manifest")
    quotas = selection["quota_tokens"]
    holdout_manifest = args.holdout_root / "authority" / "manifest.json"
    if sha256(holdout_manifest) != args.holdout_manifest_sha256:
        raise SystemExit("repository holdout manifest digest mismatch")
    holdout_shingles, holdout_identifiers = _holdout_signatures(args.holdout_root)
    probe_sha256s = _probe_text_sha256s(args.probe_panel)
    sources = (*BOOK_SOURCES, *LONGFORM_SOURCES)
    candidates: dict[str, list[dict]] = {}
    for source in sources:
        path = args.index_dir / f"{source}.candidates.jsonl"
        candidates[source] = [json.loads(line) for line in path.open()]

    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    counts: Counter = Counter()
    source_receipts: dict[str, dict] = {}
    touched_shards: dict[str, set[int]] = {source: set() for source in sources}
    output_offset = 0
    selected_text_sha256s: set[str] = set()

    try:
        with (outputs["tokens"].open("wb", buffering=4 << 20) as token_out,
              outputs["mask"].open("wb", buffering=4 << 20) as mask_out,
              outputs["index"].open("wb", buffering=1 << 20) as index_out,
              outputs["metadata"].open("w", buffering=1 << 20) as metadata_out):
            for source in sources:
                shards = _shards(args.input_root, source)
                source_counts: Counter = Counter()
                accepted_tokens = 0
                # Group candidate lines by shard so each touched shard is
                # decompressed exactly once (candidates are in rank order).
                by_shard: dict[int, dict[int, dict]] = {}
                stop = False
                for rank, item in enumerate(candidates[source], 1):
                    if stop:
                        break
                    by_shard.setdefault(item["shard"], {})[item["line"]] = {
                        **item, "rank": rank}
                pending: list[dict] = []
                for shard_number in sorted(by_shard):
                    if stop:
                        break
                    wanted = by_shard[shard_number]
                    touched_shards[source].add(shard_number)
                    shard_name = shards[shard_number].name
                    for line_number, row in _iter_rows(shards[shard_number]):
                        hit = wanted.get(line_number)
                        if hit is None:
                            continue
                        del wanted[line_number]
                        if "_error" in row:
                            source_counts["excluded_json_errors"] += 1
                            continue
                        text = row.get("text")
                        if not isinstance(text, str) or not text:
                            source_counts["excluded_empty_records"] += 1
                            continue
                        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
                        if text_sha256 in probe_sha256s:
                            source_counts["excluded_probe_panel_text"] += 1
                            continue
                        if text_sha256 in selected_text_sha256s:
                            source_counts["excluded_duplicate_text_sha256"] += 1
                            continue
                        reason = _overlap_reason(
                            text, holdout_shingles, holdout_identifiers)
                        if reason is not None:
                            source_counts[f"excluded_overlap:{reason}"] += 1
                            continue
                        pending.append({
                            "rank": hit["rank"], "shard": shard_number,
                            "shard_name": shard_name, "line": line_number,
                            "text": text, "text_sha256": text_sha256,
                        })
                        if not wanted:
                            break
                pending.sort(key=lambda item: item["rank"])
                for item in pending:
                    if accepted_tokens >= quotas[source]:
                        stop = True
                        break
                    token_values = encoding.encode_ordinary(item["text"])
                    token_values.append(EOT_TOKEN)
                    total_tokens = len(token_values)
                    segments = max(1, -(-total_tokens // SEGMENT_TOKENS))
                    for segment in range(segments):
                        chunk = token_values[segment * SEGMENT_TOKENS:
                                             (segment + 1) * SEGMENT_TOKENS]
                        token_count = len(chunk)
                        identity = (f"{source}:{item['shard_name']}:{item['line']}"
                                    f":rank{item['rank']}")
                        if segments > 1:
                            identity += f":segment{segment + 1}of{segments}"
                        effective_targets = token_count - 1
                        token_out.write(struct.pack(f"<{token_count}I", *chunk))
                        # Token zero has no same-document predecessor; keeping
                        # it unsupervised matches the commapile document-causal
                        # authority and stays valid under boundary-aware packs.
                        mask_out.write(b"\x00" + bytes([1]) * effective_targets)
                        split = int.from_bytes(
                            _stable_hash("commapile-doc-anchor-v1", identity),
                            "little") % 100 == 0
                        index_out.write(RECORD_INDEX.pack(
                            output_offset, token_count, effective_targets, int(split)))
                        metadata_out.write(json.dumps({
                            "id": identity,
                            "source": f"commapile:{source}",
                            "source_name": source,
                            "source_shard": f"{source}/{item['shard_name']}",
                            "source_line": item["line"],
                            "source_rank": item["rank"],
                            "text_sha256": item["text_sha256"],
                            "segment": segment + 1 if segments > 1 else None,
                            "segments": segments if segments > 1 else None,
                            "split": int(split),
                            "tokens": token_count,
                            "causal_targets": effective_targets,
                        }, sort_keys=True) + "\n")
                        output_offset += token_count
                        accepted_tokens += token_count
                        source_counts["records"] += 1
                        source_counts["segmented_records"] += int(segments > 1)
                        source_counts["tokens"] += token_count
                        source_counts["causal_targets"] += effective_targets
                        counts["records"] += 1
                        counts["tokens"] += token_count
                        counts["causal_targets"] += effective_targets
                        counts["validation_records" if split else "train_records"] += 1
                    selected_text_sha256s.add(item["text_sha256"])
                    source_counts["documents"] += 1
                source_receipts[source] = {
                    "quota_tokens": quotas[source],
                    "accepted_tokens": accepted_tokens,
                    **dict(sorted(source_counts.items())),
                }
                if accepted_tokens < quotas[source]:
                    raise RuntimeError(
                        f"{source}: candidate ceiling exhausted at {accepted_tokens} "
                        f"tokens before the {quotas[source]}-token quota")
        shard_receipts = []
        for source in sources:
            for shard_number in sorted(touched_shards[source]):
                shard_receipts.append(
                    _lfs_identity(args.input_root, _shards(args.input_root, source)[shard_number]))
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "training_eligible": True,
            "purpose": "long-document anchor library: whole-document long-tail "
                       "selection from CommaPile book/long-form sources",
            "cohort": "e97-long-document-anchor-library-v1",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_readme_sha256": sha256(args.input_root / "README.md"),
            "selection_manifest_sha256": sha256(args.selection),
            "probe_panel_manifest_sha256": sha256(args.probe_panel),
            "probe_exclusion": ("probe-panel document texts are excluded from "
                                "training so the frozen NLL panel measures drift "
                                "without train-on-probe contamination"),
            "tokenizer": TOKENIZER,
            "tokenizer_cache_sha256": TOKENIZER_SHA256,
            "book_sources": list(BOOK_SOURCES),
            "longform_sources": list(LONGFORM_SOURCES),
            "book_min_chars": selection["book_min_chars"],
            "selection_rule": selection["selection_rule"],
            "max_record_tokens": MAX_RECORD_TOKENS,
            "segmentation": ("documents longer than 65537 tokens are segmented "
                             "only at the 64K pack boundary: consecutive records "
                             "in reading order, reset per segment (the pre-training "
                             "window-start semantics); the final segment carries EOT"),
            "holdout_manifest_sha256": args.holdout_manifest_sha256,
            "holdout_content_shingle_width": 5,
            "holdout_content_shingles": len(holdout_shingles),
            "holdout_identifiers": list(holdout_identifiers),
            "forbidden_sources": ["github_archive"],
            "splits": "1% deterministic hash validation split (commapile-doc-anchor-v1 salad)",
            "counts": {
                **dict(counts),
                "assistant_target_tokens": int(counts["causal_targets"]),
            },
            "sources": source_receipts,
            "input_shards": shard_receipts,
            "outputs": {name: {"path": path.name, "bytes": path.stat().st_size,
                               "sha256": sha256(path)}
                        for name, path in outputs.items()},
        }
        manifest_path = args.output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except BaseException:
        import shutil
        shutil.rmtree(args.output_root, ignore_errors=True)
        raise
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest},
                     sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("--input-root", type=Path, required=True)
    p_index.add_argument("--index-dir", type=Path, required=True)
    p_index.add_argument("--source", action="append",
                         default=[*BOOK_SOURCES, *LONGFORM_SOURCES,
                                  "libretexts", "news", "stackexchange"])
    p_index.add_argument("--book-min-chars", type=int, default=20_000)
    p_index.add_argument("--longform-keep", type=int, default=2_000,
                         help="top-K longest documents kept for non-book sources")
    p_index.add_argument("--workers", type=int, default=48)
    p_index.set_defaults(func=_cmd_index)

    p_select = sub.add_parser("select")
    p_select.add_argument("--index-dir", type=Path, required=True)
    p_select.add_argument("--output", type=Path, required=True)
    p_select.add_argument("--book-min-chars", type=int, default=20_000)
    p_select.add_argument("--chars-per-token", type=float, default=4.0)
    p_select.add_argument("--margin", type=float, default=0.25)
    p_select.add_argument("--histogram-sample", type=int, default=50_000)
    p_select.add_argument("--quota", action="append", default=[],
                          help="source=tokens override")
    p_select.set_defaults(func=_cmd_select)

    p_probe = sub.add_parser("freeze-probe")
    p_probe.add_argument("--input-root", type=Path, required=True)
    p_probe.add_argument("--index-dir", type=Path, required=True)
    p_probe.add_argument("--panel-root", type=Path, required=True)
    p_probe.add_argument("--probe-top", type=int, default=1_000)
    p_probe.add_argument("--max-scored-tokens", type=int, default=32_768)
    p_probe.set_defaults(func=_cmd_freeze_probe)

    p_build = sub.add_parser("build")
    p_build.add_argument("--input-root", type=Path, required=True)
    p_build.add_argument("--index-dir", type=Path, required=True)
    p_build.add_argument("--selection", type=Path, required=True)
    p_build.add_argument("--probe-panel", type=Path, required=True)
    p_build.add_argument("--holdout-root", type=Path, required=True)
    p_build.add_argument("--holdout-manifest-sha256", required=True)
    p_build.add_argument("--output-root", type=Path, required=True)
    p_build.set_defaults(func=_cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
