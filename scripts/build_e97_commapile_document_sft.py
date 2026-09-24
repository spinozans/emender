#!/usr/bin/env python3
"""Build a source-filtered, document-complete CommaPile causal authority.

The combined 1 TB byte stream intentionally omits source labels, so it cannot
support repository-exclusion claims. This builder instead reads immutable
source shards from the pinned Hugging Face checkout, rejects GitHub and frozen
holdout overlap, tokenizes whole documents, and fills source quotas only after
64K eligibility filtering.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import struct
import subprocess
from typing import Iterable

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
try:  # Support both ``python -m`` and the documented script-path invocation.
    from scripts import build_e97_tulu3_sft as codec
    from scripts.build_commapile_mainmix import COMMA_MAIN_EFFECTIVE_TOKENS_B, stable_seed
except ModuleNotFoundError:  # pragma: no cover - exercised by subprocess CLIs
    import build_e97_tulu3_sft as codec
    from build_commapile_mainmix import COMMA_MAIN_EFFECTIVE_TOKENS_B, stable_seed

DATASET_ID = "common-pile/comma_v0.1_training_dataset"
DATASET_REVISION = "5afc546db324e7f39f297ba757c9a60547151e7c"
TOKENIZER = "p50k_base"
SCHEMA = AUTHORITY_SCHEMA


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True).strip()


def _lfs_identity(root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    pointer = _git(root, "show", f"{DATASET_REVISION}:{relative}")
    oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.MULTILINE)
    size_match = re.search(r"^size ([0-9]+)$", pointer, re.MULTILINE)
    if oid_match is None or size_match is None:
        raise RuntimeError(f"tracked shard is not an immutable Git LFS object: {relative}")
    expected_size = int(size_match.group(1))
    expected_sha = oid_match.group(1)
    if path.stat().st_size != expected_size or sha256(path) != expected_sha:
        raise RuntimeError(f"Git LFS payload mismatch: {relative}")
    return {"path": relative, "bytes": expected_size, "sha256": expected_sha}


def _normalized_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _shingles(text: str, width: int = 5) -> set[str]:
    lines = _normalized_lines(text)
    result = set()
    for start in range(0, max(0, len(lines) - width + 1)):
        value = "\n".join(lines[start:start + width])
        if len(value) >= 160:
            result.add(hashlib.sha256(value.encode()).hexdigest())
    return result


def _holdout_signatures(root: Path) -> tuple[set[str], tuple[str, ...]]:
    manifest_path = root / "authority" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "emender-e97-real-repo-holdout-v1":
        raise RuntimeError("unsupported repository holdout schema")
    identifiers = set()
    for name, receipt in manifest["repositories"].items():
        identifiers.add(name.casefold())
        identifiers.add(receipt["url"].rsplit("/", 1)[-1].removesuffix(".git").casefold())
    signatures: set[str] = set()
    for path in sorted((root / "sources").rglob("*")):
        if not path.is_file() or ".git" in path.parts or path.stat().st_size > (8 << 20):
            continue
        try:
            signatures.update(_shingles(path.read_text(errors="replace")))
        except OSError:
            continue
    if not signatures:
        raise RuntimeError("repository holdout produced no content-overlap signatures")
    return signatures, tuple(sorted(identifiers))


def _overlap_reason(
    text: str, holdout_shingles: set[str], identifiers: tuple[str, ...],
) -> str | None:
    folded = text.casefold()
    for identity in identifiers:
        if identity in folded:
            return f"repository_identity:{identity}"
    if _shingles(text) & holdout_shingles:
        return "repository_content_shingle"
    return None


def _iter_source_rows(
    root: Path, source: str, seed: int, max_epochs: int | None = None,
) -> Iterable[tuple[int, Path, int, dict]]:
    shards = sorted((root / source).glob("*.jsonl.gz"))
    if not shards:
        raise RuntimeError(f"source contains no jsonl.gz shards: {source}")
    epoch = 0
    while max_epochs is None or epoch < max_epochs:
        order = list(shards)
        random.Random(stable_seed(seed, source, epoch)).shuffle(order)
        for shard in order:
            with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        yield epoch, shard, line_number, {"_error": "json"}
                        continue
                    yield epoch, shard, line_number, row
        epoch += 1


def _load_excluded_authorities(
    roots: list[Path], manifest_sha256s: list[str],
) -> tuple[set[tuple[str, str, int]], set[str], list[dict[str, object]]]:
    if len(roots) != len(manifest_sha256s):
        raise RuntimeError(
            "exclude-authority-root and exclude-authority-manifest-sha256 counts differ")
    identities: set[tuple[str, str, int]] = set()
    text_sha256s: set[str] = set()
    receipts = []
    for root, expected_sha256 in zip(roots, manifest_sha256s, strict=True):
        manifest_path = root / "manifest.json"
        if sha256(manifest_path) != expected_sha256:
            raise RuntimeError(f"excluded authority manifest SHA-256 mismatch: {root}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise RuntimeError(f"unsupported excluded authority: {root}")
        if manifest.get("training_eligible") is not True:
            raise RuntimeError(f"excluded authority is non-trainable or lacks explicit training eligibility: {root}")
        metadata_receipt = manifest["outputs"]["metadata"]
        metadata_path = root / Path(metadata_receipt["path"]).name
        if (metadata_path.stat().st_size != int(metadata_receipt["bytes"])
                or sha256(metadata_path) != metadata_receipt["sha256"]):
            raise RuntimeError(f"excluded authority metadata mismatch: {root}")
        before = len(identities)
        with metadata_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                identities.add((
                    str(record["source_name"]), str(record["source_shard"]),
                    int(record["source_line"])))
                text_sha256s.add(str(record["text_sha256"]))
        receipts.append({
            "root": str(root.resolve()),
            "manifest_sha256": expected_sha256,
            "metadata_sha256": metadata_receipt["sha256"],
            "records": len(identities) - before,
        })
    return identities, text_sha256s, receipts


def _entry(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--holdout-manifest-sha256", required=True)
    parser.add_argument("--include-source", action="append", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-record-tokens", type=int, default=65_537)
    parser.add_argument("--seed", type=int, default=974120)
    parser.add_argument("--exclude-authority-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--exclude-authority-manifest-sha256", action="append", default=[])
    args = parser.parse_args()
    sources = tuple(sorted(set(args.include_source)))
    if len(sources) != len(args.include_source):
        raise SystemExit("include-source values must be unique")
    unknown = set(sources) - set(COMMA_MAIN_EFFECTIVE_TOKENS_B)
    if unknown:
        raise SystemExit(f"unknown CommaPile sources: {sorted(unknown)}")
    if "github_archive" in sources:
        raise SystemExit("github_archive is forbidden for repository-holdout training")
    if args.target_tokens <= 0 or args.max_record_tokens < 2:
        raise SystemExit("target-tokens and max-record-tokens must be positive")
    if _git(args.input_root, "rev-parse", "HEAD") != DATASET_REVISION:
        raise SystemExit("CommaPile checkout revision mismatch")
    if _git(args.input_root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("CommaPile checkout has modified tracked files")
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_root:
        raise SystemExit("TIKTOKEN_CACHE_DIR must bind the verified p50k cache")
    cache_object = Path(cache_root) / codec.TOKENIZER_CACHE_KEY
    if not cache_object.is_file() or sha256(cache_object) != codec.TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache object is missing or corrupt")
    holdout_manifest = args.holdout_root / "authority" / "manifest.json"
    if sha256(holdout_manifest) != args.holdout_manifest_sha256:
        raise SystemExit("repository holdout manifest digest mismatch")
    holdout_shingles, holdout_identifiers = _holdout_signatures(args.holdout_root)
    excluded_identities, excluded_text_sha256s, exclusion_receipts = (
        _load_excluded_authorities(
            args.exclude_authority_root, args.exclude_authority_manifest_sha256))
    encoding = tiktoken.get_encoding(TOKENIZER)

    total_weight = sum(COMMA_MAIN_EFFECTIVE_TOKENS_B[source] for source in sources)
    quotas = {
        source: max(1, round(args.target_tokens * COMMA_MAIN_EFFECTIVE_TOKENS_B[source] / total_weight))
        for source in sources
    }
    # Preserve the exact requested aggregate by assigning rounding drift to the
    # highest-weight source. Actual output may exceed quotas by one whole doc.
    drift = args.target_tokens - sum(quotas.values())
    quotas[max(sources, key=COMMA_MAIN_EFFECTIVE_TOKENS_B.__getitem__)] += drift

    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    counts = Counter()
    source_receipts = {}
    touched_shards: set[Path] = set()
    output_offset = 0
    selected_text_sha256s: set[str] = set()
    try:
        with (outputs["tokens"].open("wb") as token_out,
              outputs["mask"].open("wb") as mask_out,
              outputs["index"].open("wb") as index_out,
              outputs["metadata"].open("w") as metadata_out):
            for source in sources:
                accepted_targets = 0
                source_counts = Counter()
                for epoch, shard, line_number, row in _iter_source_rows(
                        args.input_root, source, args.seed, max_epochs=1):
                    source_counts["input_records"] += 1
                    touched_shards.add(shard)
                    source_shard = shard.relative_to(args.input_root).as_posix()
                    if (source, source_shard, line_number) in excluded_identities:
                        source_counts["excluded_prior_record_identity"] += 1
                        continue
                    if "_error" in row:
                        source_counts["json_errors"] += 1
                        continue
                    text = row.get("text")
                    if not isinstance(text, str) or not text:
                        source_counts["empty_records"] += 1
                        continue
                    text_sha256 = hashlib.sha256(text.encode()).hexdigest()
                    if text_sha256 in excluded_text_sha256s:
                        source_counts["excluded_prior_text_sha256"] += 1
                        continue
                    if text_sha256 in selected_text_sha256s:
                        source_counts["excluded_duplicate_text_sha256"] += 1
                        continue
                    reason = _overlap_reason(
                        text, holdout_shingles, holdout_identifiers)
                    if reason is not None:
                        source_counts[f"excluded_overlap:{reason}"] += 1
                        continue
                    token_values = encoding.encode_ordinary(text)
                    token_values.append(encoding.eot_token)
                    token_count = len(token_values)
                    if token_count > args.max_record_tokens:
                        source_counts["excluded_oversize_records"] += 1
                        source_counts["excluded_oversize_tokens"] += token_count
                        continue
                    if token_count < 2:
                        source_counts["excluded_short_records"] += 1
                        continue
                    identity = f"{source}:{shard.name}:{line_number}:{epoch}"
                    split = int.from_bytes(hashlib.sha256(
                        f"commapile-doc-v1\0{identity}".encode()).digest()[:8], "little") % 100 == 0
                    effective_targets = token_count - 1
                    token_out.write(struct.pack(f"<{token_count}I", *token_values))
                    # Token zero has no same-document predecessor. Keeping it
                    # unsupervised makes this authority valid under both legacy
                    # per-record execution and boundary-aware packing.
                    mask_out.write(b"\x00" + bytes([1]) * effective_targets)
                    index_out.write(RECORD_INDEX.pack(
                        output_offset, token_count, effective_targets, int(split)))
                    metadata_out.write(json.dumps({
                        "id": identity,
                        "source": f"commapile:{source}",
                        "source_name": source,
                        "source_shard": source_shard,
                        "source_line": line_number,
                        "source_epoch": epoch,
                        "text_sha256": text_sha256,
                        "split": int(split),
                        "tokens": token_count,
                        "causal_targets": effective_targets,
                    }, sort_keys=True) + "\n")
                    selected_text_sha256s.add(text_sha256)
                    output_offset += token_count
                    accepted_targets += effective_targets
                    source_counts["records"] += 1
                    source_counts["tokens"] += token_count
                    source_counts["causal_targets"] += effective_targets
                    counts["records"] += 1
                    counts["tokens"] += token_count
                    counts["causal_targets"] += effective_targets
                    counts["validation_records" if split else "train_records"] += 1
                    if accepted_targets >= quotas[source]:
                        break
                if accepted_targets < quotas[source]:
                    raise RuntimeError(
                        f"{source}: requested {quotas[source]} fresh causal targets but "
                        f"only {accepted_targets} survived one no-replacement source pass")
                source_receipts[source] = {
                    "requested_causal_targets": quotas[source],
                    **dict(sorted(source_counts.items())),
                }

        shard_receipts = [_lfs_identity(args.input_root, path)
                          for path in sorted(touched_shards)]
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "training_eligible": True,
            "purpose": "source-filtered document-complete CommaPile causal replay",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_readme_sha256": sha256(args.input_root / "README.md"),
            "tokenizer": TOKENIZER,
            "tokenizer_cache_sha256": codec.TOKENIZER_SHA256,
            "seed": args.seed,
            "max_record_tokens": args.max_record_tokens,
            "included_sources": list(sources),
            "forbidden_sources": ["github_archive"],
            "quota_basis": "causal targets after whole-document eligibility and overlap filtering",
            "requested_causal_targets": args.target_tokens,
            "holdout_manifest_sha256": args.holdout_manifest_sha256,
            "holdout_content_shingle_width": 5,
            "holdout_content_shingles": len(holdout_shingles),
            "holdout_identifiers": list(holdout_identifiers),
            "sampling_mode": "without-replacement",
            "excluded_authorities": exclusion_receipts,
            "counts": {
                **dict(counts),
                "assistant_target_tokens": int(counts["causal_targets"]),
            },
            "sources": source_receipts,
            "input_shards": shard_receipts,
            "outputs": {name: _entry(path) for name, path in outputs.items()},
        }
        manifest_path = args.output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except BaseException:
        for path in outputs.values():
            path.unlink(missing_ok=True)
        try:
            args.output_root.rmdir()
        except OSError:
            pass
        raise
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
