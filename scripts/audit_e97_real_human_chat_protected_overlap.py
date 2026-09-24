#!/usr/bin/env python3
"""Protected-panel entity-overlap audit for the e97 real-human-chat intake.

Same sealed fixed protected panels and the same entity domains as the
hybrid/session overlap audits (ndm.e97_protected_overlap machinery), adapted
to a raw conversation authority: the candidate side is the FULL decoded text
of the authority token stream. Records are decoded one at a time (whole-record
boundaries from records.idx, each record <= the 65,537-token pack ceiling) so
there is no cross-record token bleed and no normalization seam; the raw text
is the exact stream a reader would decode. Protected panel entities
(task ids, family ids, repositories, fixture paths, prompt templates, fixture
contents, exact scalars) are checked for containment in the raw text, and
their normalized forms in the per-record normalized text (same public
normalize_content as the protected side). EXACT SCALARS use the exact
set-mode semantics of the established audits: extract_exact_scalars is
applied to both sides (protected fixture contents; our per-record decoded
texts) and the sets are intersected, significant from 8 bytes up — prose
substring containment is not meaningful for scalar values (a common word or
one-line idiom inside a JSON/patch fixture is not a copied value). For the
containment domains (leak signals), the significance bars are a documented
adaptation of exact-and-significant-entity-v1: task ids, family ids,
repositories, fixture paths, prompt templates and FULL fixture contents are
collisions at any size; a full fixture content is significant from 16 bytes
up (the established content bar); a NORMALIZED fixture content is
significant only when its original fixture content is at least 64 bytes
(substantial snippet) — normalized forms of sub-64-byte fixtures are
single-line structural idioms (the trivial alpha/1 content class).
Trivial hits are counted in the receipt, never dropped silently. Fail closed
on any significant entity collision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import struct
from pathlib import Path

import tiktoken

from ndm.e97_protected_overlap import (
    _domains, extract_exact_scalars, load_protected_panel, normalize_content)
from scripts import build_e97_tulu3_sft as codec

SCHEMA = "emender-e97-real-human-chat-protected-overlap-audit-v1"
RECORD_INDEX = struct.Struct("<QQQB7x")
FIXED = (
    ("/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/manifest.json",
     "/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/records.jsonl"),
    ("/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/manifest.json",
     "/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/records.jsonl"),
    ("/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/manifest.json",
     "/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/tasks.jsonl"),
)
# significance bars: set-mode scalar bar of exact-and-significant-entity-v1;
# containment-mode content bars (see module docstring)
SIGNIFICANT_CONTENT_BYTES = 16
SIGNIFICANT_SCALAR_BYTES = 8
SIGNIFICANT_NORMALIZED_SOURCE_BYTES = 64
_TEXTS: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_authority(root: Path):
    """Verify the four payload outputs against the manifest; return layout."""
    manifest = json.loads((root / "manifest.json").read_text())
    paths = {}
    for key in ("tokens", "mask", "index", "metadata"):
        descriptor = manifest["outputs"][key]
        path = Path(descriptor["path"])
        if not path.is_absolute():
            path = root / path
        if path.stat().st_size != descriptor["bytes"] or sha(path) != descriptor["sha256"]:
            raise ValueError(f"authority output identity: {key}")
        paths[key] = path
    index = Path(paths["index"]).read_bytes()
    metadata = Path(paths["metadata"]).read_text().splitlines()
    if len(index) != RECORD_INDEX.size * len(metadata):
        raise ValueError("authority shape")
    records = [RECORD_INDEX.unpack_from(index, i * RECORD_INDEX.size)
               for i in range(len(metadata))]
    return manifest, paths, records


def decode_texts(root: Path, encoding):
    """Decode the authority stream per record into raw + normalized text."""
    manifest, paths, records = load_authority(root)
    tokens = Path(paths["tokens"]).read_bytes()
    raw_parts = []
    norm_parts = []
    scalar_set = set()
    decoded_bytes = 0
    for offset, length, _targets, _split in records:
        chunk = tokens[offset * 4:(offset + length) * 4]
        text = encoding.decode(struct.unpack(f"<{length}I", chunk))
        raw_parts.append(text)
        decoded_bytes += len(text.encode("utf-8"))
        scalar_set |= extract_exact_scalars([text])
        # normalize per record; split oversized records at line boundaries so
        # the bounded public normalize_content applies without seams
        if len(text.encode("utf-8")) <= (1 << 20):
            norm_parts.append(normalize_content(text))
        else:
            pieces = []
            start = 0
            while start < len(text):
                stop = min(start + (1 << 20), len(text))
                if stop < len(text):
                    newline = text.rfind("\n", start, stop)
                    stop = newline + 1 if newline != -1 else stop
                pieces.append(normalize_content(text[start:stop]))
                start = stop
            norm_parts.append("".join(pieces))
    return manifest, "".join(raw_parts), "\n".join(norm_parts), decoded_bytes, scalar_set


def _contains(job):
    domain, needle, haystack, significant = job
    return job, needle in _TEXTS[haystack]


def audit(args):
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_root:
        raise SystemExit("TIKTOKEN_CACHE_DIR must bind the verified p50k cache")
    cache_object = Path(cache_root) / codec.TOKENIZER_CACHE_KEY
    if not cache_object.is_file() or sha(cache_object) != codec.TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache object is missing or corrupt")
    encoding = tiktoken.get_encoding(codec.TOKENIZER)
    manifest, raw_text, norm_text, decoded_bytes, candidate_scalars = \
        decode_texts(args.candidate, encoding)
    _TEXTS["raw"] = raw_text
    _TEXTS["norm"] = norm_text
    protected = []
    manifest_shas = []
    record_shas = []
    for manifest_path, records_path in FIXED:
        msha, rsha, items = load_protected_panel(Path(manifest_path), Path(records_path))
        manifest_shas.append(msha)
        record_shas.append(rsha)
        protected.extend(items)
    domains = _domains(protected)
    original_by_normalized = {}
    for record in protected:
        for fixture in record["fixture_files"]:
            original_by_normalized[normalize_content(fixture["content"])] = fixture["content"]
    jobs = []
    for domain in ("task_ids", "family_ids", "repositories",
                   "fixture_paths_full", "prompt_templates_full"):
        jobs += [(domain, needle, "raw", True) for needle in domains[domain]]
    for domain in ("fixture_paths_normalized", "prompt_templates_normalized"):
        jobs += [(domain, needle, "norm", True) for needle in domains[domain]]
    for needle in domains["fixture_contents_normalized"]:
        original = original_by_normalized.get(needle, "")
        significant = len(original.encode("utf-8")) >= SIGNIFICANT_NORMALIZED_SOURCE_BYTES
        jobs.append(("fixture_contents_normalized", needle, "norm", significant))
    for needle in domains["fixture_contents_full"]:
        significant = len(needle.encode("utf-8")) >= SIGNIFICANT_CONTENT_BYTES
        jobs.append(("fixture_contents_full", needle, "raw", significant))
    scalar_hits = candidate_scalars & domains["exact_scalars"]
    scalar_trivial = {value for value in scalar_hits
                      if len(value.encode("utf-8")) < SIGNIFICANT_SCALAR_BYTES}
    scalar_collisions = scalar_hits - scalar_trivial
    found = {}
    trivial = {}
    if scalar_trivial:
        trivial["exact_scalars"] = len(scalar_trivial)
    if scalar_collisions:
        found["exact_scalars"] = len(scalar_collisions)
    with mp.Pool(args.workers) as pool:
        for job, hit in pool.imap_unordered(_contains, jobs, chunksize=16):
            domain, _needle, _haystack, significant = job
            bucket = found if significant else trivial
            if hit:
                bucket[domain] = bucket.get(domain, 0) + 1
    collision = bool(found)
    receipt = {
        "schema": SCHEMA,
        "status": "pass" if not collision else "collisions",
        "candidate_manifest_sha256": sha(Path(args.candidate) / "manifest.json"),
        "candidate_records": manifest["counts"]["records"],
        "candidate_tokens": manifest["counts"]["tokens"],
        "candidate_decoded_bytes": decoded_bytes,
        "candidate_normalized_bytes": len(norm_text.encode("utf-8")),
        "candidate_method": ("per-record decode of the whole authority token stream "
                             "(whole-record boundaries, no cross-record bleed; raw text "
                             "is the exact stream), normalized side = per-record "
                             "normalize_content, same function as the protected side; "
                             "exact scalars by set-mode extract_exact_scalars on the "
                             "same per-record texts"),
        "fixed_protected_manifest_sha256s": sorted(manifest_shas),
        "fixed_protected_record_sha256s": sorted(record_shas),
        "entity_collision_counts": {domain: found.get(domain, 0)
                                    for domain in sorted(set(j[0] for j in jobs) | {"exact_scalars"})},
        "trivial_entity_counts": {domain: trivial.get(domain, 0)
                                  for domain in sorted(set(j[0] for j in jobs) | {"exact_scalars"})},
        "entity_policy": ("exact-and-significant-entity-v1: exact scalars use "
                          "the established set-mode semantics (extract_exact_scalars "
                          "on both sides — protected fixture contents and our "
                          "per-record decoded texts — set intersection, significant "
                          ">=8 bytes); containment domains (the leak signals) adapt "
                          "the bars for raw-text substring containment: task ids, "
                          "family ids, repositories, fixture paths and prompt "
                          "templates collide at any size; full fixture contents from "
                          "16 bytes; normalized fixture contents only when the "
                          "original fixture is >=64 bytes (substantial snippet) — "
                          "shorter normalized forms are single-line structural "
                          "idioms (trivial alpha/1 content class); trivial hits are "
                          "reported in trivial_entity_counts, never dropped "
                          "silently"),
        "checker_sha256": sha(Path(__file__)),
        "training_eligible": False,
        "packing_authorized": False,
        "optimizer_updates_authorized": 0,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("REAL_HUMAN_CHAT_PROTECTED_OVERLAP", receipt["status"],
          json.dumps(receipt["entity_collision_counts"], sort_keys=True),
          sha(args.output))
    if collision:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True,
                        help="real-human-chat authority root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
