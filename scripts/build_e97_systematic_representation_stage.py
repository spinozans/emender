#!/usr/bin/env python3
"""Build the sealed token-matched E97 action/private-analysis SFT stage.

The two arms share exact source records outside Open-SWE and the exact same
Open-SWE trajectory identity set.  Private analysis sees one occurrence of
selected trajectories; action-only deterministically repeats a subset so the
Open-SWE assistant-target total is exactly matched when possible and otherwise
minimizes the integral residual.  No record is split or truncated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Iterable

import numpy as np

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_atomic import publish_directory_no_replace

SCHEMA = "emender-e97-systematic-representation-stage-v1"
SEED = 9_740_121
MAX_RECORD_TOKENS = 65_537
MATCHED_VALIDATION_SHA256 = "08541a03dc041e91cc4317209537ee735928c2d5406a760533aaace4b802101d"
MATCHED_PACK_VALIDATION_SHA256 = "9b3745f844896792dac2ff79a2db58d52ce9fc61d48616c017e883c67e7bc4ee"
MATCHED_TRAJECTORY_SET_SHA256 = "b993eb9b4fc013074f9d0f6eeacf8763455ea73428abecbd58cce0769e6f2a81"

# Exact reviewed inputs.  Legacy None eligibility predates the mandatory field;
# this builder supersedes it only for these byte identities.  The two Open-SWE
# candidates require the cross-representation receipts above.
SOURCES = {
    "private-analysis": (
        Path("/mnt/nvme2n1/erikg/sft/e97-4b-open-swe-private-analysis-64k-v3"),
        "248a02e6d977b83474eba6e48589a691e4fc36115602c931cfac4960b73dc196"),
    "action-only": (
        Path("/mnt/nvme2n1/erikg/sft/e97-4b-open-swe-matched-action-64k-v3"),
        "f1226ebc6f74cbd528887ff742d28d4a65ea42207d02e676e9dc849cda961726"),
    "conversation": (
        Path("/mnt/nvme1n1/erikg/sft/e97-4b-smoltalk2-admitted-v1"),
        "d64f51abc615c097910900ffa7fc88f020ae5567b89958f607008078c271c7f6"),
    "compositional": (
        Path("/mnt/nvme1n1/erikg/sft/e97-4b-pi-compositional-retention-mix-v1"),
        "79e7981077fc387c3e3759728ddd0f9e1ac16f687c628542e8313fb735c4cfb7"),
    "core-retention": (
        Path("/mnt/nvme1n1/erikg/sft/e97-4b-pi-live-aligned-all-assistant-v1"),
        "4a1cf86f9089cc3f2f79884f845d26d487d88b20fb32b589a8095d4824bc6a20"),
    "documents": (
        Path("/mnt/nvme2n1/erikg/sft/e97-4b-commapile-document-causal-fresh-1b-v2"),
        "6fae902f9c3afa214c251384fa317dfb793abf01e844ff823c0c72b74e901d59"),
}
TARGETS = {
    "agent": 15_000_000,
    "conversation": 17_850_000,
    "compositional": 7_300_000,
    "core-retention": 2_350_000,
    "documents": 7_500_000,
}


@dataclass
class Source:
    name: str
    root: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    records: np.memmap
    token_handle: Any
    mask_handle: Any
    token_map: mmap.mmap
    mask_map: mmap.mmap

    def close(self) -> None:
        del self.records
        self.token_map.close()
        self.mask_map.close()
        self.token_handle.close()
        self.mask_handle.close()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def entry(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_receipt(path: Path, expected: str, schema: str) -> dict[str, Any]:
    if sha256(path) != expected:
        raise RuntimeError(f"receipt SHA-256 mismatch: {path}")
    value = json.loads(path.read_text())
    if value.get("schema") != schema or value.get("status") != "passed":
        raise RuntimeError(f"receipt is not a passing {schema}: {path}")
    return value


def output_path(root: Path, descriptor: Any, name: str) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{name}: invalid output descriptor")
    candidate = Path(descriptor["path"])
    if candidate.name != candidate.as_posix().rsplit("/", 1)[-1]:
        raise RuntimeError(f"{name}: invalid output path")
    path = root / candidate.name
    if (not path.is_file() or path.stat().st_size != int(descriptor["bytes"])
            or sha256(path) != descriptor["sha256"]):
        raise RuntimeError(f"{name}: output payload mismatch")
    return path


def open_source(name: str) -> Source:
    root, digest = SOURCES[name]
    manifest_path = root / "manifest.json"
    if sha256(manifest_path) != digest:
        raise RuntimeError(f"{name}: source manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != AUTHORITY_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError(f"{name}: unsupported source authority")
    if name in {"private-analysis", "action-only"}:
        if manifest.get("training_eligible") is not False:
            raise RuntimeError(f"{name}: matched candidate eligibility changed")
    elif manifest.get("training_eligible") is not None:
        raise RuntimeError(f"{name}: legacy source eligibility unexpectedly changed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"tokens", "mask", "index", "metadata"}:
        raise RuntimeError(f"{name}: invalid output set")
    paths = {key: output_path(root, outputs[key], f"{name}:{key}") for key in outputs}
    index_path = paths["index"]
    if index_path.stat().st_size % RECORD_INDEX.size:
        raise RuntimeError(f"{name}: record index is misaligned")
    records = np.memmap(index_path, mode="r", dtype=np.dtype([
        ("offset", "<u8"), ("tokens", "<u8"), ("targets", "<u8"),
        ("split", "u1"), ("pad", "V7")]))
    if len(records) != int(manifest["counts"]["records"]):
        raise RuntimeError(f"{name}: record count mismatch")
    token_handle = paths["tokens"].open("rb")
    mask_handle = paths["mask"].open("rb")
    return Source(
        name, root, digest, manifest, records, token_handle, mask_handle,
        mmap.mmap(token_handle.fileno(), 0, access=mmap.ACCESS_READ),
        mmap.mmap(mask_handle.fileno(), 0, access=mmap.ACCESS_READ))


def hash_order(namespace: str, value: str | int) -> bytes:
    return hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).digest()


def closest_prefix(items: Iterable[tuple[Any, int]], target: int) -> list[Any]:
    chosen: list[Any] = []
    total = 0
    for identity, weight in items:
        if weight <= 0:
            continue
        before = abs(target - total)
        after = abs(target - (total + weight))
        if total >= target and after >= before:
            break
        chosen.append(identity)
        total += weight
        if total >= target:
            break
    if not chosen:
        raise RuntimeError("target selection produced no records")
    return chosen


def affine_order(eligible: list[int], namespace: str) -> Iterable[int]:
    n = len(eligible)
    digest = hash_order(namespace, n)
    multiplier = int.from_bytes(digest[:8], "little") % n
    while math.gcd(multiplier, n) != 1:
        multiplier = (multiplier + 1) % n
    offset = int.from_bytes(digest[8:16], "little") % n
    for index in range(n):
        yield eligible[(multiplier * index + offset) % n]


def select_shared(source: Source, target: int) -> list[int]:
    eligible = [
        record_id for record_id, record in enumerate(source.records)
        if int(record["split"]) == 0 and 1 < int(record["tokens"]) <= MAX_RECORD_TOKENS
        and int(record["targets"]) > 0
    ]
    available = sum(int(source.records[i]["targets"]) for i in eligible)
    if available < target:
        raise RuntimeError(f"{source.name}: only {available} eligible targets for quota {target}")
    ordered = ((record_id, int(source.records[record_id]["targets"]))
               for record_id in affine_order(eligible, source.name))
    return closest_prefix(ordered, target)


def agent_groups(source: Source) -> dict[str, list[int]]:
    metadata_path = output_path(source.root, source.manifest["outputs"]["metadata"], f"{source.name}:metadata")
    groups: dict[str, list[int]] = defaultdict(list)
    rows = 0
    with metadata_path.open() as handle:
        for record_id, line in enumerate(handle):
            rows += 1
            value = json.loads(line)
            if int(source.records[record_id]["split"]) != int(value["split"]):
                raise RuntimeError(f"{source.name}: metadata split mismatch")
            if int(value["tokens"]) != int(source.records[record_id]["tokens"]):
                raise RuntimeError(f"{source.name}: metadata token mismatch")
            if int(value["targets"]) != int(source.records[record_id]["targets"]):
                raise RuntimeError(f"{source.name}: metadata target mismatch")
            if int(value["split"]) == 0:
                groups[str(value["trajectory_identity"])].append(record_id)
    if rows != len(source.records):
        raise RuntimeError(f"{source.name}: metadata/index count mismatch")
    return dict(groups)


def choose_second_pass(weights: dict[str, int], required: int) -> tuple[list[str], int]:
    order = sorted(weights, key=lambda identity: hash_order("agent-action-epoch-1", identity))
    chosen: list[str] = []
    total = 0
    cursor = 0
    while cursor < len(order) and total + weights[order[cursor]] <= required:
        chosen.append(order[cursor])
        total += weights[order[cursor]]
        cursor += 1
    residual = required - total
    remaining = order[cursor:]
    if residual == 0:
        return chosen, total
    by_weight: dict[int, str] = {}
    for identity in remaining:
        by_weight.setdefault(weights[identity], identity)
    if residual in by_weight:
        chosen.append(by_weight[residual])
        return chosen, required
    # Deterministically search an exact pair, then the closest single/pair.
    for identity in remaining:
        complement = residual - weights[identity]
        other = by_weight.get(complement)
        if other is not None and other != identity:
            chosen.extend((identity, other))
            return chosen, required
    best: tuple[int, tuple[str, ...], int] | None = None
    bounded = remaining[: min(4096, len(remaining))]
    for identity in bounded:
        value = weights[identity]
        candidate = (abs(required - (total + value)), (identity,), total + value)
        if best is None or candidate < best:
            best = candidate
    # Pair search through the bounded set using weights nearest each complement.
    sorted_weighted = sorted((weights[i], i) for i in bounded)
    import bisect
    values = [item[0] for item in sorted_weighted]
    for left, identity in sorted_weighted:
        wanted = residual - left
        at = bisect.bisect_left(values, wanted)
        for position in (at - 1, at, at + 1):
            if 0 <= position < len(sorted_weighted):
                right, other = sorted_weighted[position]
                if other == identity:
                    continue
                value = total + left + right
                candidate = (abs(required - value), tuple(sorted((identity, other))), value)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        return chosen, total
    chosen.extend(best[1])
    return chosen, best[2]


def payload(source: Source, record_id: int) -> tuple[bytes, bytes, int, int]:
    record = source.records[record_id]
    offset = int(record["offset"])
    tokens = int(record["tokens"])
    targets = int(record["targets"])
    token_bytes = source.token_map[offset * 4:(offset + tokens) * 4]
    mask_bytes = source.mask_map[offset:offset + tokens]
    if len(token_bytes) != tokens * 4 or len(mask_bytes) != tokens or sum(mask_bytes) != targets:
        raise RuntimeError(f"{source.name}:{record_id}: payload range/target mismatch")
    return token_bytes, mask_bytes, tokens, targets


def build_arm(
    destination: Path, representation: str, sources: dict[str, Source],
    agent_occurrences: list[tuple[str, int, int]], shared: dict[str, list[int]],
    receipts: dict[str, Any], recipe_sha256: str,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temp:
        stage = Path(temp) / "authority"
        stage.mkdir(mode=0o700)
        paths = {
            "tokens": stage / "tokens.uint32.bin",
            "mask": stage / "assistant_mask.uint8.bin",
            "index": stage / "records.idx",
            "metadata": stage / "records.jsonl",
        }
        counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0,
                  "train_records": 0, "validation_records": 0,
                  "unique_source_records": 0, "unique_trajectories": len({x[0] for x in agent_occurrences})}
        source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {
            "occurrences": 0, "unique_records": 0, "tokens": 0, "assistant_target_tokens": 0})
        unique_records: dict[str, set[int]] = defaultdict(set)
        offset = 0
        with (paths["tokens"].open("wb", buffering=16 << 20) as token_out,
              paths["mask"].open("wb", buffering=16 << 20) as mask_out,
              paths["index"].open("wb", buffering=4 << 20) as index_out,
              paths["metadata"].open("w", buffering=4 << 20) as metadata_out):
            ordered: list[tuple[str, int, int, str | None]] = [
                (representation, record_id, occurrence, trajectory)
                for trajectory, record_id, occurrence in agent_occurrences
            ]
            for name in ("conversation", "compositional", "core-retention", "documents"):
                ordered.extend((name, record_id, 0, None) for record_id in shared[name])
            for ordinal, (name, record_id, occurrence, trajectory) in enumerate(ordered):
                source = sources[name]
                token_bytes, mask_bytes, tokens, targets = payload(source, record_id)
                token_out.write(token_bytes)
                mask_out.write(mask_bytes)
                index_out.write(RECORD_INDEX.pack(offset, tokens, targets, 0))
                record_identity = hashlib.sha256(canonical({
                    "arm": representation, "ordinal": ordinal, "source_manifest_sha256": source.manifest_sha256,
                    "source_record_id": record_id, "occurrence": occurrence,
                })).hexdigest()
                metadata_out.write(json.dumps({
                    "id": f"systematic-{representation}-{ordinal:09d}",
                    "identity_sha256": record_identity,
                    "source": "agent" if name == representation else name,
                    "source_representation": representation if name == representation else None,
                    "source_manifest_sha256": source.manifest_sha256,
                    "source_record_id": record_id,
                    "source_occurrence": occurrence,
                    "trajectory_identity": trajectory,
                    "split": 0, "tokens": tokens, "targets": targets,
                }, sort_keys=True) + "\n")
                offset += tokens
                counts["records"] += 1
                counts["tokens"] += tokens
                counts["assistant_target_tokens"] += targets
                counts["train_records"] += 1
                source_counts[name]["occurrences"] += 1
                source_counts[name]["tokens"] += tokens
                source_counts[name]["assistant_target_tokens"] += targets
                unique_records[name].add(record_id)
        for name, ids in unique_records.items():
            source_counts[name]["unique_records"] = len(ids)
            counts["unique_source_records"] += len(ids)
        manifest = {
            "schema": AUTHORITY_SCHEMA,
            "status": "complete",
            "training_eligible": True,
            "purpose": "sealed 50M-target systematic E97 representation-comparison SFT stage",
            "stage_schema": SCHEMA,
            "representation": representation,
            "agent_protocol": ("e97-pi-agent-analysis-v1" if representation == "private-analysis" else "e97-pi-agent-v2"),
            "selection_seed": SEED,
            "selection": "deterministic without-replacement source traversals; complete records; no truncation",
            "max_record_tokens": MAX_RECORD_TOKENS,
            "counts": counts,
            "source_counts": dict(source_counts),
            "source_receipts": receipts,
            "recipe_sha256": recipe_sha256,
            "operator_authorization": {
                "scope": "build, validate, qualify, and run the matched systematic local training stage",
                "conversation_date": "2026-09-08",
                "authorization_text_sha256": hashlib.sha256(
                    b"Go. Do it. You've got all night. Don't give up until we are actually running training on an appropriate data set."
                ).hexdigest(),
            },
            "outputs": {key: entry(path) for key, path in paths.items()},
            "builder_source_sha256": sha256(Path(__file__)),
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        expected = {path.name: path.read_bytes() for path in paths.values()}
        expected["manifest.json"] = manifest_path.read_bytes()
        publish_directory_no_replace(stage, destination, expected_payloads=expected)
    return {"root": str(destination), "manifest_sha256": sha256(destination / "manifest.json"), **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--matched-validation", type=Path, required=True)
    parser.add_argument("--matched-pack-validation", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_root.is_absolute() or args.output_root.exists():
        raise SystemExit("output-root must be a new absolute path")
    if sha256(args.recipe) != hashlib.sha256(args.recipe.read_bytes()).hexdigest():
        raise AssertionError("unreachable recipe hash mismatch")
    recipe = json.loads(args.recipe.read_text())
    expected_recipe = {
        "schema": SCHEMA, "status": "frozen", "seed": SEED,
        "assistant_target_tokens": TARGETS,
        "source_manifest_sha256s": {name: digest for name, (_, digest) in SOURCES.items()},
        "matched_validation_sha256": MATCHED_VALIDATION_SHA256,
        "matched_pack_validation_sha256": MATCHED_PACK_VALIDATION_SHA256,
        "matched_trajectory_identity_sha256": MATCHED_TRAJECTORY_SET_SHA256,
    }
    if recipe != expected_recipe:
        raise SystemExit("recipe bytes do not encode the exact reviewed stage policy")
    recipe_sha256 = sha256(args.recipe)
    matched = load_receipt(
        args.matched_validation, MATCHED_VALIDATION_SHA256,
        "emender-e97-open-swe-matched-representation-validation-v1")
    packs = load_receipt(
        args.matched_pack_validation, MATCHED_PACK_VALIDATION_SHA256,
        "emender-e97-open-swe-matched-pack-validation-v1")
    if (matched["included_trajectory_identity_sha256"] != MATCHED_TRAJECTORY_SET_SHA256
            or matched["private_manifest_sha256"] != SOURCES["private-analysis"][1]
            or matched["action_manifest_sha256"] != SOURCES["action-only"][1]):
        raise SystemExit("matched validation identity mismatch")
    if (packs.get("private", {}).get("authority_manifest_sha256") != SOURCES["private-analysis"][1]
            or packs.get("action_only", {}).get("authority_manifest_sha256") != SOURCES["action-only"][1]
            or packs.get("matched_representation_validation_sha256") != MATCHED_VALIDATION_SHA256):
        raise SystemExit("matched pack validation identity mismatch")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(mode=0o700)
    try:
        sources = {name: open_source(name) for name in SOURCES}
        private_groups = agent_groups(sources["private-analysis"])
        action_groups = agent_groups(sources["action-only"])
        if set(private_groups) != set(action_groups):
            raise RuntimeError("agent trajectory groups are not matched")
        private_weights = {
            identity: sum(int(sources["private-analysis"].records[i]["targets"]) for i in ids)
            for identity, ids in private_groups.items()
        }
        trajectory_order = sorted(private_groups, key=lambda identity: hash_order("agent-private-epoch-0", identity))
        selected = closest_prefix(((identity, private_weights[identity]) for identity in trajectory_order), TARGETS["agent"])
        private_agent_targets = sum(private_weights[i] for i in selected)
        action_weights = {
            identity: sum(int(sources["action-only"].records[i]["targets"]) for i in action_groups[identity])
            for identity in selected
        }
        base_action_targets = sum(action_weights.values())
        if base_action_targets > private_agent_targets:
            raise RuntimeError("action-only base unexpectedly exceeds private-analysis targets")
        second, second_targets = choose_second_pass(action_weights, private_agent_targets - base_action_targets)
        action_agent_targets = base_action_targets + second_targets
        private_occurrences = [
            (identity, record_id, 0) for identity in selected for record_id in private_groups[identity]
        ]
        action_occurrences = [
            (identity, record_id, 0) for identity in selected for record_id in action_groups[identity]
        ] + [
            (identity, record_id, 1) for identity in second for record_id in action_groups[identity]
        ]
        shared = {
            name: select_shared(sources[name], TARGETS[name])
            for name in ("conversation", "compositional", "core-retention", "documents")
        }
        shared_targets = {
            name: sum(int(sources[name].records[i]["targets"]) for i in ids)
            for name, ids in shared.items()
        }
        receipts = {
            "matched_validation": {"path": str(args.matched_validation), "sha256": MATCHED_VALIDATION_SHA256},
            "matched_pack_validation": {"path": str(args.matched_pack_validation), "sha256": MATCHED_PACK_VALIDATION_SHA256},
            "matched_trajectory_identity_sha256": MATCHED_TRAJECTORY_SET_SHA256,
            "selected_agent_trajectories": len(selected),
            "selected_agent_trajectory_identity_sha256": hashlib.sha256(("\n".join(sorted(selected)) + "\n").encode()).hexdigest(),
            "private_agent_targets": private_agent_targets,
            "action_agent_base_targets": base_action_targets,
            "action_agent_second_pass_trajectories": len(second),
            "action_agent_targets": action_agent_targets,
            "agent_target_difference": action_agent_targets - private_agent_targets,
            "shared_targets": shared_targets,
            "source_manifest_sha256s": {name: source.manifest_sha256 for name, source in sources.items()},
        }
        private_result = build_arm(
            args.output_root / "private-analysis", "private-analysis", sources,
            private_occurrences, shared, receipts, recipe_sha256)
        action_result = build_arm(
            args.output_root / "action-only", "action-only", sources,
            action_occurrences, shared, receipts, recipe_sha256)
        stage_receipt = {
            "schema": SCHEMA, "status": "complete", "training_eligible": True,
            "recipe_sha256": recipe_sha256, "selection_receipts": receipts,
            "private_analysis": private_result, "action_only": action_result,
            "assistant_target_difference": action_result["assistant_target_tokens"] - private_result["assistant_target_tokens"],
            "relative_target_difference": abs(action_result["assistant_target_tokens"] - private_result["assistant_target_tokens"]) / private_result["assistant_target_tokens"],
            "policy": "token-matched primary comparison; same unique Open-SWE trajectories and exact shared replay records",
            "builder_source_sha256": sha256(Path(__file__)),
        }
        receipt_path = args.output_root / "stage-receipt.json"
        receipt_path.write_text(json.dumps(stage_receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"stage_receipt": str(receipt_path), "stage_receipt_sha256": sha256(receipt_path), **stage_receipt}, sort_keys=True))
    except BaseException:
        shutil.rmtree(args.output_root, ignore_errors=True)
        raise
    finally:
        for source in locals().get("sources", {}).values():
            source.close()


if __name__ == "__main__":
    main()
