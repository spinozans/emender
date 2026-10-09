#!/usr/bin/env python3
"""Freeze validated lake-expansion tasks into the running bank's pool.

The bank's global pool is its documented lake-agnostic intake surface:
claim = one atomic rename pool/pending -> pool/claims (rl_bank.claim_pool_task
claims ANY valid pending task file; the coordinator only counts files and
never parses them; lanes grade purely from the frozen task body's own
task_lake binding). This driver grows the claimable work under the RUNNING
bank without stopping, restarting, or reconfiguring it.

Every frozen task body is produced with the same binding discipline the
bank's own refresh_pool uses (rl_gym_tasks.gym_task_body), plus the
admission-receipt v3 + operator-allowlist verification that the bank's
schema-v1-era pull layer cannot perform:

  * the admitted collection's admission receipt (v3) is verified end to end:
    registry + tasks digests, task-lake registry/collection validation,
    per-bundle spec digest binding, byte-verified pinned validator program,
    operator allowlist entry + zero-collision protected-overlap receipt
    (ndm.e97_protected_overlap.validate_authorized_overlap_receipt);
  * fixture archives are extracted with the CURRENT sealed
    safe_extract_fixture_archive (era-correct WIDENED tree digest for the
    new collections' pins);
  * task files mirror refresh_pool's exact shape (schema, task_id, attempts,
    round, body, task_sha256, generator, authoring, source_commit) with
    lakeexp-* ids and round: 0 (coordinator rounds start at 1), so the
    tranche is a distinguishable bucket in session telemetry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

REPO = Path("/home/erikg/emender")
LOOP = Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1")
WORK = Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1")
TASK_LAKE = Path("/mnt/nvme2n1/erikg/task_lake")
BANK = LOOP / "bank"
TASK_SCHEMA = "emender-rl-loop-seed-task-v1"
GYM_BINDING_SCHEMA = "emender-rl-loop-gym-task-binding-v1"
VALIDATOR_PROGRAM = REPO / "scripts" / "e97_first_party_validator.py"
# Era-3 (first-action criterion) and era-4 (protocol-breadth curriculum,
# operator directive 2026-09-28, repo commit 25f2dca9) sealed validators:
# the retained programs that era-3- and era-4-pinned private specs resolve
# to. The era-2 path above stays byte-identical (72820d63…) because
# in-flight bank pool tasks pin that absolute path; gym_task_body resolves
# the program PER SPEC PIN so all eras coexist under their own shas.
VALIDATOR_PROGRAM_FIRST_ACTION = REPO / "scripts" / "e97_first_party_validator_first_action.py"
VALIDATOR_PROGRAM_PROTOCOL_BREADTH = REPO / "scripts" / "e97_first_party_validator_protocol_breadth.py"

sys.path.insert(0, str(LOOP / "scripts"))
sys.path.insert(0, str(Path(os.environ["F9_STAGE_ROOT"]) / "loop"))

from rl_bank import bank_paths, freeze_pool_task  # noqa: E402


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    from ndm.e97_onpolicy_records import canonical_json
    return canonical_json(value)


def admitted_root(index: int) -> Path:
    return TASK_LAKE / f"e97-firstparty-lakeexp-v1-seed-{index:04d}-admitted"


def pull_validated_lakeexp(index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify one admitted lake-expansion collection end to end (v3-aware)."""
    from ndm.e97_protected_overlap import validate_authorized_overlap_receipt
    from ndm.e97_task_lake import validate_source_registry, validate_task_collection

    root = admitted_root(index)
    receipt = json.loads((root / "admission-receipt.json").read_bytes())
    if receipt.get("schema") != "emender-e97-first-party-admission-receipt-v3":
        raise SystemExit(f"seed-{index:04d}: admission receipt schema mismatch")
    registry_payload = (root / "source-registry.json").read_bytes()
    tasks_payload = (root / "tasks.jsonl").read_bytes()
    registry_sha, tasks_sha = hashlib.sha256(registry_payload).hexdigest(), \
        hashlib.sha256(tasks_payload).hexdigest()
    if registry_sha != receipt["registry_sha256"]:
        raise SystemExit(f"seed-{index:04d}: registry drifted")
    if tasks_sha != receipt["tasks_sha256"]:
        raise SystemExit(f"seed-{index:04d}: tasks drifted")
    registry = validate_source_registry(json.loads(registry_payload))
    tasks = validate_task_collection(
        [json.loads(line) for line in tasks_payload.decode().splitlines() if line.strip()],
        registry=registry)
    generation = json.loads((root / "generation-receipt.json").read_bytes())
    overlap_payload = (root / "protected-overlap-receipt.json").read_bytes()
    validate_authorized_overlap_receipt(
        json.loads(overlap_payload),
        receipt_sha256=hashlib.sha256(overlap_payload).hexdigest(),
        registry_sha256=generation["registry_sha256"],
        generation_receipt_sha256=hashlib.sha256(
            (root / "generation-receipt.json").read_bytes()).hexdigest(),
        tasks_sha256=generation["tasks_sha256"],
        archive_root_sha256=generation["archive_root_sha256"],
        generator_manifest_sha256=generation["generator_component_manifest_sha256"],
        source_archive_sha256=generation["generator_source_archive_sha256"],
        source_revision=generation["source_revision"],
        controller_source_sha256=generation["controller_source_sha256"],
    )
    summary = {
        "lake_root": str(root),
        "registry_sha256": registry_sha,
        "tasks_sha256": tasks_sha,
        "admission_receipt_sha256": _sha_file(root / "admission-receipt.json"),
        "tasks": len(tasks),
    }
    return tasks, summary


def gym_task_body(bundle: Mapping[str, Any], lake: Path) -> dict[str, Any]:
    """Adapt one validated bundle into the loop's task body (bank binding)."""
    from ndm.e97_first_party_read_observe import safe_extract_fixture_archive
    from ndm.e97_onpolicy_records import sha256_json

    identity = bundle["task"]["identity"]
    spec_path = lake / "private_validators" / f"{identity}.json"
    spec = json.loads(spec_path.read_bytes())
    if sha256_json(spec) != bundle["validator"]["spec_digest"]:
        raise SystemExit(f"validator spec digest mismatch for {identity[:16]}")
    # Resolve the pinned validator program PER SPEC PIN (fail-closed): the
    # era-2 repo bytes for era-2 pins (in-flight bank tasks), the era-3
    # first-action program for new pins.
    program = VALIDATOR_PROGRAM
    program_sha = _sha_file(program)
    if spec["program_sha256"] != program_sha:
        for retained in (VALIDATOR_PROGRAM_FIRST_ACTION,
                        VALIDATOR_PROGRAM_PROTOCOL_BREADTH,
                        Path(os.environ["F9_STAGE_ROOT"]) / "scripts/e97_first_party_validator_diversity.py"):
            if spec["program_sha256"] == _sha_file(retained):
                program = retained
                program_sha = spec["program_sha256"]
                break
        else:
            raise SystemExit(f"validator spec program pin matches no retained "
                             f"validator era for {identity[:16]}")
    for key, expected in (
        ("task_identity", identity),
        ("fixture_tree_digest", bundle["task"]["fixture_tree_digest"]),
        ("archive_sha256", bundle["fixture"]["artifact_sha256"]),
        ("program_sha256", program_sha),
    ):
        if spec[key] != expected:
            raise SystemExit(f"validator spec {key} mismatch for {identity[:16]}")
    archive_payload = (lake / bundle["fixture"]["artifact_path"]).read_bytes()
    if hashlib.sha256(archive_payload).hexdigest() != bundle["fixture"]["artifact_sha256"]:
        raise SystemExit(f"sealed fixture archive sha mismatch {identity[:16]}")
    workspace_files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="lakeexp-inject-") as scratch:
        fixture_root = Path(scratch) / "fixture"
        fixture_root.mkdir()
        # era-correct extraction: the CURRENT sealed verifier computes the
        # WIDENED tree digest these new collections pin
        safe_extract_fixture_archive(
            archive_payload, fixture_root,
            expected_sha256=bundle["fixture"]["artifact_sha256"],
            expected_tree_digest=bundle["task"]["fixture_tree_digest"],
            disk_limit=bundle["limits"]["disk_bytes"])
        for path in sorted(fixture_root.rglob("*")):
            if path.is_file():
                workspace_files[path.relative_to(fixture_root).as_posix()] = \
                    path.read_text(encoding="utf-8")
    if not workspace_files:
        raise SystemExit(f"fixture extracted no files {identity[:16]}")
    split = bundle["split"]
    return {
        "id": identity[:16],
        "family": bundle["task"]["family_id"],
        "template": "first-party-read-observe",
        "prompt": bundle["task"]["prompt"],
        "workspace_files": workspace_files,
        "setup": [],
        "split": split,
        "receipt_eligible": split == "train",
        "task_lake": {
            "schema": GYM_BINDING_SCHEMA,
            "lake_root": str(lake),
            "task_identity": identity,
            "task_bundle_sha256": hashlib.sha256(
                _canonical(dict(bundle)).encode("utf-8")).hexdigest(),
            "fixture_tree_digest": bundle["task"]["fixture_tree_digest"],
            "difficulty": bundle["task"]["difficulty"],
            "validator": {
                "spec_path": str(spec_path),
                "spec_sha256": _sha_file(spec_path),
                "spec_digest": bundle["validator"]["spec_digest"],
                "milestone_digest": bundle["validator"]["milestone_digest"],
                "minefield_digest": bundle["validator"]["minefield_digest"],
                "focused_argv": list(bundle["validator"]["focused_argv"]),
                "regression_argv": list(bundle["validator"]["regression_argv"]),
                "program_path": str(program),
                "program_sha256": program_sha,
                "interpreter_pinned_sha256": spec["interpreter_sha256"],
                "expected_token": spec["expected_token"],
                "required_read_path": spec["required_read_path"],
                "required_first_action": spec.get("required_first_action"),
                "expected_final": spec.get("expected_final"),
                "required_grounded_reads": spec.get("required_grounded_reads"),
                "required_error_read": spec.get("required_error_read"),
                "required_workspace_writes": spec.get("required_workspace_writes"),
                "allowed_tools": list(spec["minefield"]["allowed_tools"]),
            },
            "limits": dict(bundle["limits"]),
        },
    }


def inject(first: int, last: int) -> None:
    bank = bank_paths(BANK)
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    program_sha = _sha_file(VALIDATOR_PROGRAM)
    record_dir = WORK / "bank-integration"
    record_dir.mkdir(parents=True, exist_ok=True)
    frozen_all = []
    for index in range(first, last + 1):
        lake = admitted_root(index)
        bundles, summary = pull_validated_lakeexp(index)
        for bundle in bundles:
            body = gym_task_body(bundle, lake)
            task_id = f"lakeexp-{bundle['task']['identity'][:16]}"
            task = {
                "schema": TASK_SCHEMA,
                "task_id": task_id,
                "attempts": 0,
                "round": 0,
                "body": body,
                "task_sha256": hashlib.sha256(
                    _canonical(body).encode("utf-8")).hexdigest(),
                "generator": (f"sealed first-party task-lake authority "
                              f"({lake.name}; lake-expansion-v1, operator "
                              f"directive 2026-09-27)"),
                "authoring": ("sealed first-party read-observe bundle; "
                              "admission-receipt-v3 + operator-allowlist + "
                              "zero-collision protected-overlap receipt "
                              "verified; task-lake validation layer verified; "
                              f"pinned validator program sha256 "
                              f"{program_sha[:16]}… bound"),
                "source_commit": source_commit,
            }
            freeze_pool_task(bank, task)
            frozen_all.append({
                "task_id": task_id,
                "seed": index,
                "identity": bundle["task"]["identity"],
                "split": bundle["split"],
                "family": bundle["task"]["family_id"],
                "receipt_eligible": body["receipt_eligible"],
                "task_sha256": task["task_sha256"],
                "lake_root": str(lake),
            })
            print(f"POOL_FROZE {task_id} seed={index:04d} "
                  f"split={bundle['split']}", flush=True)
        record = {
            "schema": "emender-lakeexp-pool-injection-v1",
            "first_seed": first, "last_seed": last,
            "frozen_unix": time.time(),
            "source_commit": source_commit,
            "program_sha256": program_sha,
            "collections": [summary],
            "frozen": frozen_all,
        }
    out = record_dir / f"injection-seeds-{first:04d}-{last:04d}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"INJECTED seeds {first:04d}..{last:04d} tasks={len(frozen_all)} "
          f"receipt={out.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    args = parser.parse_args()
    inject(args.first, args.last)


if __name__ == "__main__":
    main()
