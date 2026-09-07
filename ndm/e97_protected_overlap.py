"""Privacy-preserving overlap checks for sealed E97 evaluation panels.

Protected records are parsed only in this process and transformed immediately to
comparison domains.  Receipts contain only manifest/records digests, candidate
roots, checker digest, counts, collision counts, and status.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tarfile
from typing import Any, Iterable, Mapping
import unicodedata

from ndm.e97_onpolicy_records import canonical_json, sha256_text
from ndm.e97_task_lake import validate_task_collection

PI_V3_SCHEMA = "emender-e97-pi-core-eval-authority-v3"
PI_V4_SCHEMA = "emender-e97-pi-core-eval-authority-v4"
REAL_REPO_SCHEMA = "emender-e97-real-repo-holdout-v1"
OVERLAP_RECEIPT_SCHEMA = "emender-e97-protected-overlap-receipt-v1"
OVERLAP_AUTHORIZATION_SCHEMA = "emender-e97-firstparty-overlap-authorization-v3"

_EXPECTED_MANIFEST_SHA256S = {
    PI_V3_SCHEMA: "ef481c637fde5916b8b0fe1f80cc2b4f0a6b88262088cbb33aefe0fed6bd6d09",
    PI_V4_SCHEMA: "8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8",
    REAL_REPO_SCHEMA: "939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60",
}
_EXPECTED_RECORD_SHA256S = {
    PI_V3_SCHEMA: "9ae13ee4d19075e8ab581a78b1edce15086398731823ab7b6db5a516037f96f2",
    PI_V4_SCHEMA: "fc391133e82cc85f05e0e29a00bd521d3d2fdac526f14def1893e166db363dac",
    REAL_REPO_SCHEMA: "3b93fc433cc6e23e98bfcd62b294f5d5c2e5d4144bfe3983cd661f53b2414676",
}
_REAL_REPOSITORY_IDS = {
    "markupsafe": "pallets/markupsafe",
    "humanize": "python-humanize/humanize",
    "more-itertools": "more-itertools/more-itertools",
    "prettytable": "prettytable/prettytable",
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_QUOTED = re.compile(r"`[^`]*`|'[^']*'|\"[^\"]*\"")
_JSON_KEY = re.compile(r'\"([^\"]*)\"(?=\s*:)')
_PATHLIKE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)+[A-Za-z0-9][A-Za-z0-9_.-]*")
_LONG_HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{8,}(?![0-9a-f])")
_DIGITS = re.compile(r"\d+")


class OverlapError(ValueError):
    """Protected overlap inputs or receipt bindings are malformed."""


def _sha_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise OverlapError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_fields(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OverlapError(f"{name} fields mismatch")
    return value


def _bounded_text(value: Any, name: str, *, allow_empty: bool = False, maximum: int = 1 << 20) -> str:
    if (not isinstance(value, str) or (not allow_empty and not value)
            or len(value.encode("utf-8")) > maximum):
        raise OverlapError(f"{name} must be bounded text")
    return value


def _safe_relative_path(value: Any, name: str) -> str:
    path = _bounded_text(value, name, maximum=4096)
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or path != candidate.as_posix():
        raise OverlapError(f"{name} must be a safe relative path")
    return path


def normalize_path(value: str) -> str:
    """Preserve path layout while replacing numeric and opaque variable segments."""

    value = _bounded_text(value, "overlap path", allow_empty=True, maximum=4096)
    normalized = unicodedata.normalize("NFKC", value).casefold()
    segments = []
    for segment in normalized.split("/"):
        segment = _LONG_HEX.sub("<hex>", segment)
        segment = _DIGITS.sub("<num>", segment)
        segments.append(segment)
    return "/".join(segments)


def _normalize_variables(value: str, *, quoted: bool) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    # Keep JSON field names so structured fixture shape remains comparable.
    if quoted:
        protected_keys: dict[str, str] = {}

        def keep_key(match: re.Match[str]) -> str:
            token = f"@@jsonkey{len(protected_keys)}@@"
            protected_keys[token] = match.group(0)
            return token

        normalized = _JSON_KEY.sub(keep_key, normalized)
        normalized = _QUOTED.sub("<quoted>", normalized)
        for token, key in protected_keys.items():
            normalized = normalized.replace(token, key)
    normalized = _PATHLIKE.sub("<path>", normalized)
    normalized = _LONG_HEX.sub("<hex>", normalized)
    normalized = _DIGITS.sub("<num>", normalized)
    return " ".join(normalized.split())


def normalize_template(value: str) -> str:
    """Normalize prompts while retaining instruction words and punctuation structure."""

    return _normalize_variables(_bounded_text(value, "prompt template"), quoted=True)


def normalize_content(value: str) -> str:
    """Normalize fixture content without treating arbitrary prose words as scalars."""

    return _normalize_variables(_bounded_text(value, "fixture content"), quoted=True)


def normalize_text(value: str) -> str:
    """Backward-compatible template normalization entry point."""

    return normalize_template(value)


def _scalar_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return canonical_json(value)
    raise OverlapError("structured scalar is invalid")


def _structured_scalars(value: Any) -> set[str]:
    scalars: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            scalars.update(_structured_scalars(item))
    elif isinstance(value, list):
        for item in value:
            scalars.update(_structured_scalars(item))
    else:
        scalars.add(_scalar_text(value))
    return scalars


def extract_exact_scalars(values: Iterable[str]) -> set[str]:
    """Extract JSON leaves and key=value RHS values, never arbitrary prose tokens."""

    scalars: set[str] = set()
    for text in values:
        _bounded_text(text, "candidate scalar source", allow_empty=True)
        try:
            scalars.update(_structured_scalars(json.loads(text)))
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            if "=" in line:
                _, rhs = line.split("=", 1)
                if rhs.strip():
                    scalars.add(rhs.strip())
    return scalars


def _parse_jsonl_records(path: Path) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise OverlapError("protected records are unreadable JSONL") from exc
    if not lines or any(not line.strip() for line in lines):
        raise OverlapError("protected records must be non-empty strict JSONL")
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise OverlapError("protected records are invalid JSONL") from exc


def _output_descriptor(value: Any, name: str) -> Mapping[str, Any]:
    descriptor = _exact_fields(value, {"path", "bytes", "sha256"}, name)
    if (not isinstance(descriptor["path"], str) or not descriptor["path"]
            or isinstance(descriptor["bytes"], bool) or not isinstance(descriptor["bytes"], int)
            or descriptor["bytes"] < 0):
        raise OverlapError(f"{name} descriptor is invalid")
    _digest(descriptor["sha256"], f"{name} SHA-256")
    return descriptor


def _fixture_files(value: Any, name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise OverlapError(f"{name} must be a list")
    files: list[dict[str, str]] = []
    paths: set[str] = set()
    for index, item in enumerate(value):
        item = _exact_fields(item, {"path", "content"}, f"{name} {index}")
        path = _safe_relative_path(item["path"], f"{name} path")
        if path in paths:
            raise OverlapError(f"{name} paths must be unique")
        paths.add(path)
        files.append({"path": path, "content": _bounded_text(item["content"], f"{name} content", allow_empty=True)})
    return files


def _pi_record(value: Any, *, schema: str) -> dict[str, Any]:
    record = _exact_fields(value, {"id", "kind", "source", "split", "task", "user"}, "protected Pi record")
    if (not isinstance(record["split"], int) or isinstance(record["split"], bool)
            or record["split"] != 1):
        raise OverlapError("protected Pi split is invalid")
    task = record["task"]
    base_task_fields = {"fixtures", "expected_calls", "postconditions", "final_contains"}
    permitted_task_fields = (base_task_fields, base_task_fields | {"expected_exit_codes"})
    if not isinstance(task, Mapping) or set(task) not in permitted_task_fields:
        raise OverlapError("protected Pi task fields mismatch")
    files = _fixture_files(task["fixtures"], "protected Pi fixtures")
    if not isinstance(task["expected_calls"], list) or not isinstance(task["postconditions"], list):
        raise OverlapError("protected Pi task execution metadata is invalid")
    final_contains = task["final_contains"]
    if (not isinstance(final_contains, list)
            or any(not isinstance(item, str) or not item for item in final_contains)
            or len(final_contains) != len(set(final_contains))):
        raise OverlapError("protected Pi final_contains is invalid")
    if "expected_exit_codes" in task and (
            not isinstance(task["expected_exit_codes"], list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in task["expected_exit_codes"])):
        raise OverlapError("protected Pi expected_exit_codes is invalid")
    _bounded_text(record["source"], "protected Pi source")
    return {
        "task_id": _bounded_text(record["id"], "protected Pi ID"),
        "family_id": _bounded_text(record["kind"], "protected Pi kind"),
        "repository": f"evaluation/{schema}",
        "prompt_template": _bounded_text(record["user"], "protected Pi user"),
        "fixture_files": files,
        "exact_scalars": sorted(set(final_contains).union(extract_exact_scalars(file["content"] for file in files))),
    }


def _real_repo_record(value: Any) -> dict[str, Any]:
    record = _exact_fields(value, {
        "clean", "commit", "expected_patch", "focused_test", "id", "mutated", "path", "prompt",
        "repository", "setup_files", "split", "url",
    }, "protected real-repository record")
    if (not isinstance(record["split"], int) or isinstance(record["split"], bool)
            or record["split"] != 1):
        raise OverlapError("protected real-repository split is invalid")
    repository = _bounded_text(record["repository"], "protected real-repository repository")
    normalized_repository = _REAL_REPOSITORY_IDS.get(repository)
    if normalized_repository is None:
        raise OverlapError("protected real-repository identity is unknown")
    patch = _exact_fields(record["expected_patch"], {"path", "oldText", "newText"}, "protected expected_patch")
    patch_path = _safe_relative_path(patch["path"], "protected expected_patch path")
    patch_fields = {
        "path": patch_path,
        "oldText": _bounded_text(patch["oldText"], "protected expected_patch oldText", allow_empty=True),
        "newText": _bounded_text(patch["newText"], "protected expected_patch newText", allow_empty=True),
    }
    setup = record["setup_files"]
    if not isinstance(setup, Mapping):
        raise OverlapError("protected setup_files is invalid")
    setup_files = []
    for path, content in sorted(setup.items()):
        setup_files.append({
            "path": _safe_relative_path(path, "protected setup path"),
            "content": _bounded_text(content, "protected setup content", allow_empty=True),
        })
    if len({item["path"] for item in setup_files}) != len(setup_files):
        raise OverlapError("protected setup paths are duplicated")
    structured = {
        "clean": _bounded_text(record["clean"], "protected clean", allow_empty=True),
        "commit": _bounded_text(record["commit"], "protected commit"),
        "expected_patch": patch_fields,
        "focused_test": _bounded_text(record["focused_test"], "protected focused_test"),
        "mutated": _bounded_text(record["mutated"], "protected mutated", allow_empty=True),
        "path": _safe_relative_path(record["path"], "protected source path"),
        "setup_files": {item["path"]: item["content"] for item in setup_files},
        "url": _bounded_text(record["url"], "protected URL"),
    }
    files = [{"path": patch_path, "content": canonical_json({"oldText": patch_fields["oldText"], "newText": patch_fields["newText"]})}]
    files.extend(setup_files)
    return {
        "task_id": _bounded_text(record["id"], "protected real-repository ID"),
        "family_id": "real-repository-repair",
        "repository": normalized_repository,
        "prompt_template": _bounded_text(record["prompt"], "protected real-repository prompt"),
        "fixture_files": files,
        "exact_scalars": sorted(_structured_scalars(structured)),
    }


def _load_pi_panel(manifest: Mapping[str, Any], records_path: Path, *, schema: str, records_sha256: str) -> list[dict[str, Any]]:
    fields = {
        "schema", "status", "purpose", "records", "seed", "kinds", "kind_counts",
        "training_exclusion", "outputs",
    }
    manifest = _exact_fields(manifest, fields, "protected Pi manifest")
    if manifest["schema"] != schema or manifest["status"] != "complete":
        raise OverlapError("protected Pi manifest schema/status is invalid")
    if isinstance(manifest["records"], bool) or not isinstance(manifest["records"], int) or manifest["records"] <= 0:
        raise OverlapError("protected Pi manifest records count is invalid")
    if (not isinstance(manifest["kinds"], list) or not manifest["kinds"]
            or any(not isinstance(kind, str) or not kind for kind in manifest["kinds"])
            or len(manifest["kinds"]) != len(set(manifest["kinds"]))):
        raise OverlapError("protected Pi manifest kinds are invalid")
    kind_counts = manifest["kind_counts"]
    if (not isinstance(kind_counts, Mapping) or set(kind_counts) != set(manifest["kinds"])
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in kind_counts.values())
            or sum(kind_counts.values()) != manifest["records"]):
        raise OverlapError("protected Pi manifest kind counts are invalid")
    outputs = _exact_fields(manifest["outputs"], {"metadata"}, "protected Pi outputs")
    metadata = _output_descriptor(outputs["metadata"], "protected Pi metadata")
    if metadata["sha256"] != records_sha256 or metadata["bytes"] != records_path.stat().st_size:
        raise OverlapError("protected Pi manifest metadata bytes/hash do not match records")
    records = [_pi_record(record, schema=schema) for record in _parse_jsonl_records(records_path)]
    if len(records) != manifest["records"]:
        raise OverlapError("protected Pi manifest record count does not match records")
    if any(record["family_id"] not in manifest["kinds"] for record in records):
        raise OverlapError("protected Pi record kind is not declared")
    observed_counts = {kind: sum(record["family_id"] == kind for record in records) for kind in manifest["kinds"]}
    if observed_counts != dict(kind_counts):
        raise OverlapError("protected Pi records do not match declared kind counts")
    return records


def _load_real_repo_panel(manifest: Mapping[str, Any], records_path: Path, *, records_sha256: str) -> list[dict[str, Any]]:
    fields = {"schema", "status", "purpose", "tasks", "repositories", "training_exclusion", "outputs"}
    manifest = _exact_fields(manifest, fields, "protected real-repository manifest")
    if manifest["schema"] != REAL_REPO_SCHEMA or manifest["status"] != "complete":
        raise OverlapError("protected real-repository manifest schema/status is invalid")
    if isinstance(manifest["tasks"], bool) or not isinstance(manifest["tasks"], int) or manifest["tasks"] <= 0:
        raise OverlapError("protected real-repository task count is invalid")
    if not isinstance(manifest["repositories"], Mapping) or not manifest["repositories"]:
        raise OverlapError("protected real-repository repositories are invalid")
    outputs = _exact_fields(manifest["outputs"], {"tasks"}, "protected real-repository outputs")
    descriptor = _output_descriptor(outputs["tasks"], "protected real-repository tasks")
    if descriptor["sha256"] != records_sha256 or descriptor["bytes"] != records_path.stat().st_size:
        raise OverlapError("protected real-repository manifest task bytes/hash do not match records")
    records = [_real_repo_record(record) for record in _parse_jsonl_records(records_path)]
    if len(records) != manifest["tasks"]:
        raise OverlapError("protected real-repository manifest task count does not match records")
    return records


def load_protected_panel(manifest_path: Path, records_path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    """Verify one fixed sealed manifest/records pair and adapt it in memory."""

    manifest_sha256, records_sha256 = _sha_path(manifest_path), _sha_path(records_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OverlapError("protected manifest is unreadable") from exc
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("schema"), str):
        raise OverlapError("protected manifest schema is missing")
    schema = manifest["schema"]
    if (schema not in _EXPECTED_MANIFEST_SHA256S
            or manifest_sha256 != _EXPECTED_MANIFEST_SHA256S[schema]
            or records_sha256 != _EXPECTED_RECORD_SHA256S[schema]):
        raise OverlapError("protected manifest or records SHA-256 does not match its sealed schema")
    if schema in {PI_V3_SCHEMA, PI_V4_SCHEMA}:
        records = _load_pi_panel(manifest, records_path, schema=schema, records_sha256=records_sha256)
    else:
        records = _load_real_repo_panel(manifest, records_path, records_sha256=records_sha256)
    return manifest_sha256, records_sha256, records


def _archive_files(archive: Path, *, disk_limit: int) -> list[dict[str, str]]:
    """Read candidate fixture text without retaining protected data anywhere."""

    try:
        with tarfile.open(archive, "r:") as stream:
            names: set[str] = set()
            expanded = 0
            files: list[dict[str, str]] = []
            for member in stream.getmembers():
                path = _safe_relative_path(member.name, "candidate fixture path")
                if path in names:
                    raise OverlapError("candidate archive has duplicate members")
                names.add(path)
                if member.isdir():
                    continue
                if not member.isreg() or member.issym() or member.islnk():
                    raise OverlapError("candidate archive has unsafe member")
                expanded += member.size
                if expanded > disk_limit:
                    raise OverlapError("candidate archive exceeds disk limit")
                payload = stream.extractfile(member)
                if payload is None:
                    raise OverlapError("candidate archive member is unreadable")
                try:
                    text = payload.read().decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise OverlapError("candidate fixture must be UTF-8 text") from exc
                files.append({"path": path, "content": text})
    except tarfile.TarError as exc:
        raise OverlapError("candidate archive is invalid") from exc
    return sorted(files, key=lambda item: item["path"])


def candidate_collection_archive_root(tasks: Iterable[Mapping[str, Any]], candidate_root: Path) -> str:
    """Hash exact candidate archive paths, bytes, and SHA-256s in canonical order."""

    entries = []
    for task in tasks:
        relative = task["fixture"]["artifact_path"]
        path = candidate_root / relative
        payload = path.read_bytes()
        entries.append({"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return sha256_text(canonical_json(sorted(entries, key=lambda item: item["path"])))


def _domains(records: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    fields = {
        "task_ids": set(), "family_ids": set(), "repositories": set(),
        "prompt_templates_full": set(), "prompt_templates_normalized": set(),
        "fixture_paths_full": set(), "fixture_paths_normalized": set(),
        "fixture_contents_full": set(), "fixture_contents_normalized": set(),
        "exact_scalars": set(),
    }
    for record in records:
        fields["task_ids"].add(record["task_id"])
        fields["family_ids"].add(record["family_id"])
        fields["repositories"].add(record["repository"])
        fields["prompt_templates_full"].add(record["prompt_template"])
        fields["prompt_templates_normalized"].add(normalize_template(record["prompt_template"]))
        for fixture in record["fixture_files"]:
            fields["fixture_paths_full"].add(fixture["path"])
            fields["fixture_paths_normalized"].add(normalize_path(fixture["path"]))
            fields["fixture_contents_full"].add(fixture["content"])
            fields["fixture_contents_normalized"].add(normalize_content(fixture["content"]))
        fields["exact_scalars"].update(record["exact_scalars"])
    return fields


def candidate_records(tasks: list[Mapping[str, Any]], candidate_root: Path) -> list[dict[str, Any]]:
    result = []
    for task in tasks:
        files = _archive_files(candidate_root / task["fixture"]["artifact_path"], disk_limit=task["limits"]["disk_bytes"])
        result.append({
            "task_id": task["task"]["identity"],
            "family_id": task["task"]["family_id"],
            "repository": task["source"]["repository"],
            "prompt_template": task["task"]["prompt"],
            "fixture_files": files,
            "exact_scalars": sorted(extract_exact_scalars(file["content"] for file in files)),
        })
    return result


def check_protected_overlap(
    *, registry: Mapping[str, Any], candidate_collection: Path, candidate_root: Path,
    protected_panels: Iterable[tuple[Path, Path]],
) -> dict[str, Any]:
    """Compare a candidate only to the three fixed sealed evaluation panels."""

    try:
        tasks = [json.loads(line) for line in candidate_collection.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise OverlapError("candidate collection is invalid JSONL") from exc
    tasks = validate_task_collection(tasks, registry=registry)
    expected_panels = sorted(_EXPECTED_MANIFEST_SHA256S.values())
    registry_panels = sorted(item["manifest_sha256"] for item in registry["protected_evaluation"])
    if registry_panels != expected_panels:
        raise OverlapError("registry does not bind exactly the fixed protected panels")
    loaded = [load_protected_panel(manifest, records) for manifest, records in protected_panels]
    observed_panels = sorted(item[0] for item in loaded)
    observed_records = sorted(item[1] for item in loaded)
    if (observed_panels != expected_panels
            or observed_records != sorted(_EXPECTED_RECORD_SHA256S.values())
            or len(loaded) != len(expected_panels)):
        raise OverlapError("protected panel inputs do not exactly match the fixed sealed pairs")
    protected_records = [record for _, _, panel_records in loaded for record in panel_records]
    protected_domains = _domains(protected_records)
    candidate_domains = _domains(candidate_records(tasks, candidate_root))
    collision_counts = {
        field: len(candidate_domains[field].intersection(protected_domains[field]))
        for field in sorted(candidate_domains)
    }
    return {
        "schema": OVERLAP_RECEIPT_SCHEMA,
        "status": "pass" if not any(collision_counts.values()) else "fail",
        "candidate_collection_sha256": _sha_path(candidate_collection),
        "candidate_archive_root_sha256": candidate_collection_archive_root(tasks, candidate_root),
        "checker_source_sha256": _sha_path(Path(__file__)),
        "protected_manifest_sha256s": observed_panels,
        "protected_records_sha256s": observed_records,
        "field_domain_counts": {
            field: {"candidate": len(candidate_domains[field]), "protected": len(protected_domains[field])}
            for field in sorted(candidate_domains)
        },
        "collision_counts": collision_counts,
    }


def validate_overlap_authorization(value: Any) -> dict[str, Any]:
    """Validate static checker authorization without treating it as semantic clearance."""

    receipt = _exact_fields(value, {
        "schema", "authority_scope", "checker_source_sha256", "protected_manifest_sha256s",
        "semantic_clearance",
    }, "overlap authorization")
    if receipt["schema"] != OVERLAP_AUTHORIZATION_SCHEMA:
        raise OverlapError("overlap authorization schema is invalid")
    if receipt["authority_scope"] != "checker-code-and-protected-manifest-metadata-only":
        raise OverlapError("overlap authorization scope is invalid")
    if receipt["semantic_clearance"] is not False:
        raise OverlapError("overlap authorization must not claim semantic clearance")
    if _digest(receipt["checker_source_sha256"], "overlap authorization checker SHA-256") != _sha_path(Path(__file__)):
        raise OverlapError("overlap authorization checker bytes changed")
    expected = sorted(_EXPECTED_MANIFEST_SHA256S.values())
    if receipt["protected_manifest_sha256s"] != expected:
        raise OverlapError("overlap authorization does not bind protected manifest metadata")
    return dict(receipt)


def validate_overlap_receipt(
    value: Any, *, candidate_collection_sha256: str, candidate_archive_root_sha256: str,
) -> dict[str, Any]:
    """Validate a privacy-preserving executed pass receipt for collection admission."""

    receipt = _exact_fields(value, {
        "schema", "status", "candidate_collection_sha256", "candidate_archive_root_sha256",
        "checker_source_sha256", "protected_manifest_sha256s", "protected_records_sha256s",
        "field_domain_counts", "collision_counts",
    }, "overlap receipt")
    if receipt["schema"] != OVERLAP_RECEIPT_SCHEMA or receipt["status"] != "pass":
        raise OverlapError("protected overlap receipt is not a pass")
    if (_digest(receipt["candidate_collection_sha256"], "candidate collection SHA-256") != candidate_collection_sha256
            or _digest(receipt["candidate_archive_root_sha256"], "candidate archive root SHA-256") != candidate_archive_root_sha256
            or _digest(receipt["checker_source_sha256"], "checker SHA-256") != _sha_path(Path(__file__))):
        raise OverlapError("protected overlap receipt input binding mismatch")
    expected_panels = sorted(_EXPECTED_MANIFEST_SHA256S.values())
    expected_records = sorted(_EXPECTED_RECORD_SHA256S.values())
    if receipt["protected_manifest_sha256s"] != expected_panels:
        raise OverlapError("protected overlap receipt does not bind every fixed panel")
    records = receipt["protected_records_sha256s"]
    if records != expected_records:
        raise OverlapError("protected overlap receipt does not bind every fixed panel records SHA-256")
    fields = set(_domains([]))
    if not isinstance(receipt["field_domain_counts"], Mapping) or set(receipt["field_domain_counts"]) != fields:
        raise OverlapError("protected overlap field counts are invalid")
    if not isinstance(receipt["collision_counts"], Mapping) or set(receipt["collision_counts"]) != fields:
        raise OverlapError("protected overlap collision counts are invalid")
    for field in fields:
        counts = receipt["field_domain_counts"][field]
        if (not isinstance(counts, Mapping) or set(counts) != {"candidate", "protected"}
                or any(isinstance(counts[key], bool) or not isinstance(counts[key], int) or counts[key] < 0 for key in counts)):
            raise OverlapError("protected overlap field count is invalid")
        if receipt["collision_counts"][field] != 0:
            raise OverlapError("protected overlap receipt contains a collision")
    return json.loads(canonical_json(dict(receipt)))
