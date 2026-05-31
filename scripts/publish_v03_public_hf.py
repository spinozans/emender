#!/usr/bin/env python3
"""Publish validated v0.3 artifacts to the public Hugging Face model repos.

This is the v0.3 analogue of ``scripts/publish_v02_public_hf.py``. It is
intentionally narrow and approval-gated: it uploads only the files already
listed in the local v0.3 validation manifest, verifies an unauthenticated
readback at the uploaded commit before tagging, and never deletes or recreates
the existing ``v0.1`` / ``v0.2`` tags.

Adaptations vs the v0.2 recipe (changed only as needed for the new revision):
  * RELEASE_TAG is ``v0.3`` and the manifest marker is ``v0.3-rc-local``.
  * The approval gate requires the v0.3 authorization text + the three repos.
  * BOTH prior release tags (``v0.1`` and ``v0.2``) are captured at preflight
    and asserted byte-identical (same target commit) at postflight, instead of
    hard-coding only the v0.1 SHAs. This task must not move/modify v0.1 OR v0.2.

The publish target repos are the SAME canonical public repos as v0.1/v0.2
(``poietic-pbc/{emender-e88-1.3b,gdn-1.3b,m2rnn-cma-1.3b}``); only a new tag is
added. No new repo name is invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RevisionNotFoundError
from safetensors import safe_open


DEFAULT_WORKDIR = Path("/tmp/release-v03-local-hf-candidates-agent-672")
DEFAULT_LOCAL_SMOKE = Path("/tmp/release-v03-docker-local-hf-artifact-smoke-agent-672/summary.json")
DEFAULT_OUTPUT = Path(f"/tmp/release-v03-public-hf-publish-{os.environ.get('WG_AGENT_ID', 'agent')}/summary.json")
RELEASE_TAG = "v0.3"
UPLOAD_REVISION = "main"
MANIFEST_MARKER = "v0.3-rc-local"

# Prior release tags that MUST NOT be moved or modified by this publish. Their
# current target commits are captured at preflight and asserted unchanged at
# postflight (capture-and-compare; no hard-coded SHAs to drift).
PRESERVE_TAGS = ("v0.1", "v0.2")

MODEL_LABELS = {
    "e88": "Emender/E88",
    "gdn": "GDN",
    "m2rnn": "M2RNN-CMA",
}

REQUIRED_PUBLIC_RESOLVE_FILES = [
    "config.json",
    "configuration_ndm.py",
    "modeling_ndm.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tiktoken/tokenizer.model",
    "model.safetensors",
]

DISALLOWED_SUFFIXES = (".pt", ".pth", ".pdf")
DISALLOWED_PARTS = {".cache", "__pycache__"}

T = TypeVar("T")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def task_log(message: str) -> None:
    task_id = os.environ.get("WG_TASK_ID")
    if not task_id:
        return
    subprocess.run(["wg", "log", task_id, message], check=False)


def emit_event(event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    print(json.dumps(record, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(128 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retry(label: str, func: Callable[[], T], *, attempts: int = 8, delay: float = 2.0) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except (HfHubHTTPError, RevisionNotFoundError, EntryNotFoundError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if data.get("release_candidate") != MANIFEST_MARKER:
        raise RuntimeError(f"unexpected release candidate marker in {path}: {data.get('release_candidate')!r}")
    return data


def load_local_smoke(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if data.get("ok") is not True:
        raise RuntimeError(f"local Docker smoke summary is not ok: {path}")
    if data.get("gpu_status") != "available":
        raise RuntimeError(f"local Docker smoke summary did not include GPU pass: {data.get('gpu_status')!r}")
    return data


def ensure_approval(args: argparse.Namespace) -> None:
    if not args.approved_public_v03_publication:
        raise SystemExit("--approved-public-v03-publication is required for any public HF write")
    note = args.approval_note.strip().lower()
    required = [
        "authorizes public hugging face v0.3 publication",
        "poietic-pbc/emender-e88-1.3b",
        "poietic-pbc/gdn-1.3b",
        "poietic-pbc/m2rnn-cma-1.3b",
    ]
    missing = [text for text in required if text not in note]
    if missing:
        raise SystemExit(f"--approval-note is missing required approval text: {missing}")


def manifest_models(manifest: dict[str, Any], selected: list[str]) -> list[dict[str, Any]]:
    by_key = {model["key"]: model for model in manifest["models"]}
    missing = sorted(set(selected) - set(by_key))
    if missing:
        raise RuntimeError(f"manifest is missing selected models: {missing}")
    return [by_key[key] for key in selected]


def validate_artifact_files(model: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = Path(model["artifact_dir"])
    if not artifact_dir.exists():
        raise FileNotFoundError(artifact_dir)

    files = sorted(str(path.relative_to(artifact_dir)) for path in artifact_dir.rglob("*") if path.is_file())
    expected_files = sorted(model["files"])
    if files != expected_files:
        raise RuntimeError(f"{model['key']} local artifact file list differs from manifest: {files} != {expected_files}")

    for rel in files:
        path = Path(rel)
        if path.suffix in DISALLOWED_SUFFIXES or DISALLOWED_PARTS.intersection(path.parts):
            raise RuntimeError(f"{model['key']} artifact contains disallowed file: {rel}")

    config = json.loads((artifact_dir / "config.json").read_text())
    checkpoint = model["checkpoint"]
    if config.get("repo_id") != model["repo_id"]:
        raise RuntimeError(f"{model['key']} config repo mismatch: {config.get('repo_id')} != {model['repo_id']}")
    if config.get("release_revision_name") != RELEASE_TAG:
        raise RuntimeError(f"{model['key']} config release revision mismatch: {config.get('release_revision_name')}")
    if config.get("source_checkpoint_sha256") != checkpoint["sha256"]:
        raise RuntimeError(f"{model['key']} config source checkpoint sha mismatch")
    if int(config.get("checkpoint_step")) != int(checkpoint["step"]):
        raise RuntimeError(f"{model['key']} config checkpoint step mismatch")

    safetensors_path = artifact_dir / "model.safetensors"
    if safetensors_path.stat().st_size != int(model["weights"]["size"]):
        raise RuntimeError(f"{model['key']} safetensors size differs from manifest")
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()
    if len(keys) != int(model["weights"]["keys"]):
        raise RuntimeError(f"{model['key']} safetensors key count differs from manifest")
    if metadata.get("checkpoint_step") != str(checkpoint["step"]):
        raise RuntimeError(f"{model['key']} safetensors checkpoint step metadata differs from manifest")

    file_sha256: dict[str, str] = {}
    file_sizes: dict[str, int] = {}
    for rel in files:
        path = artifact_dir / rel
        file_sizes[rel] = path.stat().st_size
        file_sha256[rel] = sha256_file(path)

    return {
        "artifact_dir": str(artifact_dir),
        "files": files,
        "config": {
            "repo_id": config.get("repo_id"),
            "release_revision_name": config.get("release_revision_name"),
            "source_checkpoint_sha256": config.get("source_checkpoint_sha256"),
            "checkpoint_step": config.get("checkpoint_step"),
            "checkpoint_loss": config.get("checkpoint_loss"),
        },
        "safetensors_metadata": metadata,
        "file_sizes": file_sizes,
        "file_sha256": file_sha256,
    }


def validate_local_smoke_for_model(model: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    key = model["key"]
    checkpoint = model["checkpoint"]
    rows = [row for row in smoke["rows"] if row.get("model") == key]
    devices = {row.get("device") for row in rows if row.get("ok") is True}
    if devices != {"cpu", "cuda"}:
        raise RuntimeError(f"{key} local Docker smoke did not pass on CPU and CUDA: {devices}")
    for row in rows:
        if row.get("source_checkpoint_sha256") != checkpoint["sha256"]:
            raise RuntimeError(f"{key} local Docker smoke source checkpoint sha mismatch")
        if int(row.get("checkpoint_step")) != int(checkpoint["step"]):
            raise RuntimeError(f"{key} local Docker smoke checkpoint step mismatch")
        if row.get("missing_keys") or row.get("unexpected_keys") or row.get("mismatched_keys"):
            raise RuntimeError(f"{key} local Docker smoke reported load key issues")
    return {"rows": rows, "devices": sorted(devices)}


def refs_dict(api: HfApi, repo_id: str, *, token: bool | str | None) -> dict[str, list[dict[str, str | None]]]:
    refs = api.list_repo_refs(repo_id, repo_type="model", token=token)
    return {
        "branches": [{"name": branch.name, "target_commit": branch.target_commit} for branch in refs.branches],
        "tags": [{"name": tag.name, "target_commit": tag.target_commit} for tag in refs.tags],
    }


def ref_target(refs: dict[str, list[dict[str, str | None]]], kind: str, name: str) -> str | None:
    for ref in refs[kind]:
        if ref["name"] == name:
            return ref["target_commit"]
    return None


def get_public_info(api: HfApi, repo_id: str, revision: str, *, files_metadata: bool = False) -> Any:
    return retry(
        f"model_info({repo_id}, revision={revision})",
        lambda: api.model_info(repo_id, revision=revision, token=False, files_metadata=files_metadata),
    )


def capture_preserved_tags(api: HfApi, repo_id: str, refs: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Record both identifiers of each tag we must NOT move.

    For an annotated tag the ref ``target_commit`` is the tag-OBJECT sha, which
    differs from the underlying commit sha that ``model_info(revision=tag).sha``
    resolves to. Both are stable; we capture both and (at postflight) assert
    neither changed — that is the real "tag not moved/modified" guarantee. We do
    NOT assert the two are equal to each other (they legitimately differ).
    """
    preserved: dict[str, dict[str, str]] = {}
    for tag in PRESERVE_TAGS:
        target = ref_target(refs, "tags", tag)
        if target is None:
            raise RuntimeError(f"{repo_id} expected preserved tag {tag} is missing before upload")
        info = get_public_info(api, repo_id, tag)
        resolved = getattr(info, "sha", None)
        if resolved is None:
            raise RuntimeError(f"{repo_id} {tag} did not resolve to a commit before upload")
        preserved[tag] = {"resolved_sha": resolved, "ref_target": target}
    return preserved


def preflight_public_refs(api: HfApi, model: dict[str, Any]) -> dict[str, Any]:
    repo_id = model["repo_id"]
    main = get_public_info(api, repo_id, UPLOAD_REVISION)
    if getattr(main, "private", None) is not False:
        raise RuntimeError(f"{repo_id} is not public before upload: private={getattr(main, 'private', None)}")
    refs = refs_dict(api, repo_id, token=False)
    preserved = capture_preserved_tags(api, repo_id, refs)
    return {
        "repo_id": repo_id,
        "main_sha": getattr(main, "sha", None),
        "preserved_tags_before": preserved,
        "v0_3_target_before": ref_target(refs, "tags", RELEASE_TAG),
        "private": getattr(main, "private", None),
        "refs": refs,
    }


def remote_file_metadata(repo_id: str, revision: str, filename: str) -> dict[str, Any]:
    url = hf_hub_url(repo_id, filename=filename, repo_type="model", revision=revision)
    meta = retry(
        f"HEAD {repo_id}@{revision}:{filename}",
        lambda: get_hf_file_metadata(url, token=False, timeout=30),
    )
    return {
        "commit_hash": meta.commit_hash,
        "etag": meta.etag,
        "size": meta.size,
    }


def download_and_hash_small_file(repo_id: str, revision: str, filename: str, cache_dir: Path) -> str:
    path = retry(
        f"download {repo_id}@{revision}:{filename}",
        lambda: hf_hub_download(
            repo_id,
            filename=filename,
            revision=revision,
            repo_type="model",
            token=False,
            cache_dir=str(cache_dir),
        ),
    )
    return sha256_file(Path(path))


def verify_remote_revision(
    api: HfApi,
    model: dict[str, Any],
    local: dict[str, Any],
    revision: str,
    cache_dir: Path,
) -> dict[str, Any]:
    repo_id = model["repo_id"]
    info = get_public_info(api, repo_id, revision, files_metadata=True)
    if getattr(info, "private", None) is not False:
        raise RuntimeError(f"{repo_id}@{revision} is not public: private={getattr(info, 'private', None)}")

    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    missing = sorted(set(local["files"]) - set(siblings))
    if missing:
        raise RuntimeError(f"{repo_id}@{revision} is missing uploaded files: {missing}")

    small_hashes: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for rel in local["files"]:
        if rel == "model.safetensors":
            sibling = siblings[rel]
            lfs = getattr(sibling, "lfs", None)
            if getattr(sibling, "size", None) != local["file_sizes"][rel]:
                raise RuntimeError(f"{repo_id}@{revision} model.safetensors size mismatch")
            if getattr(lfs, "sha256", None) != local["file_sha256"][rel]:
                raise RuntimeError(f"{repo_id}@{revision} model.safetensors LFS sha mismatch")
            metadata[rel] = remote_file_metadata(repo_id, revision, rel)
            if metadata[rel]["etag"] != local["file_sha256"][rel]:
                raise RuntimeError(f"{repo_id}@{revision} model.safetensors HEAD etag mismatch")
            continue

        if getattr(siblings[rel], "size", None) != local["file_sizes"][rel]:
            raise RuntimeError(f"{repo_id}@{revision} {rel} size mismatch")
        remote_sha = download_and_hash_small_file(repo_id, revision, rel, cache_dir)
        if remote_sha != local["file_sha256"][rel]:
            raise RuntimeError(f"{repo_id}@{revision} {rel} sha mismatch")
        small_hashes[rel] = remote_sha
        if rel in REQUIRED_PUBLIC_RESOLVE_FILES:
            metadata[rel] = remote_file_metadata(repo_id, revision, rel)

    config_path = retry(
        f"download {repo_id}@{revision}:config.json",
        lambda: hf_hub_download(
            repo_id,
            filename="config.json",
            revision=revision,
            repo_type="model",
            token=False,
            cache_dir=str(cache_dir),
        ),
    )
    config = json.loads(Path(config_path).read_text())
    checkpoint = model["checkpoint"]
    if config.get("source_checkpoint_sha256") != checkpoint["sha256"]:
        raise RuntimeError(f"{repo_id}@{revision} config source checkpoint sha mismatch")
    if int(config.get("checkpoint_step")) != int(checkpoint["step"]):
        raise RuntimeError(f"{repo_id}@{revision} config checkpoint step mismatch")
    if config.get("release_revision_name") != RELEASE_TAG:
        raise RuntimeError(f"{repo_id}@{revision} config release revision mismatch")

    for rel in REQUIRED_PUBLIC_RESOLVE_FILES:
        if rel not in metadata:
            metadata[rel] = remote_file_metadata(repo_id, revision, rel)

    return {
        "repo_id": repo_id,
        "revision": revision,
        "resolved_sha": getattr(info, "sha", None),
        "private": getattr(info, "private", None),
        "required_files": {
            rel: {
                "size": metadata[rel]["size"],
                "etag": metadata[rel]["etag"],
                "commit_hash": metadata[rel]["commit_hash"],
            }
            for rel in REQUIRED_PUBLIC_RESOLVE_FILES
        },
        "config": {
            "source_checkpoint_sha256": config.get("source_checkpoint_sha256"),
            "checkpoint_step": config.get("checkpoint_step"),
            "checkpoint_loss": config.get("checkpoint_loss"),
            "release_revision_name": config.get("release_revision_name"),
        },
        "small_file_sha256": small_hashes,
    }


def upload_and_tag_model(
    api: HfApi,
    model: dict[str, Any],
    local: dict[str, Any],
    preflight: dict[str, Any],
    cache_dir: Path,
) -> dict[str, Any]:
    key = model["key"]
    repo_id = model["repo_id"]
    existing_v03 = preflight["v0_3_target_before"]
    if existing_v03:
        task_log(f"{repo_id}: existing {RELEASE_TAG} tag found at {existing_v03}; verifying it matches validated artifacts.")
        emit_event("existing_release_readback_start", repo_id=repo_id, sha=existing_v03)
        existing_readback = verify_remote_revision(api, model, local, RELEASE_TAG, cache_dir / key / "existing-v03")
        if existing_readback["resolved_sha"] != existing_v03:
            raise RuntimeError(f"{repo_id} {RELEASE_TAG} ref target/readback mismatch")
        task_log(f"{repo_id}: existing {RELEASE_TAG} tag verified at {existing_v03}; prior tags untouched.")
        emit_event("existing_release_readback_finish", repo_id=repo_id, sha=existing_v03)
        return {
            "repo_id": repo_id,
            "action": "existing_v0_3_verified",
            "upload_commit": existing_v03,
            "tag_sha": existing_v03,
            "pre_tag_readback": existing_readback,
        }

    artifact_dir = local["artifact_dir"]
    task_log(f"{repo_id}: starting public {RELEASE_TAG} upload from validated local artifacts in {artifact_dir}.")
    emit_event("upload_start", repo_id=repo_id, artifact_dir=artifact_dir)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=artifact_dir,
        repo_type="model",
        revision=UPLOAD_REVISION,
        parent_commit=preflight["main_sha"],
        token=True,
        commit_message=f"Upload public {RELEASE_TAG} artifacts for {MODEL_LABELS[key]}",
        commit_description=(
            f"Approved public {RELEASE_TAG} upload from validated local artifacts.\n\n"
            f"Source checkpoint SHA256: {model['checkpoint']['sha256']}\n"
            f"Checkpoint step: {model['checkpoint']['step']}\n"
            f"v0.1 and v0.2 tags are intentionally left untouched."
        ),
        allow_patterns=local["files"],
        ignore_patterns=[".cache/**", "__pycache__/**", "*.pt", "*.pth", "*.pdf"],
    )
    upload_sha = commit.oid
    task_log(f"{repo_id}: upload finished at commit {upload_sha}; starting pre-tag unauthenticated readback.")
    emit_event("upload_finish", repo_id=repo_id, upload_sha=upload_sha)
    pre_tag_readback = verify_remote_revision(api, model, local, upload_sha, cache_dir / key / "pre-tag")

    if pre_tag_readback["resolved_sha"] != upload_sha:
        raise RuntimeError(f"{repo_id} upload readback resolved {pre_tag_readback['resolved_sha']} != {upload_sha}")

    task_log(f"{repo_id}: pre-tag readback verified commit {upload_sha}; creating {RELEASE_TAG} tag.")
    emit_event("pre_tag_readback_finish", repo_id=repo_id, upload_sha=upload_sha)
    api.create_tag(
        repo_id=repo_id,
        repo_type="model",
        tag=RELEASE_TAG,
        revision=upload_sha,
        tag_message=f"{RELEASE_TAG} public release for {MODEL_LABELS[key]}",
        token=True,
        exist_ok=False,
    )
    tagged_info = get_public_info(api, repo_id, RELEASE_TAG)
    if getattr(tagged_info, "sha", None) != upload_sha:
        raise RuntimeError(f"{repo_id} {RELEASE_TAG} tag resolved {getattr(tagged_info, 'sha', None)} != {upload_sha}")

    task_log(f"{repo_id}: created {RELEASE_TAG} tag at {upload_sha}.")
    emit_event("tag_created", repo_id=repo_id, tag=RELEASE_TAG, sha=upload_sha)
    return {
        "repo_id": repo_id,
        "action": "uploaded_and_tagged",
        "upload_commit": upload_sha,
        "commit_url": commit.commit_url,
        "tag_sha": getattr(tagged_info, "sha", None),
        "pre_tag_readback": pre_tag_readback,
    }


def postflight_refs(api: HfApi, model: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    repo_id = model["repo_id"]
    release = get_public_info(api, repo_id, RELEASE_TAG)
    refs_after = refs_dict(api, repo_id, token=False)
    preserved_after: dict[str, dict[str, str]] = {}
    for tag, before in preflight["preserved_tags_before"].items():
        info = get_public_info(api, repo_id, tag)
        after_sha = getattr(info, "sha", None)
        after_target = ref_target(refs_after, "tags", tag)
        if after_sha != before["resolved_sha"]:
            raise RuntimeError(
                f"{repo_id} {tag} resolved sha changed after publish: {after_sha} != {before['resolved_sha']}"
            )
        if after_target != before["ref_target"]:
            raise RuntimeError(
                f"{repo_id} {tag} ref target changed after publish: {after_target} != {before['ref_target']}"
            )
        preserved_after[tag] = {"resolved_sha": after_sha, "ref_target": after_target}
    return {
        "repo_id": repo_id,
        "preserved_tags_after": preserved_after,
        "v0_3_sha": getattr(release, "sha", None),
        "v0_3_private": getattr(release, "private", None),
        "refs": refs_after,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--local-smoke-summary", type=Path, default=DEFAULT_LOCAL_SMOKE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_LABELS), default=sorted(MODEL_LABELS))
    parser.add_argument("--approved-public-v03-publication", action="store_true")
    parser.add_argument("--approval-note", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_approval(args)
    task_log("Approval confirmed in task context; public v0.3 HF write path is now permitted.")
    emit_event("approval_confirmed", release_tag=RELEASE_TAG)
    if RELEASE_TAG in ("v0.1", "v0.2"):
        raise SystemExit(f"refusing to publish to a preserved tag: {RELEASE_TAG}")

    manifest = load_manifest(args.workdir / "validation_manifest.json")
    smoke = load_local_smoke(args.local_smoke_summary)
    selected_models = manifest_models(manifest, args.models)
    output_root = args.output.parent
    cache_dir = output_root / "readback-cache"
    output_root.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    auth = api.whoami(token=True)
    summary: dict[str, Any] = {
        "ok": False,
        "release_tag": RELEASE_TAG,
        "upload_revision": UPLOAD_REVISION,
        "preserve_tags": list(PRESERVE_TAGS),
        "approval_present": True,
        "approval_note_record": args.approval_note,
        "workdir": str(args.workdir),
        "local_smoke_summary": str(args.local_smoke_summary),
        "auth": {
            "name": auth.get("name"),
            "type": auth.get("type"),
            "orgs": sorted(org.get("name") for org in auth.get("orgs", []) if org.get("name")),
        },
        "models": [],
    }
    write_json(args.output, summary)

    for model in selected_models:
        key = model["key"]
        local = validate_artifact_files(model)
        local_smoke = validate_local_smoke_for_model(model, smoke)
        preflight = preflight_public_refs(api, model)
        publish = upload_and_tag_model(api, model, local, preflight, cache_dir)
        postflight = postflight_refs(api, model, preflight)
        if postflight["v0_3_sha"] != publish["tag_sha"]:
            raise RuntimeError(f"{model['repo_id']} postflight {RELEASE_TAG} tag mismatch")
        task_log(
            f"{model['repo_id']}: postflight public refs verified; "
            f"{RELEASE_TAG}={postflight['v0_3_sha']} and preserved={postflight['preserved_tags_after']}."
        )
        summary["models"].append(
            {
                "key": key,
                "identity": MODEL_LABELS[key],
                "repo_id": model["repo_id"],
                "source_checkpoint_sha256": model["checkpoint"]["sha256"],
                "checkpoint_step": model["checkpoint"]["step"],
                "local": {
                    "artifact_dir": local["artifact_dir"],
                    "file_sizes": local["file_sizes"],
                    "model_safetensors_sha256": local["file_sha256"]["model.safetensors"],
                    "config": local["config"],
                },
                "local_smoke": local_smoke,
                "preflight": preflight,
                "publish": publish,
                "postflight": postflight,
            }
        )
        write_json(args.output, summary)
        print(
            json.dumps(
                {
                    "ok": True,
                    "model": key,
                    "repo_id": model["repo_id"],
                    "action": publish["action"],
                    "preserved": postflight["preserved_tags_after"],
                    RELEASE_TAG: postflight["v0_3_sha"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary["ok"] = True
    summary["public_v0_3_shas"] = {row["key"]: row["postflight"]["v0_3_sha"] for row in summary["models"]}
    summary["public_preserved_shas"] = {
        row["key"]: row["postflight"]["preserved_tags_after"] for row in summary["models"]
    }
    write_json(args.output, summary)
    print(f"summary: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
