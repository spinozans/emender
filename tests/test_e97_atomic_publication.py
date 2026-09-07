import hashlib
import os
import threading

import pytest

from ndm.e97_artifact_store import ArtifactStore, ArtifactStoreError, completed_receipt_relative_path, receipt_relative_path
from ndm.e97_atomic import publish_bytes_no_replace, publish_directory_no_replace


def test_no_replace_publication_rejects_partial_conflict_allows_retry_and_is_concurrent(tmp_path):
    target = tmp_path / "receipt.json"
    payload = b'{"receipt":"complete"}'
    target.write_bytes(payload[:8])
    with pytest.raises(ValueError, match="conflicts"):
        publish_bytes_no_replace(target, payload)
    assert target.read_bytes() == payload[:8]

    target.unlink()
    outcomes = []
    failures = []

    def publish():
        try:
            outcomes.append(publish_bytes_no_replace(target, payload))
        except Exception as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    workers = [threading.Thread(target=publish) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not failures
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7
    assert target.read_bytes() == payload
    assert publish_bytes_no_replace(target, payload) is False


def test_directory_no_replace_concurrent_distinct_authorities_has_one_winner(tmp_path):
    destination = tmp_path / "authority"
    stages = []
    for seed in ("seed-a", "seed-b"):
        stage = tmp_path / f"stage-{seed}"
        stage.mkdir()
        (stage / "seed.txt").write_text(seed)
        stages.append(stage)
    outcomes = []
    conflicts = []

    def publish(stage):
        try:
            publish_directory_no_replace(stage, destination)
            outcomes.append(stage.name)
        except FileExistsError:
            conflicts.append(stage.name)

    workers = [threading.Thread(target=publish, args=(stage,)) for stage in stages]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(outcomes) == len(conflicts) == 1
    assert destination.is_dir()
    assert destination.joinpath("seed.txt").read_text() in {"seed-a", "seed-b"}
    assert sum(stage.exists() for stage in stages) == 1


def test_orphan_content_receipt_is_not_completion_authority(tmp_path):
    store = ArtifactStore(tmp_path)
    task_id = "a" * 64
    payload = b'{"schema":"receipt","task_id":"' + task_id.encode() + b'"}'
    digest = hashlib.sha256(payload).hexdigest()
    content = tmp_path / receipt_relative_path(digest)
    content.parent.mkdir()
    content.write_bytes(payload)
    reference = {"sha256": digest, "path": completed_receipt_relative_path(task_id, digest)}
    with pytest.raises(ArtifactStoreError, match="completed receipt marker"):
        store.resolve_published_receipt(reference, task_id=task_id)
    marker = tmp_path / reference["path"]
    marker.parent.mkdir(parents=True)
    marker.write_bytes(payload)
    assert store.resolve_published_receipt(reference, task_id=task_id)["task_id"] == task_id


def test_artifact_resolver_rejects_symlink_swap_before_open(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    payload = b"safe artifact"
    reference = store.put_bytes(payload, suffix=".bin")
    artifact = tmp_path / "store" / reference["path"]
    outside = tmp_path / "outside"
    outside.write_bytes(payload)
    original_open = os.open

    def swapped_open(path, flags, *args, **kwargs):
        if path == artifact.name and kwargs.get("dir_fd") is not None:
            artifact.unlink()
            artifact.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("ndm.e97_artifact_store.os.open", swapped_open)
    with pytest.raises(ArtifactStoreError, match="cannot be opened safely"):
        store.resolve_bytes(reference, suffix=".bin")


def test_artifact_resolver_rejects_symlinked_parent_component(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    payload = b"same digest does not authorize path escape"
    reference = store.put_bytes(payload, suffix=".bin")
    artifact = tmp_path / "store" / reference["path"]
    prefix = artifact.parent
    saved_prefix = prefix.with_name(prefix.name + "-saved")
    prefix.rename(saved_prefix)
    outside_prefix = tmp_path / "outside-prefix"
    outside_prefix.mkdir()
    (outside_prefix / artifact.name).write_bytes(payload)
    prefix.symlink_to(outside_prefix, target_is_directory=True)
    with pytest.raises(ArtifactStoreError, match="cannot be opened safely"):
        store.resolve_bytes(reference, suffix=".bin")


def test_publisher_and_artifact_resolver_reject_symlink_targets(tmp_path):
    payload = b"immutable"
    outside = tmp_path / "outside"
    outside.write_bytes(payload)
    target = tmp_path / "receipt.json"
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        publish_bytes_no_replace(target, payload)

    store = ArtifactStore(tmp_path / "store")
    reference = store.put_bytes(payload, suffix=".bin")
    artifact = tmp_path / "store" / reference["path"]
    artifact.unlink()
    artifact.symlink_to(outside)
    with pytest.raises(ArtifactStoreError, match="cannot be opened safely"):
        store.resolve_bytes(reference, suffix=".bin")
