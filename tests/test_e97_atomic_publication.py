import ctypes
import hashlib
import os
import stat
import threading

import pytest

from ndm.e97_artifact_store import ArtifactStore, ArtifactStoreError, completed_receipt_relative_path, receipt_relative_path
import ndm.e97_atomic as atomic
from ndm.e97_atomic import cleanup_directory_best_effort, publish_bytes_no_replace, publish_directory_no_replace


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


def test_same_byte_retry_fsyncs_exact_parent_before_return(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"; target.write_bytes(b"durable")
    parent_info = tmp_path.stat()
    parent = (parent_info.st_dev, parent_info.st_ino, parent_info.st_mode)
    events = []
    original_fsync = os.fsync

    def tracked_fsync(fd):
        info = os.fstat(fd)
        events.append(("fsync", info.st_dev, info.st_ino, info.st_mode))
        return original_fsync(fd)

    monkeypatch.setattr("ndm.e97_atomic.os.fsync", tracked_fsync)
    assert publish_bytes_no_replace(target, b"durable") is False
    # Absolute parent traversal flushes ancestor directories too; the retry
    # contract requires exactly one final fsync of this containing directory.
    assert [event for event in events if event[1:] == parent] == [("fsync", *parent)]


def test_nested_publication_orders_parent_fsyncs_around_mkdir_link_and_unlink(tmp_path, monkeypatch):
    target = tmp_path / "new-a" / "new-b" / "receipt.json"
    events = []
    original_fsync, original_mkdir = os.fsync, os.mkdir
    original_link, original_unlink = os.link, os.unlink

    def identity(fd):
        info = os.fstat(fd)
        return info.st_dev, info.st_ino, info.st_mode

    def tracked_fsync(fd):
        events.append(("fsync", *identity(fd)))
        return original_fsync(fd)

    def tracked_mkdir(path, mode=0o777, *, dir_fd=None):
        events.append(("mkdir", *identity(dir_fd), path))
        return original_mkdir(path, mode, dir_fd=dir_fd)

    def tracked_link(source, destination, *args, **kwargs):
        events.append(("link", *identity(kwargs["dst_dir_fd"]), destination))
        return original_link(source, destination, *args, **kwargs)

    def tracked_unlink(path, *args, **kwargs):
        events.append(("unlink", *identity(kwargs["dir_fd"]), path))
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr("ndm.e97_atomic.os.fsync", tracked_fsync)
    monkeypatch.setattr("ndm.e97_atomic.os.mkdir", tracked_mkdir)
    monkeypatch.setattr("ndm.e97_atomic.os.link", tracked_link)
    monkeypatch.setattr("ndm.e97_atomic.os.unlink", tracked_unlink)
    assert publish_bytes_no_replace(target, b"durable") is True
    def path_identity(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_mode

    parent_ids = {
        "top": path_identity(tmp_path),
        "new-a": path_identity(tmp_path / "new-a"),
        "new-b": path_identity(tmp_path / "new-a" / "new-b"),
    }
    known_identities = set(parent_ids.values())
    directory_events = [
        event for event in events
        if event[1:4] in known_identities and (event[0] != "fsync" or stat.S_ISDIR(event[3]))
    ]
    assert [event[:4] for event in directory_events] == [
        ("mkdir", *parent_ids["top"]),
        ("fsync", *parent_ids["top"]),
        ("mkdir", *parent_ids["new-a"]),
        ("fsync", *parent_ids["new-a"]),
        ("link", *parent_ids["new-b"]),
        ("fsync", *parent_ids["new-b"]),
        ("unlink", *parent_ids["new-b"]),
        ("fsync", *parent_ids["new-b"]),
    ]
    assert [event[4] if len(event) > 4 else None for event in directory_events[:5]] == [
        "new-a", None, "new-b", None, "receipt.json",
    ]
    assert directory_events[6][4].startswith(".receipt.json.")
    assert directory_events[6][4].endswith(".tmp")
    assert directory_events[7][4:] == ()
    assert target.read_bytes() == b"durable"


def test_publisher_rejects_symlinked_parent_and_final_without_path_races(tmp_path):
    payload = b"immutable"
    outside = tmp_path / "outside"; outside.mkdir()
    parent_link = tmp_path / "linked-parent"; parent_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="parent cannot be opened safely"):
        publish_bytes_no_replace(parent_link / "receipt.json", payload)
    target = tmp_path / "receipt.json"
    target.symlink_to(outside / "target")
    with pytest.raises(ValueError, match="regular file"):
        publish_bytes_no_replace(target, payload)


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
            publish_directory_no_replace(stage, destination, expected_payloads={"seed.txt": stage.joinpath("seed.txt").read_bytes()})
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
    # Source stages are snapshotted into an API-owned child; they are never
    # consumed by renaming a caller-controlled basename.
    assert all(stage.is_dir() for stage in stages)
    assert not list(tmp_path.glob(".authority.publish.*"))


def test_concurrent_parent_creator_is_flushed_by_the_observing_publisher(tmp_path, monkeypatch):
    """A publisher that sees another creator's directory still flushes it."""

    target = tmp_path / "shared-parent" / "receipt.json"
    top_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    creator_ready = threading.Event()
    contender_flushed_parent = threading.Event()
    allow_creator = threading.Event()
    original_fsync = os.fsync

    def tracked_fsync(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == top_identity:
            if threading.current_thread().name == "creator":
                creator_ready.set()
                assert allow_creator.wait(timeout=5)
            elif threading.current_thread().name == "contender":
                contender_flushed_parent.set()
        return original_fsync(fd)

    monkeypatch.setattr("ndm.e97_atomic.os.fsync", tracked_fsync)
    failures = []

    def publish(name):
        try:
            publish_bytes_no_replace(target, b"same")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    creator = threading.Thread(target=publish, args=("creator",), name="creator")
    creator.start()
    assert creator_ready.wait(timeout=5)
    contender = threading.Thread(target=publish, args=("contender",), name="contender")
    contender.start()
    try:
        assert contender_flushed_parent.wait(timeout=5)
    finally:
        allow_creator.set()
    creator.join(timeout=5)
    contender.join(timeout=5)
    assert not failures
    assert target.read_bytes() == b"same"


def test_fifo_target_is_rejected_without_blocking(tmp_path):
    target = tmp_path / "receipt.fifo"
    os.mkfifo(target)
    with pytest.raises(ValueError, match="not a regular file"):
        publish_bytes_no_replace(target, b"payload")


def test_directory_publication_renames_only_a_private_snapshot_under_stage_exchange(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "seed.txt").write_text("validated")
    destination = tmp_path / "authority"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "seed.txt").write_text("attacker")
    saved_stage = tmp_path / "saved-stage"
    real_libc = ctypes.CDLL(None, use_errno=True)
    real_renameat2 = real_libc.renameat2
    real_renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    real_renameat2.restype = ctypes.c_int

    class RaceRename:
        def __init__(self):
            self.swapped = False

        def __call__(self, *args):
            if not self.swapped:
                self.swapped = True
                stage.rename(saved_stage)
                stage.symlink_to(outside, target_is_directory=True)
            return real_renameat2(*args)

    class RaceLibc:
        renameat2 = RaceRename()

    monkeypatch.setattr("ndm.e97_atomic.ctypes.CDLL", lambda *_args, **_kwargs: RaceLibc())
    publish_directory_no_replace(stage, destination, expected_payloads={"seed.txt": b"validated"})
    assert destination.is_dir()
    assert destination.joinpath("seed.txt").read_text() == "validated"
    assert stage.is_symlink()
    assert not list(tmp_path.glob(".authority.publish.*"))


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_directory_publication_rejects_child_exchange_against_approved_payloads(tmp_path, monkeypatch, kind):
    stage = tmp_path / "stage"
    child = stage / "child"
    child.mkdir(parents=True)
    (child / "payload.txt").write_text("approved")
    destination = tmp_path / "authority"
    expected = {"child/payload.txt": b"approved"}
    original_open = os.open
    swapped = False

    def exchange_before_child_open(name, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and name == "child" and kwargs.get("dir_fd") is not None:
            swapped = True
            replacement = stage / "replacement"
            if kind == "file":
                # A same-type child-file exchange happens immediately before
                # copy; it must not enter the published authority.
                (child / "payload.txt").write_text("attacker")
            else:
                child.rename(replacement)
                child.mkdir()
                (child / "payload.txt").write_text("attacker")
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr("ndm.e97_atomic.os.open", exchange_before_child_open)
    with pytest.raises(ValueError, match="differs from approved"):
        publish_directory_no_replace(stage, destination, expected_payloads=expected)
    assert not destination.exists()


def test_post_publication_cleanup_fsync_failure_does_not_recast_success(tmp_path, monkeypatch):
    stage = tmp_path / "stage"; stage.mkdir()
    (stage / "payload").write_bytes(b"approved")
    destination = tmp_path / "authority"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    cleanup_removed = False
    real_rmdir, real_fsync = os.rmdir, os.fsync

    def tracked_rmdir(name, *args, **kwargs):
        nonlocal cleanup_removed
        result = real_rmdir(name, *args, **kwargs)
        if str(name).startswith(".authority.publish."):
            cleanup_removed = True
        return result

    def fail_cleanup_only_fsync(fd):
        info = os.fstat(fd)
        if cleanup_removed and (info.st_dev, info.st_ino) == parent_identity:
            raise OSError("injected cleanup fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr("ndm.e97_atomic.os.rmdir", tracked_rmdir)
    monkeypatch.setattr("ndm.e97_atomic.os.fsync", fail_cleanup_only_fsync)
    publish_directory_no_replace(stage, destination, expected_payloads={"payload": b"approved"})
    assert cleanup_removed
    assert destination.joinpath("payload").read_bytes() == b"approved"


def test_post_durable_temporary_cleanup_fsync_failure_does_not_recast_byte_success(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    temporary_removed = False
    real_unlink, real_fsync = os.unlink, os.fsync

    def track_temporary_unlink(name, *args, **kwargs):
        nonlocal temporary_removed
        result = real_unlink(name, *args, **kwargs)
        if str(name).startswith(".receipt.json.") and str(name).endswith(".tmp"):
            temporary_removed = True
        return result

    def fail_only_cleanup_fsync(fd):
        info = os.fstat(fd)
        if temporary_removed and (info.st_dev, info.st_ino) == parent_identity:
            raise OSError("injected post-durable temporary cleanup fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr("ndm.e97_atomic.os.unlink", track_temporary_unlink)
    monkeypatch.setattr("ndm.e97_atomic.os.fsync", fail_only_cleanup_fsync)
    assert publish_bytes_no_replace(target, b"durable") is True
    assert temporary_removed
    assert target.read_bytes() == b"durable"


def test_post_durable_close_failure_does_not_recast_immutable_byte_success(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    durable, injected = False, False
    real_fsync, real_close = os.fsync, os.close

    def mark_durable(fd):
        nonlocal durable
        result = real_fsync(fd)
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == parent_identity and target.exists():
            durable = True
        return result

    def fail_first_post_durable_close(fd):
        nonlocal injected
        if durable and not injected:
            injected = True
            real_close(fd)
            raise OSError("injected post-durable close failure")
        return real_close(fd)

    monkeypatch.setattr("ndm.e97_atomic.os.fsync", mark_durable)
    monkeypatch.setattr("ndm.e97_atomic.os.close", fail_first_post_durable_close)
    assert publish_bytes_no_replace(target, b"durable") is True
    assert injected
    assert target.read_bytes() == b"durable"


def test_post_durable_descriptor_close_failure_does_not_recast_directory_success(tmp_path, monkeypatch):
    stage = tmp_path / "stage"; stage.mkdir()
    (stage / "payload").write_bytes(b"approved")
    destination = tmp_path / "authority"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    durable, injected = False, False
    real_fsync, real_close = os.fsync, os.close

    def mark_durable(fd):
        nonlocal durable
        result = real_fsync(fd)
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == parent_identity and destination.exists():
            durable = True
        return result

    def fail_first_post_durable_close(fd):
        nonlocal injected
        if durable and not injected:
            injected = True
            real_close(fd)
            raise OSError("injected post-durable close failure")
        return real_close(fd)

    monkeypatch.setattr("ndm.e97_atomic.os.fsync", mark_durable)
    monkeypatch.setattr("ndm.e97_atomic.os.close", fail_first_post_durable_close)
    publish_directory_no_replace(stage, destination, expected_payloads={"payload": b"approved"})
    assert injected
    assert destination.joinpath("payload").read_bytes() == b"approved"


def test_pre_durability_descriptor_close_failure_stays_fail_closed(tmp_path, monkeypatch):
    stage = tmp_path / "stage"; stage.mkdir()
    (stage / "payload").write_bytes(b"approved")
    destination = tmp_path / "authority"
    real_close = os.close

    def fail_close(fd):
        real_close(fd)
        raise OSError("injected pre-durable close failure")

    monkeypatch.setattr("ndm.e97_atomic.os.close", fail_close)
    with pytest.raises(OSError, match="pre-durable"):
        publish_directory_no_replace(stage, destination, expected_payloads={"payload": b"approved"})
    assert not destination.exists()


def test_cleanup_directory_suppresses_close_errors_after_publication_cleanup(tmp_path, monkeypatch):
    stage = tmp_path / "stage"; stage.mkdir()
    (stage / "payload").write_text("safe")
    armed = False
    real_remove, real_close = atomic._remove_private_tree, os.close

    def mark_cleanup_complete(fd):
        nonlocal armed
        result = real_remove(fd)
        armed = True
        return result

    def fail_cleanup_close(fd):
        if armed:
            real_close(fd)
            raise OSError("injected cleanup close failure")
        return real_close(fd)

    monkeypatch.setattr(atomic, "_remove_private_tree", mark_cleanup_complete)
    monkeypatch.setattr("ndm.e97_atomic.os.close", fail_cleanup_close)
    cleanup_directory_best_effort(stage)
    assert not stage.exists()


def test_post_publication_stage_cleanup_is_best_effort_and_never_follows_exchange_symlink(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "payload").write_text("safe")
    outside = tmp_path / "outside"
    outside.mkdir()
    stage.rename(tmp_path / "saved-stage")
    stage.symlink_to(outside, target_is_directory=True)
    cleanup_directory_best_effort(stage)
    assert outside.is_dir()
    assert not stage.exists()


def test_temporary_name_cleanup_is_fsynced_after_its_exact_unlink(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"
    parent_info = tmp_path.stat()
    parent_identity = (parent_info.st_dev, parent_info.st_ino, parent_info.st_mode)
    events = []
    original_fsync, original_link, original_unlink = os.fsync, os.link, os.unlink

    def identity(fd):
        info = os.fstat(fd)
        return info.st_dev, info.st_ino, info.st_mode

    def tracked_fsync(fd):
        if identity(fd) == parent_identity:
            events.append(("fsync", *identity(fd)))
        return original_fsync(fd)

    def tracked_link(source, destination, *args, **kwargs):
        if identity(kwargs["dst_dir_fd"]) == parent_identity:
            events.append(("link", *parent_identity, destination))
        return original_link(source, destination, *args, **kwargs)

    def tracked_unlink(path, *args, **kwargs):
        if identity(kwargs["dir_fd"]) == parent_identity:
            events.append(("unlink", *parent_identity, path))
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr("ndm.e97_atomic.os.fsync", tracked_fsync)
    monkeypatch.setattr("ndm.e97_atomic.os.link", tracked_link)
    monkeypatch.setattr("ndm.e97_atomic.os.unlink", tracked_unlink)
    assert publish_bytes_no_replace(target, b"durable") is True
    assert [event[:4] for event in events] == [
        ("link", *parent_identity),
        ("fsync", *parent_identity),
        ("unlink", *parent_identity),
        ("fsync", *parent_identity),
    ]
    assert events[0][4] == "receipt.json"
    assert events[2][4].startswith(".receipt.json.") and events[2][4].endswith(".tmp")
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


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


def test_artifact_store_rejects_payloads_its_resolver_cannot_bound(tmp_path, monkeypatch):
    import ndm.e97_artifact_store as artifacts

    monkeypatch.setattr(artifacts, "_MAX_ARTIFACT_BYTES", 4)
    store = ArtifactStore(tmp_path / "store")
    assert store.put_bytes(b"four", suffix=".bin")["sha256"] == hashlib.sha256(b"four").hexdigest()
    with pytest.raises(ArtifactStoreError, match="bounded store limit"):
        store.put_bytes(b"five!", suffix=".bin")


def test_artifact_resolver_rejects_fifo_without_blocking(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    reference = store.put_bytes(b"artifact", suffix=".bin")
    artifact = store.root / reference["path"]
    artifact.unlink()
    os.mkfifo(artifact)
    with pytest.raises(ArtifactStoreError, match="bounded regular"):
        store.resolve_bytes(reference, suffix=".bin")


@pytest.mark.parametrize("authority", ("content", "marker"))
def test_published_receipt_resolver_rejects_fifo_authorities_without_blocking(tmp_path, authority):
    store = ArtifactStore(tmp_path / "store")
    task_id = "b" * 64
    payload = b'{"schema":"receipt","task_id":"' + task_id.encode() + b'"}'
    digest = hashlib.sha256(payload).hexdigest()
    marker = store.root / completed_receipt_relative_path(task_id, digest)
    content = store.root / receipt_relative_path(digest)
    marker.parent.mkdir(parents=True)
    content.parent.mkdir(parents=True)
    marker.write_bytes(payload)
    content.write_bytes(payload)
    target = content if authority == "content" else marker
    target.unlink()
    os.mkfifo(target)
    with pytest.raises(ArtifactStoreError, match="bounded regular"):
        store.resolve_published_receipt(
            {"sha256": digest, "path": completed_receipt_relative_path(task_id, digest)},
            task_id=task_id,
        )


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
