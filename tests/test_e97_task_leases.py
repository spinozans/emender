import hashlib
import json
import subprocess
import sys
import threading

import pytest

from ndm.e97_onpolicy_records import completion_marker_relative_path
from ndm.e97_task_leases import Lease, LeaseError, TaskLeases


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def task(name):
    return {"task_id": digest("task:" + name), "bundle_sha256": digest("bundle:" + name)}


def completion(item, terminal="terminal"):
    return {
        "task_id": item["task_id"], "bundle_sha256": item["bundle_sha256"],
        "archive_sha256": digest("archive"), "failed_terminal_sha256": digest("failed-" + terminal),
        "accepted_terminal_sha256": digest("accepted-" + terminal),
        "validator_receipt_sha256": digest("validator"), "validator_status": "pass",
    }


class Clock:
    def __init__(self):
        self.value = 1_000_000_000

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += int(seconds * 1_000_000_000)


def test_task_leases_orders_fsyncs_for_each_created_hierarchy_entry(tmp_path, monkeypatch):
    import os

    events = []
    original_fsync, original_mkdir = os.fsync, os.mkdir

    def identity(fd):
        info = os.fstat(fd)
        return info.st_dev, info.st_ino, info.st_mode

    def tracked_fsync(fd):
        events.append(("fsync", *identity(fd)))
        return original_fsync(fd)

    def tracked_mkdir(path, mode=0o777, *, dir_fd=None):
        events.append(("mkdir", *identity(dir_fd), path))
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr("ndm.e97_atomic.os.fsync", tracked_fsync)
    monkeypatch.setattr("ndm.e97_atomic.os.mkdir", tracked_mkdir)
    leases = TaskLeases(tmp_path / "new" / "nested" / "leases", clock_ns=Clock())
    try:
        root = tmp_path / "new" / "nested" / "leases"
        assert (root / "receipts").is_dir()
        assert (root / "completed").is_dir()

        def path_identity(path):
            info = path.stat()
            return info.st_dev, info.st_ino, info.st_mode

        identities = {
            "top": path_identity(tmp_path),
            "new": path_identity(tmp_path / "new"),
            "nested": path_identity(tmp_path / "new" / "nested"),
            "leases": path_identity(root),
        }
        known_identities = set(identities.values())
        hierarchy_events = [event for event in events if event[1:4] in known_identities]
        # Every pre-existing ancestor is also flushed when receipts/completed
        # are opened.  The exact sequence proves each new entry is flushed by
        # its containing inode before the next authority child is used.
        assert [event[:4] for event in hierarchy_events] == [
            ("mkdir", *identities["top"]),
            ("fsync", *identities["top"]),
            ("mkdir", *identities["new"]),
            ("fsync", *identities["new"]),
            ("mkdir", *identities["nested"]),
            ("fsync", *identities["nested"]),
            ("fsync", *identities["top"]),
            ("fsync", *identities["new"]),
            ("fsync", *identities["nested"]),
            ("mkdir", *identities["leases"]),
            ("fsync", *identities["leases"]),
            ("fsync", *identities["top"]),
            ("fsync", *identities["new"]),
            ("fsync", *identities["nested"]),
            ("mkdir", *identities["leases"]),
            ("fsync", *identities["leases"]),
            ("fsync", *identities["leases"]),
        ]
        assert [event[4] if len(event) > 4 else None for event in hierarchy_events] == [
            "new", None, "nested", None, "leases", None,
            None, None, None, "receipts", None,
            None, None, None, "completed", None, None,
        ]
    finally:
        leases.close()


def test_fair_claim_prefers_never_attempted_then_retry_and_restart(tmp_path):
    clock = Clock()
    root = tmp_path / "leases"
    tasks = [task("z"), task("a")]
    first = TaskLeases(root, clock_ns=clock)
    first.initialize(tasks)
    claimed = []

    def claim(owner):
        store = TaskLeases(root, clock_ns=clock)
        try:
            claimed.append(store.claim(owner, lease_seconds=1))
        finally:
            store.close()

    workers = [threading.Thread(target=claim, args=(f"owner-{index}",)) for index in range(2)]
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]
    assert {lease.task_id for lease in claimed if lease} == {item["task_id"] for item in tasks}
    clock.advance(2)
    recovered = first.claim("recovered", lease_seconds=1)
    assert recovered.task_id == min(item["task_id"] for item in tasks) and recovered.attempt == 2
    first.close()


def test_recovery_handles_before_and_after_receipt_publication(tmp_path):
    clock = Clock()
    root = tmp_path / "leases"
    item = task("a")
    leases = TaskLeases(root, clock_ns=clock)
    leases.initialize([item])
    lease = leases.claim("worker", lease_seconds=1)
    receipt = completion(item)

    prepared = leases.complete(lease, receipt, publish=False)
    assert leases.status()[0]["prepared_sha256"] == prepared
    clock.advance(2)
    reclaimed = leases.claim("recovery", lease_seconds=1)
    assert reclaimed and reclaimed.attempt == 2

    published = leases.complete(reclaimed, receipt, publish=False)
    payload = leases.db.execute("SELECT prepared_payload FROM tasks WHERE task_id=?", (item["task_id"],)).fetchone()[0]
    (root / "receipts" / f"{published}.json").write_bytes(payload)
    marker = root / completion_marker_relative_path(item["task_id"], published)
    marker.parent.mkdir(parents=True)
    marker.write_bytes(payload)
    leases.close()
    resumed = TaskLeases(root, clock_ns=clock)
    assert resumed.status()[0]["completion_sha256"] == published
    assert resumed.claim("other", lease_seconds=1) is None
    resumed.close()


def test_active_unpublished_is_unclaimable_expiry_reclaims_and_corruption_fails_closed(tmp_path):
    clock = Clock(); root = tmp_path / "leases"; item = task("a")
    leases = TaskLeases(root, clock_ns=clock); leases.initialize([item])
    lease = leases.claim("worker", lease_seconds=10)
    prepared = leases.complete(lease, completion(item), publish=False)
    # Active preparation is not a new lease candidate, even after explicit recovery.
    leases.recover(); assert leases.claim("other", lease_seconds=1) is None
    clock.advance(11)
    reclaimed = leases.claim("other", lease_seconds=1)
    assert reclaimed is not None and reclaimed.attempt == 2
    prepared = leases.complete(reclaimed, completion(item), publish=False)
    leases.db.execute("UPDATE tasks SET prepared_payload=? WHERE task_id=?", (b"corrupt", item["task_id"]))
    with pytest.raises(LeaseError, match="corrupt"):
        leases.recover()
    leases.close()


def test_expired_prepared_worker_cannot_publish_or_block_new_finalization(tmp_path):
    clock = Clock(); root = tmp_path / "leases"; item = task("stale")
    leases = TaskLeases(root, clock_ns=clock); leases.initialize([item])
    old = leases.claim("old", lease_seconds=1)
    old_digest = leases.complete(old, completion(item, terminal="old"), publish=False)
    old_payload = leases.db.execute(
        "SELECT prepared_payload FROM tasks WHERE task_id=?", (item["task_id"],)
    ).fetchone()[0]
    # Content evidence by itself is harmless after preparation expires.
    (root / "receipts" / f"{old_digest}.json").write_bytes(old_payload)
    clock.advance(2)
    current = leases.claim("new", lease_seconds=10)
    assert current is not None and current.attempt == 2
    with pytest.raises(LeaseError, match="lease is not active|no longer bound"):
        leases.complete(old, completion(item, terminal="old"), publish=True)
    assert not (root / completion_marker_relative_path(item["task_id"], old_digest)).exists()

    current_digest = leases.complete(current, completion(item, terminal="new"), publish=True)
    assert (root / "receipts" / f"{old_digest}.json").read_bytes() == old_payload
    assert (root / completion_marker_relative_path(item["task_id"], current_digest)).is_file()
    assert leases.status()[0]["completion_sha256"] == current_digest
    leases.close()


def test_recovery_fails_closed_when_completed_sqlite_row_lacks_marker_or_content(tmp_path):
    clock = Clock(); root = tmp_path / "leases"; item = task("durability")
    leases = TaskLeases(root, clock_ns=clock); leases.initialize([item])
    lease = leases.claim("worker", lease_seconds=10)
    digest = leases.complete(lease, completion(item), publish=True)
    marker = root / completion_marker_relative_path(item["task_id"], digest)
    marker.unlink()
    with pytest.raises(LeaseError, match="completed row lacks durable"):
        leases.recover()
    leases.close()


def test_completed_marker_authority_survives_sqlite_index_loss(tmp_path):
    clock = Clock(); root = tmp_path / "leases"; item = task("index-loss")
    leases = TaskLeases(root, clock_ns=clock); leases.initialize([item])
    original = leases.claim("worker", lease_seconds=10)
    digest_value = leases.complete(original, completion(item), publish=True)
    leases.close()
    # SQLite is only an index.  Removing every SQLite sidecar must not permit
    # the finalized marker/content authority to be reclaimed or re-completed.
    for path in root.glob("leases.sqlite3*"):
        path.unlink()
    recovered = TaskLeases(root, clock_ns=clock)
    try:
        recovered.initialize([item])
        row = recovered.status()[0]
        assert row["completion_sha256"] == digest_value
        assert recovered.claim("different-worker", lease_seconds=10) is None
    finally:
        recovered.close()


def test_recovery_rejects_unknown_or_multiple_marker_authorities(tmp_path):
    import shutil

    clock = Clock(); root = tmp_path / "leases"; item = task("marker-conflict")
    primary = TaskLeases(root, clock_ns=clock); primary.initialize([item])
    primary_digest = primary.complete(primary.claim("primary", lease_seconds=10), completion(item), publish=True)
    primary.close()

    # A separately finalized marker for the same sealed task is a conflicting
    # authority, even if its content itself is canonical and digest-matching.
    alternate_root = tmp_path / "alternate"
    alternate = TaskLeases(alternate_root, clock_ns=Clock()); alternate.initialize([item])
    alternate_digest = alternate.complete(
        alternate.claim("alternate", lease_seconds=10), completion(item, terminal="alternate"), publish=True)
    alternate.close()
    shutil.copyfile(alternate_root / "receipts" / f"{alternate_digest}.json", root / "receipts" / f"{alternate_digest}.json")
    alternate_marker = alternate_root / completion_marker_relative_path(item["task_id"], alternate_digest)
    target_marker = root / completion_marker_relative_path(item["task_id"], alternate_digest)
    target_marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(alternate_marker, target_marker)
    with pytest.raises(LeaseError, match="multiple or unknown"):
        TaskLeases(root, clock_ns=clock)

    # The same inventory also refuses a canonical marker for a task absent from
    # this sealed collection rather than silently treating it as an orphan.
    unknown = task("unknown-marker")
    foreign_root = tmp_path / "foreign"
    foreign = TaskLeases(foreign_root, clock_ns=Clock()); foreign.initialize([unknown])
    foreign_digest = foreign.complete(foreign.claim("foreign", lease_seconds=10), completion(unknown), publish=True)
    foreign.close()
    shutil.copyfile(foreign_root / "receipts" / f"{foreign_digest}.json", root / "receipts" / f"{foreign_digest}.json")
    foreign_marker = foreign_root / completion_marker_relative_path(unknown["task_id"], foreign_digest)
    target_marker = root / completion_marker_relative_path(unknown["task_id"], foreign_digest)
    target_marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(foreign_marker, target_marker)
    # Remove the duplicate first so this assertion reaches unknown-task handling.
    (root / completion_marker_relative_path(item["task_id"], primary_digest)).unlink()
    (root / completion_marker_relative_path(item["task_id"], alternate_digest)).unlink()
    with pytest.raises(LeaseError, match="unknown to sealed collection"):
        TaskLeases(root, clock_ns=clock)


@pytest.mark.parametrize("row_state", ("prepared", "completed"))
@pytest.mark.parametrize("authority", ("content", "marker"))
@pytest.mark.parametrize("component", ("final", "parent"))
def test_recovery_rejects_symlinked_receipt_authority_components(
        tmp_path, row_state, authority, component):
    """Recovery may not promote or retain authority through a linked path."""

    clock = Clock()
    root = tmp_path / "leases"
    item = task(f"{row_state}-{authority}-{component}")
    leases = TaskLeases(root, clock_ns=clock)
    leases.initialize([item])
    lease = leases.claim("worker", lease_seconds=10)
    if row_state == "completed":
        digest_value = leases.complete(lease, completion(item), publish=True)
    else:
        digest_value = leases.complete(lease, completion(item), publish=False)
        payload = leases.db.execute(
            "SELECT prepared_payload FROM tasks WHERE task_id=?", (item["task_id"],)
        ).fetchone()[0]
        content = root / "receipts" / f"{digest_value}.json"
        marker = root / completion_marker_relative_path(item["task_id"], digest_value)
        marker.parent.mkdir(parents=True, exist_ok=True)
        content.write_bytes(payload)
        marker.write_bytes(payload)
    payload = (root / "receipts" / f"{digest_value}.json").read_bytes()
    target = (root / "receipts" / f"{digest_value}.json" if authority == "content"
              else root / completion_marker_relative_path(item["task_id"], digest_value))
    outside = tmp_path / f"outside-{row_state}-{authority}-{component}"
    if component == "final":
        outside.write_bytes(payload)
        target.unlink()
        target.symlink_to(outside)
    else:
        original_parent = target.parent
        saved_parent = original_parent.with_name(original_parent.name + "-saved")
        original_parent.rename(saved_parent)
        outside.mkdir()
        (outside / target.name).write_bytes(payload)
        original_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(LeaseError, match="cannot be opened safely"):
        leases.recover()
    leases.close()


def test_heartbeat_mints_new_identity_and_rejects_old_prepared_and_expired_leases(tmp_path):
    clock = Clock()
    leases = TaskLeases(tmp_path / "leases", clock_ns=clock)
    item = task("a")
    leases.initialize([item])
    original = leases.claim("worker", lease_seconds=10)
    renewed = leases.heartbeat(original, lease_seconds=10)
    assert renewed.identity != original.identity
    with pytest.raises(LeaseError, match="stale"):
        leases.heartbeat(original, lease_seconds=10)
    prepared = leases.complete(renewed, completion(item), publish=False)
    assert prepared
    with pytest.raises(LeaseError, match="prepared or completed"):
        leases.heartbeat(renewed, lease_seconds=10)
    leases.close()

    expiring = TaskLeases(tmp_path / "expiry", clock_ns=clock)
    expiring.initialize([task("b")])
    lease = expiring.claim("worker", lease_seconds=1)
    clock.advance(2)
    with pytest.raises(LeaseError, match="expired"):
        expiring.heartbeat(lease, lease_seconds=1)
    expiring.close()


def _wait_for_begin(connection):
    entered = threading.Event()

    def trace(statement):
        if statement == "BEGIN IMMEDIATE":
            entered.set()

    connection.set_trace_callback(trace)
    return entered


def test_claim_samples_clock_after_writer_lock_wait(tmp_path):
    clock = Clock(); root = tmp_path / "leases"; item = task("claim-lock")
    owner = TaskLeases(root, clock_ns=clock); owner.initialize([item])
    contender = TaskLeases(root, clock_ns=clock)
    owner.db.execute("BEGIN IMMEDIATE")
    result = []
    entered = _wait_for_begin(contender.db)
    worker = threading.Thread(target=lambda: result.append(contender.claim("worker", lease_seconds=1)))
    worker.start()
    assert entered.wait(timeout=5)
    clock.advance(5)
    owner.db.execute("COMMIT")
    worker.join(timeout=5)
    assert result and result[0] is not None
    assert result[0].deadline_ns == clock.value + 1_000_000_000
    contender.close(); owner.close()


@pytest.mark.parametrize("operation", ("heartbeat", "complete"))
def test_lease_deadline_is_checked_after_writer_lock_wait(tmp_path, operation):
    clock = Clock(); root = tmp_path / "leases"; item = task(f"{operation}-lock")
    owner = TaskLeases(root, clock_ns=clock); owner.initialize([item])
    lease = owner.claim("worker", lease_seconds=1)
    contender = TaskLeases(root, clock_ns=clock)
    owner.db.execute("BEGIN IMMEDIATE")
    failures = []
    entered = _wait_for_begin(contender.db)

    def act():
        try:
            if operation == "heartbeat":
                contender.heartbeat(lease, lease_seconds=1)
            else:
                contender.complete(lease, completion(item), publish=False)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=act)
    worker.start()
    assert entered.wait(timeout=5)
    clock.advance(2)
    owner.db.execute("COMMIT")
    worker.join(timeout=5)
    assert len(failures) == 1
    assert isinstance(failures[0], LeaseError)
    assert "expired" in str(failures[0]) or "not active" in str(failures[0])
    contender.close(); owner.close()


@pytest.mark.parametrize("mutate", (
    lambda lease: {**lease, "attempt": True},
    lambda lease: {**lease, "attempt": 1.0},
    lambda lease: {**lease, "deadline_ns": 1.0},
    lambda lease: {**lease, "owner": ""},
    lambda lease: {**lease, "extra": "forbidden"},
    lambda _lease: ["foreign"],
))
def test_complete_cli_rejects_malformed_lease_before_preparation_or_publication(tmp_path, mutate):
    root = tmp_path / "leases"; item = task("malformed-cli")
    leases = TaskLeases(root, clock_ns=Clock())
    leases.initialize([item])
    claimed = leases.claim("worker", lease_seconds=30)
    receipt_path = tmp_path / "receipt.json"; receipt_path.write_text(json.dumps(completion(item)))
    malformed = mutate(dict(claimed.__dict__))
    leases.close()
    outcome = subprocess.run([
        sys.executable, "-m", "scripts.e97_task_leases", "--root", str(root), "complete",
        "--lease-json", json.dumps(malformed), "--receipt-json", str(receipt_path),
    ], capture_output=True, text=True)
    assert outcome.returncode != 0
    inspection = TaskLeases(root, clock_ns=Clock())
    try:
        row = inspection.status()[0]
        assert row["prepared_sha256"] is None and row["completion_sha256"] is None
        assert not list((root / "receipts").glob("*.json"))
        assert not list((root / "completed").rglob("*.json"))
    finally:
        inspection.close()


def test_task_lease_rejects_foreign_or_typed_lease_before_database_mutation(tmp_path):
    clock = Clock(); root = tmp_path / "leases"; item = task("strict-lease")
    leases = TaskLeases(root, clock_ns=clock); leases.initialize([item])
    valid = leases.claim("worker", lease_seconds=30)

    class ForeignLease:
        task_id = valid.task_id
        owner = valid.owner
        attempt = valid.attempt
        deadline_ns = valid.deadline_ns
        identity = valid.identity

    malformed = [
        Lease(valid.task_id, valid.owner, True, valid.deadline_ns, valid.identity),
        Lease(valid.task_id, valid.owner, valid.attempt, float(valid.deadline_ns), valid.identity),
        ForeignLease(),
    ]
    for lease in malformed:
        with pytest.raises(LeaseError, match="lease"):
            leases.complete(lease, completion(item))
    row = leases.status()[0]
    assert row["prepared_sha256"] is None and row["completion_sha256"] is None
    assert not list((root / "receipts").glob("*.json"))
    assert not list((root / "completed").rglob("*.json"))
    leases.close()


def test_recover_samples_expiry_after_writer_lock_wait(tmp_path):
    clock = Clock(); root = tmp_path / "leases"; item = task("recover-lock")
    owner = TaskLeases(root, clock_ns=clock); owner.initialize([item])
    lease = owner.claim("worker", lease_seconds=1)
    owner.complete(lease, completion(item), publish=False)
    contender = TaskLeases(root, clock_ns=clock)
    owner.db.execute("BEGIN IMMEDIATE")
    entered = _wait_for_begin(contender.db)
    worker = threading.Thread(target=contender.recover)
    worker.start()
    assert entered.wait(timeout=5)
    clock.advance(2)
    owner.db.execute("COMMIT")
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert contender.status()[0]["prepared_sha256"] is None
    contender.close(); owner.close()


def test_complete_publication_second_transaction_rechecks_expiry(tmp_path):
    class SecondTransactionExpiryClock(Clock):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __call__(self):
            self.calls += 1
            # Initial construction/claim use the base time.  The two complete
            # transactions see active then expired time respectively.
            return self.value if self.calls < 5 else self.value + 2_000_000_000

    clock = SecondTransactionExpiryClock(); root = tmp_path / "leases"; item = task("second-transaction")
    leases = TaskLeases(root, clock_ns=clock); leases.initialize([item])
    lease = leases.claim("worker", lease_seconds=1)
    with pytest.raises(LeaseError, match="no longer bound"):
        leases.complete(lease, completion(item), publish=True)
    row = leases.status()[0]
    assert row["prepared_sha256"] is not None and row["completion_sha256"] is None
    assert not list((root / "receipts").glob("*.json"))
    assert not list((root / "completed").rglob("*.json"))
    leases.close()


def test_owner_utf8_bound_and_completion_receipt_bound_fail_before_publication(tmp_path, monkeypatch):
    import ndm.e97_task_leases as lease_module

    clock = Clock(); root = tmp_path / "leases"; item = task("bounded-owner")
    leases = TaskLeases(root, clock_ns=clock); leases.initialize([item])
    owner = "é" * (lease_module._MAX_OWNER_UTF8_BYTES // 2)
    lease = leases.claim(owner, lease_seconds=1)
    assert lease is not None
    # A boundary-sized valid owner remains recoverable after publication.
    digest_value = leases.complete(lease, completion(item), publish=True)
    leases.close()
    resumed = TaskLeases(root, clock_ns=clock)
    assert resumed.status()[0]["completion_sha256"] == digest_value
    resumed.close()

    oversized = TaskLeases(tmp_path / "oversized", clock_ns=Clock())
    oversized.initialize([item])
    with pytest.raises(LeaseError, match="UTF-8 bytes"):
        oversized.claim("é" * (lease_module._MAX_OWNER_UTF8_BYTES // 2 + 1), lease_seconds=1)
    row = oversized.status()[0]
    assert row["attempts"] == 0 and row["prepared_sha256"] is None
    oversized.close()

    constrained = TaskLeases(tmp_path / "receipt-bound", clock_ns=Clock())
    constrained.initialize([item])
    lease = constrained.claim("worker", lease_seconds=1)
    # Exercise the canonical payload guard before it is prepared/published.
    monkeypatch.setattr(lease_module, "_MAX_RECEIPT_BYTES", 1)
    with pytest.raises(LeaseError, match="canonical completion receipt"):
        constrained.complete(lease, completion(item))
    row = constrained.status()[0]
    assert row["prepared_sha256"] is None and row["completion_sha256"] is None
    constrained.close()


@pytest.mark.parametrize("lease_seconds", [True, float("nan"), float("inf"), float("-inf"), 0, -1, 1e-12, 10 ** 20, "1"])
def test_claim_rejects_nonfinite_boolean_subnanosecond_and_unrepresentable_durations(tmp_path, lease_seconds):
    leases = TaskLeases(tmp_path / "leases", clock_ns=Clock())
    item = task("strict-duration"); leases.initialize([item])
    with pytest.raises(LeaseError, match="lease_seconds"):
        leases.claim("worker", lease_seconds=lease_seconds)
    assert leases.status()[0]["attempts"] == 0
    leases.close()


def test_lease_duration_cli_fails_closed_for_nonfinite_and_subnanosecond_values(tmp_path):
    root = tmp_path / "leases"; item = task("cli-duration")
    tasks = tmp_path / "tasks.jsonl"; tasks.write_text(json.dumps(item) + "\n")
    subprocess.run([
        sys.executable, "-m", "scripts.e97_task_leases", "--root", str(root), "initialize",
        "--tasks-jsonl", str(tasks),
    ], check=True, capture_output=True, text=True)
    for value in ("nan", "inf", "0", "0.0000000001"):
        result = subprocess.run([
            sys.executable, "-m", "scripts.e97_task_leases", "--root", str(root), "claim",
            "--owner", "worker", "--lease-seconds", value,
        ], capture_output=True, text=True)
        assert result.returncode != 0
    leases = TaskLeases(root, clock_ns=Clock())
    assert leases.status()[0]["attempts"] == 0
    leases.close()


def test_heartbeat_cli_returns_the_new_bound_lease(tmp_path):
    root = tmp_path / "leases"
    item = task("cli")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(item) + "\n")
    subprocess.run([
        sys.executable, "-m", "scripts.e97_task_leases", "--root", str(root), "initialize",
        "--tasks-jsonl", str(tasks),
    ], check=True, capture_output=True, text=True)
    claimed = json.loads(subprocess.run([
        sys.executable, "-m", "scripts.e97_task_leases", "--root", str(root), "claim",
        "--owner", "cli", "--lease-seconds", "30",
    ], check=True, capture_output=True, text=True).stdout)
    renewed = json.loads(subprocess.run([
        sys.executable, "-m", "scripts.e97_task_leases", "--root", str(root), "heartbeat",
        "--lease-json", json.dumps(claimed), "--lease-seconds", "30",
    ], check=True, capture_output=True, text=True).stdout)
    assert renewed["identity"] != claimed["identity"]


def test_sealed_collection_and_stale_completion_leave_no_receipt(tmp_path):
    clock = Clock()
    leases = TaskLeases(tmp_path / "leases", clock_ns=clock)
    item = task("a")
    leases.initialize([item])
    with pytest.raises(LeaseError, match="sealed collection"):
        leases.initialize([item, task("b")])
    lease = leases.claim("worker", lease_seconds=1)
    clock.advance(2)
    with pytest.raises(LeaseError, match="not active"):
        leases.complete(lease, completion(item))
    assert not list((tmp_path / "leases" / "receipts").glob("*.json"))
    leases.close()
