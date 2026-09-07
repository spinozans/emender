import hashlib
import json
import subprocess
import sys
import threading

import pytest

from ndm.e97_onpolicy_records import completion_marker_relative_path
from ndm.e97_task_leases import LeaseError, TaskLeases


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
