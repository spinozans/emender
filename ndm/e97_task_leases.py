"""Fail-closed SQLite task leases with recoverable prepared completions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable, Iterable, Mapping

from ndm.e97_atomic import fsync_directory, publish_bytes_no_replace
from ndm.e97_onpolicy_records import completion_marker_relative_path

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class LeaseError(RuntimeError):
    """The requested lease transition is unsafe."""


@dataclass(frozen=True)
class Lease:
    task_id: str
    owner: str
    attempt: int
    deadline_ns: int
    identity: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise LeaseError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _lease_identity(task_id: str, owner: str, attempt: int, deadline_ns: int) -> str:
    return _sha(_canonical({"task_id": task_id, "owner": owner, "attempt": attempt,
                            "deadline_ns": deadline_ns}))


class TaskLeases:
    """Immutable task collection and receipt-backed, crash-recoverable leases."""

    def __init__(self, root: Path | str, *, clock_ns: Callable[[], int] = time.time_ns) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipts = self.root / "receipts"
        self.receipts.mkdir(exist_ok=True)
        self.completed = self.root / "completed"
        self.completed.mkdir(exist_ok=True)
        self.clock_ns = clock_ns
        self.db = sqlite3.connect(self.root / "leases.sqlite3", isolation_level=None,
                                  timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS collection (digest TEXT PRIMARY KEY, task_count INTEGER NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, bundle_sha256 TEXT NOT NULL, task_json TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, owner TEXT, deadline_ns INTEGER,
            lease_identity TEXT, prepared_sha256 TEXT, prepared_payload BLOB,
            completion_sha256 TEXT)""")
        # Migration from the first uncommitted slice.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(tasks)")}
        if "prepared_payload" not in columns:
            self.db.execute("ALTER TABLE tasks ADD COLUMN prepared_payload BLOB")
        self.recover()

    def close(self) -> None:
        self.db.close()

    def initialize(self, tasks: Iterable[Mapping[str, object]]) -> None:
        rows = []
        for task in tasks:
            if not isinstance(task, Mapping) or set(task) != {"task_id", "bundle_sha256"}:
                raise LeaseError("lease task must contain only task_id and bundle_sha256")
            rows.append((_digest(task["task_id"], "task_id"),
                         _digest(task["bundle_sha256"], "bundle_sha256"),
                         _canonical(dict(task)).decode("utf-8")))
        if not rows or len({row[0] for row in rows}) != len(rows):
            raise LeaseError("collection must have unique task IDs")
        rows.sort()
        collection = _sha(_canonical([{"task_id": item[0], "bundle_sha256": item[1]} for item in rows]))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT digest,task_count FROM collection").fetchone()
            if existing is None:
                self.db.execute("INSERT INTO collection(digest,task_count) VALUES (?,?)", (collection, len(rows)))
                self.db.executemany("INSERT INTO tasks(task_id,bundle_sha256,task_json) VALUES (?,?,?)", rows)
            elif existing != (collection, len(rows)):
                raise LeaseError("lease collection differs from sealed collection")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _completion_marker(self, task_id: str, digest: str) -> Path:
        try:
            return self.root / completion_marker_relative_path(task_id, digest)
        except ValueError as exc:
            raise LeaseError("completion marker identity is invalid") from exc

    @staticmethod
    def _matching_bytes(path: Path, payload: bytes) -> bool:
        try:
            return path.read_bytes() == payload
        except FileNotFoundError:
            return False

    def recover(self) -> None:
        """Finalize only marker-backed preparations; expired content-only receipts orphan."""

        now = self.clock_ns()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute(
                "SELECT task_id,prepared_sha256,prepared_payload,deadline_ns FROM tasks WHERE prepared_sha256 IS NOT NULL"
            ).fetchall()
            for task_id, digest, payload, deadline in rows:
                if not isinstance(payload, bytes) or _sha(payload) != digest:
                    raise LeaseError("prepared completion binding is corrupt")
                content = self.receipts / f"{digest}.json"
                marker = self._completion_marker(task_id, digest)
                content_matches = self._matching_bytes(content, payload)
                marker_matches = self._matching_bytes(marker, payload)
                if marker_matches and content_matches:
                    self.db.execute(
                        "UPDATE tasks SET completion_sha256=?,prepared_sha256=NULL,prepared_payload=NULL WHERE task_id=?",
                        (digest, task_id),
                    )
                elif marker_matches and not content_matches:
                    raise LeaseError("completion marker exists without matching content receipt")
                elif deadline is not None and deadline <= now:
                    # A content-only receipt is harmless orphan evidence.  It
                    # cannot finalize or block a subsequent lease attempt.
                    self.db.execute("UPDATE tasks SET prepared_sha256=NULL,prepared_payload=NULL WHERE task_id=?", (task_id,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def claim(self, owner: str, *, lease_seconds: float) -> Lease | None:
        if not isinstance(owner, str) or not owner or lease_seconds <= 0:
            raise LeaseError("owner and positive lease_seconds are required")
        self.recover()
        now = self.clock_ns()
        deadline = now + int(lease_seconds * 1_000_000_000)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("""SELECT task_id,attempts FROM tasks
                WHERE completion_sha256 IS NULL AND prepared_sha256 IS NULL
                  AND (deadline_ns IS NULL OR deadline_ns <= ?)
                ORDER BY CASE WHEN attempts=0 THEN 0 ELSE 1 END, attempts, task_id LIMIT 1""", (now,)).fetchone()
            if row is None:
                self.db.execute("COMMIT")
                return None
            task_id, attempts = row
            attempt = int(attempts) + 1
            identity = _lease_identity(task_id, owner, attempt, deadline)
            self.db.execute("UPDATE tasks SET attempts=?,owner=?,deadline_ns=?,lease_identity=? WHERE task_id=?", (attempt, owner, deadline, identity, task_id))
            self.db.execute("COMMIT")
            return Lease(task_id, owner, attempt, deadline, identity)
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def heartbeat(self, lease: Lease, *, lease_seconds: float) -> Lease:
        """Atomically renew one active lease and invalidate its old identity."""

        if not isinstance(lease, Lease) or lease_seconds <= 0:
            raise LeaseError("an active lease and positive lease_seconds are required")
        now = self.clock_ns()
        requested_deadline = now + int(lease_seconds * 1_000_000_000)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT owner,attempts,deadline_ns,lease_identity,prepared_sha256,completion_sha256 "
                "FROM tasks WHERE task_id=?", (lease.task_id,),
            ).fetchone()
            if row is None:
                raise LeaseError("lease task is unknown")
            owner, attempt, deadline, identity, prepared, completed = row
            if prepared is not None or completed is not None:
                raise LeaseError("prepared or completed leases cannot be renewed")
            if deadline is None or deadline <= now:
                raise LeaseError("expired leases cannot be renewed")
            if (owner, attempt, deadline, identity) != (
                lease.owner, lease.attempt, lease.deadline_ns, lease.identity,
            ):
                raise LeaseError("stale lease cannot be renewed")
            # A same-tick heartbeat must still mint a new identity.
            renewed_deadline = max(requested_deadline, int(deadline) + 1)
            renewed_identity = _lease_identity(lease.task_id, lease.owner, lease.attempt, renewed_deadline)
            self.db.execute(
                "UPDATE tasks SET deadline_ns=?,lease_identity=? WHERE task_id=?",
                (renewed_deadline, renewed_identity, lease.task_id),
            )
            self.db.execute("COMMIT")
            return Lease(lease.task_id, lease.owner, lease.attempt, renewed_deadline, renewed_identity)
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _publish_finalized_receipt(self, task_id: str, digest: str, payload: bytes) -> None:
        """Publish content then its task marker while the SQLite writer lock is held."""

        try:
            publish_bytes_no_replace(self.receipts / f"{digest}.json", payload)
            marker = self._completion_marker(task_id, digest)
            marker.parent.mkdir(parents=True, exist_ok=True)
            fsync_directory(marker.parent.parent)
            publish_bytes_no_replace(marker, payload)
        except ValueError as exc:
            raise LeaseError("receipt or completion marker digest collision") from exc

    def complete(self, lease: Lease, receipt: Mapping[str, object], *, publish: bool = True) -> str:
        required = {"task_id", "bundle_sha256", "archive_sha256", "failed_terminal_sha256", "accepted_terminal_sha256", "validator_receipt_sha256", "validator_status"}
        if not isinstance(receipt, Mapping) or set(receipt) != required:
            raise LeaseError("completion binding fields mismatch")
        for key in required - {"validator_status"}:
            _digest(receipt[key], key)
        if receipt["validator_status"] != "pass":
            raise LeaseError("validator status must be pass")
        payload = _canonical({"schema": "emender-e97-task-completion-receipt-v3", "task_id": lease.task_id,
                              "lease": lease.__dict__, "receipt": dict(receipt)})
        digest = _sha(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT bundle_sha256,owner,attempts,deadline_ns,lease_identity,prepared_sha256,completion_sha256 FROM tasks WHERE task_id=?", (lease.task_id,)).fetchone()
            if row is None or receipt["task_id"] != lease.task_id or receipt["bundle_sha256"] != row[0]:
                raise LeaseError("completion task/bundle binding mismatch")
            if row[6] is not None:
                if row[6] == digest:
                    self.db.execute("COMMIT")
                    return digest
                raise LeaseError("conflicting completion receipt")
            if row[5] not in (None, digest):
                raise LeaseError("conflicting prepared completion")
            if row[5] is None and (tuple(row[1:5]) != (lease.owner, lease.attempt, lease.deadline_ns, lease.identity) or lease.deadline_ns <= self.clock_ns()):
                raise LeaseError("lease is not active")
            self.db.execute("UPDATE tasks SET prepared_sha256=?,prepared_payload=? WHERE task_id=?", (digest, payload, lease.task_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        if not publish:
            return digest

        # Reacquire the sole SQLite writer lock before any authority file is
        # published.  This prevents an expired prepared lease from publishing
        # after recovery has cleared it and a different claimant has won.
        now = self.clock_ns()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT owner,attempts,deadline_ns,lease_identity,prepared_sha256,prepared_payload,completion_sha256 "
                "FROM tasks WHERE task_id=?", (lease.task_id,),
            ).fetchone()
            if row is None:
                raise LeaseError("lease task is unknown")
            owner, attempt, deadline, identity, prepared_digest, prepared_payload, completed = row
            if completed is not None:
                if completed == digest:
                    self.db.execute("COMMIT")
                    return digest
                raise LeaseError("conflicting completion receipt")
            if (prepared_digest != digest or prepared_payload != payload
                    or deadline is None or deadline <= now
                    or (owner, attempt, deadline, identity) != (
                        lease.owner, lease.attempt, lease.deadline_ns, lease.identity,
                    )):
                raise LeaseError("prepared completion is no longer bound to an active lease")
            self._publish_finalized_receipt(lease.task_id, digest, payload)
            self.db.execute(
                "UPDATE tasks SET completion_sha256=?,prepared_sha256=NULL,prepared_payload=NULL WHERE task_id=?",
                (digest, lease.task_id),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return digest

    def status(self) -> list[dict[str, object]]:
        rows = self.db.execute("SELECT task_id,attempts,owner,deadline_ns,lease_identity,prepared_sha256,completion_sha256 FROM tasks ORDER BY task_id")
        return [dict(zip(("task_id", "attempts", "owner", "deadline_ns", "lease_identity", "prepared_sha256", "completion_sha256"), row)) for row in rows]
