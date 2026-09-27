"""Fail-closed SQLite task leases with recoverable prepared completions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable, Iterable, Mapping

from ndm.e97_atomic import (
    durable_directory,
    fsync_directory,
    open_directory_no_follow,
    publish_bytes_no_replace,
    read_regular_file_no_follow,
)
from ndm.e97_onpolicy_records import completion_marker_relative_path

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_RECEIPT_BYTES = 16 << 20
_MAX_OWNER_UTF8_BYTES = 4096
_MAX_SQLITE_INTEGER = (1 << 63) - 1
_NANOSECONDS_PER_SECOND = 1_000_000_000


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


def _owner(value: object) -> str:
    """Accept a nonempty owner whose receipt representation remains bounded."""

    if not isinstance(value, str) or not value:
        raise LeaseError("lease owner must be non-empty text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LeaseError("lease owner must be valid UTF-8 text") from exc
    if len(encoded) > _MAX_OWNER_UTF8_BYTES:
        raise LeaseError("lease owner UTF-8 bytes exceed the sealed bound")
    return value


def _lease_duration_ns(value: object) -> int:
    """Convert one finite non-Boolean duration to a positive SQLite-safe ns count."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise LeaseError("lease_seconds must be a finite non-Boolean numeric value")
    if isinstance(value, int):
        if value <= 0:
            raise LeaseError("lease_seconds must be positive")
        nanoseconds = value * _NANOSECONDS_PER_SECOND
    else:
        try:
            seconds = float(value)
        except (OverflowError, ValueError) as exc:
            raise LeaseError("lease_seconds must be finite") from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise LeaseError("lease_seconds must be finite and positive")
        scaled = seconds * _NANOSECONDS_PER_SECOND
        if not math.isfinite(scaled):
            raise LeaseError("lease_seconds nanoseconds are not representable")
        nanoseconds = int(scaled)
    if not 0 < nanoseconds <= _MAX_SQLITE_INTEGER:
        raise LeaseError("lease_seconds nanoseconds are not representable")
    return nanoseconds


def _deadline_ns(now: object, duration_ns: int) -> int:
    """Return a positive signed-64-bit deadline without SQLite coercion."""

    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise LeaseError("lease clock nanoseconds are invalid")
    if duration_ns > _MAX_SQLITE_INTEGER - now:
        raise LeaseError("lease deadline_ns is not representable")
    deadline = now + duration_ns
    if deadline <= 0:
        raise LeaseError("lease deadline_ns must be positive")
    return deadline


def validate_lease(value: object) -> Lease:
    """Accept only one exact, self-authenticating lease value at every ingress."""

    if type(value) is not Lease:
        raise LeaseError("lease must be an exact Lease value")
    lease = value
    task_id = _digest(lease.task_id, "lease task_id")
    _owner(lease.owner)
    if isinstance(lease.attempt, bool) or not isinstance(lease.attempt, int) or lease.attempt <= 0:
        raise LeaseError("lease attempt must be a positive exact integer")
    if isinstance(lease.deadline_ns, bool) or not isinstance(lease.deadline_ns, int) or lease.deadline_ns <= 0:
        raise LeaseError("lease deadline_ns must be a positive exact integer")
    identity = _digest(lease.identity, "lease identity")
    if identity != _lease_identity(task_id, lease.owner, lease.attempt, lease.deadline_ns):
        raise LeaseError("lease identity does not bind lease fields")
    return Lease(task_id, lease.owner, lease.attempt, lease.deadline_ns, identity)


class TaskLeases:
    """Immutable task collection and receipt-backed, crash-recoverable leases."""

    def __init__(self, root: Path | str, *, clock_ns: Callable[[], int] = time.time_ns) -> None:
        self.root = Path(root)
        # The hierarchy itself is completion authority.  Persist every new
        # parent entry before SQLite or receipt publication can reference it.
        durable_directory(self.root)
        self.receipts = self.root / "receipts"
        durable_directory(self.receipts)
        self.completed = self.root / "completed"
        durable_directory(self.completed)
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
        # SQLite creates its durable files inside the already durable root.
        # Flush its directory entry before a lease can be claimed.
        fsync_directory(self.root)
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
            # A recreated/rolled-back SQLite file is only an index.  Once this
            # call seals the task set, canonical marker/content authority must
            # be reconstructed before a claimant can observe any task.
            self._recover_locked(None)
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
        """Compare one bounded receipt through all-component no-follow reads."""

        try:
            return read_regular_file_no_follow(path, maximum=_MAX_RECEIPT_BYTES) == payload
        except FileNotFoundError:
            return False
        except ValueError as exc:
            raise LeaseError("receipt authority cannot be opened safely") from exc

    @staticmethod
    def _validated_completion_authority(task_id: str, digest: str, payload: bytes) -> tuple[Lease, Mapping[str, object]]:
        """Validate one canonical marker/content payload before it can recover an index."""

        if _sha(payload) != digest:
            raise LeaseError("completion authority digest is corrupt")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LeaseError("completion authority is malformed") from exc
        if _canonical(decoded) != payload:
            raise LeaseError("completion authority is not canonical")
        required = {"schema", "task_id", "lease", "receipt"}
        if (not isinstance(decoded, Mapping) or set(decoded) != required
                or decoded["schema"] != "emender-e97-task-completion-receipt-v3"
                or decoded["task_id"] != task_id or not isinstance(decoded["lease"], Mapping)
                or not isinstance(decoded["receipt"], Mapping)):
            raise LeaseError("completion authority fields are malformed")
        lease_raw = decoded["lease"]
        if set(lease_raw) != {"task_id", "owner", "attempt", "deadline_ns", "identity"}:
            raise LeaseError("completion authority lease is malformed")
        try:
            lease = validate_lease(Lease(
                task_id=lease_raw["task_id"], owner=lease_raw["owner"], attempt=lease_raw["attempt"],
                deadline_ns=lease_raw["deadline_ns"], identity=lease_raw["identity"],
            ))
        except (KeyError, LeaseError) as exc:
            raise LeaseError("completion authority lease is malformed") from exc
        receipt = decoded["receipt"]
        required_receipt = {
            "task_id", "bundle_sha256", "archive_sha256", "failed_terminal_sha256",
            "accepted_terminal_sha256", "validator_receipt_sha256", "validator_status",
        }
        if set(receipt) != required_receipt or receipt.get("task_id") != task_id or receipt.get("validator_status") != "pass":
            raise LeaseError("completion authority receipt is malformed")
        try:
            for key in required_receipt - {"validator_status"}:
                _digest(receipt[key], key)
        except LeaseError as exc:
            raise LeaseError("completion authority receipt is malformed") from exc
        if lease.task_id != task_id:
            raise LeaseError("completion authority task binding is malformed")
        return lease, receipt

    def _inventory_completed_authorities(self) -> dict[str, tuple[str, bytes, Lease, Mapping[str, object]]]:
        """Read every finalized marker through no-follow directory descriptors.

        ``completed/<task>/<digest>.json`` is the durable completion authority;
        SQLite is deliberately not consulted while deciding which markers exist.
        Unknown entries, malformed names, linked components, and duplicate task
        markers all fail closed rather than permitting a post-rollback reclaim.
        """

        try:
            completed_fd = open_directory_no_follow(self.completed)
        except ValueError as exc:
            raise LeaseError("completion authority directory cannot be opened safely") from exc
        inventory: dict[str, tuple[str, bytes, Lease, Mapping[str, object]]] = {}
        try:
            try:
                task_names = sorted(os.listdir(completed_fd))
            except OSError as exc:
                raise LeaseError("completion authority directory cannot be listed safely") from exc
            for task_id in task_names:
                if not _DIGEST.fullmatch(task_id):
                    raise LeaseError("completion authority task identity is malformed")
                try:
                    task_fd = os.open(
                        task_id, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=completed_fd,
                    )
                except OSError as exc:
                    raise LeaseError("completion authority task directory cannot be opened safely") from exc
                try:
                    try:
                        marker_names = sorted(os.listdir(task_fd))
                    except OSError as exc:
                        raise LeaseError("completion authority task directory cannot be listed safely") from exc
                finally:
                    os.close(task_fd)
                # A crashed/unlinked publication can leave an empty task
                # directory.  It carries no marker authority; a completed
                # SQLite row for it is rejected below, while an index rebuild
                # cannot mistake it for a finalized completion.
                if not marker_names:
                    continue
                if len(marker_names) != 1:
                    raise LeaseError("completion authority has multiple or unknown markers")
                marker_name = marker_names[0]
                if not marker_name.endswith(".json") or not _DIGEST.fullmatch(marker_name[:-5]):
                    raise LeaseError("completion authority marker identity is malformed")
                digest = marker_name[:-5]
                marker = self.completed / task_id / marker_name
                content = self.receipts / f"{digest}.json"
                try:
                    marker_payload = read_regular_file_no_follow(marker, maximum=_MAX_RECEIPT_BYTES)
                    content_payload = read_regular_file_no_follow(content, maximum=_MAX_RECEIPT_BYTES)
                except FileNotFoundError as exc:
                    raise LeaseError("completion marker lacks matching content receipt") from exc
                except ValueError as exc:
                    raise LeaseError("completion authority cannot be opened safely") from exc
                if marker_payload != content_payload:
                    raise LeaseError("completion marker differs from matching content receipt")
                lease, receipt = self._validated_completion_authority(task_id, digest, marker_payload)
                inventory[task_id] = (digest, marker_payload, lease, receipt)
        finally:
            os.close(completed_fd)
        return inventory

    def _recover_locked(self, now: int | None) -> None:
        """Reconcile authoritative markers into the current SQLite index lock."""

        collection = self.db.execute("SELECT digest FROM collection").fetchone()
        if collection is None:
            # A newly recreated SQLite index has no sealed task collection yet.
            # ``initialize`` invokes this same routine after sealing supplied
            # tasks, at which point retained markers become bindable authority.
            return
        task_rows = self.db.execute(
            "SELECT task_id,bundle_sha256,attempts,completion_sha256 FROM tasks"
        ).fetchall()
        tasks = {row[0]: row for row in task_rows}
        if not tasks:
            raise LeaseError("sealed lease collection has no tasks")
        inventory = self._inventory_completed_authorities()
        for task_id, (digest, _payload, lease, receipt) in inventory.items():
            row = tasks.get(task_id)
            if row is None:
                raise LeaseError("completion authority task is unknown to sealed collection")
            if receipt["bundle_sha256"] != row[1]:
                raise LeaseError("completion authority bundle does not match sealed task")
            existing = row[3]
            if existing is not None and existing != digest:
                raise LeaseError("completed task has conflicting completion authority")
            # The marker/content pair is authoritative after an SQLite loss or
            # rollback.  Retain its lease metadata for audit while ensuring a
            # recovered completed row cannot ever become claimable again.
            self.db.execute(
                "UPDATE tasks SET attempts=MAX(attempts,?),owner=?,deadline_ns=?,lease_identity=?,"
                "prepared_sha256=NULL,prepared_payload=NULL,completion_sha256=? WHERE task_id=?",
                (lease.attempt, lease.owner, lease.deadline_ns, lease.identity, digest, task_id),
            )

        completed_rows = self.db.execute(
            "SELECT task_id,completion_sha256 FROM tasks WHERE completion_sha256 IS NOT NULL"
        ).fetchall()
        for task_id, digest in completed_rows:
            authority = inventory.get(task_id)
            if authority is None or authority[0] != digest:
                raise LeaseError("completed row lacks durable receipt authority")

        rows = self.db.execute(
            "SELECT task_id,prepared_sha256,prepared_payload,deadline_ns FROM tasks WHERE prepared_sha256 IS NOT NULL"
        ).fetchall()
        if rows and now is None:
            now = self.clock_ns()
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
                # Content-only evidence is explicitly non-authoritative.  It
                # may remain for audit but never blocks a later bounded lease.
                self.db.execute("UPDATE tasks SET prepared_sha256=NULL,prepared_payload=NULL WHERE task_id=?", (task_id,))

    def recover(self) -> None:
        """Reconcile marker/content authority and expire only unpublished preparations."""

        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._recover_locked(self.clock_ns())
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def claim(self, owner: str, *, lease_seconds: float) -> Lease | None:
        owner = _owner(owner)
        duration_ns = _lease_duration_ns(lease_seconds)
        self.recover()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = self.clock_ns()
            deadline = _deadline_ns(now, duration_ns)
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

        lease = validate_lease(lease)
        duration_ns = _lease_duration_ns(lease_seconds)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = self.clock_ns()
            requested_deadline = _deadline_ns(now, duration_ns)
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
            # ``publish_bytes_no_replace`` creates and fsyncs every missing
            # marker parent through no-follow descriptors.
            publish_bytes_no_replace(marker, payload)
        except ValueError as exc:
            raise LeaseError("receipt or completion marker digest collision") from exc

    def complete(self, lease: Lease, receipt: Mapping[str, object], *, publish: bool = True) -> str:
        lease = validate_lease(lease)
        required = {"task_id", "bundle_sha256", "archive_sha256", "failed_terminal_sha256", "accepted_terminal_sha256", "validator_receipt_sha256", "validator_status"}
        if not isinstance(receipt, Mapping) or set(receipt) != required:
            raise LeaseError("completion binding fields mismatch")
        for key in required - {"validator_status"}:
            _digest(receipt[key], key)
        if receipt["validator_status"] != "pass":
            raise LeaseError("validator status must be pass")
        payload = _canonical({"schema": "emender-e97-task-completion-receipt-v3", "task_id": lease.task_id,
                              "lease": {"task_id": lease.task_id, "owner": lease.owner,
                                        "attempt": lease.attempt, "deadline_ns": lease.deadline_ns,
                                        "identity": lease.identity}, "receipt": dict(receipt)})
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise LeaseError("canonical completion receipt exceeds the sealed bound")
        digest = _sha(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = self.clock_ns()
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
            if row[5] is None and (tuple(row[1:5]) != (lease.owner, lease.attempt, lease.deadline_ns, lease.identity) or lease.deadline_ns <= now):
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
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = self.clock_ns()
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
