"""Bank plumbing for the e97-rl-loop-v1 8-GPU DiLoCo-synced bank (tranche 3).

Filesystem-only coordination, per the 2026-07-25 no-database amendment of
docs/RESILIENT_DILOCO_COMPUTE_POOL.md: NO SQLite, NO shared lock file, NO
heartbeat store.  Every piece of coordination state below is either an
atomic-rename claim (single filesystem, one winner) or a durable,
digest-linked artifact:

  * GLOBAL TASK POOL (one pool dir, work-stealing across all lanes):
      pool/pending/<task_id>.json     frozen task (claimable)
      pool/claims/<task_id>.claim     live claim (lease semantics:
                                      lane id + deadline + heartbeat inside
                                      the claim block; a crashed claimant's
                                      claim goes stale and is re-queueable)
      pool/done/<task_id>.json        retired task (with outcome block)
    claim  = ONE atomic rename pending -> claims (exactly one lane can win);
    lease  = the claim block's deadline_unix, refreshed by heartbeats;
    stale  = no valid claim block (crash between rename and enrich; the
             file mtime bounds it) or deadline passed -> any sweeper renames
             the claim back to pending (crash recovery from durable state).
  * LANE STATE bank/lanes/lane-NN/lane.json — owned and atomically rewritten
    by that lane process ONLY (the coordinator never writes lane state; it
    reads it and keeps its own watermarks in bank/coordinator.json).
  * MERGES bank/merges/ — the coordinator publishes each merged checkpoint
    atomically (temp file + fsync + os.replace; ADR-003 R07/NDP15: the
    pointer advances only from the readable complete file), appends a
    digest-linked merge receipt to bank/merges/stream.jsonl (hash chain),
    and atomically rewrites merges/current.json (operator convenience;
    NEVER authority — the receipts and immutable copies are).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from ndm.e97_onpolicy_records import canonical_json

REPO_ROOT = Path("/home/erikg/emender")

CLAIM_SCHEMA = "emender-rl-loop-bank-claim-v1"
LANE_SCHEMA = "emender-rl-loop-bank-lane-state-v1"
MERGE_RECEIPT_SCHEMA = "emender-rl-loop-bank-merge-receipt-v1"
MERGE_CURRENT_SCHEMA = "emender-rl-loop-bank-merge-current-v1"
COORDINATOR_SCHEMA = "emender-rl-loop-bank-coordinator-state-v1"
POOL_PULL_SCHEMA = "emender-rl-loop-bank-pool-pull-v1"

# Re-import is not replay. Any future replay allowance is bounded per content
# identity by the durable replay_count on frozen tasks (legacy tasks count as 0).
MAX_TASK_REPLAYS = 0

# The admitted lake's sealed validator program pin.  The D re-pin tranche
# (dc8fffd7) advanced the repo copy to 72820d63…, but every bundle in the
# ADMITTED lake pins program_sha256 344a1209… — the lake's pinned specs
# govern its grading, so the bank binds the byte-verified pinned program
# (recovered byte-exact from git 2002eded:scripts/e97_first_party_validator.py).
# Both program versions coexist under their own shas; no bytes unverified.
PINNED_VALIDATOR_SHA256 = ("344a1209ed32e44f58ce6b2be2723de9f60cc9f137dfd"
                           "169c8480e49a54efaf3")
PINNED_VALIDATOR_SOURCE_COMMIT = "2002eded"
PINNED_VALIDATOR_SOURCE_PATH = "scripts/e97_first_party_validator.py"

TASK_SCHEMA = "emender-rl-loop-seed-task-v1"
RECLAIM_SCHEMA = "emender-rl-loop-bank-reclaim-record-v1"

# Low-disk fail-closed guard: the programme's standing run.sh 100GiB-guard
# precedent, applied inside the bank so an indefinite session PAUSES instead
# of dying mid-write when free space drops below this bound.
MIN_FREE_BYTES = 100 * (1 << 30)


def disk_free_bytes(path: str | Path) -> int:
    import shutil

    return shutil.disk_usage(str(path)).free


def disk_low(path: str | Path, *, min_free_bytes: int = MIN_FREE_BYTES) -> bool:
    return disk_free_bytes(path) < int(min_free_bytes)


def sha256_text_payload(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    from ndm.e97_atomic import fsync_directory as _fsync

    _fsync(Path(path))


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Rewrite a small mutable file atomically (temp + os.replace + fsync)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")
    os.replace(temporary, path)
    fsync_directory(path.parent)


def publish_json_immutable(path: Path, value: Mapping[str, Any]) -> str:
    """Publish one immutable 0400 JSON file; return its sha256."""
    from ndm.e97_atomic import publish_bytes_no_replace

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    try:
        publish_bytes_no_replace(path, payload, mode=0o400)
    except ValueError as exc:
        raise FileExistsError(f"refusing to overwrite {path}: {exc}") from exc
    fsync_directory(path.parent)
    import hashlib

    return hashlib.sha256(payload).hexdigest()


# ------------------------------------------------------------------ paths
def bank_paths(bank_root: str | Path) -> dict[str, Path]:
    base = Path(bank_root)
    return {
        "root": base,
        "pool_pending": base / "pool" / "pending",
        "pool_claims": base / "pool" / "claims",
        "pool_done": base / "pool" / "done",
        "lanes": base / "lanes",
        "merges": base / "merges",
        "merge_stream": base / "merges" / "stream.jsonl",
        "merge_current": base / "merges" / "current.json",
        "reclaim_log": base / "reclaim-log.jsonl",
        "sealed": base / "sealed",
        "state": base / "state",
        "coordinator": base / "state" / "coordinator.json",
        "report": base / "report",
        "stop": base / "STOP",
    }


def ensure_bank_layout(paths: Mapping[str, Path]) -> None:
    for key in ("pool_pending", "pool_claims", "pool_done", "lanes",
                "merges", "sealed", "state", "report"):
        paths[key].mkdir(parents=True, exist_ok=True)


def lane_root(paths: Mapping[str, Path], lane: int) -> Path:
    root = paths["lanes"] / f"lane-{lane:02d}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def lane_paths(bank: Mapping[str, Path], lane: int) -> dict[str, Path]:
    """The per-lane workspace: the v1 loop layout rooted at the lane dir.

    The queue keys are REDIRECTED to the one global task pool (work-stealing:
    every lane claims from the same pool); everything else (receipts stream,
    immutable receipt copies, episodes, packs, training, cycles, state) is
    lane-private so the proven v1 machinery operates unchanged per lane.
    """
    root = lane_root(bank, lane)
    return {
        "root": root,
        "lane": root / "lane.json",
        # the ONE global pool (lease semantics, not the v1 active/ rename)
        "queue_pending": bank["pool_pending"],
        "queue_active": bank["pool_claims"],
        "queue_done": bank["pool_done"],
        "receipts": root / "receipts",
        "stream": root / "receipts" / "stream.jsonl",
        "episodes": root / "episodes",
        "packs": root / "packs",
        "training": root / "training",
        "cycles": root / "cycles",
        "state": root / "state",
    }


# ------------------------------------------------------- pinned validator
def ensure_pinned_validator(paths: Mapping[str, Path]) -> Path:
    """Materialize the byte-verified sealed validator pinned by the lake.

    The admitted lake's private specs pin program_sha256 344a1209…; the repo
    copy has advanced past that pin (D re-pin tranche), so the bank grades
    with the pinned bytes recovered byte-exact from git history.  Fail
    closed if the recovered bytes do not hash to the pin.
    """
    target = paths["sealed"] / "e97_first_party_validator.py"
    if target.is_file():
        if sha256_file(target) != PINNED_VALIDATOR_SHA256:
            raise SystemExit(f"pinned validator copy drifted: {target}")
        return target
    payload = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show",
         f"{PINNED_VALIDATOR_SOURCE_COMMIT}:{PINNED_VALIDATOR_SOURCE_PATH}"],
        capture_output=True, check=True).stdout
    import hashlib

    observed = hashlib.sha256(payload).hexdigest()
    if observed != PINNED_VALIDATOR_SHA256:
        raise SystemExit(f"recovered validator bytes do not match the lake "
                         f"pin: {observed}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o444)
    os.replace(temporary, target)
    fsync_directory(target.parent)
    return target


# ------------------------------------------------------------------- pool
def freeze_pool_task(paths: Mapping[str, Path], task: Mapping[str, Any]) -> Path:
    """Write one immutable training task file into the global pool (pending)."""
    body = task.get("body", {})
    if body.get("split") == "development" or body.get("receipt_eligible") is False:
        raise ValueError("receipt-ineligible task cannot enter the training pool")
    task_id = task["task_id"]
    if "/" in task_id or task_id != task_id.strip():
        raise ValueError("task id must be a clean single path component")
    target = paths["pool_pending"] / f"{task_id}.json"
    payload = (canonical_json(dict(task)) + "\n").encode("utf-8")
    from ndm.e97_atomic import publish_bytes_no_replace

    try:
        publish_bytes_no_replace(target, payload, mode=0o400)
    except ValueError as exc:
        raise FileExistsError(f"task already frozen: {target}") from exc
    fsync_directory(paths["pool_pending"])
    return target


def _claim_block_valid(task: Mapping[str, Any], *, now: float) -> bool:
    claim = task.get("claim")
    return (isinstance(claim, Mapping) and claim.get("schema") == CLAIM_SCHEMA
            and isinstance(claim.get("deadline_unix"), (int, float))
            and float(claim["deadline_unix"]) > now)


def claim_pool_task(paths: Mapping[str, Path], lane: int, *,
                    ttl_seconds: float, skip: set[str] | None = None,
                    max_stale_sweeps: int = 2) -> dict[str, Any] | None:
    """Claim the next pool task for this lane; ONE atomic rename wins.

    Lease semantics on the filesystem (no database, no lock): the claim is
    deadline-bounded inside the claim block; a crashed claimant's claim is
    swept back to pending by any participant once its deadline passes.
    """
    ensure_bank_layout(paths)
    sweeps = 0
    while True:
        for candidate in sorted(paths["pool_pending"].glob("*.json")):
            task_id = candidate.name[:-len(".json")]
            if skip is not None and task_id in skip:
                continue
            destination = paths["pool_claims"] / f"{task_id}.claim"
            try:
                os.rename(candidate, destination)
            except FileNotFoundError:
                continue  # a concurrent lane won the race
            fsync_directory(paths["pool_claims"])
            now = time.time()
            task = json.loads(destination.read_text())
            task["claim"] = {
                "schema": CLAIM_SCHEMA,
                "task_id": task_id,
                "lane": int(lane),
                "claimed_unix": now,
                "deadline_unix": now + float(ttl_seconds),
                "heartbeat_unix": now,
            }
            _rewrite_claim(destination, task)
            return task
        # pending empty: one stale-claim sweep may surface requeueable work
        if sweeps >= max_stale_sweeps:
            return None
        sweeps += 1
        if sweep_stale_claims(paths) == 0:
            return None


def _rewrite_claim(path: Path, task: Mapping[str, Any]) -> None:
    """Atomically rewrite one claim file (enrichment + heartbeats)."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(task, sort_keys=True, indent=1) + "\n")
    os.replace(temporary, path)
    fsync_directory(path.parent)


def heartbeat_claim(paths: Mapping[str, Path], task: Mapping[str, Any], *,
                    lane: int, ttl_seconds: float) -> bool:
    """Extend this lane's live claim lease (atomic claim-file rewrite)."""
    claim = task.get("claim")
    if not (isinstance(claim, Mapping) and claim.get("lane") == int(lane)):
        return False
    path = paths["pool_claims"] / f"{task['task_id']}.claim"
    try:
        current = json.loads(path.read_text())
    except FileNotFoundError:
        return False
    current_claim = current.get("claim")
    if not (isinstance(current_claim, Mapping)
            and current_claim.get("lane") == int(lane)):
        return False
    now = time.time()
    current["claim"]["heartbeat_unix"] = now
    current["claim"]["deadline_unix"] = now + float(ttl_seconds)
    _rewrite_claim(path, current)
    return True


def sweep_stale_claims(paths: Mapping[str, Path], *, grace_fraction: float = 0.5,
                      ttl_default: float = 3600.0) -> int:
    """Return expired/crashed claims to pending; return how many moved.

    A claim is stale when its claim block's deadline passed, or when it has
    no valid claim block (crash between the winning rename and enrichment)
    and its file mtime is older than grace_fraction * ttl_default.
    """
    ensure_bank_layout(paths)
    now = time.time()
    moved = 0
    for candidate in sorted(paths["pool_claims"].glob("*.claim")):
        try:
            task = json.loads(candidate.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        claim = task.get("claim") if isinstance(task, Mapping) else None
        if isinstance(claim, Mapping) and claim.get("schema") == CLAIM_SCHEMA:
            stale = float(claim.get("deadline_unix", 0)) <= now
        else:
            stale = (now - candidate.stat().st_mtime) > \
                grace_fraction * ttl_default
        if not stale:
            continue
        # NOTE: the stale claim block stays inside the requeued file — it is
        # replaced atomically by the next claimant's enrichment and ignored
        # everywhere else (only *.claim files carry claim state).
        destination = paths["pool_pending"] / f"{candidate.name[:-len('.claim')]}.json"
        try:
            # ONE atomic rename decides the sweep: a concurrent sweeper or
            # claimant that moved the file first makes this fail cleanly.
            os.rename(candidate, destination)
        except FileNotFoundError:
            continue
        moved += 1
    if moved:
        fsync_directory(paths["pool_pending"])
        fsync_directory(paths["pool_claims"])
    return moved


def requeue_claim(paths: Mapping[str, Path], task_id: str) -> bool:
    """Voluntarily return a live claim to pending (lease could not be served).

    ONE atomic rename decides it: if a sweeper already requeued this task,
    the rename fails cleanly and we return False.  The stale claim block
    inside the file is harmless — the next claimant's enrichment replaces
    it atomically.
    """
    source = paths["pool_claims"] / f"{task_id}.claim"
    destination = paths["pool_pending"] / f"{task_id}.json"
    try:
        os.rename(source, destination)
    except FileNotFoundError:
        return False
    fsync_directory(paths["pool_pending"])
    fsync_directory(paths["pool_claims"])
    return True


def requeue_lane_claims(paths: Mapping[str, Path], lane: int) -> int:
    """Return every one of this lane's live claims to pending (crash path).

    Called only after the lane's collect subprocess has exited, so its own
    claims have no live claimant anymore.  ONE atomic rename per claim; a
    concurrent sweeper winning the same rename fails cleanly here.
    """
    moved = 0
    for candidate in sorted(paths["pool_claims"].glob("*.claim")):
        try:
            task = json.loads(candidate.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        claim = task.get("claim") if isinstance(task, Mapping) else None
        if not (isinstance(claim, Mapping)
                and claim.get("lane") == int(lane)):
            continue
        destination = paths["pool_pending"] / \
            f"{candidate.name[:-len('.claim')]}.json"
        try:
            os.rename(candidate, destination)
        except FileNotFoundError:
            continue
        moved += 1
    if moved:
        fsync_directory(paths["pool_pending"])
        fsync_directory(paths["pool_claims"])
    return moved


def retire_task(paths: Mapping[str, Path], task: Mapping[str, Any], *,
                outcome: Mapping[str, Any]) -> None:
    """Retire a finished task (any stage outcome) to done/ with evidence."""
    task = dict(task)
    task.pop("claim", None)
    attempts = int(task.get("attempts", 0)) + 1
    task["attempts"] = attempts
    task["retired"] = {
        "schema": "emender-rl-loop-bank-task-retirement-v1",
        "outcome": dict(outcome),
        "retired_unix": time.time(),
    }
    source = paths["pool_claims"] / f"{task['task_id']}.claim"
    destination = paths["pool_done"] / f"{task['task_id']}.json"
    if destination.is_file():
        raise SystemExit(f"double retirement refused for {task['task_id']} "
                         "(a retired task is never re-executable in-round)")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(task, sort_keys=True, indent=1) + "\n")
    os.replace(temporary, destination)
    source.unlink(missing_ok=True)
    fsync_directory(paths["pool_done"])
    fsync_directory(paths["pool_claims"])


def pool_status(paths: Mapping[str, Path]) -> dict[str, Any]:
    now = time.time()
    claims_live = 0
    claims_stale = 0
    for candidate in paths["pool_claims"].glob("*.claim"):
        try:
            task = json.loads(candidate.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            claims_stale += 1
            continue
        if _claim_block_valid(task, now=now):
            claims_live += 1
        else:
            claims_stale += 1
    return {
        "schema": "emender-rl-loop-bank-pool-status-v1",
        "pending": len(list(paths["pool_pending"].glob("*.json"))),
        "claims_live": claims_live,
        "claims_stale": claims_stale,
        "done": len(list(paths["pool_done"].glob("*.json"))),
        "claimable": (len(list(paths["pool_pending"].glob("*.json")))
                      + claims_stale),
        "observed_unix": now,
    }


# -------------------------------------------------------------- lane state
def read_lane_state(lane_dir: str | Path) -> dict[str, Any] | None:
    path = Path(lane_dir) / "lane.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_lane_state(lane_dir: str | Path, state: Mapping[str, Any]) -> None:
    if state.get("schema") != LANE_SCHEMA:
        raise ValueError("lane state schema mismatch")
    atomic_write_json(Path(lane_dir) / "lane.json", state)


def init_lane_state(bank: Mapping[str, Path], lane: int, *,
                    checkpoint: Path, checkpoint_sha256: str,
                    note: str) -> dict[str, Any]:
    root = lane_root(bank, lane)
    existing = read_lane_state(root)
    if existing is not None:
        return existing
    state = {
        "schema": LANE_SCHEMA,
        "lane": int(lane),
        "lane_cycle": 0,
        "lineage_path": str(checkpoint),
        "lineage_sha256": checkpoint_sha256,
        "merge_epoch": 0,
        "updates_total": 0,
        "receipts_total": 0,
        "window_attempts": 0,
        "window_passes": 0,
        "window_receipts": 0,
        "superseded_lineages": [],
        "gpu": None,
        "status": "init",
        "note": note,
        "updated_unix": time.time(),
    }
    write_lane_state(root, state)
    return state


# ------------------------------------------------------- reclaim (RECLAIM-LOG)
def append_reclaim_record(paths: Mapping[str, Path],
                          record: Mapping[str, Any]) -> None:
    """Append one reclaim record to the bank's reclaim log (durable,
    fsynced).  Every deleted checkpoint is logged with its sha + reason:
    superseded lineage weights are CACHE, not evidence — every receipt and
    merge receipt already binds the exact sha of the reclaimed bytes."""
    line = (json.dumps({"schema": RECLAIM_SCHEMA, **dict(record)},
                       sort_keys=True) + "\n").encode("utf-8")
    log = Path(paths["reclaim_log"])
    with log.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def all_lane_current_lineages(paths: Mapping[str, Path]) -> set[str]:
    """Resolved paths of every lane's CURRENT lineage (never reclaimable)."""
    current: set[str] = set()
    for lane_dir in sorted(Path(paths["lanes"]).glob("lane-*")):
        state = read_lane_state(lane_dir)
        if state is not None:
            current.add(str(Path(state["lineage_path"]).resolve()))
    return current


def reclaim_superseded_lineages(paths: Mapping[str, Path], state: dict[str, Any],
                                *, grace_seconds: float = 900.0) -> list[dict]:
    """RECLAIM-LOG discipline: delete THIS lane's superseded lineage
    checkpoints after a successful advance (supervisor-approved retention).

    Fail-closed refusals — NOTHING is deleted when the candidate:
      * was superseded less than grace_seconds ago (an in-flight coordinator
        merge may still be loading the lineage it snapshotted);
      * is ANY lane's CURRENT lineage (cross-lane refusal);
      * is the merge-current checkpoint or lives under bank/merges (merged
        checkpoints are lineage authorities + gate-read candidates — kept);
      * lives outside this lane's own training tree (the shared v1 seed
        checkpoint, another lane's or workspace's artifacts — not ours);
      * no longer hashes to the sha recorded when it was superseded
        (identity drift — fail closed, retried next cycle).
    The lane's superseded_lineages ledger is the durable queue; reclaimed
    and already-gone entries leave it, permanently-refused entries leave it
    (the file is kept forever), and the rest stay for the next attempt.
    Returns the reclaim records written this pass.
    """
    now = time.time()
    ledger = list(state.get("superseded_lineages") or [])
    if not ledger:
        return []
    current = all_lane_current_lineages(paths)
    merged = read_merge_current(paths)
    merged_path = (str(Path(merged["merged_checkpoint_path"]).resolve())
                   if merged else None)
    merges_tree = Path(paths["merges"]).resolve()
    training_tree = (Path(paths["lanes"]) / f"lane-{int(state['lane']):02d}"
                     / "training").resolve()
    kept: list[dict[str, Any]] = []
    reclaimed: list[dict[str, Any]] = []
    for entry in ledger:
        path = Path(entry["path"])
        resolved = str(path.resolve())
        drop = False
        if now - float(entry["superseded_unix"]) < float(grace_seconds):
            pass  # grace window: retry later
        elif not path.is_file():
            drop = True  # already gone (earlier reclaim or operator)
        elif (merged_path is not None and resolved == merged_path) \
                or Path(resolved).is_relative_to(merges_tree):
            drop = True  # merged checkpoints are kept forever
        elif resolved in current:
            pass  # some lane's CURRENT lineage: retry later
        elif not Path(resolved).is_relative_to(training_tree):
            drop = True  # not this lane's training artifact
        else:
            observed = sha256_file(path)
            if observed != entry["sha256"]:
                pass  # identity drift: fail closed, retry (never delete)
            else:
                record = {
                    "lane": int(state["lane"]),
                    "path": str(path), "sha256": observed,
                    "bytes": path.stat().st_size,
                    "superseded_unix": float(entry["superseded_unix"]),
                    "reclaimed_unix": now,
                    "reason": ("superseded lineage checkpoint (cache, not "
                               "evidence: receipts bind this sha)"),
                }
                os.unlink(path)
                append_reclaim_record(paths, record)
                reclaimed.append(record)
                drop = True
        if not drop:
            kept.append(entry)
    if reclaimed or len(kept) != len(ledger):
        state["superseded_lineages"] = kept
    return reclaimed


# ------------------------------------------------------------ merge stream
def merge_stream_tail(paths: Mapping[str, Path]) -> str | None:
    path = paths["merge_stream"]
    if not path.is_file():
        return None
    tail = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                tail = line
    return json.loads(tail)["receipt_sha256"] if tail else None


def append_merge_receipt(paths: Mapping[str, Path],
                         receipt: Mapping[str, Any]) -> str:
    """Append one merge receipt to the digest-linked merge chain + copy."""
    import hashlib

    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    if digest != receipt["receipt_sha256"]:
        raise ValueError("merge receipt digest does not bind its body")
    if receipt["schema"] != MERGE_RECEIPT_SCHEMA:
        raise ValueError("merge receipt schema mismatch")
    copy_path = paths["merges"] / f"merge-{int(receipt['merge_index']):04d}" / \
        f"{digest}.json"
    publish_json_immutable(copy_path, receipt)
    line = canonical_json(dict(receipt)) + "\n"
    with paths["merge_stream"].open("a") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def read_merge_current(paths: Mapping[str, Path]) -> dict[str, Any] | None:
    path = paths["merge_current"]
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def publish_merge_current(paths: Mapping[str, Path],
                          value: Mapping[str, Any]) -> None:
    if value.get("schema") != MERGE_CURRENT_SCHEMA:
        raise ValueError("merge-current schema mismatch")
    atomic_write_json(paths["merge_current"], value)


# -------------------------------------------------------- coordinator state
def read_coordinator_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    path = paths["coordinator"]
    if not path.is_file():
        return {
            "schema": COORDINATOR_SCHEMA,
            "pool_round": 0,
            "merge_epoch": 0,
            "merge_count": 0,
            "merge_skip_count": 0,
            "counter_snapshot": {},
            "updated_unix": time.time(),
        }
    return json.loads(path.read_text())


def write_coordinator_state(paths: Mapping[str, Path],
                            state: Mapping[str, Any]) -> None:
    if state.get("schema") != COORDINATOR_SCHEMA:
        raise ValueError("coordinator state schema mismatch")
    atomic_write_json(paths["coordinator"], dict(state, updated_unix=time.time()))


# ------------------------------------------------------------ pool refresh
def pool_task_replay_counts(bank: Mapping[str, Path]) -> dict[str, int]:
    """Global content identities, including in-flight and legacy round IDs."""
    counts: dict[str, int] = {}
    for key, pattern in (("pool_pending", "*.json"),
                         ("pool_claims", "*.claim"), ("pool_done", "*.json")):
        for path in bank[key].glob(pattern):
            try:
                task = json.loads(path.read_text())
            except FileNotFoundError:
                continue  # a lane moved it; the later claims/done scan sees it
            digest = task["task_sha256"]
            counts[digest] = max(counts.get(digest, 0),
                                 int(task.get("replay_count", 0)))
    return counts


def refresh_pool(bank: Mapping[str, Path], *, lake: Path, round_number: int,
                 program_path: Path, count: int = 0) -> dict[str, Any]:
    """Freeze the next round of sealed first-party gym tasks into the pool.

    Pulls through the task-lake validation layer (registry + collection +
    admission-receipt digests + per-bundle validator spec digest binding +
    program sha binding against the lake's PINNED program bytes) exactly as
    tranche 2 proved on these bundles. Content identities, not round IDs,
    control admission. Development tasks never enter this training pool;
    known training tasks are skipped under the explicit bounded replay policy.
    """
    import hashlib

    from rl_gym_tasks import gym_task_body, pull_validated_tasks

    bundles, summary = pull_validated_tasks(Path(lake))
    selected = list(bundles)
    if count:
        selected = selected[:count]
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    known = pool_task_replay_counts(bank)
    frozen = []
    skipped = []
    for bundle in selected:
        identity = bundle["task"]["identity"]
        if bundle["split"] != "train":
            skipped.append({"identity": identity, "split": bundle["split"],
                            "reason": "non-training-split"})
            print(f"POOL_SKIPPED {identity} split={bundle['split']} "
                  "reason=non-training-split", flush=True)
            continue
        body = gym_task_body(bundle, Path(lake), program_path=Path(program_path))
        task_sha256 = hashlib.sha256(
            canonical_json(body).encode("utf-8")).hexdigest()
        replay_count = 0
        if task_sha256 in known:
            if known[task_sha256] >= MAX_TASK_REPLAYS:
                skipped.append({"identity": identity, "split": bundle["split"],
                                "task_sha256": task_sha256,
                                "replay_count": known[task_sha256],
                                "reason": "replay-limit"})
                print(f"POOL_SKIPPED {identity} task_sha256={task_sha256} "
                      f"reason=replay-limit replay_count={known[task_sha256]}",
                      flush=True)
                continue
            replay_count = known[task_sha256] + 1
        task_id = f"bank-r{round_number:04d}-{bundle['task']['identity'][:12]}"
        task = {
            "schema": TASK_SCHEMA,
            "task_id": task_id,
            "attempts": 0,
            "round": int(round_number),
            "body": body,
            "task_sha256": task_sha256,
            "replay_count": replay_count,
            "generator": ("sealed first-party task-lake authority "
                          "(e97-firstparty-cpu-phase-bc-v1-admitted)"),
            "authoring": ("sealed first-party read-observe bundle; task-lake "
                          "validation layer verified; pinned validator program "
                          f"sha256 {PINNED_VALIDATOR_SHA256[:16]}… bound"),
            "source_commit": source_commit,
        }
        freeze_pool_task(bank, task)
        known[task_sha256] = replay_count
        frozen.append({
            "task_id": task_id,
            "identity": bundle["task"]["identity"],
            "split": bundle["split"],
            "family": bundle["task"]["family_id"],
            "receipt_eligible": body["receipt_eligible"],
            "task_sha256": task["task_sha256"],
            "replay_count": replay_count,
        })
        print(f"POOL_FROZE {task_id} split={bundle['split']}", flush=True)
    record = {
        **summary,
        "schema": POOL_PULL_SCHEMA,
        "round": int(round_number),
        "program_path": str(program_path),
        "program_sha256": sha256_file(program_path),
        "frozen": frozen,
        "skipped": skipped,
        "max_task_replays": MAX_TASK_REPLAYS,
        "frozen_unix": time.time(),
    }
    publish_json_immutable(bank["state"] /
                           f"pool-pull-round-{round_number:04d}.json", record)
    print(f"POOL_REFRESH round={round_number} tasks={len(frozen)}", flush=True)
    return record
