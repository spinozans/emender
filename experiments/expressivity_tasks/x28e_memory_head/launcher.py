"""X28E per-arm launcher + persistent watcher — the house standard, M2-gated.

Binding: REGISTRATION.md "Resource plan" and "Standard chain". Assembled at
the M2 engineering freeze; NOT EXECUTED at freeze time (no launch, zero
optimizer updates — this module exists so the post-M2 reviewed/pushed freeze
has its launch machinery frozen with it, exactly the X25/X27
commit-before-launch ancestry).

M2-review corrections applied to this module:

  * F3 — runtime artifacts live OUTSIDE the frozen package: launch/worker/
    guard/terminal/watcher receipts are written to the sibling house-standard
    results directory ``experiments/expressivity_tasks/
    results_x28e_memory_head/`` (created at launch, never inside the
    package), so the package subtree stays verifiably-clean.
  * F2/F4 — the M2 gate verifies the ACTUAL EXECUTING package with
    UNCONDITIONAL worktree+HEAD checks, including the manifest-bound
    dependency identities (the ``../ndm/...`` native implementation files
    and ``../scripts/gpu_lease.sh``); changes there invalidate the freeze.
  * F9 — identity-safe TERM->KILL escalation is scheduled INDEPENDENTLY of
    the 60s poll cycle (a timer fires 30s after TERM), and the escalation
    reaches ANY surviving process in the owned process group even if the
    worker identity vanished (group members are receipted at TERM time and
    re-verified by identity or group ownership at KILL time; a recycled
    recorded PID is never signalled). Terminal/result/marker receipts are
    hash-validated per the house standard (``trainer.receipt_payload_hash``):
    a tampered receipt classifies the arm as ``error``, never ``complete``.

House-standard elements (X23E/X25/X27 lineage):

  * M2 GATE FIRST: the freeze manifest is verified (worktree AND HEAD,
    ``trainer.verify_freeze``) before any spawn; the manifest hash is passed
    to the trainer as ``--frozen-freeze-hash`` — the trainer independently
    re-verifies and refuses to update the optimizer otherwise.
  * CARDS 0-6 ONLY: physical card 7 is excluded (the live X25
    big-E97-softplus run; left alone). Per-arm PREFERENCES are registered
    one-per-free-card; the X27q queue rule applies — registered card if free
    at launch, else next-free ascending physical index, actual card
    receipted in the per-arm launch receipt.
  * atomic idle/unleased shell leases via ``scripts/gpu_lease.sh``
    (``acquire 1 --no-wait`` inside the worker; the lease's EXIT/INT/TERM
    trap auto-releases; lease file copied into the per-arm receipt).
  * identity-safe guards: worker identity receipts (pid, /proc starttime,
    host); kills target ONLY the OWNED process group — TERM then KILL after
    30s on an independent schedule — never foreign PIDs.
  * disk guard >= 400 GiB free before/while launching; per-run bytes guard
    <= 40 GiB; wall guard <= 7 days (right-censored infrastructure, never a
    scientific verdict).
  * marker/terminal receipts: per-arm launch.json / worker.json / lease.txt /
    guard.json / terminal.json (all hash-validated); watcher polls.jsonl,
    watcher_status.json, watcher_terminal.json, watcher_wiring.json
    (supervisor native arming probe included, exit 2 = still nonterminal).

Usage (post-M2 only):
    python launcher.py --launch        # verify freeze, lease, spawn queued arms
    python launcher.py --once          # supervisor probe (exit 2 = nonterminal)
    python launcher.py --terminal <arm> --exit-code N   # worker-side terminal
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from trainer import receipt_payload_hash, verify_freeze  # noqa: E402

# F3: runtime artifacts NEVER inside the frozen package (sibling,
# house-standard results directory; created at launch).
RESULTS_ROOT = PACKAGE_ROOT.parent / "results_x28e_memory_head"
RECEIPTS = RESULTS_ROOT / "receipts"
GPU_LEASE = REPO_ROOT / "scripts" / "gpu_lease.sh"
PYTHON = sys.executable

# Registered card preferences (cards 0-6 only; card 7 EXCLUDED — live X25 run)
CARDS = {
    "tape_B0": 0,
    "tape_B0x4": 1,
    "tape_B0x16": 2,
    "tape_unbounded": 3,
    "tape_appendonly": 4,
    "gdn2_native": 5,
    "e97_native": 6,
}
QUEUE_RULE = ("registered card if free at launch, else next-free ascending "
              "physical index, actual card receipted per-arm; cards 0-6 only, "
              "card 7 excluded (live X25 run); atomic idle/unleased shell "
              "leases only, no preemption, no WorksGood")
FREE_BYTES = 400 * 1024 ** 3        # disk guard (launch precondition + runtime)
RUN_BYTES = 40 * 1024 ** 3          # per-arm run-bytes guard
WALL_SECONDS = 7 * 86400            # per-arm wall guard
KILL_GRACE_SECONDS = 30             # TERM -> KILL window (F9: independent timer)
POLL_SECONDS = 60


def identity(pid: int) -> dict:
    """Host-unique process identity (pid + /proc starttime + host) — the
    identity-safe guard basis: a kill is issued only against a PID whose
    starttime still matches the receipted worker."""
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "starttime": stat[19], "host": os.uname().nodename}


def write_json(path: Path, row: dict):
    """Hash-validated receipt write (house standard, F9): the payload carries
    ``payload_sha256`` over its canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row["payload_sha256"] = receipt_payload_hash(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(row, indent=2, default=int) + "\n")
    tmp.replace(path)


def read_receipt(path: Path):
    """Read a hash-validated receipt. Returns (row, hash_ok); a tampered,
    missing-hash, or mismatching receipt reports hash_ok=False and the
    caller classifies the arm as ``error`` (never ``complete``)."""
    row = json.loads(path.read_text())
    stored = row.pop("payload_sha256", None)
    return row, bool(stored is not None and stored == receipt_payload_hash(row))


def freeze_manifest_hash() -> str:
    manifest = json.loads((PACKAGE_ROOT / "freeze_manifest.json").read_text())
    return manifest["manifest_hash"]


def list_free_cards() -> set:
    out = subprocess.check_output(["bash", str(GPU_LEASE), "list-free"], text=True)
    return {int(c) for c in re.split(r"[,\s]+", out.strip()) if c} & set(CARDS.values())


def arm_state(arm: str, frozen_hash: str = "") -> str:
    """queued | running | complete | halt | censored | error (marker/marker-
    receipt derived; result.json is the trainer's terminal marker). All
    terminal/result receipts are HASH-VALIDATED AND IDENTITY-BOUND
    (M2-review-2 findings 7/8): a tampered, malformed, wrong-arm,
    wrong-freeze, or completion-less receipt is ``error`` or ``censored``,
    never ``complete``. A validated identity-matching guard.json is the
    registered resource-censor disposition (right-censoring, never an
    error)."""
    run = RESULTS_ROOT / f"{arm}_seed0"
    receipts = RECEIPTS / f"{arm}_seed0"
    launch = receipts / "launch.json"
    # M2-review-5 finding 1: the launch receipt itself is hash-validated
    # (read_receipt) — tampering its freeze field without re-hashing never
    # binds identity. spawn writes launch.json with payload_sha256.
    expect_arm_ok = False
    if frozen_hash and launch.is_file():
        try:
            lrow, lok = read_receipt(launch)
        except Exception:
            lrow, lok = None, False
        if lok and lrow is not None:
            expect_arm_ok = (lrow.get("arm") == arm
                            and lrow.get("freeze_manifest_hash")
                            == frozen_hash)
    if (run / "result.json").is_file():
        try:
            row, ok = read_receipt(run / "result.json")
        except Exception:
            return "error"            # malformed JSON is an error (finding 7)
        if not ok:
            return "error"
        if not (row.get("arm") == arm and expect_arm_ok):
            return "error"            # identity mismatch (finding 7)
        status = row.get("status")
        if status == "complete":
            # completion requires the trainer's expected evidence: either
            # the zero-update attestations (verify-only/refusal paths) or
            # the TRAINER's completion evidence fields (total_updates for
            # trained runs, updates for the historical schema). A genuine
            # trained completion is never misclassified (M2-review-4 F1).
            has_evidence = (row.get("zero_updates") is True
                            or row.get("updates") is not None
                            or row.get("total_updates") is not None)
            if not has_evidence:
                return "error"
            return "complete"
        if status == "halt_stage_gate_failed":
            return "halt"
    # M2-review-4 finding 2: the censor receipt is validated exactly like
    # every other terminal receipt — hash-valid, arm-matching, and
    # identity-bound (expect_arm_ok). Existence alone proves nothing.
    censor = receipts / "censor.json"
    if censor.is_file():
        try:
            crow, cok = read_receipt(censor)
        except Exception:
            crow, cok = None, False
        if cok and crow is not None and crow.get("arm") == arm \
                and expect_arm_ok:
            return "censored"
        return "error"
    # M2-review-2 finding 8: a validated, identity-matching guard.json is
    # the registered resource-censor disposition (TERM/KILL by the
    # resource guard), right-censored - never classified as an error.
    guard = receipts / "guard.json"
    if guard.is_file():
        try:
            grow, gok = read_receipt(guard)
        except Exception:
            grow, gok = None, False
        if gok and grow is not None and grow.get("arm") == arm \
                and expect_arm_ok:
            return "censored"
        # M2-review-5 finding 3: invalid guard evidence (malformed,
        # hash-invalid, or identity-mismatched) is terminal-seam corruption —
        # an error, never a silent fall-through to running/queued.
        return "error"
    worker = receipts / "worker.json"
    if worker.is_file():
        # M2-review-5 finding 2 + M2-review-6 finding 3: the worker
        # receipt is hash-validated with the FULL schema (pid, starttime,
        # finite numeric launched_at) BEFORE liveness classification — a
        # tampered or schema-broken worker receipt can never crash the
        # watcher nor masquerade as live.
        rec = read_worker_record(receipts, arm, frozen_hash)
        if rec is None:
            return "error"
        if alive(rec):
            return "running"
        if (receipts / "terminal.json").is_file():
            try:
                term, ok = read_receipt(receipts / "terminal.json")
            except Exception:
                return "error"        # malformed terminal receipt (finding 7)
            if not ok:
                return "error"
            # M2-review-2 finding 7: exit 0 is NOT sufficient for
            # ``complete`` — the trainer's validated completed result.json
            # is required (handled above); a bare exit-0 terminal without
            # it is an error, never a verdict.
            return "error"
        return "error"
    return "queued"


def valid_pid(value) -> bool:
    """A positive integer PID, booleans excluded (M2-review-8 finding 1)."""
    return isinstance(value, int) and not isinstance(value, bool) \
        and value > 0


def valid_starttime(value) -> bool:
    """A /proc starttime representation: a nonempty digit string."""
    return isinstance(value, str) and value.isdigit()


def guard_valid_for_signalling(row, arm: str) -> bool:
    """M2-review-7 finding 2: a guard receipt consumed by a SIGNAL path
    must be checksum-valid (checked by the caller), arm-bound to the arm
    being signalled, and its recorded group members must be well-formed
    identity records. Corrupt evidence means no signalling — fail-closed."""
    if not isinstance(row, dict) or row.get("arm") != arm:
        return False
    members = row.get("group_members", [])
    if not isinstance(members, list):
        return False
    for m in members:
        if not isinstance(m, dict) or not valid_pid(m.get("pid")) \
                or not valid_starttime(m.get("starttime")):
            return False
    return True


def read_worker_record(receipt: Path, arm: str = None,
                       frozen_hash: str = None):
    """Hash-validated, typed, identity-bound worker record, or None. The
    schema: positive-integer pid, digit-string starttime, finite numeric
    launched_at, and — when the caller binds context (M2-review-8
    finding 2) — arm and freeze_manifest_hash fields matching the expected
    launch identity. A receipt failing ANY check is corrupt evidence;
    callers must contain the failure, never signal on it."""
    worker = receipt / "worker.json"
    if not worker.is_file():
        return None
    try:
        rec, ok = read_receipt(worker)
    except Exception:
        return None
    if not ok or not isinstance(rec, dict):
        return None
    if not valid_pid(rec.get("pid")) or not valid_starttime(
            rec.get("starttime")):
        return None
    t = rec.get("launched_at")
    if not isinstance(t, (int, float)) or isinstance(t, bool) \
            or not math.isfinite(t):
        return None
    if arm is not None and rec.get("arm") != arm:
        return None
    if frozen_hash is not None \
            and rec.get("freeze_manifest_hash") != frozen_hash:
        return None
    return rec


def alive(record: dict) -> bool:
    """Identity-verified liveness: the PID must exist AND its /proc starttime
    must match the receipted starttime (a recycled PID never counts).
    Schema-broken records (M2-review-5 finding 2) are dead, never fatal."""
    if not isinstance(record, dict) or "pid" not in record \
            or "starttime" not in record:
        return False
    try:
        stat = Path(f"/proc/{record['pid']}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, ValueError, TypeError,
            OSError):
        return False
    starttime = stat.rsplit(") ", 1)[1].split()[19]
    return starttime == str(record["starttime"])


def group_members(pgid: int) -> list:
    """Identities (pid, starttime) of every live process in the owned process
    group (F9). The worker is spawned with start_new_session=True, so the
    receipted worker pid IS the owned pgid."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == pgid:                     # pgrp
            members.append({"pid": int(entry.name), "starttime": fields[19]})
    return members


def worker_script(arm: str, card: int, frozen_hash: str) -> str:
    """The per-arm worker (house shell pattern): acquire the lease atomically
    inside a fresh shell whose EXIT/INT/TERM trap auto-releases, receipt the
    lease + command, exec the trainer (which re-verifies the freeze), then
    write the terminal receipt through this module. Built by placeholder
    substitution (NOT an f-string: the embedded receipt-helper heredoc is
    itself Python with braces of its own)."""
    receipt = RECEIPTS / f"{arm}_seed0"
    trainer = PACKAGE_ROOT / "trainer.py"
    template = r"""set -euo pipefail
cd @REPO_ROOT@
export GPU_LEASE_VISIBLE=@CARD@
lease_code="$(bash @GPU_LEASE@ acquire 1 --no-wait)"
eval "$lease_code"
cp "${GPU_LEASE_DIR:-$PWD/.wg/gpu_leases}/$CUDA_VISIBLE_DEVICES" @RECEIPT@/lease.txt
export X28E_LAUNCHER_LAUNCH_JSON='@RECEIPT@/launch.json'
cmd=(@PYTHON@ @TRAINER@ --arm @ARM@ --frozen-freeze-hash @HASH@)
printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES" > @RECEIPT@/command.txt
printf '%q ' "${cmd[@]}" >> @RECEIPT@/command.txt
printf '\n' >> @RECEIPT@/command.txt
"${cmd[@]}" > @RECEIPT@/console.txt 2>&1 &
train_pid=$!
@PYTHON@ - "$train_pid" @RECEIPT@/trainer.json <<'RECEIPT_HELPER'
import json, pathlib, socket, sys
pid = int(sys.argv[1])
stat = pathlib.Path('/proc/' + str(pid) + '/stat').read_text().rsplit(') ', 1)[1].split()
pathlib.Path(sys.argv[2]).write_text(json.dumps(
    {'pid': pid, 'starttime': stat[19], 'host': socket.gethostname()}))
RECEIPT_HELPER
set +e
wait "$train_pid"
rc=$?
train_pid=""
@PYTHON@ @LAUNCHER@ --terminal @ARM@ --exit-code "$rc"
exit "$rc"
"""
    return (template
            .replace("@REPO_ROOT@", str(REPO_ROOT))
            .replace("@GPU_LEASE@", str(GPU_LEASE))
            .replace("@CARD@", str(card))
            .replace("@RECEIPT@", str(receipt))
            .replace("@PYTHON@", repr(PYTHON))
            .replace("@TRAINER@", str(trainer))
            .replace("@LAUNCHER@", str(Path(__file__).resolve()))
            .replace("@ARM@", arm)
            .replace("@HASH@", frozen_hash))


def spawn(arm: str, card: int, frozen_hash: str):
    receipt = RECEIPTS / f"{arm}_seed0"
    if receipt.is_dir():
        raise ValueError(f"per-arm fail-closed launch guard: {receipt} exists")
    receipt.mkdir(parents=True)
    write_json(receipt / "launch.json", {
        "arm": arm, "registered_card": CARDS[arm], "card": card,
        "queue_rule": QUEUE_RULE, "launched_at": time.time(),
        "commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "freeze_manifest_hash": frozen_hash,
        "watcher": identity(os.getpid())})
    script = worker_script(arm, card, frozen_hash)
    with (receipt / "worker_console.txt").open("w") as stream:
        process = subprocess.Popen(["bash", "-c", script],
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write_json(receipt / "worker.json", {**identity(process.pid),
                                        "arm": arm,
                                        "freeze_manifest_hash": frozen_hash,
                                        "card": card,
                                        "command": ["bash", "-c",
                                                     script.replace(frozen_hash, "<hash>")],
                                        "launched_at": time.time(),
                                        "pgid_note": "start_new_session: the "
                                        "worker pid IS the owned process-group "
                                        "id (the F9 escalation target)"})


def launch_queued(states, frozen_hash):
    if not any(state == "queued" for state in states.values()):
        return
    free = list_free_cards()
    for arm, preferred in CARDS.items():
        if states[arm] != "queued" or not free:
            continue
        # Re-verify the freeze against HEAD immediately before each spawn
        # (the queue can wait days; the standard chain requires it).
        verify_freeze(frozen_hash)
        card = preferred if preferred in free else min(free)
        spawn(arm, card, frozen_hash)
        free.remove(card)


def arm_run_bytes(arm: str) -> int:
    root = RESULTS_ROOT / f"{arm}_seed0"
    size = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                size += path.stat().st_size
        except FileNotFoundError:
            pass
    return size


def _escalate(arm: str):
    """F9: the KILL escalation, fired by an INDEPENDENT timer 30s after TERM
    (never only on a poll cycle). Reaches ANY surviving process in the owned
    group even if the worker identity vanished: recorded members are killed
    only when their /proc starttime still matches (a recycled recorded PID
    is never signalled); unrecorded live processes in the owned pgid are
    killed on group ownership and receipted."""
    receipt = RECEIPTS / f"{arm}_seed0"
    guard = receipt / "guard.json"
    if not guard.is_file():
        return
    try:
        row, ok = read_receipt(guard)
    except Exception:
        return                    # corrupt guard evidence: never signalled
    if not ok or row.get("escalated_at") is not None:
        return
    if not guard_valid_for_signalling(row, arm):
        return      # unbound or malformed guard: never signalled (F2)
    record = read_worker_record(receipt, arm, freeze_manifest_hash())
    if record is None:
        return        # corrupt worker evidence: no signalling (F1)
    pgid = record["pid"]
    recorded = {m["pid"]: m["starttime"]
                for m in row.get("group_members", [])}
    # M2-review-6 finding 2: unrecorded group members are signalled ONLY
    # under proven original-group continuity -- the recorded leader must
    # STILL be alive with its receipted starttime. A replacement group
    # (reused PGID) is never touched: its recycled leader is skipped and
    # its unrecorded children are left alone.
    leader_alive = alive(record)
    kills = []
    for member in group_members(pgid):
        pid, starttime = member["pid"], member["starttime"]
        if pid in recorded and recorded[pid] != starttime:
            continue                      # recycled PID: never signalled
        if pid not in recorded and not leader_alive:
            continue        # ownership unproven: replacement group member
        try:
            os.kill(pid, signal.SIGKILL)
            kills.append({**member,
                          "basis": ("recorded owned-group member (starttime "
                                    "verified)" if pid in recorded else
                                   "owned-pgid member (leader alive: group "
                                    "continuity proven)"),
                          "signal": "SIGKILL"})
        except ProcessLookupError:
            pass
    if leader_alive:
        try:
            os.killpg(pgid, signal.SIGKILL)
            kills.append({"pid": pgid, "starttime": record["starttime"],
                          "basis": "identity-verified leader: killpg SIGKILL",
                          "signal": "SIGKILL"})
        except ProcessLookupError:
            pass
    row["escalated_at"] = time.time()
    row["escalation_kills"] = kills
    row["escalation_note"] = ("scheduled independently of the poll cycle "
                              f"({KILL_GRACE_SECONDS}s after TERM; F9)")
    write_json(guard, row)


def guard_kill(arm: str, reason_row: dict):
    """Identity-verified OWN process-group guard (F9): TERM at the guard
    decision, KILL after the 30s grace on an INDEPENDENT schedule; the
    escalation reaches any surviving owned-group process even if the worker
    identity vanished."""
    receipt = RECEIPTS / f"{arm}_seed0"
    guard = receipt / "guard.json"
    if not guard.is_file():
        write_json(guard, {**reason_row, "at": time.time(),
                          "decision": "infrastructure censoring (right-"
                                      "censored, never a scientific verdict); "
                                      "TERM then KILL own process group after "
                                      f"{KILL_GRACE_SECONDS}s on an "
                                      "independent schedule"})
    try:
        row, gok = read_receipt(guard) if guard.is_file() else ({}, False)
    except Exception:
        return            # corrupt guard evidence: never re-signed (F1)
    if not gok or not row or row.get("term_sent_at") is not None:
        return                            # escalation already scheduled
    if not guard_valid_for_signalling(row, arm):
        return      # unbound or malformed guard: never re-signed (F2)
    record = read_worker_record(receipt, arm, freeze_manifest_hash())
    if record is None:
        return        # corrupt worker evidence: no signalling (F1)
    pgid = record["pid"]
    members = group_members(pgid)
    leader_alive = alive(record)
    term_mode = None
    try:
        if leader_alive:
            os.killpg(pgid, signal.SIGTERM)
            term_mode = "killpg TERM (identity-verified leader)"
        else:
            # M2-review-6 finding 2: with the leader gone, ownership of
            # the group is unproven -- TERM reaches only identity-matching
            # RECORDED members; a replacement group is never signalled.
            recorded = {m["pid"]: m["starttime"]
                       for m in row.get("group_members", [])}
            termed = []
            for member in members:
                if recorded.get(member["pid"]) == member["starttime"]:
                    try:
                        os.kill(member["pid"], signal.SIGTERM)
                        termed.append(member)
                    except ProcessLookupError:
                        pass
            term_mode = ("per-member TERM restricted to identity-matching "
                         "recorded members (leader identity vanished; group "
                         f"ownership unproven; {len(termed)} signalled)")
    except ProcessLookupError:
        term_mode = "group already gone at TERM time"
    row["term_sent_at"] = time.time()
    row["term_mode"] = term_mode
    # M2-review-7 finding 1: the escalation's ownership evidence is ONLY
    # the trusted member set — under proven leader continuity the current
    # owned-group snapshot; without it, only previously-receipted
    # identity-matching members. The unproven current snapshot is NEVER
    # promoted into group_members (a reused-PGID replacement group must
    # not become a KILL target list 30 seconds later).
    if leader_alive:
        trusted_members = members
    else:
        prev = {m["pid"]: m["starttime"]
                for m in row.get("group_members", [])}
        trusted_members = [m for m in members
                           if prev.get(m["pid"]) == m["starttime"]]
    row["group_members"] = trusted_members
    row["kill_grace_seconds"] = KILL_GRACE_SECONDS
    write_json(guard, row)
    # F9: the escalation is scheduled INDEPENDENTLY of the 60s poll cycle.
    threading.Timer(KILL_GRACE_SECONDS, _escalate, args=(arm,)).start()


def watch(initial: bool = False, once: bool = False) -> int:
    frozen_hash = freeze_manifest_hash()
    if initial:
        # F3: the runtime artifacts root lives OUTSIDE the frozen package.
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(PACKAGE_ROOT).free < FREE_BYTES:
            raise OSError("disk guard: less than 400 GiB free before launch")
        # M2 gate: the freeze must verify (worktree AND HEAD, package AND
        # dependency identities) BEFORE any spawn.
        verify_freeze(frozen_hash)
        write_json(RECEIPTS / "watcher_wiring.json", {
            "watcher": identity(os.getpid()),
            "freeze_manifest_hash": frozen_hash,
            "results_root": str(RESULTS_ROOT),
            "status_path": str(RECEIPTS / "watcher_status.json"),
            "terminal_path": str(RECEIPTS / "watcher_terminal.json"),
            "poll_seconds": POLL_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS,
            "guard_escalation": "independent of the poll cycle: a timer "
                                f"fires {KILL_GRACE_SECONDS}s after TERM and "
                                "KILLs any surviving owned-group process even "
                                "if the worker identity vanished (F9)",
            "receipt_hashing": "terminal/result/marker receipts carry "
                               "payload_sha256 and are validated on read; a "
                               "tampered receipt is error, never complete",
            "supervisor_probe": [PYTHON, str(Path(__file__).resolve()), "--once"],
            "native_arming": "supervisor-owned; this launcher claims only its "
                            "independent resource guard and exposed wiring"})
    while True:
        states = {arm: arm_state(arm, frozen_hash=frozen_hash)
                  for arm in CARDS}
        free = shutil.disk_usage(PACKAGE_ROOT).free
        if not once and free >= FREE_BYTES:
            launch_queued(states, frozen_hash)
            states = {arm: arm_state(arm, frozen_hash=frozen_hash)
                      for arm in CARDS}
        for arm, state in states.items():
            if state != "running":
                continue
            # M2-review-6 finding 3: the watchdog consumes the SAME
            # validated record the classifier does (hash + full schema
            # incl. finite numeric launched_at); a corrupt receipt is
            # contained -- no KeyError/TypeError can kill the watcher, and
            # the arm reclassifies as error on the next arm_state poll.
            record = read_worker_record(RECEIPTS / f"{arm}_seed0", arm,
                                        frozen_hash)
            if record is None:
                continue
            size, wall = arm_run_bytes(arm), time.time() - record["launched_at"]
            if (size > RUN_BYTES or wall > WALL_SECONDS or free < FREE_BYTES):
                guard_kill(arm, {"free_bytes": free, "run_bytes": size,
                                 "wall_seconds": wall, "arm": arm})
        row = {"at": time.time(), "states": states, "free_bytes": free,
               "run_bytes": {arm: arm_run_bytes(arm) for arm in CARDS}}
        with (RECEIPTS / "watcher_polls.jsonl").open("a") as stream:
            stream.write(json.dumps(row, default=int) + "\n")
        write_json(RECEIPTS / "watcher_status.json", row)
        print(json.dumps(row, default=int), flush=True)
        if all(s in ("complete", "halt", "censored", "error") for s in states.values()):
            write_json(RECEIPTS / "watcher_terminal.json", row)
            return int("error" in states.values())
        if once:
            return 2
        time.sleep(POLL_SECONDS)


def terminal(arm: str, exit_code: int):
    if arm not in CARDS:
        raise ValueError("unauthorized arm")
    write_json(RECEIPTS / f"{arm}_seed0" / "terminal.json", {
        "arm": arm, "exit_code": exit_code, "at": time.time(),
        "identity": identity(os.getpid())})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--terminal")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    if args.terminal:
        terminal(args.terminal, args.exit_code)
        return 0
    if args.launch or args.once:
        return watch(initial=args.launch, once=args.once)
    parser.error("nothing to do: --launch, --once or --terminal required")


if __name__ == "__main__":
    raise SystemExit(main())
