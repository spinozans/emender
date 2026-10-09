#!/usr/bin/env python3
"""Standing curriculum supply (Path B at production scale): keep the bank fed.

Sizing (measured 2026-09-27 from live bank telemetry): the 8-lane bank
retires ~200 tasks/hour total and ate the 520 injected lakeexp tasks at
~178/hour on new tasks alone, so a 100-task buffer is ~30 minutes of
queue.  The pilot authoring rate is 78.3s/call at ~46 bundles/h/lane, so
matching consumption needs ~4-6 sustained parallel authoring lanes
(hard cap 8; teacher corrections have ABSOLUTE priority over supply —
see the back-off rules below).

Pipeline per tranche (identical admission machinery to the pilot; the
quality bar is NOT relaxed by volume):

  author  GLM-5.3-flash-background (lunaroute, background slots) authors
          sealed-shape task specs in N parallel lanes with a rotating
          genre directive for diversity; fail-closed author guards
          (teacher_author.guards: strict JSON, one sealed token file,
          safe paths, bounded UTF-8 fixtures, token+tree uniqueness vs
          the WHOLE lake and within the tranche, benchmark/eval
          vocabulary bans); author-failures are RECORDED AND REJECTED,
          never retried into admission.
  build   teacher_author.build_tranche: sealed quarantine + a MECHANICAL
          SOLVABILITY PROOF per bundle (scripted solver + real sealed
          executor + validate_replay running BOTH sealed validators);
          a proof failure discards the whole tranche build (never
          re-worked, never admitted).
  overlap ndm.e97_protected_overlap.check_protected_overlap against the
          three fixed sealed panels; zero-collision receipt published
          immutable (0400).
  allowlist  one operator-authorized entry per collection in the checked-in
          trust root configs/pi/e97-firstparty-collection-authorizations-v1.json
          (pure per-collection digest tuple; machinery unchanged),
          machine-committed to main with an attributed message
          "[standing-supply daemon] tranche NNN: tas-* collections, …"
          (supervisor-approved 2026-09-27; commit scope is ONLY the
          allowlist file — anything else dirty in the tree aborts the
          tranche fail-closed).
  admit   ndm.e97_first_party_read_observe.admit_generated_collection
          (fail-closed on every digest binding; publishes by atomic
          rename as e97-firstparty-teacher-authored-tranche-NNN-admitted).
  inject  admission-receipt v3 + allowlist + overlap verified per bundle,
          era-correct widened tree-digest extraction, byte-verified pinned
          validator binding, frozen into bank/pool/pending via the pool's
          documented claim contract with NEW task id prefix tas-* (teacher
          -authored standing supply; distinguishes the standing-authority
          stream from pilot lakeexp-* in session telemetry) and round: 0.

Survival mechanism (survives the launching session end):

  * the daemon runs detached: setsid + nohup equivalent
    (start_new_session, stdio -> logs/standing-supply.log);
  * it holds an exclusive flock on standing-supply/standing-supply.lock
    for its lifetime, so a second daemon can never double-run;
  * a cron watchdog (*/10, user crontab) checks pid + heartbeat
    freshness and relaunches the daemon (fully detached) if it died or
    hung; concurrent watchdogs are flock-guarded;
  * STATUS.md is the running ledger (per-tranche lines appended by the
    daemon); standing-supply/metrics.jsonl is the machine-readable
    telemetry (one record per tranche + probes).

Teacher-priority back-off (bank corrections never yield to supply):
the bank's internal per-call teacher latency is not externally observable
without touching the bank, so degradation is measured non-invasively by
(1) a tiny control probe against the SAME lunaroute model/endpoint the
bank's teacher corrections use (probe latency is the shared-congestion
signal; >30s degrades one lane step, >60s or an error drops two steps
and pauses) and (2) watching bank lane logs for new LANE_FAILURE lines
(immediate back-off to <=4 lanes).  Recovery re-scales one step at a
time after consecutive clean tranches.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO = Path("/home/erikg/emender")
WORK = Path(os.environ["F9_AUTHOR_WORK"])
LOOP = Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1")
# Bank root for admitted-task injection (bank pool freeze).
# Configurable since 2026-10-07 (single-learner restart). Resolution order:
#   1. daemon-state.json "bank_root" (durable — survives the cron
#      watchdog's env-less relaunch; the watchdog respawns `daemon` without
#      passing BANK_ROOT, so an env-only binding is lost on respawn)
#   2. BANK_ROOT environment variable
#   3. legacy default (the audit-frozen bank)
def _persisted_bank_roots():
    # bank-root.txt: ONE BANK ROOT PER LINE, owned by the operator, only read
    # by the daemon (never rewritten by it) — survives daemon state rewrites
    # and the cron watchdog's env-less respawns. Multiple banks each receive
    # every admitted tranche (independent pools, independent consumption).
    try:
        lines = [l.strip() for l in
                 (WORK / "standing-supply" / "bank-root.txt").read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        return [Path(l) for l in lines] or [Path(os.environ.get("BANK_ROOT", str(LOOP / "bank")))]
    except Exception:
        return [Path(os.environ.get("BANK_ROOT", str(LOOP / "bank")))]

BANKS = _persisted_bank_roots()
BANK = BANKS[0]  # legacy single-bank references (metrics/reporting)
TASK_LAKE = Path("/mnt/nvme2n1/erikg/task_lake")
SUPPLY = WORK / "standing-supply"
LOGS = WORK / "logs"
STATE_PATH = SUPPLY / "daemon-state.json"
METRICS_PATH = SUPPLY / "metrics.jsonl"
LOCK_PATH = SUPPLY / "standing-supply.lock"
WATCHDOG_LOCK_PATH = SUPPLY / "watchdog.lock"
PID_PATH = SUPPLY / "daemon.pid"
STATUS_PATH = WORK / "STATUS.md"
ALLOWLIST_REL = "configs/pi/e97-firstparty-collection-authorizations-v1.json"
ALLOWLIST = REPO / ALLOWLIST_REL
VENV_PYTHON = REPO / ".venv" / "bin" / "python"
POOL_PREFIX = "tas"
TRANCHE_SIZE = 48
LOW_WATER = 100
HIGH_WATER = 288
LANE_MIN, LANE_START, LANE_SOFT_CAP, LANE_HARD_CAP = 16, 16, 24, 24
# LANE_MIN is the normal operating target, not a congestion safety floor.
BACKOFF_FLOOR = 2
PROBE_DEGRADE_S = 30.0
PROBE_SEVERE_S = 60.0
HEARTBEAT_STALE_S = 1800.0
AUTHOR_DEADLINE_S = 90 * 60
STATUS_MARKER = "## Standing curriculum supply (operator directive 2026-09-27)"
# Curriculum rotation (operator directive 2026-09-28 — the collapse fix):
# the bank's merge-5 read 0/96 on the frozen execution suite (down from its
# substrate's 63/96) — twenty hours of RL on the gym's single read-observe
# protocol destroyed multi-step completion under execution-style
# instruction shapes.  Curriculum breadth is therefore LOAD-BEARING: the
# authoring rotation now mixes the read-observe core, the first-action
# criterion genre, the era-4 protocol-breadth families (multi-step
# edit / lookup / recovery / sum with real workspace operations, plus
# chat-shaped episodes), and the era-6 writing family (judged drafting:
# strict-JSON rubric, floor-proofed here and judge-graded on the bank
# side per the era-6 directive), in the shares below.  Shares are exact
# over one
# ROTATION_CYCLE (= 60 authoring calls, ~2.5 tranches) and reported per
# tranche; the interleaving is a deterministic largest-remainder schedule.
ROTATION_SHARES = {"first_action": 16, "edit": 6, "lookup": 6, "recovery": 6,
                   "sum": 6, "chat": 6, "writing": 6, "core": 1, "terminal": 4, "conversation": 3}
ROTATION_ORDER = ("first_action", "edit", "lookup", "recovery", "sum",
                  "chat", "writing", "core", "terminal", "conversation")
ROTATION_CYCLE = sum(ROTATION_SHARES.values())
assert ROTATION_CYCLE == 60 and set(ROTATION_ORDER) == set(ROTATION_SHARES)
# Era8 adds terminal=4/conversation=3 by reducing first_action 20->16,
# chat 9->6; all other shares and the 60-call cycle are unchanged.


def _build_rotation() -> tuple[str, ...]:
    """Deterministic largest-remainder interleave of the rotation shares."""
    remaining = dict(ROTATION_SHARES)
    schedule: list[str] = []
    while any(remaining.values()):
        best = max((kind for kind in ROTATION_ORDER if remaining[kind] > 0),
                   key=lambda kind: (remaining[kind] / ROTATION_SHARES[kind],
                                    -ROTATION_ORDER.index(kind)))
        schedule.append(best)
        remaining[best] -= 1
    return tuple(schedule)


ROTATION = _build_rotation()
assert Counter(ROTATION) == Counter(ROTATION_SHARES)
FIRST_ACTION_EVERY = 3  # retained for the historical tranche-127 share note

sys.path.insert(0, str(WORK / "scripts"))
sys.path.insert(0, str(LOOP / "scripts"))
sys.path.insert(0, str(Path(os.environ["F9_STAGE_ROOT"]) / "loop"))

GENRES = [
    "hot-air ballooning club depot", "deep-sea research vessel",
    "mountain rescue outpost", "community seed library",
    "artisan bakery back office", "regional rail maintenance depot",
    "amateur radio society", "university observatory annex",
    "harbor master's office", "theatrical prop storage",
    "glaciology field camp", "municipal water utility",
    "beekeepers' association records", "vintage print shop",
    "wind farm operations", "archaeological dig headquarters",
    "orchestra library", "small regional airport operations",
    "canal lock keepers' guild", "cartography society archive",
]
STATE_SCHEMA = "emender-standing-supply-daemon-state-v1"
TRANCHE_METRIC_SCHEMA = "emender-standing-supply-tranche-v1"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    from ndm.e97_onpolicy_records import canonical_json
    return canonical_json(value)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------ state
def load_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_bytes())
    return {
        "schema": STATE_SCHEMA, "lanes": LANE_START, "clean_streak": 0,
        "heartbeat_unix": 0.0, "started_unix": 0.0,
        "last_probe": None, "bank_log_offsets": {}, "backoff_events": [],
        "scaleup_events": [], "next_tranche": None,
    }


def save_state(state: dict[str, Any]) -> None:
    SUPPLY.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, STATE_PATH)


def heartbeat(state: dict[str, Any]) -> None:
    state["heartbeat_unix"] = time.time()
    save_state(state)


def append_metrics(record: dict[str, Any]) -> None:
    SUPPLY.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_status(line: str) -> None:
    with STATUS_PATH.open("a") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ensure_status_section() -> None:
    """Append the standing-supply documentation section once (idempotent)."""
    if STATUS_MARKER in STATUS_PATH.read_text():
        return
    append_status(f"""
{STATUS_MARKER}

Supervisor-approved standing authority (2026-09-27): Path B (teacher
authoring) runs as a standing production pipeline that keeps the bank's
pool fed; the pool must never fully drain (target: >= {LOW_WATER} pending,
author-ahead buffer; pause authoring at {HIGH_WATER} pending).  Decisions:

- SIZING: bank consumption measured from live telemetry ~178-212
  tasks/hour (520 new tasks retired in 2.92h during the full-lake drain;
  ~203 claims/h over the trailing 6h).  Authoring target therefore
  ~200 bundles/h sustained: {TRANCHE_SIZE}-bundle tranches, parallel
  authoring lanes (start {LANE_START}, scale to {LANE_SOFT_CAP} when clean, hard cap
  {LANE_HARD_CAP}); measured authoring rate per tranche is recorded in
  standing-supply/metrics.jsonl and reported per tranche below.
- ADMISSION: every bundle through the SAME machinery as pilot tranche
  001 — author guards, mechanical solvability proof (validate_replay +
  both sealed validators), zero-collision protected overlap, operator
  allowlist entry, admission receipt v3, receipts; author-failures are
  rejected and recorded, never retried into admission.
- ALLOWLIST AUTHORITY (escalated, APPROVED): the daemon auto-writes and
  auto-commits each tranche's allowlist entry to main with machine-
  attributed messages ("[standing-supply daemon] tranche NNN: tas-*
  collections, …"); commit scope is ONLY the allowlist file (a tree dirty
  with anything else aborts the tranche fail-closed); each entry remains
  a pure per-collection digest tuple (machinery unchanged).
- TEACHER PRIORITY: bank teacher corrections have absolute priority; the
  supply lanes back off FIRST on any congestion signal (control probe
  against the same lunaroute model/endpoint the bank's corrections use;
  new LANE_FAILURE lines in bank logs) and re-scale only after clean
  tranches.
- POOL IDS: standing-supply tasks use the NEW prefix tas-* (teacher-
  authored standing supply; round: 0 bucket as before) to distinguish
  the standing stream from pilot lakeexp-* in telemetry.
- SURVIVAL: daemon is detached (start_new_session; stdio ->
  logs/standing-supply.log) and holds an exclusive flock on
  standing-supply/standing-supply.lock; a cron watchdog (*/10, user
  crontab -> scripts/standing_supply.py watchdog) relaunches it if the
  pid dies or the heartbeat goes stale; STATUS.md is the running ledger.

Tranche log (append-only, one line per tranche):
""")


# -------------------------------------------------------------- telemetry
def pool_counts() -> dict[str, int]:
    return {
        "pending": len(list((BANK / "pool" / "pending").glob("*.json"))),
        "claims": len(list((BANK / "pool" / "claims").glob("*.claim"))),
        "done": len(list((BANK / "pool" / "done").glob("*.json"))),
    }


def consumption(hours: float = 3.0) -> dict[str, Any]:
    """Retirement rate from the bank's own retirement receipts (claims/h)."""
    now = time.time()
    cutoff = now - hours * 3600
    per_prefix: Counter[str] = Counter()
    lane_cycle: Counter[tuple[int, int]] = Counter()
    total = 0
    for path in (BANK / "pool" / "done").glob("*.json"):
        ts = None
        lane_cycle_key = None
        try:
            task = json.loads(path.read_bytes())
            retired = task.get("retired", {})
            ts = retired.get("retired_unix")
            outcome = retired.get("outcome", {})
            if isinstance(outcome.get("lane"), int) and isinstance(outcome.get("cycle"), int):
                lane_cycle_key = (outcome["lane"], outcome["cycle"])
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        if ts is None:
            ts = path.stat().st_mtime
        if ts >= cutoff:
            total += 1
            name = path.name
            prefix = ("tas" if name.startswith(f"{POOL_PREFIX}-")
                      else "lakeexp" if name.startswith("lakeexp-")
                      else "oldlake")
            per_prefix[prefix] += 1
            if lane_cycle_key is not None:
                lane_cycle[lane_cycle_key] += 1
    return {
        "window_h": hours,
        "retired": total,
        "retire_rate_per_h": round(total / hours, 1),
        "per_prefix": dict(per_prefix),
        "per_prefix_rate_per_h": {key: round(value / hours, 1)
                                   for key, value in per_prefix.items()},
        "tasks_per_lane_cycle": (round(sum(lane_cycle.values()) / len(lane_cycle), 2)
                                 if lane_cycle else None),
    }


def control_probe() -> dict[str, Any]:
    """Tiny call to the same model/endpoint the bank's teacher corrections use."""
    import urllib.request
    token = json.loads(
        Path(os.path.expanduser("~/.pi/agent/auth.json")).read_text())["lunaroute"]["access"]
    url = os.environ.get("LUNAROUTE_ROUTING_URL",
                         "https://gw.lunaroute.com/v1") + "/chat/completions"
    body = json.dumps({
        "model": "deepseek-4.1-flash-background",
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "temperature": 0, "max_tokens": 16}).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Authorization": f"Bearer {token}",
                                "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            json.load(response)
        return {"ok": True, "latency_s": round(time.monotonic() - started, 2),
                "unix": time.time()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "latency_s": round(time.monotonic() - started, 2), "unix": time.time()}


def scan_bank_failures(state: dict[str, Any]) -> list[str]:
    """New LANE_FAILURE lines in the bank's lane logs since the last scan."""
    events: list[str] = []
    offsets: dict[str, int] = state.setdefault("bank_log_offsets", {})
    for log in sorted(BANK.glob("lane-*.log")):
        key = log.name
        offset = int(offsets.get(key, log.stat().st_size))
        with log.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offsets[key] = handle.tell()
        for line in chunk.decode("utf-8", "replace").splitlines():
            if "LANE_FAILURE" in line:
                events.append(f"{key}: {line.strip()[:160]}")
    return events


# ------------------------------------------------------- protected screen
_PROTECTED_DOMAINS: dict[str, set[str]] | None = None


def protected_domains() -> dict[str, set[str]]:
    """All 10 overlap domains of the three fixed sealed panels (cached)."""
    global _PROTECTED_DOMAINS
    if _PROTECTED_DOMAINS is None:
        from lake_expand import PANELS
        from ndm.e97_protected_overlap import _domains, load_protected_panel
        loaded = [load_protected_panel(manifest, records)
                  for manifest, records in PANELS]
        protected_records = [record for _, _, panel_records in loaded
                             for record in panel_records]
        _PROTECTED_DOMAINS = _domains(protected_records)
    return _PROTECTED_DOMAINS


def protected_prescreen(spec: Mapping[str, Any]) -> None:
    """Author-time per-spec screen against the sealed protected-overlap domains.

    Same criterion, same checker functions as the authoritative
    collection-level check (ndm.e97_protected_overlap), applied EARLIER:
    a spec that would collide with a protected panel is rejected as an
    author-failure and never admitted (the teacher authors a replacement;
    the rejected spec is not retried).  Colliding values are logged as
    hashes only, never in clear (the receipt layer counts, never prints).
    """
    from ndm.e97_protected_overlap import (
        _domains, extract_exact_scalars, normalize_content, normalize_path,
        normalize_template,
    )
    family = spec.get("family", "read")
    record = {
        "task_id": "authoring-candidate",
        "family_id": f"firstparty-authored-{family}-v1",
        "repository": f"firstparty/authored-{family}",
        "prompt_template": spec["prompt"],
        "fixture_files": [{"path": file["path"], "content": file["content"]}
                          for file in spec["files"]],
        "exact_scalars": sorted(extract_exact_scalars(
            file["content"] for file in spec["files"])),
    }
    candidate = _domains([record])
    protected = protected_domains()
    for field in ("family_ids", "repositories", "prompt_templates_full",
                  "prompt_templates_normalized", "fixture_paths_full",
                  "fixture_paths_normalized", "fixture_contents_full",
                  "fixture_contents_normalized", "exact_scalars"):
        collisions = candidate[field].intersection(protected[field])
        if collisions:
            sample = sorted(collisions)[0]
            raise ValueError(
                f"protected-overlap pre-screen: {field} collision "
                f"({len(collisions)} value(s), sample sha {_sha_text(sample)[:12]})")
    execution_prescreen(spec)


# The frozen execution-suite panels (bank-gate-v1 and the dual-gate replica)
# are strictly off-limits fixtures (the criterion-lane discipline): fresh
# protocol-breadth authoring may share the FAMILY STYLE but never copy a
# panel task.  This screen enforces the anti-copy lines mechanically at
# author time: verbatim/normalized prompt templates, verbatim/normalized
# fixture contents, and long exact scalars (the panel's answers, tokens and
# identifiers).  Generic path names and short common words are NOT screened
# — they are the shared vocabulary of the family style, not contamination.
EXECUTION_PANELS = (
    Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/"
         "bank-gate-v1/execution/panel.json"),
    Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/"
         "e97-e2-chat-agent-u256-dual-gate-v1/execution/panel.json"),
)


_EXECUTION_DOMAINS: dict[str, set[str]] | None = None


def execution_domains() -> dict[str, set[str]]:
    """The frozen execution panels' anti-copy domains (cached)."""
    global _EXECUTION_DOMAINS
    if _EXECUTION_DOMAINS is None:
        from ndm.e97_protected_overlap import (
            extract_exact_scalars, normalize_content, normalize_template,
        )
        domains: dict[str, set[str]] = {
            "prompt_templates_full": set(), "prompt_templates_normalized": set(),
            "fixture_contents_full": set(), "fixture_contents_normalized": set(),
            "exact_scalars": set(),
        }
        for panel_path in EXECUTION_PANELS:
            panel = json.loads(panel_path.read_bytes())
            for case in panel.get("cases", []):
                domains["prompt_templates_full"].add(case["prompt"])
                domains["prompt_templates_normalized"].add(
                    normalize_template(case["prompt"]))
                for content in case.get("files", {}).values():
                    domains["fixture_contents_full"].add(content)
                    domains["fixture_contents_normalized"].add(
                        normalize_content(content))
                    domains["exact_scalars"].update(
                        scalar for scalar in extract_exact_scalars([content])
                        if len(scalar) >= 8)
                for scalar in extract_exact_scalars(
                        [json.dumps(case.get("expected_output"))
                         if case.get("expected_output") is not None else "",
                         str(case.get("answer") or "")]):
                    if len(scalar) >= 8:
                        domains["exact_scalars"].add(scalar)
        _EXECUTION_DOMAINS = domains
    return _EXECUTION_DOMAINS


def execution_prescreen(spec: Mapping[str, Any]) -> None:
    """Reject any spec that copies a frozen execution-panel task."""
    from ndm.e97_protected_overlap import (
        extract_exact_scalars, normalize_content, normalize_template,
    )
    protected = execution_domains()
    checks = {
        "prompt_templates_full": {spec["prompt"]},
        "prompt_templates_normalized": {normalize_template(spec["prompt"])},
        "fixture_contents_full": {file["content"] for file in spec["files"]},
        "fixture_contents_normalized": {normalize_content(file["content"])
                                        for file in spec["files"]},
        "exact_scalars": {scalar for scalar in extract_exact_scalars(
            file["content"] for file in spec["files"]) if len(scalar) >= 8},
    }
    for field, values in checks.items():
        collisions = values.intersection(protected[field])
        if collisions:
            sample = sorted(collisions)[0]
            raise ValueError(
                f"execution-panel pre-screen: {field} collision "
                f"({len(collisions)} value(s), sample sha {_sha_text(sample)[:12]})")


# ---------------------------------------------------------------- author
def author_parallel(tranche: int, count: int, lanes: int,
                    state: dict[str, Any]) -> dict[str, Any]:
    """Parallel teacher authoring with the curriculum rotation (fail-closed).

    The rotation (operator directive 2026-09-28) mixes the read-observe core,
    the first-action criterion genre, and the era-4 protocol-breadth
    families per ROTATION; every spec passes the same fail-closed guards,
    the protected-overlap + execution-panel pre-screens, and lake-wide
    token/fixture-tree uniqueness before acceptance.
    """
    import teacher_author as ta
    tranche_dir = WORK / "teacher-authored" / f"tranche-{tranche:03d}"
    tranche_dir.mkdir(parents=True, exist_ok=True)
    (tranche_dir / "rejects.jsonl").write_text("")
    (tranche_dir / "teacher-metrics.jsonl").write_text("")
    known_tokens, known_trees = ta.lake_token_tree_index()
    known_prompts, _ = ta.diversity_author.lake_dedupe_index(TASK_LAKE)
    corpus_path = Path(os.environ["F9_AUTHOR_WORK"]) / "seed-corpus.json"
    seeds = json.loads(corpus_path.read_bytes()) if corpus_path.is_file() else []
    lock = threading.Lock()
    stats: dict[str, Any] = {"calls": 0, "accepted": 0, "rejected": 0,
                             "prescreen_rejects": 0, "first_action_accepted": 0,
                             "family_accepted": {}, "transport_error": None,
                             "latencies": []}
    stop = threading.Event()
    started = time.monotonic()

    def worker() -> None:
        while not stop.is_set():
            with lock:
                if stats["accepted"] >= count or stats["transport_error"]:
                    return
                call_index = stats["calls"]
                stats["calls"] += 1
                genre = GENRES[call_index % len(GENRES)]
                kind = ROTATION[call_index % len(ROTATION)]
                first_action = kind == "first_action"
                family = None if kind in ("core", "first_action") else kind
            instruction = ta.author_instruction(genre, first_action, family, seed=(seeds[call_index % len(seeds)] if seeds else None))
            try:
                content, entry = ta.lunaroute_call(
                    ta.TEACHER_MODEL,
                    [{"role": "system", "content": instruction},
                     {"role": "user", "content": f"Author task {call_index + 1} now."}],
                    0.7, 8192, ta.TeacherMetrics())
            except RuntimeError as exc:
                stop.set()
                with lock:
                    stats["transport_error"] = str(exc)
                return
            with lock:
                stats["latencies"].append(entry.get("latency_s"))
                with (tranche_dir / "teacher-metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
                state["heartbeat_unix"] = time.time()
                save_state(state)
            spec = None
            reason = None
            genre_used = genre
            kind_used = kind
            try:
                spec = ta.parse_author_json(content)
                ta.guards(spec, known_tokens, known_trees,
                          require_first_action=first_action)
                ta.diversity_author.admission_prescreen(spec, protected_prescreen)
                ta.diversity_author.check_dedupe(spec, known_prompts)
            except ValueError as exc:
                reason = str(exc)
                spec = None
                if reason.startswith("protected-overlap pre-screen") \
                        or reason.startswith("execution-panel pre-screen"):
                    with lock:
                        stats["prescreen_rejects"] += 1
            if spec is not None:
                tree = ta.fixture_tree_digest(
                    {file["path"]: file["content"] for file in spec["files"]})
                with lock:
                    if stats["accepted"] >= count or stats["transport_error"]:
                        return
                    prompt_sha = ta._sha_text(spec["prompt"])
                    if prompt_sha in known_prompts:
                        reason, spec = "prompt_sha256 collides within tranche", None
                    elif spec["token"] in known_tokens:
                        reason, spec = "token collides within tranche", None
                    elif tree in known_trees:
                        reason, spec = "fixture tree collides within tranche", None
                    if spec is not None:
                        index = stats["accepted"]
                        stats["accepted"] += 1
                        if first_action:
                            stats["first_action_accepted"] += 1
                        if family is not None:
                            stats["family_accepted"][family] = \
                                stats["family_accepted"].get(family, 0) + 1
                        known_prompts.add(prompt_sha)
                        known_tokens.add(spec["token"])
                        known_trees.add(tree)
                        spec["genre"] = genre_used
                        spec["curriculum_kind"] = kind_used
                        spec["authoring_call_index"] = call_index
                        spec["accepted_index"] = index
                        (tranche_dir / f"authored-{index:04d}.json").write_text(
                            json.dumps(spec, indent=2, sort_keys=True) + "\n")
            if spec is None:
                with lock:
                    stats["rejected"] += 1
                    record = {"index": call_index, "reason": reason,
                              "genre": genre_used, "curriculum_kind": kind_used,
                              "first_action": first_action,
                              "raw_sha256": _sha_text(content)}
                    with (tranche_dir / "rejects.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                continue

    threads = [threading.Thread(target=worker, name=f"author-{i}", daemon=True)
               for i in range(lanes)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + AUTHOR_DEADLINE_S
    for thread in threads:
        thread.join(timeout=max(deadline - time.monotonic(), 1.0))
    if any(thread.is_alive() for thread in threads):
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
        stats["transport_error"] = stats["transport_error"] or \
            f"authoring deadline exceeded ({AUTHOR_DEADLINE_S}s)"
    latencies = [value for value in stats["latencies"] if value is not None]
    return {
        "calls": stats["calls"], "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "prescreen_rejects": stats["prescreen_rejects"],
        "first_action_accepted": stats["first_action_accepted"],
        "family_accepted": dict(stats["family_accepted"]),
        "transport_error": stats["transport_error"],
        "author_seconds": round(time.monotonic() - started, 1),
        "mean_call_latency_s": (round(sum(latencies) / len(latencies), 1)
                                if latencies else None),
        "lanes": lanes,
        "bundles_per_hour_achieved": (round(stats["accepted"] * 3600
                                            / max(time.monotonic() - started, 1), 1)),
    }


# ------------------------------------------------------- overlap/allowlist
def run_overlap(tranche: int, quarantine: Path) -> Path:
    from lake_expand import PANELS
    from ndm.e97_protected_overlap import check_protected_overlap
    from ndm.e97_task_lake import validate_source_registry
    from ndm.e97_atomic import publish_bytes_no_replace
    registry = validate_source_registry(json.loads((quarantine / "source-registry.json").read_bytes()))
    receipt = check_protected_overlap(
        registry=registry, candidate_collection=quarantine / "tasks.jsonl",
        candidate_root=quarantine, protected_panels=PANELS)
    out = WORK / "teacher-authored" / f"tranche-{tranche:03d}" / \
        "protected-overlap-receipt.json"
    publish_bytes_no_replace(out, (_canonical(receipt) + "\n").encode(), mode=0o400)
    if receipt["status"] != "pass":
        raise SystemExit(f"tranche-{tranche:03d}: protected overlap FAILED: "
                         f"{receipt['collision_counts']}")
    return out


def mint_allowlist_entry(quarantine: Path, overlap_path: Path) -> dict[str, Any]:
    generation = json.loads((quarantine / "generation-receipt.json").read_bytes())
    return {
        "scope": "production-collection-admission",
        "registry_sha256": generation["registry_sha256"],
        "generation_receipt_sha256": _sha_file(quarantine / "generation-receipt.json"),
        "protected_overlap_receipt_sha256": _sha_file(overlap_path),
        "tasks_sha256": generation["tasks_sha256"],
        "archive_root_sha256": generation["archive_root_sha256"],
        "generator_manifest_sha256": generation["generator_component_manifest_sha256"],
        "source_archive_sha256": generation["generator_source_archive_sha256"],
        "source_revision": generation["source_revision"],
        "controller_source_sha256": generation["controller_source_sha256"],
    }


def write_allowlist_entry(entry: dict[str, Any]) -> str:
    from ndm.e97_protected_overlap import load_canonical_collection_authorizations
    _, _, existing = load_canonical_collection_authorizations()
    by_key = {(item["generation_receipt_sha256"], item["tasks_sha256"]): item
              for item in existing}
    key = (entry["generation_receipt_sha256"], entry["tasks_sha256"])
    if key in by_key and by_key[key] != entry:
        raise SystemExit("allowlist entry conflicts with an existing authorization")
    by_key[key] = entry
    ordered = sorted(by_key.values(),
                     key=lambda item: (item["scope"], item["generation_receipt_sha256"]))
    text = json.dumps({"schema": "emender-e97-firstparty-collection-authorization-v1",
                       "status": "authorized" if ordered else "candidate",
                       "authorizations": ordered},
                      indent=2, sort_keys=True) + "\n"
    ALLOWLIST.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def commit_allowlist(tranche: int, entry: dict[str, Any]) -> str:
    """Machine-commit ONLY the allowlist file (fail-closed on any other dirt)."""
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
        text=True)
    dirty = {line[3:].strip() for line in status.splitlines() if line.strip()}
    if dirty - {ALLOWLIST_REL}:
        raise SystemExit("emender tree is dirty beyond the allowlist (daemon commit "
                         f"scope is allowlist-only): {sorted(dirty - {ALLOWLIST_REL})}")
    if ALLOWLIST_REL not in dirty:
        raise SystemExit("allowlist entry produced no change to commit")
    message = (f"[standing-supply daemon] tranche {tranche:03d}: {POOL_PREFIX}-* "
               f"collections, tasks sha {entry['tasks_sha256'][:12]}, overlap receipt "
               f"sha {entry['protected_overlap_receipt_sha256'][:12]}")
    subprocess.check_call(["git", "-C", str(REPO), "add", "--", ALLOWLIST_REL])
    subprocess.check_call(["git", "-C", str(REPO), "commit", "-m", message,
                            "--", ALLOWLIST_REL])
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


# ----------------------------------------------------------------- admit
def admit_tranche(tranche: int, quarantine: Path, overlap_path: Path) -> Path:
    from ndm.e97_first_party_read_observe import admit_generated_collection
    output = TASK_LAKE / f"e97-firstparty-teacher-authored-tranche-{tranche:03d}-admitted"
    admit_generated_collection(quarantine, overlap_path, output)
    (WORK / "teacher-authored" / f"tranche-{tranche:03d}" /
     "admission-receipt.json").write_bytes((output / "admission-receipt.json").read_bytes())
    return output


def verify_admitted(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Admission-receipt v3 + allowlist + overlap verification (fail-closed)."""
    from ndm.e97_protected_overlap import validate_authorized_overlap_receipt
    from ndm.e97_task_lake import validate_source_registry, validate_task_collection
    receipt = json.loads((root / "admission-receipt.json").read_bytes())
    if receipt.get("schema") != "emender-e97-first-party-admission-receipt-v3":
        raise SystemExit(f"{root.name}: admission receipt schema mismatch")
    registry_payload = (root / "source-registry.json").read_bytes()
    tasks_payload = (root / "tasks.jsonl").read_bytes()
    registry_sha = hashlib.sha256(registry_payload).hexdigest()
    tasks_sha = hashlib.sha256(tasks_payload).hexdigest()
    if registry_sha != receipt["registry_sha256"]:
        raise SystemExit(f"{root.name}: registry drifted")
    if tasks_sha != receipt["tasks_sha256"]:
        raise SystemExit(f"{root.name}: tasks drifted")
    registry = validate_source_registry(json.loads(registry_payload))
    tasks = validate_task_collection(
        [json.loads(line) for line in tasks_payload.decode().splitlines() if line.strip()],
        registry=registry)
    generation = json.loads((root / "generation-receipt.json").read_bytes())
    overlap_payload = (root / "protected-overlap-receipt.json").read_bytes()
    validate_authorized_overlap_receipt(
        json.loads(overlap_payload),
        receipt_sha256=hashlib.sha256(overlap_payload).hexdigest(),
        registry_sha256=generation["registry_sha256"],
        generation_receipt_sha256=hashlib.sha256(
            (root / "generation-receipt.json").read_bytes()).hexdigest(),
        tasks_sha256=generation["tasks_sha256"],
        archive_root_sha256=generation["archive_root_sha256"],
        generator_manifest_sha256=generation["generator_component_manifest_sha256"],
        source_archive_sha256=generation["generator_source_archive_sha256"],
        source_revision=generation["source_revision"],
        controller_source_sha256=generation["controller_source_sha256"])
    summary = {"lake_root": str(root), "registry_sha256": registry_sha,
               "tasks_sha256": tasks_sha,
               "admission_receipt_sha256": _sha_file(root / "admission-receipt.json"),
               "tasks": len(tasks)}
    return tasks, summary


def inject_tranche(root: Path, tranche: int,
                   frozen_record_path: Path) -> list[dict[str, Any]]:
    """Freeze verified bundles into bank/pool/pending as tas-* (round: 0)."""
    import inject_pool
    from rl_bank import bank_paths, freeze_pool_task
    bundles, summary = verify_admitted(root)
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    program_sha = _sha_file(inject_pool.VALIDATOR_PROGRAM_FIRST_ACTION)
    frozen_all = []
    skipped_ineligible = []
    first_action_bundles = 0
    protocol_breadth_bundles = 0
    for bundle in bundles:
        body = inject_pool.gym_task_body(bundle, root)
        identity = bundle["task"]["identity"]
        spec = json.loads((root / "private_validators" /
                           f"{identity}.json").read_bytes())
        if "required_first_action" in spec:
            first_action_bundles += 1
        if "expected_final" in spec:
            protocol_breadth_bundles += 1
        task_id = f"{POOL_PREFIX}-{identity[:16]}"
        task = {
            "schema": inject_pool.TASK_SCHEMA,
            "task_id": task_id,
            "attempts": 0,
            "round": 0,
            "body": body,
            "task_sha256": hashlib.sha256(
                _canonical(body).encode("utf-8")).hexdigest(),
            "generator": (f"sealed first-party task-lake authority ({root.name}; "
                          f"standing curriculum supply, operator directive "
                          f"2026-09-27)"),
            "authoring": ("teacher-authored standing supply (GLM-5.3-flash-background "
                          "via lunaroute background slots); author guards + sealed "
                          "mechanical solvability proof (validate_replay or the era-4 "
                          "gym-shaped workspace-write proof, both sealed "
                          "validators) + zero-collision protected-overlap receipt + "
                          "execution-panel anti-copy pre-screen + admission-receipt "
                          "v3 + machine-committed operator-allowlist entry verified; "
                          "task-lake validation layer verified; pinned validator "
                          f"program sha256 {program_sha[:16]}… bound"
                          + ("; first-action criterion: required_first_action declared "
                             "(the era-3 sealed validator grades the first emitted "
                             "action: exact tool + exact declared key arguments)"
                             if "required_first_action" in spec else "")
                          + ("; protocol-breadth outcome mode: expected_final + "
                             "mechanical outcome verification (exact answer, verbatim "
                             "read-backs, error-diagnosis ordering, write/edit "
                             "chains) via the era-4 sealed validator"
                             if "expected_final" in spec else "")),
            "source_commit": source_commit,
        }
        # E6 collision fix (2026-10-08): freeze_pool_task RAISES on
        # receipt-ineligible tasks (development split) — correct for bank-side
        # callers, but this supply-side loop was crashing and discarding every
        # tranche at injection (tranches ~1001-1024 lost to the pool). Filter
        # here instead: skip + log ineligible bundles; freeze only eligible.
        if not body["receipt_eligible"]:
            skipped_ineligible.append(task_id)
            continue
        for _bank_root in BANKS:
            freeze_pool_task(bank_paths(_bank_root), task)
        frozen_all.append({
            "task_id": task_id, "identity": identity,
            "split": bundle["split"], "family": bundle["task"]["family_id"],
            "first_action": "required_first_action" in spec,
            "receipt_eligible": body["receipt_eligible"],
            "task_sha256": task["task_sha256"], "lake_root": str(root),
        })
    record = {
        "schema": "emender-lakeexp-pool-injection-v1",
        "tranche": tranche, "frozen_unix": time.time(),
        "source_commit": source_commit, "program_sha256": program_sha,
        "first_action_bundles": first_action_bundles,
        "protocol_breadth_bundles": protocol_breadth_bundles,
        "skipped_ineligible": skipped_ineligible,
        "collections": [summary], "frozen": frozen_all,
    }
    frozen_record_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return frozen_all


# ---------------------------------------------------------------- tranche
def tranche_dirs_used() -> set[int]:
    used: set[int] = set()
    for base in (WORK / "teacher-authored", WORK / "teacher-authored" / "discarded"):
        if not base.is_dir():
            continue
        for path in base.iterdir():
            match = re.fullmatch(r"tranche-(\d{3,})(?:-.*)?", path.name)
            if match:
                used.add(int(match.group(1)))
    return used


def next_tranche() -> int:
    used = tranche_dirs_used()
    index = 2
    while index in used:
        index += 1
    return index


def discard_tranche(tranche: int, reason: str) -> Path:
    """A failed/rejected build is discarded, never retried into admission."""
    source = WORK / "teacher-authored" / f"tranche-{tranche:03d}"
    target_root = WORK / "teacher-authored" / "discarded"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"tranche-{tranche:03d}-discarded-{int(time.time())}"
    if source.exists():
        os.replace(source, target)
    (target / "discard-reason.txt").write_text(
        f"{_now_iso()} discarded by standing-supply daemon: {reason}\n")
    return target


def run_tranche(tranche: int, lanes: int, size: int,
                state: dict[str, Any]) -> dict[str, Any]:
    import teacher_author as ta
    record: dict[str, Any] = {
        "schema": TRANCHE_METRIC_SCHEMA, "tranche": tranche,
        "started_unix": time.time(), "lanes": lanes, "size": size,
        "pool_before": pool_counts(),
    }
    started = time.monotonic()
    try:
        author = author_parallel(tranche, size, lanes, state)
        record["authoring"] = author
        if author["transport_error"] or author["accepted"] < size:
            raise RuntimeError(f"authoring incomplete: {author}")
        quarantine = ta.build_tranche(f"{tranche:03d}")
        record["build_seconds"] = round(time.monotonic() - started - author["author_seconds"], 1)
        proofs = json.loads((quarantine.parent / "proofs.json").read_bytes())
        bad = [item for item in proofs
               if item.get("status") != "pass"
               or any(run["status"] != "pass" for run in item["validators"].values())]
        if bad:
            raise RuntimeError(f"solvability proof failures: {len(bad)}")
        overlap_path = run_overlap(tranche, quarantine)
        entry = mint_allowlist_entry(quarantine, overlap_path)
        write_allowlist_entry(entry)
        record["allowlist_commit"] = commit_allowlist(tranche, entry)
        root = admit_tranche(tranche, quarantine, overlap_path)
        frozen = inject_tranche(
            root, tranche,
            WORK / "bank-integration" / f"injection-teacher-tranche-{tranche:03d}.json")
        record["admitted_root"] = str(root)
        record["frozen_tasks"] = len(frozen)
        record["proofs_pass"] = len(proofs)
    except BaseException as exc:  # noqa: BLE001 - discard + record, never retry
        reason = f"{type(exc).__name__}: {exc}"
        discard_tranche(tranche, reason)
        record["status"] = "failed"
        record["reason"] = reason[:500]
        record["pool_after"] = pool_counts()
        record["finished_unix"] = time.time()
        append_metrics(record)
        append_status(f"- {_now_iso()} (standing supply) TRANCHE {tranche:03d} DISCARDED "
                      f"({reason[:220]}): rejected, not retried into admission; specs "
                      f"retained under teacher-authored/discarded/.")
        return record
    genres = Counter()
    curriculum_kinds = Counter()
    direct = 0
    first_action_bundles = 0
    protocol_breadth_bundles = 0
    for spec_path in sorted((WORK / "teacher-authored" / f"tranche-{tranche:03d}"
                             ).glob("authored-*.json")):
        spec = json.loads(spec_path.read_bytes())
        genres[spec.get("genre", "unknown")] += 1
        curriculum_kinds[spec.get("curriculum_kind", "core")] += 1
        if "required_first_action" in spec:
            first_action_bundles += 1
        elif "family" in spec:
            protocol_breadth_bundles += 1
        elif spec["token_file_path"] in spec["prompt"]:
            direct += 1
    record["status"] = "admitted"
    record["genre_distribution"] = dict(genres)
    record["curriculum_kind_distribution"] = dict(curriculum_kinds)
    record["first_action_bundles"] = first_action_bundles
    record["first_action_share"] = round(first_action_bundles / size, 3)
    record["protocol_breadth_bundles"] = protocol_breadth_bundles
    record["protocol_breadth_share"] = round(protocol_breadth_bundles / size, 3)
    record["direct_prompt_token_path"] = direct
    # read-observe-core discovery split (legacy core families only; the
    # first-action genre is direct by guard, era-4 families are excluded)
    legacy_core_bundles = size - protocol_breadth_bundles - first_action_bundles
    record["discovery"] = legacy_core_bundles - direct
    record["pool_after"] = pool_counts()
    record["consumption_snapshot"] = consumption(3.0)
    record["finished_unix"] = time.time()
    record["wall_seconds"] = round(time.monotonic() - started, 1)
    append_metrics(record)
    author = record["authoring"]
    append_status(
        f"- {_now_iso()} (standing supply) TRANCHE {tranche:03d}: {size} {POOL_PREFIX}-* "
        f"bundles admitted+injected; authoring {author['calls']} calls / "
        f"{author['rejected']} rejects, {lanes} lanes, {author['author_seconds']}s "
        f"({author['bundles_per_hour_achieved']} bundles/h at {lanes} lanes); "
        f"{record['proofs_pass']}/proofs pass; overlap zero-collision; allowlist "
        f"machine-commit {record['allowlist_commit'][:12]}; first-action criterion "
        f"{record['first_action_bundles']}/{size} bundles "
        f"({record['first_action_share']:.0%}); protocol-breadth "
        f"{record['protocol_breadth_bundles']}/{size} bundles "
        f"({record['protocol_breadth_share']:.0%}: "
        f"{', '.join(f'{kind}={count}' for kind, count in sorted(curriculum_kinds.items()) if kind not in ('core', 'first_action'))}); "
        f"pool pending "
        f"{record['pool_before']['pending']}->{record['pool_after']['pending']}; "
        f"consumption {record['consumption_snapshot']['retire_rate_per_h']}/h "
        f"(3h window).")
    return record


# ----------------------------------------------------------------- daemon
def _acquire_daemon_lock() -> "int | None":
    SUPPLY.mkdir(parents=True, exist_ok=True)
    handle = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(handle)
        return None
    os.write(handle, str(os.getpid()).encode())
    os.ftruncate(handle, len(str(os.getpid()).encode()))
    return handle


def backoff_lanes(lanes: int, steps: int) -> int:
    """Emergency reduction must never increase even a below-floor state."""
    return min(lanes, max(BACKOFF_FLOOR, lanes - steps))


def recovery_lanes(lanes: int, clean_streak: int, pending: int) -> int:
    """Recover gradually after clean tranches, retaining normal/hard caps."""
    if lanes < LANE_SOFT_CAP and clean_streak >= 2:
        return min(lanes + 2, LANE_SOFT_CAP)
    if lanes < LANE_HARD_CAP and clean_streak >= 4 and pending < LOW_WATER:
        return min(lanes + 2, LANE_HARD_CAP)
    return lanes


def health_adjust(state: dict[str, Any]) -> dict[str, Any]:
    """Teacher-priority back-off / re-scale based on probe + bank signals."""
    probe = control_probe()
    state["last_probe"] = probe
    lanes = int(state["lanes"])
    events: list[str] = []
    if not probe["ok"] or probe["latency_s"] >= PROBE_SEVERE_S:
        lanes = backoff_lanes(lanes, 2)
        events.append(f"probe severe ({probe.get('latency_s')}{probe.get('error', '')}) -> {lanes}")
    elif probe["latency_s"] >= PROBE_DEGRADE_S:
        lanes = backoff_lanes(lanes, 1)
        events.append(f"probe degraded ({probe['latency_s']}s) -> {lanes}")
    failures = scan_bank_failures(state)
    if failures:
        lanes = min(lanes, 4)
        events.append(f"{len(failures)} new LANE_FAILURE line(s) -> {lanes}")
        for line in failures:
            events.append(line)
    if events:
        state["clean_streak"] = 0
        state["lanes"] = lanes
        state["backoff_events"] = (state["backoff_events"] +
                                   [{"unix": time.time(), "events": events}])[-50:]
        append_metrics({"schema": "emender-standing-supply-backoff-v1",
                        "unix": time.time(), "events": events, "lanes": lanes,
                        "probe": probe})
    return probe


def daemon_iteration(state: dict[str, Any]) -> None:
    heartbeat(state)
    counts = pool_counts()
    if counts["pending"] >= HIGH_WATER:
        time.sleep(120)
        return
    probe = health_adjust(state)
    save_state(state)
    tranche = next_tranche()
    record = run_tranche(tranche, int(state["lanes"]), TRANCHE_SIZE, state)
    if record["status"] == "admitted":
        transport_clean = not record["authoring"]["transport_error"]
        if transport_clean and probe["ok"] and probe["latency_s"] < PROBE_DEGRADE_S:
            state["clean_streak"] = int(state["clean_streak"]) + 1
        else:
            state["clean_streak"] = 0
        lanes = int(state["lanes"])
        state["lanes"] = recovery_lanes(
            lanes, int(state["clean_streak"]), record["pool_after"]["pending"])
        if state["lanes"] != lanes:
            state["scaleup_events"].append(
                {"unix": time.time(), "from": lanes, "to": state["lanes"],
                 "reason": ("2 consecutive clean tranches" if lanes < LANE_SOFT_CAP
                            else "4 clean tranches with pool below low water (catch-up)")})
    else:
        state["clean_streak"] = 0
        if record["authoring"]["transport_error"]:
            state["lanes"] = backoff_lanes(int(state["lanes"]), 1)
    save_state(state)
    if pool_counts()["pending"] < LOW_WATER:
        return  # catch-up: next tranche immediately
    time.sleep(60)


def cmd_daemon(lanes_start: int) -> None:
    handle = _acquire_daemon_lock()
    if handle is None:
        print("another standing-supply daemon holds the lock; exiting", flush=True)
        raise SystemExit(2)
    PID_PATH.write_text(str(os.getpid()) + "\n")
    state = load_state()
    if not state.get("started_unix"):
        state["lanes"] = lanes_start
    state["started_unix"] = time.time()
    state["heartbeat_unix"] = time.time()
    save_state(state)
    ensure_status_section()
    probe = control_probe()
    state["last_probe"] = probe
    append_status(f"- {_now_iso()} (standing supply) DAEMON START pid={os.getpid()} "
                  f"lanes={state['lanes']} size={TRANCHE_SIZE} waters="
                  f"{LOW_WATER}/{HIGH_WATER} probe={probe.get('latency_s')}s; "
                  f"mechanism: detached daemon + cron watchdog (see section header).")
    save_state(state)
    while True:
        try:
            daemon_iteration(state)
        except Exception as exc:  # noqa: BLE001 - stay alive, record honestly
            state["backoff_events"].append(
                {"unix": time.time(), "events": [f"iteration error: "
                                                 f"{type(exc).__name__}: {exc}"]})
            save_state(state)
            append_status(f"- {_now_iso()} (standing supply) DAEMON iteration error "
                          f"({type(exc).__name__}: {str(exc)[:200]}); continuing.")
            time.sleep(120)


def cmd_watchdog() -> None:
    SUPPLY.mkdir(parents=True, exist_ok=True)
    guard = os.open(WATCHDOG_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another watchdog is running; exiting")
        return
    state = load_state()
    pid_text = PID_PATH.read_text().strip() if PID_PATH.is_file() else ""
    alive = bool(pid_text) and Path(f"/proc/{pid_text}").exists()
    stale = (time.time() - float(state.get("heartbeat_unix") or 0)) > HEARTBEAT_STALE_S
    if alive and not stale:
        print(f"{_now_iso()} daemon pid={pid_text} alive, heartbeat fresh; nothing to do")
        return
    if alive and stale:
        try:
            os.kill(int(pid_text), 15)
            time.sleep(10)
        except ProcessLookupError:
            pass
        print(f"{_now_iso()} daemon pid={pid_text} heartbeat STALE; killed for relaunch")
        append_status(f"- {_now_iso()} (standing supply) WATCHDOG: daemon pid={pid_text} "
                      f"heartbeat stale >{HEARTBEAT_STALE_S:.0f}s; killed for relaunch.")
    log = (LOGS / "standing-supply.log").open("ab")
    subprocess.Popen([str(VENV_PYTHON), str(Path(__file__).resolve()), "daemon"],
                     stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(WORK))
    print(f"{_now_iso()} daemon relaunched (detached)")
    append_status(f"- {_now_iso()} (standing supply) WATCHDOG: daemon relaunched "
                  f"(detached, start_new_session).")


def install_cron() -> None:
    line = (f"*/10 * * * * {VENV_PYTHON} {Path(__file__).resolve()} watchdog "
            f">> {LOGS / 'standing-watchdog.log'} 2>&1")
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    existing = current.stdout if current.returncode == 0 else ""
    if "standing_supply.py watchdog" in existing:
        print("cron watchdog entry already present")
        return
    merged = existing.rstrip("\n") + ("\n" if existing else "") + line + "\n"
    subprocess.run(["crontab", "-"], input=merged, check=True, text=True)
    print(f"cron watchdog installed:\n{line}")


# ----------------------------------------------------------------- status
def cmd_status() -> None:
    state = load_state()
    counts = pool_counts()
    cons = consumption(3.0)
    cons6 = consumption(6.0)
    records = []
    if METRICS_PATH.is_file():
        for line in METRICS_PATH.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("schema") == TRANCHE_METRIC_SCHEMA:
                records.append(record)
    admitted = [r for r in records if r["status"] == "admitted"]
    genres: Counter[str] = Counter()
    for record in admitted:
        genres.update(record.get("genre_distribution", {}))
    calls = sum(r["authoring"]["calls"] for r in admitted)
    rejects = sum(r["authoring"]["rejected"] for r in admitted)
    supply_hours = (sum(r["wall_seconds"] for r in admitted) / 3600
                    if admitted else None)
    print(json.dumps({
        "now": _now_iso(),
        "pool": counts,
        "waters": {"low": LOW_WATER, "high": HIGH_WATER},
        "consumption": {"3h": cons, "6h": cons6},
        "daemon": {"lanes": state.get("lanes"), "clean_streak": state.get("clean_streak"),
                   "heartbeat_age_s": round(time.time() - float(state.get("heartbeat_unix") or 0)),
                   "pid": PID_PATH.read_text().strip() if PID_PATH.is_file() else None,
                   "last_probe": state.get("last_probe")},
        "supply": {"tranches_attempted": len(records),
                   "tranches_admitted": len(admitted),
                   "bundles_admitted": sum(r["frozen_tasks"] for r in admitted),
                   "authoring_calls": calls, "authoring_rejects": rejects,
                   "reject_rate": round(rejects / calls, 3) if calls else None,
                   "mean_bundles_per_hour": (round(sum(r["frozen_tasks"] for r in admitted)
                                                   / supply_hours, 1)
                                             if supply_hours else None),
                   "per_lane_bundles_per_hour": (
                       round(sum(r["frozen_tasks"] for r in admitted) / supply_hours
                             / (sum(r["lanes"] for r in admitted) / len(admitted)), 1)
                       if supply_hours and admitted else None),
                   "genre_distribution": dict(genres),
                   "direct_vs_discovery": {
                       "direct": sum(r["direct_prompt_token_path"] for r in admitted),
                       "discovery": sum(r["discovery"] for r in admitted)}},
        "backoff_events": state.get("backoff_events", [])[-3:],
        "scaleup_events": state.get("scaleup_events", [])[-3:],
    }, indent=2, sort_keys=True))


def main() -> None:
    if os.environ.get("F9_DEPLOYMENT_APPROVED") != "1":
        raise SystemExit("F9 staged supply cannot run daemon/admission: operator deployment required")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["tranche", "daemon", "watchdog",
                                            "status", "probe", "install-cron"])
    parser.add_argument("--tranche", type=int, default=None,
                        help="explicit tranche number (tranche command)")
    parser.add_argument("--size", type=int, default=TRANCHE_SIZE)
    parser.add_argument("--lanes", type=int, default=LANE_START)
    args = parser.parse_args()
    if args.command == "tranche":
        state = load_state()
        tranche = args.tranche if args.tranche is not None else next_tranche()
        record = run_tranche(tranche, args.lanes, args.size, state)
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("consumption_snapshot",)}, indent=2, default=str))
        if record["status"] != "admitted":
            raise SystemExit(1)
    elif args.command == "daemon":
        cmd_daemon(args.lanes)
    elif args.command == "watchdog":
        cmd_watchdog()
    elif args.command == "status":
        cmd_status()
    elif args.command == "probe":
        print(json.dumps(control_probe(), indent=2))
    elif args.command == "install-cron":
        install_cron()


if __name__ == "__main__":
    main()
