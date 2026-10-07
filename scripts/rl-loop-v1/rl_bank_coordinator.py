#!/usr/bin/env python3
"""The DiLoCo sync coordinator for the e97-rl-loop-v1 bank (tranche 3).

A separate CPU process that owns the bank-level sync geometry per the locked
supervisor defaults:

  * every lane counts its own completed optimizer updates (monotonic
    ``updates_total`` in its lane state — the lane NEVER resets it);
  * when ANY lane's updates_total exceeds its counter snapshot at the last
    merge by K=32, the coordinator runs the DiLoCo merge across ALL lane
    lineages (weighted soup by cycle pass-rate; winner-take-all fallback;
    see rl_bank_merge), publishes the merged checkpoint atomically
    (ADR-003 R07/NDP15), and advances merges/current.json — every lane
    re-serves from the merged checkpoint at its next cycle boundary
    (receipts already in flight complete against their old lineage sha —
    each receipt binds the exact checkpoint it was produced against, so
    that is recorded, not lost);
  * the coordinator also sweeps expired task claims back to pending (lease
    semantics, filesystem-only — the crashed claimant's deadline bounds
    recovery), refreshes the pool from the admitted sealed first-party lake
    when pending and claims are empty (global content dedupe prevents
    accidental replay of retired tasks), and assembles the
    session metrics report.

Bounded termination (R14/NDP13): the loop exits at the STOP file, the
--max-seconds deadline, or --max-failures consecutive failed iterations.
NO SQLite anywhere (R01/R10 no-database clause); coordination state is
durable JSON + the digest-linked merge receipt chain only.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rl_bank import (MIN_FREE_BYTES, bank_paths, disk_free_bytes,
                     ensure_bank_layout, ensure_pinned_validator,
                     pool_status, read_coordinator_state, read_lane_state,
                     read_merge_current, refresh_pool, sweep_stale_claims,
                     write_coordinator_state)


def lane_states(bank: Mapping[str, Path], lanes: int) -> list[dict[str, Any]]:
    rows = []
    for lane in range(lanes):
        state = read_lane_state(bank["lanes"] / f"lane-{lane:02d}")
        if state is not None:
            rows.append(state)
    return rows


def trigger_check(states: list[dict[str, Any]], snapshot: Mapping[str, int],
                  k: int) -> tuple[int, int] | None:
    """The first lane whose updates since the last merge reach K, if any."""
    for state in sorted(states, key=lambda s: s["lane"]):
        base = int(snapshot.get(str(state["lane"]), 0))
        since = int(state.get("updates_total", 0)) - base
        if since >= k:
            return int(state["lane"]), since
    return None


def assemble_session_metrics(bank: Mapping[str, Path], lanes: int, *,
                             started_unix: float) -> dict[str, Any]:
    """The first-session report: lanes, claims, receipts, merges, teacher
    volume, losses, and the gym pass-rate trajectory per pool round."""
    import tiktoken

    from rl_receipts import walk_stream

    report: dict[str, Any] = {
        "schema": "emender-rl-loop-bank-session-metrics-v1",
        "observed_unix": time.time(),
        "started_unix": started_unix,
        "pool": pool_status(bank),
    }
    enc = tiktoken.get_encoding("p50k_base")

    per_lane = []
    receipts_by_kind: Counter[str] = Counter()
    teacher_calls = 0
    step_losses: list[float] = []
    for lane in range(lanes):
        root = bank["lanes"] / f"lane-{lane:02d}"
        state = read_lane_state(root)
        lane_row: dict[str, Any] = {"lane": lane, "state": state,
                                   "live": False}
        if state is not None:
            lane_row["live"] = (time.time() - float(state.get("updated_unix", 0))
                                 < 300)
        rows = []
        stream = root / "receipts" / "stream.jsonl"
        if stream.is_file():
            rows = walk_stream({"stream": stream, "receipts": root / "receipts"},
                               enc=enc)
        lane_row["receipts"] = len(rows)
        lane_row["receipts_by_kind"] = dict(Counter(r["kind"] for r in rows))
        lane_row["receipt_chain_head"] = rows[-1]["receipt_sha256"] if rows else None
        lane_row["targets"] = sum(r["episode"]["targets"] for r in rows)
        for kind in lane_row["receipts_by_kind"]:
            receipts_by_kind[kind] += lane_row["receipts_by_kind"][kind]
        cycles = []
        for metrics_path in sorted((root / "cycles").glob("cycle-*/metrics.json")):
            metrics = json.loads(metrics_path.read_text())
            cycles.append(metrics)
            teacher_calls += int(metrics.get("teacher_calls", 0) or 0)
            if metrics.get("train"):
                step_losses.append(float(metrics["train"]["loss"]))
        lane_row["cycles"] = len(cycles)
        lane_row["updates_total"] = int(state.get("updates_total", 0)) if state else 0
        per_lane.append(lane_row)
    report["lanes_live"] = sum(1 for row in per_lane if row["live"])
    report["lanes"] = per_lane
    report["receipts_by_kind"] = dict(receipts_by_kind)

    # tasks claimed/completed per lane + per-round pass-rate trajectory
    claimed: Counter[str] = Counter()
    completed: Counter[str] = Counter()
    trajectory: dict[int, dict[str, Any]] = {}
    for done_path in sorted(bank["pool_done"].glob("*.json")):
        task = json.loads(done_path.read_text())
        retired = task.get("retired", {})
        outcome = retired.get("outcome", {})
        lane = f"lane-{int(outcome.get('lane', -1)):02d}"
        completed[lane] += 1
        round_number = int(task.get("round", 0))
        row = trajectory.setdefault(round_number, {
            "round": round_number, "attempts": 0, "policy_pass": 0,
            "teacher_pass": 0, "receipts": 0, "receipt_kinds": Counter(),
            "splits": Counter(),
        })
        row["attempts"] += 1
        row["splits"][str(task.get("body", {}).get("split", "seed"))] += 1
        if outcome.get("attempt_grade_passed"):
            row["policy_pass"] += 1
        if outcome.get("stage") == "teacher":
            row["teacher_pass"] += 1
        if outcome.get("receipt"):
            row["receipts"] += 1
            row["receipt_kinds"][str(outcome.get("receipt_kind"))] += 1
    report["tasks_completed_per_lane"] = dict(completed)
    for row in trajectory.values():
        row["policy_pass_rate"] = (row["policy_pass"] / row["attempts"]
                                   if row["attempts"] else None)
        row["receipt_kinds"] = dict(row["receipt_kinds"])
        row["splits"] = dict(row["splits"])
    report["pass_rate_trajectory"] = [trajectory[key]
                                      for key in sorted(trajectory)]
    report["teacher_calls_total"] = teacher_calls
    report["step_losses"] = step_losses

    merges = []
    stream = bank["merge_stream"]
    if stream.is_file():
        for line in stream.read_text().splitlines():
            if line.strip():
                merges.append(json.loads(line))
    report["merge_events"] = [{
        "merge_index": m["merge_index"], "status": m["status"],
        "k": m.get("k"), "trigger_lane": m.get("trigger_lane"),
        "weights": {str(l["lane"]): l["weight"] for l in m.get("lanes", [])},
        "winner_lane": m.get("winner_lane"),
        "merged_checkpoint_sha256": m.get("merged_checkpoint_sha256"),
        "receipt_sha256": m["receipt_sha256"],
    } for m in merges]
    current = read_merge_current(bank)
    report["merge_current_epoch"] = current["epoch"] if current else 0
    return report


def pool_refresh_needed(status: Mapping[str, Any]) -> bool:
    """Do not refresh while tasks remain pending or in flight."""
    return (status["pending"] == 0 and status["claims_live"] == 0
            and status["claims_stale"] == 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--lake", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-seconds", type=int, default=4 * 3600)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--refresh-min-seconds", type=float, default=60.0,
                        help="min spacing between pool round refreshes")
    args = parser.parse_args()

    bank = bank_paths(args.bank)
    ensure_bank_layout(bank)
    program = ensure_pinned_validator(bank)
    started = time.time()
    deadline = started + args.max_seconds
    failures = 0
    last_refresh = 0.0

    # the first pool round exists before any lane can claim work
    state = read_coordinator_state(bank)
    if pool_refresh_needed(pool_status(bank)):
        state["pool_round"] = int(state.get("pool_round", 0)) + 1
        refresh_pool(bank, lake=args.lake, round_number=state["pool_round"],
                     program_path=program)
        write_coordinator_state(bank, state)
        last_refresh = time.monotonic()

    while not bank["stop"].is_file() and time.time() < deadline:
        try:
            state = read_coordinator_state(bank)
            swept = sweep_stale_claims(bank)
            if swept:
                print(f"COORD swept {swept} stale claim(s) back to pending",
                      flush=True)

            # ---- K=32 DiLoCo sync trigger (any lane's own update count
            #      since the last merge reaching K)
            states = lane_states(bank, args.lanes)
            trigger = trigger_check(states, state.get("counter_snapshot", {}),
                                    args.k)
            if trigger is not None:
                lane, since = trigger
                if disk_free_bytes(bank["root"]) < MIN_FREE_BYTES:
                    # low-disk fail-closed guard: DEFER the merge (its
                    # output is a ~24.6GB checkpoint) without touching the
                    # counter snapshot — the trigger stays armed and retries
                    # until the lanes' RECLAIM pass frees space.
                    print(f"COORD merge deferred: lane {lane} reached {since} "
                          f"updates (K={args.k}) but free disk "
                          f"< {MIN_FREE_BYTES // (1 << 30)}GiB", flush=True)
                else:
                    print(f"COORD merge trigger: lane {lane} completed {since} "
                          f"updates since the last merge (K={args.k})", flush=True)
                    from rl_bank_merge import (publish_merge_current, run_merge)

                    merge_index = int(state.get("merge_count", 0)) + 1
                    receipt = run_merge(bank, merge_index=merge_index, k=args.k,
                                        trigger_lane=lane, trigger_updates=since,
                                        lane_states=states)
                    if receipt["status"] == "published":
                        epoch = int(state.get("merge_epoch", 0)) + 1
                        publish_merge_current(bank, receipt, epoch=epoch)
                        state["merge_epoch"] = epoch
                        state["merge_count"] = merge_index
                        print(f"COORD MERGED index={merge_index} epoch={epoch} "
                              f"sha={receipt['merged_checkpoint_sha256'][:16]}",
                              flush=True)
                    else:
                        state["merge_count"] = merge_index
                        state["merge_skip_count"] = int(
                            state.get("merge_skip_count", 0)) + 1
                        print(f"COORD merge skipped (no pass signal): "
                              f"{receipt['skip_reason']}", flush=True)
                    # snapshot the counters observed at this merge boundary
                    state["counter_snapshot"] = {
                        str(row["lane"]): int(row.get("updates_total", 0))
                        for row in states}
                    write_coordinator_state(bank, state)

            # ---- bounded discovery of unseen training content only, once
            #      all pending/in-flight work has retired
            status = pool_status(bank)
            if (pool_refresh_needed(status)
                    and time.monotonic() - last_refresh >=
                    args.refresh_min_seconds):
                state["pool_round"] = int(state.get("pool_round", 0)) + 1
                refresh_pool(bank, lake=args.lake,
                             round_number=state["pool_round"],
                             program_path=program)
                write_coordinator_state(bank, state)
                last_refresh = time.monotonic()

            report = assemble_session_metrics(bank, args.lanes,
                                               started_unix=started)
            from rl_bank import atomic_write_json

            atomic_write_json(bank["report"] / "session-metrics.json", report)
            failures = 0
        except Exception as exc:  # bounded: never wedge the bank (R14)
            failures += 1
            print(f"COORD_FAILURE consecutive={failures}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if failures >= args.max_failures:
                raise SystemExit(f"coordinator: {args.max_failures} "
                                  f"consecutive failed iterations")
        time.sleep(args.poll_seconds)

    report = assemble_session_metrics(bank, args.lanes, started_unix=started)
    from rl_bank import atomic_write_json

    atomic_write_json(bank["report"] / "session-metrics.json", report)
    print("COORD_EXIT", json.dumps({
        "lanes_live": report["lanes_live"],
        "receipts_by_kind": report["receipts_by_kind"],
        "merge_events": len(report["merge_events"]),
        "teacher_calls_total": report["teacher_calls_total"],
        "rounds": len(report["pass_rate_trajectory"]),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
