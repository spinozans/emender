#!/usr/bin/env python3
"""The per-lane bank driver: one continuous serve->attempt->grade->teacher-
correct->receipts->train->re-serve loop per GPU lane (e97-rl-loop-v1 tranche 3).

The lane process is a CPU ORCHESTRATOR: it owns the lane state, claims tasks
from the ONE global pool (work-stealing, lease semantics on the filesystem),
and runs the GPU stages as subprocesses exactly in the proven v1 shape
(collect = one engine load per cycle; pack build; single-GPU masked-SFT
step; re-serve probe).  It holds a GPU lease only while it has claimed work
with unconsumed receipts and releases it at every cycle boundary —
yield-when-idle for the shared box (the bank must not squat 8 GPUs).

Bounded termination (ADR-003 R14/NDP13): every stage has a hard timeout,
failed stages kill their subprocess, a crashed claim's lease expires and the
coordinator sweeps it back to pending, and the lane exits nonzero after
--max-failures consecutive failed cycles or at the session deadline / STOP
file.

Subcommands:
  init      bind the lane lineage to a seed checkpoint
  collect   GPU stage: attempt/grade/teacher-correct/receipt for pool tasks
  run       the continuous lane loop (orchestrator)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import tiktoken

from rl_bank import (CLAIM_SCHEMA, MIN_FREE_BYTES, bank_paths, claim_pool_task,
                     disk_free_bytes, ensure_bank_layout, ensure_pinned_validator,
                     heartbeat_claim, lane_paths, pool_status,
                     read_lane_state, read_merge_current, reclaim_superseded_lineages,
                     requeue_claim, requeue_lane_claims, retire_task, sweep_stale_claims,
                     write_lane_state)
from rl_correction_pool import CorrectionPool
from rl_correction_config import resolve_correction_config
from rl_common import sha256_file
from rl_receipts import (append_receipt, grade_receipt_digest,
                         stream_tail_digest)
from rl_loop_driver import (DEFAULT_TEACHER_MODEL, MANIFEST_PATH,
                            POLICY_PANEL, PROVIDER_V2, TEACHER_PANEL,
                            _merged_correction_record, _receipt_from_episode,
                            grade_episode, load_curriculum, load_pilot,
                            load_policy_engine, make_panel,
                            make_policy_generate, make_teacher_generate,
                            degeneracy_screen, prepare_workspace,
                            retained_prefix, run_episode,
                            tool_manifest)

PY = sys.executable
REPO_ROOT = Path("/home/erikg/emender")
GPU_LEASE = REPO_ROOT / "scripts" / "gpu_lease.sh"

# era-8 (2026-10-04): below this per-step pack loss a fresh-optimizer update
# carries no learning signal (the +-lr*sign first-step jitter dominates and
# scatters the lineage — verified in isolation; see the unified-collapse
# receipt). Steps under the floor are skipped: checkpoint not adopted.
NO_SIGNAL_STEP_LOSS_FLOOR = 0.35


class _LeaseKeeper:
    """Background heartbeat keeper for the lane's GPU lease (broker TTL).

    The lease is acquired with --pid <this lane process> so it is born
    confirmed to the long-lived orchestrator pid; this keeper refreshes the
    broker heartbeat while a cycle's stages run, and the broker's own
    pid/starttime guard reaps the lease if the lane dies (fail-safe).
    """

    def __init__(self, interval: float = 60.0):
        self.interval = interval
        self.gpu: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, gpu: str) -> None:
        self.gpu = gpu
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                subprocess.run(
                    ["bash", str(GPU_LEASE), "heartbeat", str(self.gpu)],
                    capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.gpu = None


# ------------------------------------------------------------------ collect
def _correction_requires_fresh_solve(body, policy_record, tools) -> bool:
    """A continuation cannot replace a retained, sealed-wrong first action."""
    from rl_loop_driver import sealed_user_script
    if sealed_user_script(body) is not None:
        return True
    if not degeneracy_screen(policy_record)[0]:
        return True
    binding = body.get("task_lake")
    if binding is None:
        return False
    validator = binding["validator"]
    spec_path = Path(validator["spec_path"])
    if sha256_file(spec_path) != validator["spec_sha256"]:
        raise SystemExit("sealed validator spec identity drift")
    from scripts.e97_first_party_validator_first_action import (
        _check_first_action, _required_first_action)
    from scripts.e97_pi_native_codec import semantic_turn

    required = _required_first_action(json.loads(spec_path.read_text()))
    if required is None:
        return False
    messages = policy_record.get("source_messages") or []
    retained, _, _, _ = retained_prefix(messages, tools)
    for index in retained:
        turn = semantic_turn(messages[index], tools)
        if turn["name"] in ("think", "finish"):
            continue
        try:
            _check_first_action([{"tool_name": turn["name"],
                                  "arguments": turn["arguments"]}], required)
        except SystemExit:
            return True
        return False
    # No retained real action: a continuation can still supply the first one.
    return False


def _execute_correction(args, task, policy_record, workspace, task_dir,
                        pilot, curriculum, tools, enc, metrics):
    from ndm.e97_atomic import publish_bytes_no_replace
    from ndm.e97_onpolicy_records import canonical_json
    body = task["body"]
    gym = body.get("task_lake")
    pi_bin = Path(tool_manifest(curriculum)["pi_bin"])
    teacher_panel = make_panel(curriculum.SYSTEM, tools, TEACHER_PANEL)
    if args.teacher == "live" and not getattr(pilot, "_correction_probe_snapshot", False):
        from rl_teacher_deadlines import harden_teacher_deadlines
        harden_teacher_deadlines(pilot)
    teacher_generate = (None if args.teacher == "fixture" else make_teacher_generate(
        args.teacher, args.teacher_model, tools, enc, pilot, metrics))
    correction_dir = task_dir / "correction"
    correction_dir.mkdir(parents=True, exist_ok=False)
    if args.teacher == "fixture":
        teacher_generate = make_teacher_generate(
            args.teacher, args.teacher_model, tools, enc, pilot, metrics, body=body)
    # era-7 (2026-10-03): a degenerate policy prefix POISONS the supervised
    # context — 925/925 teacher receipts had supervise_from=1 (clean teacher
    # frames only after a degenerate policy frame), so the model never saw
    # a clean first frame from a clean context and the degenerate mode was
    # self-perpetuating (0 screen passes in 24h). When the policy attempt
    # fails the degeneracy screen, the teacher solves FRESH: clean
    # workspace, no prefix, supervise_from=0. E5 also starts fresh when
    # a clean retained first action already violates its sealed criterion.
    # Genuinely recoverable failures keep the spliced repair continuation.
    from rl_loop_driver import sealed_user_script, is_end_state_task
    fresh_solve = _correction_requires_fresh_solve(body, policy_record, tools)
    prefix_messages = (None if fresh_solve
                       else policy_record.get("source_messages"))
    teacher_panel_eff = teacher_panel
    if gym is not None:
        dense = 0
        if prefix_messages:
            _, dense, _, _ = retained_prefix(prefix_messages, tools)
        teacher_panel_eff = make_panel(curriculum.SYSTEM, tools, {
            **TEACHER_PANEL,
            "max_turns": max(1, int(gym["limits"]["turns"]) - dense)})
    workspace_corr = (prepare_workspace(correction_dir / "workspace", body)
                      if fresh_solve else workspace)
    teacher_record = run_episode(
        episode_dir=correction_dir, workspace=workspace_corr, prompt=body["prompt"],
        panel=teacher_panel_eff, generate=teacher_generate, enc=enc,
        pi_bin=pi_bin, manifest_path=MANIFEST_PATH, pilot=pilot,
        curriculum=curriculum, seconds=teacher_panel_eff["episode_seconds"],
        prefix_messages=prefix_messages,
        prefix_policy_text=(None if fresh_solve
                            else policy_record.get("native_record")), user_script=sealed_user_script(body), end_state_task=is_end_state_task(body))
    passed, grade = grade_episode(
        body, workspace_corr if fresh_solve else workspace, pilot, task,
        record=teacher_record,
        episode_dir=correction_dir, stage="teacher", system=curriculum.SYSTEM,
        judge_metrics=metrics, judge_model=DEFAULT_TEACHER_MODEL)
    publish_bytes_no_replace(
        correction_dir / "grade.json",
        (canonical_json(dict(grade)) + "\n").encode("utf-8"), mode=0o400)
    print(f"COLLECT teacher grade passed={passed} "
          f"status={teacher_record['status']}", flush=True)
    return {"passed": passed, "grade": grade, "teacher_record": teacher_record,
            "fresh_solve": fresh_solve, "metrics": metrics.dump() if hasattr(metrics, "dump") else {"calls": []}}


def execute_correction_spec(spec):
    """Model-free child entry; it grades but NEVER appends/retire/publishes receipts."""
    from types import SimpleNamespace
    # Qualification binds an isolated, immutable pilot; production uses the
    # normal source and never supplies this probe-only snapshot field.
    snapshot = spec.get("pilot_snapshot")
    if snapshot is None:
        pilot = load_pilot()
    else:
        import importlib.util
        if sha256_file(Path(snapshot["path"])) != snapshot["sha256"]:
            raise ValueError("correction pilot snapshot identity drift")
        module_spec = importlib.util.spec_from_file_location("correction_pilot_snapshot", snapshot["path"])
        pilot = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(pilot)
        pilot._correction_probe_snapshot = True
    curriculum = load_curriculum()
    tools = tool_manifest(curriculum)["model_visible_tools"]
    enc = tiktoken.get_encoding("p50k_base")
    attempt_dir = Path(spec["task_dir"]) / "attempt"
    if sha256_file(attempt_dir / "episode-private.json") != spec["episode_artifacts"]["attempt_episode_sha256"]:
        raise ValueError("correction attempt identity drift")
    from ndm.e97_onpolicy_records import canonical_json
    if canonical_json(json.loads((attempt_dir / "episode-private.json").read_text())) != canonical_json(spec["policy_record"]):
        raise ValueError("correction snapshot differs from immutable attempt")
    grade = json.loads((attempt_dir / "grade.json").read_text())
    if grade_receipt_digest(grade) != spec["episode_artifacts"]["attempt_grade_sha256"]:
        raise ValueError("correction grade identity drift")
    if _correction_requires_fresh_solve(spec["task"]["body"], spec["policy_record"], tools) != spec["fresh_solve"]:
        raise ValueError("correction routing identity drift")
    class DurableMetrics(pilot.Metrics):
        def record(self, entry):
            super().record(entry)
            with (Path(spec["task_dir"]) / "correction-api.jsonl").open("a") as log:
                log.write(json.dumps(entry, sort_keys=True) + "\n")
    return _execute_correction(SimpleNamespace(**spec["settings"]), spec["task"],
        spec["policy_record"], Path(spec["workspace"]), Path(spec["task_dir"]),
        pilot, curriculum, tools, enc, DurableMetrics())


def bank_collect(args: argparse.Namespace) -> None:
    """One lane cycle's GPU collect over the global pool (v1 collect shape).

    Differences from the tranche-2 collect, all structural for the bank:
    the queue is the ONE global pool with lease claims (heartbeat between
    stages; a crashed claim expires and is swept back by the coordinator);
    the receipt stream is LANE-private (its own digest chain); every claimed
    task retires to pool/done with an outcome block (pass / teacher-pass /
    failed) — the pool round refresh re-attempts sealed tasks against the
    CURRENT lineages next round; teacher calls are throttled to queue
    consumption (min interval between correction calls, corrections only
    for claimed tasks that failed on-policy).
    """
    config = resolve_correction_config(args)
    args.teacher_model = config["teacher_model"]
    bank = bank_paths(args.bank)
    ensure_bank_layout(bank)
    paths = lane_paths(bank, args.lane)
    state = read_lane_state(paths["root"])
    if state is None:
        raise SystemExit("lane not initialized")
    lineage = Path(state["lineage_path"])
    if sha256_file(lineage) != state["lineage_sha256"]:
        raise SystemExit("lane lineage identity mismatch")

    pilot = load_pilot()
    curriculum = load_curriculum()
    manifest = tool_manifest(curriculum)
    pi_bin = Path(manifest["pi_bin"])
    tools = manifest["model_visible_tools"]
    enc = tiktoken.get_encoding("p50k_base")

    cycle_dir = paths["episodes"] / f"cycle-{args.cycle:04d}"
    if cycle_dir.is_dir():
        # archive a crashed collect's partial evidence; the lane cycle number
        # only advances on success, so a retry hits the same directory name
        crashed = cycle_dir.with_name(
            f"{cycle_dir.name}-crashed-{int(time.time())}")
        cycle_dir.rename(crashed)
        print(f"COLLECT archived crashed cycle evidence at {crashed.name}",
              flush=True)
    cycle_dir.mkdir(parents=True, exist_ok=False)
    print(f"COLLECT lane={args.lane} cycle={args.cycle} "
          f"checkpoint={lineage.name}", flush=True)
    engine = load_policy_engine(lineage, Path(args.args_json), device="cuda")
    policy_generate = make_policy_generate(engine, tools, enc)
    print("COLLECT policy engine ready", flush=True)

    metrics = pilot.Metrics()
    policy_panel = make_panel(curriculum.SYSTEM, tools, POLICY_PANEL)

    # the orchestrator pre-claimed one task so this GPU cycle has work;
    # claim more (work-stealing) until the pool drains or max-tasks
    claimed: dict[str, dict[str, Any]] = {}
    if args.preclaimed:
        pre_path = bank["pool_claims"] / f"{args.preclaimed}.claim"
        if not pre_path.is_file():
            raise SystemExit(f"pre-claimed task missing: {args.preclaimed}")
        claimed[args.preclaimed] = json.loads(pre_path.read_text())
    while len(claimed) < args.max_tasks:
        task = claim_pool_task(bank, args.lane, ttl_seconds=args.claim_ttl,
                               skip=set(claimed))
        if task is None:
            break
        claimed[task["task_id"]] = task

    receipts = 0
    prev_digest = stream_tail_digest(paths["stream"])
    # receipts bind the exact lineage this lane served (v1 field contract)
    policy_checkpoint_binding = {
        "checkpoint_path": state["lineage_path"],
        "checkpoint_sha256": state["lineage_sha256"],
    }
    attempts = 0
    outcomes: list[dict[str, Any]] = []
    window_attempts = window_passes = window_receipts = 0
    last_teacher_call = 0.0
    pool = (CorrectionPool(config["pool_width"], cycle_dir / "correction-pool")
            if config["async_enabled"] else None)
    pending = {}
    terminal_drain_s = 0.0

    def _finalize_correction(context, result):
        nonlocal prev_digest, receipts, window_receipts
        task, outcome = context["task"], context["outcome"]
        body = task["body"]
        eligible = bool(body.get("receipt_eligible", True))
        episode_artifacts = context["episode_artifacts"]
        policy_generations = [item for item in context["policy_record"].get("generations", [])
                              if item.get("reason") == "valid"]
        def _retire(stage, kind):
            retire_task(bank, task, outcome={
                "lane": args.lane, "cycle": args.cycle, "stage": stage,
                "receipt_kind": kind, "receipt_eligible": eligible,
                "split": body.get("split", "seed"), "receipt": outcome.get("receipt"),
                "attempt_grade_passed": outcome["attempt"]["grade_passed"]})
        if result.get("error"):
            outcome["correction"] = {"status": "error", "grade_passed": False,
                                     "reason": result["error"]}
            _retire("failed", None)
            return
        passed, grade = result["passed"], result["grade"]
        teacher_record, fresh_solve = result["teacher_record"], result["fresh_solve"]
        splice = teacher_record.get("prefix") or {}
        prefix_frames = int(splice.get("retained_frames", 0))
        teacher_identity = (
            {"mode": "live", "model": args.teacher_model,
             "api": "lunaroute /v1/chat/completions"}
            if args.teacher == "live" else
            {"mode": "fixture",
             "note": "authored deterministic continuation stub (not GLM-5.3)"})
        teacher_identity.update(
            continuation=("fresh-episode" if fresh_solve
                          else "same-transcript-splice"),
            prefix_frames=prefix_frames,
            dropped_trailing_finish=bool(splice.get("dropped_trailing_finish")),
            dropped_unresolved_call=bool(splice.get("dropped_unresolved_call")))
        outcome["correction"] = {
            "status": teacher_record["status"], "grade_passed": passed,
            "reason": teacher_record.get("reason"),
            "prefix_frames": prefix_frames,
        }
        if (passed and teacher_record["status"] == "finished"
                and teacher_record.get("close_verified", False)):
            if eligible:
                merged = _merged_correction_record(
                    teacher_record, policy_generations[:prefix_frames])
                receipt = _receipt_from_episode(
                    cycle=args.cycle, task=task, kind="teacher-corrected",
                    policy_checkpoint=policy_checkpoint_binding, record=merged,
                    supervise_from=prefix_frames, grade=grade,
                    teacher=teacher_identity,
                    attempt_link={**episode_artifacts,
                                  "policy_generations": len(policy_generations),
                                  "retained_prefix_frames": prefix_frames,
                                  "dropped_trailing_finish": bool(
                                      splice.get("dropped_trailing_finish")),
                                  "dropped_unresolved_call": bool(
                                      splice.get("dropped_unresolved_call"))},
                    enc=enc, prev=prev_digest)
                prev_digest = append_receipt(paths, receipt)
                receipts += 1
                window_receipts += 1
                outcome.update(receipt=receipt["receipt_sha256"],
                               targets=receipt["episode"]["targets"],
                               receipt_kind="teacher-corrected",
                               supervise_from=prefix_frames)
                mode = "fresh" if fresh_solve else "spliced"
                print(f"COLLECT receipt=teacher-corrected({mode}, "
                      f"supervise_from={prefix_frames}) "
                      f"{receipt['receipt_sha256'][:16]}", flush=True)
            else:
                print("COLLECT dev-split correction measured (no receipt: "
                      "sealed train/development split isolation)", flush=True)
            _retire("teacher", "teacher-corrected" if eligible else None)
        else:
            # the round refresh re-attempts this sealed task against the
            # CURRENT lineages next round; retire honestly failed
            _retire("failed", None)

    def _poll_corrections():
        if pool is not None:
            for sequence, result in pool.poll():
                context = pending.pop(sequence)
                for call in result.get("metrics", {}).get("calls", []):
                    metrics.record(call)
                _finalize_correction(context, result)
            for context in pending.values():
                heartbeat_claim(bank, context["task"], lane=args.lane,
                                ttl_seconds=args.claim_ttl)

    try:
        for task_id, task in claimed.items():
            _poll_corrections()
            attempts += 1
            heartbeat_claim(bank, task, lane=args.lane, ttl_seconds=args.claim_ttl)
            body = task["body"]
            task_dir = cycle_dir / task_id
            print(f"COLLECT task={task_id} template={body['template']}", flush=True)
            gym = body.get("task_lake")
            eligible = bool(body.get("receipt_eligible", True))
            if eligible:
                window_attempts += 1

            attempt_panel = policy_panel
            if gym is not None:
                attempt_panel = make_panel(curriculum.SYSTEM, tools, {
                    **POLICY_PANEL,
                    "max_turns": min(POLICY_PANEL["max_turns"],
                                     int(gym["limits"]["turns"]))})
            attempt_dir = task_dir / "attempt"
            workspace = prepare_workspace(attempt_dir / "workspace", body)
            from rl_loop_driver import sealed_user_script, is_end_state_task
            policy_record = run_episode(
                episode_dir=attempt_dir, workspace=workspace, prompt=body["prompt"],
                panel=attempt_panel, generate=policy_generate, enc=enc,
                pi_bin=pi_bin, manifest_path=MANIFEST_PATH, pilot=pilot,
                curriculum=curriculum, seconds=attempt_panel["episode_seconds"], user_script=sealed_user_script(body), end_state_task=is_end_state_task(body))
            heartbeat_claim(bank, task, lane=args.lane, ttl_seconds=args.claim_ttl)
            passed, grade = grade_episode(
                body, workspace, pilot, task, record=policy_record,
                episode_dir=attempt_dir, stage="policy", system=curriculum.SYSTEM,
                judge_metrics=metrics, judge_model=DEFAULT_TEACHER_MODEL)
            from ndm.e97_atomic import publish_bytes_no_replace
            from ndm.e97_onpolicy_records import canonical_json

            publish_bytes_no_replace(
                attempt_dir / "grade.json",
                (canonical_json(dict(grade)) + "\n").encode("utf-8"), mode=0o400)
            print(f"COLLECT policy grade passed={passed} "
                  f"status={policy_record['status']}", flush=True)

            episode_artifacts = {
                "attempt_grade_sha256": grade_receipt_digest(grade),
                "attempt_episode_sha256": sha256_file(
                    attempt_dir / "episode-private.json"),
                "attempt_status": policy_record["status"],
                "attempt_reason": policy_record.get("reason"),
            }
            outcome: dict[str, Any] = {
                "task_id": task_id, "template": body.get("template"),
                "split": body.get("split", "seed"), "receipt_eligible": eligible,
                "attempt": {"status": policy_record["status"],
                            "grade_passed": passed,
                            "reason": policy_record.get("reason")},
                "correction": None, "receipt": None, "targets": 0,
            }

            def _retire(stage: str, kind: str | None) -> None:
                retire_task(bank, task, outcome={
                    "lane": args.lane, "cycle": args.cycle, "stage": stage,
                    "receipt_kind": kind, "receipt_eligible": eligible,
                    "split": body.get("split", "seed"),
                    "receipt": outcome.get("receipt"),
                    "attempt_grade_passed": outcome["attempt"]["grade_passed"],
                })

            if passed and policy_record["status"] == "finished":
                if eligible:
                    receipt = _receipt_from_episode(
                        cycle=args.cycle, task=task, kind="on-policy-success",
                        policy_checkpoint=policy_checkpoint_binding,
                        record=policy_record,
                        supervise_from=0, grade=grade, teacher=None,
                        attempt_link=None, enc=enc, prev=prev_digest)
                    prev_digest = append_receipt(paths, receipt)
                    receipts += 1
                    window_passes += 1
                    window_receipts += 1
                    outcome.update(receipt=receipt["receipt_sha256"],
                                   targets=receipt["episode"]["targets"],
                                   receipt_kind="on-policy-success")
                    print(f"COLLECT receipt=on-policy-success "
                          f"{receipt['receipt_sha256'][:16]}", flush=True)
                else:
                    print("COLLECT dev-split task measured (no receipt: sealed "
                          "train/development split isolation)", flush=True)
                outcome["receipt_kind"] = "on-policy-success" if eligible else None
                _retire("policy", "on-policy-success" if eligible else None)
                outcomes.append(outcome)
                continue

            # Admission preserves the existing lane-local minimum start interval.
            _poll_corrections()
            can_admit = pool is None or len(pending) < config["pool_width"]
            if can_admit:
                interval = time.monotonic() - last_teacher_call
                if interval < args.teacher_min_interval:
                    time.sleep(args.teacher_min_interval - interval)
                last_teacher_call = time.monotonic()
            context = {"task": task, "outcome": outcome, "policy_record": policy_record,
                       "episode_artifacts": episode_artifacts}
            outcomes.append(outcome)
            if pool is None:
                result = _execute_correction(args, task, policy_record, workspace, task_dir,
                                             pilot, curriculum, tools, enc, metrics)
                heartbeat_claim(bank, task, lane=args.lane, ttl_seconds=args.claim_ttl)
                _finalize_correction(context, result)
            else:
                spec = {"settings": {"teacher": args.teacher, "teacher_model": args.teacher_model},
                        "task": task, "policy_record": policy_record,
                        "policy_checkpoint": policy_checkpoint_binding,
                        "episode_artifacts": episode_artifacts,
                        "fresh_solve": _correction_requires_fresh_solve(body, policy_record, tools),
                        "workspace": str(workspace), "task_dir": str(task_dir)}
                sequence, failure = pool.submit(spec)
                if failure is not None:
                    _finalize_correction(context, failure)
                else:
                    pending[sequence] = context

        # End-of-cycle barrier: no unfinished correction can enter training or a
        # later cycle. Every failure is finalized explicitly, without a receipt.
        if pool is not None:
            drain_started = time.monotonic()
            try:
                while pending:
                    _poll_corrections()
                    if pending:
                        time.sleep(0.2)
            finally:
                terminal_drain_s = time.monotonic() - drain_started
                pool.close()
    finally:
        if pool is not None:
            pool.close()

    summary = {
        "schema": "emender-rl-loop-collect-summary-v1",
        "cycle": args.cycle, "lane": args.lane,
        "attempts": attempts, "receipts": receipts,
        "teacher_calls": len(metrics.calls),
        "policy_checkpoint_sha256": state["lineage_sha256"],
        "teacher_mode": args.teacher,
        "correction_config": config,
        "correction_terminal_drain_s": terminal_drain_s,
        "correction_pool_exhaustions": sum(
            (outcome.get("correction") or {}).get("reason") == "pool exhausted"
            for outcome in outcomes),
        "policy_generate_state": getattr(policy_generate, "state", {}),
        "outcomes": outcomes,
    }
    publish_bytes_no_replace(
        cycle_dir / "collect-summary.json",
        (canonical_json(summary) + "\n").encode("utf-8"), mode=0o400)
    if metrics.calls:
        publish_bytes_no_replace(
            cycle_dir / "teacher-metrics.json",
            (canonical_json(metrics.dump()) + "\n").encode("utf-8"), mode=0o400)
    print(f"COLLECT_DONE lane={args.lane} cycle={args.cycle} "
          f"attempts={attempts} receipts={receipts} "
          f"window_attempts+={window_attempts} window_passes+={window_passes}",
          flush=True)


# ------------------------------------------------------------- orchestrator
class _Stop(Exception):
    pass


def adopt_merge_current(bank: Mapping[str, Path],
                        state: dict[str, Any]) -> dict[str, Any] | None:
    """Adopt the newest merged lineage when this lane hasn't yet (R07).

    Fail-closed: the pointer advances only from a readable, complete,
    sha-verified merged checkpoint (ADR-003 R07/NDP15).  On adoption the
    lane's per-window counters (window_attempts/window_passes/
    window_receipts) RESET TO ZERO: soup weights are the pass rate SINCE
    THE LAST MERGE (rl_bank_merge), so the adopting lane starts its next
    merge window here — lane-owned state write, race-free.  updates_total
    is NEVER reset (monotonic K geometry).  Returns the merge-current
    pointer on adoption, else None.
    """
    current = read_merge_current(bank)
    if not current or int(current["epoch"]) <= int(state["merge_epoch"]):
        return None
    merged = Path(current["merged_checkpoint_path"])
    observed = sha256_file(merged)
    if observed != current["merged_checkpoint_sha256"]:
        raise _Stop("merged checkpoint identity mismatch at adoption")
    # the pre-adoption lineage is now SUPERSEDED (RECLAIM-LOG candidate;
    # the adopted merged checkpoint itself lives under merges/ and is kept)
    ledger = list(state.get("superseded_lineages") or [])
    ledger.append({"path": state["lineage_path"],
                   "sha256": state["lineage_sha256"],
                   "superseded_unix": time.time()})
    state.update(merge_epoch=int(current["epoch"]),
                 lineage_path=str(merged), lineage_sha256=observed,
                 window_attempts=0, window_passes=0, window_receipts=0,
                 superseded_lineages=ledger,
                 status="merge-adopted")
    return current



def _gpu_acquire(timeout: int) -> str | None:
    """Acquire ONE exclusive GPU lease bound to THIS process pid."""
    import re

    try:
        completed = subprocess.run(
            ["bash", str(GPU_LEASE), "acquire", "1", "--wait",
             "--timeout", str(timeout), "--pid", str(os.getpid())],
            capture_output=True, text=True, timeout=timeout + 120)
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r'GPU_LEASE_HELD="(\d+)"', completed.stdout)
    if not match:
        return None
    return match.group(1)


def _gpu_release(gpu: str | None = None) -> None:
    """Release this process's GPU lease (pid+host owned; broker verified)."""
    command = ["bash", str(GPU_LEASE), "release"]
    if gpu is not None:
        command.append(str(gpu))
    environment = dict(os.environ)
    environment["GPU_LEASE_PID"] = str(os.getpid())
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=120,
                       env=environment)
    except subprocess.TimeoutExpired:
        pass


def _spawn(command: list[str], *, env: Mapping[str, str], timeout: int,
           log_path: Path) -> None:
    """Run one stage subprocess with a HARD deadline (R14 bounded termination)."""
    with log_path.open("a") as log:
        try:
            completed = subprocess.run(command, env=dict(env), timeout=timeout,
                                       stdout=log, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            raise _Stop(f"stage timeout after {timeout}s: {command[1:3]}")
    if completed.returncode != 0:
        raise _Stop(f"stage failed rc={completed.returncode}: {command[1:3]}")


def lane_run(args: argparse.Namespace) -> None:
    bank = bank_paths(args.bank)
    ensure_bank_layout(bank)
    paths = lane_paths(bank, args.lane)
    state = read_lane_state(paths["root"])
    if state is None:
        raise SystemExit("lane not initialized; run init first")
    ensure_pinned_validator(bank)
    # ---- v2: persist the train-step channel + screened parameters in lane
    #      state (surfaces in every coordinator report; lane-owned write).
    #      Default sft-receipts keeps unflagged lanes byte-identical.
    state["train_step_channel"] = args.lane_train_step
    if "block_mode" in state:
        state["block_mode"] = args.block_mode
    if args.lane_train_step == "policy-gradient":
        state["pg_parameters"] = {
            "lr": args.pg_lr, "kl_beta": args.pg_kl_beta,
            "kl_guard": args.pg_kl_guard,
            "window_cycles": args.pg_window_cycles,
            "max_episodes": args.pg_max_episodes,
            "group_by": args.pg_group_by, "min_group": args.pg_min_group,
            "anchor_mode": "parent",
            "screen": "smoke/pg-v2/screen/operator-selection.json",
        }
        state["pg_updates_total"] = int(state.get("pg_updates_total", 0))
        state["pg_fallbacks_total"] = int(state.get("pg_fallbacks_total", 0))
    write_lane_state(paths["root"], state)
    if args.block_mode == "on" and not getattr(args, "collector_only", False):
        _initialize_block_state(state, args)
        write_lane_state(paths["root"], state)
    deadline = time.monotonic() + args.max_seconds
    failures = 0
    env = dict(os.environ)
    stopped = {"flag": False}

    def _mark_stop(signum, frame):
        stopped["flag"] = True

    signal.signal(signal.SIGTERM, _mark_stop)
    signal.signal(signal.SIGINT, _mark_stop)
    log_root = paths["root"] / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    while not stopped["flag"]:
        if bank["stop"].is_file() or time.monotonic() >= deadline:
            state.update(status="stopped", updated_unix=time.time())
            write_lane_state(paths["root"], state)
            print(f"LANE_EXIT lane={args.lane} cycles={state['lane_cycle']}",
                  flush=True)
            return

        # ---- merge adoption: the pointer advances only from a readable,
        #      complete, sha-verified merged checkpoint (ADR-003 R07).
        #      Adoption resets this lane's per-window counters (soup weights
        #      are the pass rate since the LAST merge — supervisor-approved
        #      semantics; updates_total stays monotonic for the K trigger).
        adopted = (None if getattr(args, "collector_only", False) else
                   adopt_merge_current(bank, state))
        # ---- RECLAIM-LOG discipline: reclaim THIS lane's superseded lineage
        #      checkpoints once past the grace window (merged checkpoints,
        #      every lane's current lineage, and anything outside this
        #      lane's training tree are refused fail-closed inside
        #      reclaim_superseded_lineages).  The lane owns its state file,
        #      so the ledger mutation is race-free.
        reclaimed = reclaim_superseded_lineages(bank, state)
        if adopted is not None or reclaimed:
            state.update(updated_unix=time.time())
            write_lane_state(paths["root"], state)
            if adopted is not None:
                print(f"LANE_MERGE_ADOPTED lane={args.lane} "
                      f"epoch={adopted['epoch']} "
                      f"sha={adopted['merged_checkpoint_sha256'][:16]}", flush=True)
        if reclaimed:
            total_bytes = sum(int(record["bytes"]) for record in reclaimed)
            print(f"LANE_RECLAIMED lane={args.lane} files={len(reclaimed)} "
                  f"bytes={total_bytes} (reclaim-log.jsonl)", flush=True)

        # ---- low-disk fail-closed guard (the run.sh 100GiB-guard
        #      precedent): PAUSE instead of starting a cycle whose 24.6GB
        #      checkpoint write would risk ENOSPC.
        if disk_free_bytes(bank["root"]) < MIN_FREE_BYTES:
            if state.get("status") != "disk-low":
                state.update(status="disk-low", updated_unix=time.time())
                write_lane_state(paths["root"], state)
                print(f"LANE_DISK_LOW lane={args.lane} "
                      f"free<{MIN_FREE_BYTES//(1<<30)}GiB — pausing "
                      f"(no claim; in-flight state intact)", flush=True)
            time.sleep(args.poll_seconds)
            continue

        # ---- claim work from the ONE global pool (work-stealing).  A claim
        #      is CPU-side: no GPU is held until work exists.
        sweep_stale_claims(bank)
        task = claim_pool_task(bank, args.lane,
                               ttl_seconds=args.claim_ttl)
        if task is None:
            if state.get("status") != "idle":
                state.update(status="idle", updated_unix=time.time())
                write_lane_state(paths["root"], state)
                print(f"LANE_IDLE lane={args.lane} (no claimable work; "
                      f"GPU lease released — yield-when-idle)", flush=True)
            time.sleep(args.poll_seconds)
            continue

        cycle = int(state["lane_cycle"]) + 1
        cycle_log = log_root / f"cycle-{cycle:04d}.log"
        gpu = _gpu_acquire(args.lease_wait)
        if gpu is None:
            # no GPU servable right now (shared box): return the claim
            # voluntarily so another lane can serve it; bounded retry
            requeue_claim(bank, task["task_id"])
            state.update(status="lease-wait", updated_unix=time.time())
            write_lane_state(paths["root"], state)
            time.sleep(args.lease_retry_seconds)
            continue
        keeper = _LeaseKeeper()
        keeper.start(gpu)
        state.update(status="serving", gpu=gpu, updated_unix=time.time())
        write_lane_state(paths["root"], state)
        print(f"LANE_CYCLE lane={args.lane} cycle={cycle} gpu={gpu} "
              f"first_task={task['task_id']}", flush=True)
        cycle_env = {**env, "CUDA_VISIBLE_DEVICES": gpu}

        receipts = 0
        try:
            _spawn([PY, __file__, "collect", "--bank", str(args.bank),
                    "--lane", str(args.lane), "--cycle", str(cycle),
                    "--args-json", str(args.args_json),
                    "--teacher", args.teacher,
                    "--teacher-model", resolve_correction_config(args)["teacher_model"],
                    "--max-tasks", str(args.max_tasks),
                    "--claim-ttl", str(args.claim_ttl),
                    "--teacher-min-interval", str(args.teacher_min_interval),
                    "--preclaimed", task["task_id"]],
                   env=cycle_env, timeout=args.collect_timeout,
                   log_path=cycle_log)

            summary_path = paths["episodes"] / f"cycle-{cycle:04d}" / \
                "collect-summary.json"
            summary = json.loads(summary_path.read_text())
            receipts = int(summary["receipts"])
            state["lane_cycle"] = cycle
            state["receipts_total"] = int(state["receipts_total"]) + receipts
            for outcome in summary["outcomes"]:
                if outcome.get("receipt_eligible"):
                    state["window_attempts"] = int(state["window_attempts"]) + 1
                    if outcome["attempt"]["grade_passed"]:
                        state["window_passes"] = int(state["window_passes"]) + 1
            state["window_receipts"] = int(state["window_receipts"]) + receipts

            # Each channel completes train -> probe -> adopt before the next
            # channel reads lineage_path. Anchor work must not become a sibling.
            _train_cycle_channels(bank, paths, state, args, cycle, cycle_env,
                                  cycle_log, summary, gpu, receipts)
            failures = 0
        except _Stop as stop:
            failures += 1
            # reclaim this lane's still-claimed tasks: the failed collect
            # subprocess is dead, so its unretired claims are returned to the
            # pool immediately instead of waiting out the lease TTL
            reclaimed = requeue_lane_claims(bank, args.lane)
            if reclaimed:
                print(f"LANE_RECLAIMED lane={args.lane} claims={reclaimed}",
                      flush=True)
            state.update(status="failed",
                         note=str(stop), updated_unix=time.time())
            write_lane_state(paths["root"], state)
            print(f"LANE_FAILURE lane={args.lane} cycle={cycle} "
                  f"consecutive={failures}: {stop}", flush=True)
            if failures >= args.max_failures:
                state.update(status="exhausted", updated_unix=time.time())
                write_lane_state(paths["root"], state)
                raise SystemExit(f"lane {args.lane}: {args.max_failures} "
                                 f"consecutive failed cycles")
        finally:
            # ---- yield-when-idle: the GPU lease is held only around the
            #      stages that need it; released at EVERY cycle boundary
            #      (claims whose receipts were consumed; nothing squatted).
            keeper.stop()
            _gpu_release(gpu)
            state.update(gpu=None, updated_unix=time.time())
            write_lane_state(paths["root"], state)


def _probe_adopt_train_row(bank, paths, state, args, cycle, cycle_env,
                           cycle_log, summary, gpu, row, train_channel,
                           *, row_idx: int, pg_event=None) -> bool:
    probe_name = ("re-serve-probe.json" if row_idx == 0 else
                  f"re-serve-probe-{row_idx}.json")
    _spawn([PY, str(Path(__file__).resolve().parent / "rl_loop_driver.py"), "probe",
            "--args-json", str(args.args_json),
            "--checkpoint", row["checkpoint"],
            "--checkpoint-sha256", row["checkpoint_sha256"],
            "--output", str(paths["cycles"] / f"cycle-{cycle:04d}" / probe_name)],
           env=cycle_env, timeout=args.probe_timeout, log_path=cycle_log)
    probe_path = paths["cycles"] / f"cycle-{cycle:04d}" / probe_name
    try:
        probe = json.loads(probe_path.read_text())
    except (OSError, ValueError) as exc:
        raise _Stop(f"probe record unreadable: {exc}") from exc
    if probe.get("frame_valid") is not True \
            or probe.get("checkpoint_sha256") != row["checkpoint_sha256"]:
        _assemble_lane_metrics(bank, paths, args, cycle, summary, row, gpu,
                               train_channel=train_channel, pg_event=pg_event,
                               probe_name=probe_name, adopted=False)
        _unlink_orphan_checkpoint(row, "invalid probe")
        print(f"LANE_REJECTED lane={args.lane} cycle={cycle} "
              f"channel={train_channel} (probe not valid for candidate)", flush=True)
        return False
    ledger = list(state.get("superseded_lineages") or [])
    ledger.append({"path": state["lineage_path"], "sha256": state["lineage_sha256"],
                   "superseded_unix": time.time()})
    state.update(lineage_path=row["checkpoint"], lineage_sha256=row["checkpoint_sha256"],
                 superseded_lineages=ledger, updates_total=int(state["updates_total"]) + 1,
                 status="trained", updated_unix=time.time())
    channel_counter = {"sft-anchor-corpus": "anchor_updates_total",
                       "policy-gradient": "pg_updates_total"}.get(train_channel)
    if channel_counter:
        state[channel_counter] = int(state.get(channel_counter, 0)) + 1
    write_lane_state(paths["root"], state)
    if train_channel == "sft-receipts":
        _record_adopted_exposure(paths, cycle, row)
    _assemble_lane_metrics(bank, paths, args, cycle, summary, row, gpu,
                           train_channel=train_channel, pg_event=pg_event,
                           probe_name=probe_name, adopted=True)
    print(f"LANE_TRAINED lane={args.lane} cycle={cycle} channel={train_channel} "
          f"loss={row['loss']} new_ckpt={row['checkpoint_sha256'][:16]} "
          f"updates_total={state['updates_total']}", flush=True)
    return True


def _train_cycle_channels(bank, paths, state, args, cycle, cycle_env, cycle_log,
                          summary, gpu, receipts: int) -> None:
    """Sequential channels retain every adopted update in the final lineage."""
    if getattr(args, "collector_only", False):
        state.update(status="collected", updated_unix=time.time())
        write_lane_state(paths["root"], state)
        _assemble_lane_metrics(bank, paths, args, cycle, summary, None, gpu,
                               train_channel=None)
        print(f"LANE_COLLECTOR_ONLY lane={args.lane} cycle={cycle} receipts={receipts} "
              "(seed lineage retained; training disabled)", flush=True)
        return
    attempted = 0
    adopted = 0
    pg_event = None

    def probe_adopt(row, channel, event=None):
        nonlocal attempted, adopted
        if row is None:
            return False
        accepted = _probe_adopt_train_row(
            bank, paths, state, args, cycle, cycle_env, cycle_log, summary, gpu,
            row, channel, row_idx=attempted, pg_event=event)
        attempted += 1
        adopted += int(accepted)
        return accepted

    if state.get("train_step_channel") == "policy-gradient":
        row, pg_event = _pg_train_row(paths, state, args, cycle, cycle_env, cycle_log)
        if row is not None:
            probe_adopt(row, "policy-gradient", pg_event)
        else:
            state["pg_fallbacks_total"] = int(state.get("pg_fallbacks_total", 0)) + 1
            print(f"LANE_PG_FALLBACK lane={args.lane} cycle={cycle} "
                  f"outcome={(pg_event or {}).get('outcome')}", flush=True)
    if args.anchor_period > 0 and cycle % args.anchor_period == 0 \
            and args.anchor_authority_root is not None:
        probe_adopt(_anchor_train_row(paths, state, args, cycle, cycle_env, cycle_log),
                    "sft-anchor-corpus")
    if getattr(args, "block_mode", "off") == "on":
        _run_receipts_block(paths, state, args, cycle, cycle_env, cycle_log,
                            lambda row: probe_adopt(row, "sft-receipts"))
    elif receipts:
        probe_adopt(_sft_train_row(paths, state, args, cycle, cycle_env, cycle_log),
                    "sft-receipts")
    if not adopted:
        state.update(status="collected", updated_unix=time.time())
        write_lane_state(paths["root"], state)
        if not attempted:
            _assemble_lane_metrics(bank, paths, args, cycle, summary, None, gpu,
                                   train_channel=None, pg_event=pg_event)
        print(f"LANE_COLLECTED lane={args.lane} cycle={cycle} receipts={receipts} "
              "(no adopted train step)", flush=True)


def block_steps(targets: int) -> int:
    # Pre-registered density: ceil(authority assistant targets / 1400), clamped
    # to [8, 256]. --full-pass accumulates multiple packs per update if needed
    # to traverse every pack; packs repeat if the step budget exceeds their count.
    return min(256, max(8, (targets + 1399) // 1400))


def _initialize_block_state(state, args) -> None:
    if min(args.block_min_targets, args.block_max_targets,
           args.block_max_wait_cycles) <= 0:
        raise _Stop("BLOCK limits must be positive")
    if args.block_max_targets < args.block_min_targets:
        raise _Stop("BLOCK_MAX_TARGETS must be >= BLOCK_MIN_TARGETS")
    state["block_mode"] = "on"
    for key in ("block_inventory_targets", "block_wait_cycles",
                "block_attempts_total", "block_adopted_total", "block_skipped_total",
                "block_backoff_count", "block_retry_after_cycle"):
        state.setdefault(key, 0)


def _block_streams_root(args) -> Path | None:
    if getattr(args, "block_mode", "off") != "on":
        return None
    configured = getattr(args, "block_streams_root", None)
    if configured is not None:
        return Path(configured)
    # Production run args always bind --bank; lightweight legacy test callers
    # without a bank retain their workspace-only inventory.
    return Path(args.bank) / "lanes" if hasattr(args, "bank") else None


def _block_inventory(paths, streams_root: Path | None = None) -> list[dict]:
    from rl_build_pack import (load_consumed_ledger, load_selection_receipts,
                               select_window_receipts)

    receipts = load_selection_receipts(paths, enc=tiktoken.get_encoding("p50k_base"),
                                      streams_root=streams_root)
    consumed = load_consumed_ledger(paths["root"] / "consumed-receipts.jsonl")
    return select_window_receipts(receipts, consumed, len(receipts))


def _block_trigger(state, args, cycle: int, inventory: list[dict]) -> str | None:
    state["block_inventory_targets"] = sum(int(r["episode"]["targets"])
                                            for r in inventory)
    # Count completed collect cycles with inventory, not wall time or tasks.
    if state.get("block_last_inventory_cycle") != cycle:
        state["block_wait_cycles"] = (int(state["block_wait_cycles"]) + 1
                                      if inventory else 0)
        state["block_last_inventory_cycle"] = cycle
    if not inventory or cycle < int(state["block_retry_after_cycle"]):
        return None
    if state["block_inventory_targets"] >= args.block_min_targets:
        return "targets"
    if state["block_wait_cycles"] >= args.block_max_wait_cycles:
        return "max-wait"
    return None


def _run_receipts_block(paths, state, args, cycle, cycle_env, cycle_log,
                         probe_adopt) -> None:
    _initialize_block_state(state, args)
    inventory = _block_inventory(paths, _block_streams_root(args))
    reason = _block_trigger(state, args, cycle, inventory)
    write_lane_state(paths["root"], state)
    if reason is None:
        return
    state["block_attempts_total"] += 1
    print(f"BLOCK_TRIGGER lane={args.lane} cycle={cycle} reason={reason} "
          f"inventory_targets={state['block_inventory_targets']} "
          f"wait_cycles={state['block_wait_cycles']}", flush=True)
    write_lane_state(paths["root"], state)
    try:
        row = _sft_train_row(paths, state, args, cycle, cycle_env, cycle_log,
                             block=True)
        accepted = row is not None and probe_adopt(row)
    except _Stop:
        _block_skipped(paths, state, args, cycle, "stage-failure")
        raise
    if not accepted:
        _block_skipped(paths, state, args, cycle, "no-signal-or-probe-rejection")
        return
    state["block_adopted_total"] += 1
    state["block_backoff_count"] = 0
    state["block_retry_after_cycle"] = 0
    state["block_wait_cycles"] = 0
    state["block_inventory_targets"] = sum(int(r["episode"]["targets"])
                                            for r in _block_inventory(paths, _block_streams_root(args)))
    write_lane_state(paths["root"], state)
    print(f"BLOCK_ADOPTED lane={args.lane} cycle={cycle} "
          f"sha={row['checkpoint_sha256']} "
          f"inventory_targets={state['block_inventory_targets']}", flush=True)


def _block_skipped(paths, state, args, cycle, reason) -> None:
    state["block_skipped_total"] += 1
    state["block_backoff_count"] += 1
    # A rejected block leaves its inventory/wait age intact, skips the next
    # collect cycle, then retries. Bounded one-cycle cooldown, no silent drain.
    state["block_retry_after_cycle"] = cycle + 2
    write_lane_state(paths["root"], state)
    print(f"BLOCK_SKIPPED lane={args.lane} cycle={cycle} reason={reason} "
          f"backoff={state['block_backoff_count']} "
          f"retry_after_cycle={state['block_retry_after_cycle']}", flush=True)


def _adopted_exposure_rows(paths, cycle: int, row: dict) -> list[dict]:
    """Resolve only sampled packs; unsampled receipts remain window-eligible."""
    authority = paths["packs"] / f"cycle-{cycle:04d}" / "authority"
    manifest_path = authority / "manifest.json"
    if sha256_file(manifest_path) != row.get("authority_manifest_sha256"):
        raise _Stop("exposure authority manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    metadata_info = manifest["outputs"]["metadata"]
    metadata_path = authority / metadata_info["path"]
    if sha256_file(metadata_path) != metadata_info["sha256"]:
        raise _Stop("exposure receipt metadata mismatch")
    records = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    sampled = row.get("sampled_packs")
    if not sampled:
        raise _Stop("receipts checkpoint has no sampled-pack exposure log")
    receipt_packs = {}
    sources = {}
    for pack in sampled:
        for record_id in pack["record_ids"]:
            if not isinstance(record_id, int) or not 0 <= record_id < len(records):
                raise _Stop("sampled record outside receipt authority")
            record = records[record_id]
            receipt = record["receipt_sha256"]
            if "source_lane" in record:
                sources[receipt] = record["source_lane"]
            receipt_packs.setdefault(receipt, set()).add(pack["pack_id"])
    return [{"receipt_sha256": receipt, "exposed_cycle": cycle,
             **({"source_lane": sources[receipt]} if receipt in sources else {}),
             "sampled_pack_ids": sorted(pack_ids),
             "pack_manifest_sha256": row["pack_manifest_sha256"],
             "adopted_checkpoint_sha256": row["checkpoint_sha256"]}
            for receipt, pack_ids in receipt_packs.items()]


def _record_adopted_exposure(paths, cycle: int, row: dict) -> None:
    rows = _adopted_exposure_rows(paths, cycle, row)
    with (paths["root"] / "consumed-receipts.jsonl").open("a") as ledger:
        for entry in rows:
            ledger.write(json.dumps(entry, sort_keys=True) + "\n")


def _sft_train_row(paths, state, args, cycle, cycle_env, cycle_log,
                   *, block: bool = False) -> dict | None:
    """Receipts-SFT: one step by default, a complete pack pass in block mode.
    Returns the train 'checkpoint' event row, or None on no signal
    (raises _Stop on hard stage failure — unchanged v1 behavior)."""
    import subprocess as _sp

    _spawn([PY, str(Path(__file__).resolve().parent /
                    "rl_build_pack.py"),
            "--workspace", str(paths["root"]),
            "--cycle", str(cycle), "--min-targets", "1" if block else "8",
            # era-10 dose fix: pack the oldest 32 UNCONSUMED receipts from the
            # whole verified stream (FIFO, ledger-tracked) instead of this
            # cycle's 1-4. Only sampled, adopted packs are consumed.
            *(["--all-unconsumed", "--max-targets", str(args.block_max_targets)]
              if block else ["--window", "32"]),
            "--consumed-ledger", str(paths["root"] / "consumed-receipts.jsonl"),
            *(["--streams-root", str(_block_streams_root(args))]
              if _block_streams_root(args) is not None else [])],
           env=cycle_env, timeout=args.pack_timeout,
           log_path=cycle_log)
    build = json.loads((paths["packs"] / f"cycle-{cycle:04d}" /
                        "build-summary.json").read_text())
    steps = block_steps(build["targets"]) if block else 1
    if block:
        print(f"BLOCK_PACKED lane={args.lane} cycle={cycle} "
              f"targets={build['targets']} packs={build['packs']} steps={steps}",
              flush=True)
    training = paths["training"] / f"cycle-{cycle:04d}"
    source_commit = _sp.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    _spawn([PY, str(Path(__file__).resolve().parent /
                    "rl_train_step.py"),
            "--parent-checkpoint", state["lineage_path"],
            "--parent-sha256", state["lineage_sha256"],
            "--args-json", str(args.args_json),
            "--authority-root",
            str(paths["packs"] / f"cycle-{cycle:04d}" / "authority"),
            "--authority-sha256", build["authority_manifest_sha256"],
            "--pack-root",
            str(paths["packs"] / f"cycle-{cycle:04d}" / "packs"),
            "--pack-sha256", build["pack_manifest_sha256"],
            "--output-root", str(training / "checkpoints"),
            "--log-jsonl", str(training / "log.jsonl"),
            "--source-commit", source_commit, "--steps", str(steps),
            *(["--full-pass"] if block else []),
            "--context-size", str(build["context_size"]),
            "--projection-chunk-size", "2048"],
           env=cycle_env, timeout=args.train_timeout,
           log_path=cycle_log)
    train_event = None
    for line in (training / "log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if candidate.get("event") == "checkpoint":
            train_event = candidate
    if train_event is None or "checkpoint_sha256" not in train_event:
        raise _Stop("no checkpoint event from the train stage")
    if block:
        print(f"BLOCK_TRAINED lane={args.lane} cycle={cycle} "
              f"steps={steps} sha={train_event['checkpoint_sha256']}", flush=True)
        expected = build["receipts"]
        if len(_adopted_exposure_rows(paths, cycle, train_event)) != expected:
            _unlink_orphan_checkpoint(train_event, "incomplete block exposure")
            raise _Stop("block did not sample every packed receipt")
    # era-8 no-signal step skip (2026-10-04, unified-collapse finding): a
    # fresh-optimizer Adam step at near-zero pack loss carries no learning
    # signal — the update is dominated by the +-lr*sign first-step jitter
    # across all parameters, which SCATTERS the lineage (verified in
    # isolation: a step at loss 0.086 visibly degrades generation; a step
    # at loss 1.4 improves it). Lane losses fell 0.89 -> 0.09 across
    # sessions 3-7 as the model memorized the fixed anchor packs and the
    # receipt templates; the accelerating doc-NLL drift was this scatter.
    # Fail-closed against scatter: below the floor the checkpoint is NOT
    # adopted and the lineage is left unchanged.
    step_loss = float(train_event.get("loss", train_event.get("final_loss", 1e9)))
    if step_loss < NO_SIGNAL_STEP_LOSS_FLOOR:
        print(f"TRAIN no-signal step skipped (loss={step_loss:.4f} < "
              f"{NO_SIGNAL_STEP_LOSS_FLOOR}): jitter would scatter the lineage; "
              f"checkpoint {train_event['checkpoint_sha256'][:16]} NOT adopted",
              flush=True)
        _unlink_orphan_checkpoint(train_event, "no-signal skip")
        return None
    _adopted_exposure_rows(paths, cycle, train_event)
    return train_event


def _unlink_orphan_checkpoint(train_event, reason: str) -> None:
    """Delete an un-adopted trained checkpoint immediately (review-A WARN-2:
    skipped steps otherwise orphan ~24.6GB files; when the 100GiB free floor
    trips, all lanes pause and merges defer)."""
    import os as _os
    path = train_event.get("checkpoint")
    if path and _os.path.exists(path):
        _os.unlink(path)
        print(f"TRAIN orphan checkpoint removed ({reason}): "
              f"{str(path)[-80:]}", flush=True)


def _anchor_train_row(paths, state, args, cycle, cycle_env, cycle_log) -> dict | None:
    """v3 corpus-replay anchor: 1-step masked-SFT on a pack from the FROZEN
    admitted corpus (default: the E4 main-prep admission), rotating through
    screened passing sampler keys. The merge-5 fix: the gym is a shaping
    layer on a corpus-dominant diet, never the whole meal."""
    import subprocess as _sp

    keys = [int(k) for k in args.anchor_keys.split(",") if k.strip()]
    key = keys[cycle % len(keys)]
    training = paths["training"] / f"cycle-{cycle:04d}"
    source_commit = _sp.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    _spawn([PY, str(Path(__file__).resolve().parent /
                    "rl_train_step.py"),
            "--parent-checkpoint", state["lineage_path"],
            "--parent-sha256", state["lineage_sha256"],
            "--args-json", str(args.args_json),
            "--authority-root", str(args.anchor_authority_root),
            "--authority-sha256", args.anchor_authority_sha256,
            "--pack-root", str(args.anchor_pack_root),
            "--pack-sha256", args.anchor_pack_sha256,
            "--output-root", str(training / "anchor-checkpoints"),
            "--log-jsonl", str(training / "log.jsonl"),
            "--source-commit", source_commit, "--steps", "1",
            "--context-size", "65536",
            "--sampler-key", str(key),
            "--projection-chunk-size", "2048"],
           env=cycle_env, timeout=args.train_timeout,
           log_path=cycle_log)
    train_event = None
    for line in (training / "log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if candidate.get("event") == "checkpoint":
            train_event = candidate
    if train_event is None or "checkpoint_sha256" not in train_event:
        raise _Stop("no checkpoint event from the anchor train stage")
    # era-8 no-signal skip (same rationale as the receipts channel): the 24
    # fixed anchor packs are deeply memorized after seven sessions; anchor
    # steps at near-zero loss are pure scatter.
    step_loss = float(train_event.get("loss", train_event.get("final_loss", 1e9)))
    if step_loss < NO_SIGNAL_STEP_LOSS_FLOOR:
        print(f"TRAIN no-signal ANCHOR step skipped (loss={step_loss:.4f} < "
              f"{NO_SIGNAL_STEP_LOSS_FLOOR}): jitter would scatter the lineage; "
              f"checkpoint {train_event['checkpoint_sha256'][:16]} NOT adopted",
              flush=True)
        _unlink_orphan_checkpoint(train_event, "no-signal anchor skip")
        return None
    print(f"LANE_ANCHOR_UPDATE lane={args.lane} cycle={cycle} key={key} "
          f"loss={train_event.get('loss')}", flush=True)
    return train_event


def _pg_streams_root(args) -> Path | None:
    configured = getattr(args, "block_streams_root", None)
    if configured is not None:
        return Path(configured)
    root = _block_streams_root(args)
    if root is not None:
        return root
    if int(os.environ.get("COLLECTOR_LANES", "0")) > 0:
        return Path(args.bank) / "lanes"
    return None


def _pg_train_row(paths, state, args, cycle, cycle_env, cycle_log) -> tuple[dict | None, dict]:
    """The v2 policy-gradient train stage: build the trajectory batch from
    recent cycles (this lane by default, the grid when configured; ALL
    train-split attempts — passes AND failures), run the guarded PG step.
    Grid batches explicitly replay every turn under the current lineage.

    Returns (train_row_or_None, pg_event).  The caller falls back to the SFT
    stage when the row is None (no contrastive group = no PG signal, or a
    fail-closed guard trip: nonfinite loss/grad, CE-consistency mismatch, or
    realized-KL blowout) — the bank never loses a lane to the experimental
    train step.
    """
    import subprocess as _sp

    training = paths["training"] / f"cycle-{cycle:04d}"
    training.mkdir(parents=True, exist_ok=True)
    batch_path = training / "pg-batch.json"
    _spawn([PY, str(Path(__file__).resolve().parent / "rl_pg_batch.py"),
            "build", "--episodes-root", str(paths["root"]),
            "--last-cycles", str(args.pg_window_cycles),
            "--group-by", args.pg_group_by,
            "--max-episodes", str(args.pg_max_episodes),
            "--min-group", str(args.pg_min_group),
            *(["--streams-root", str(_pg_streams_root(args))]
              if _pg_streams_root(args) is not None else []),
            "--output", str(batch_path)],
           env=cycle_env, timeout=args.pg_batch_timeout,
           log_path=cycle_log)
    batch = json.loads(batch_path.read_text())
    pg_event = {"channel": "policy-gradient", "cycle": cycle,
                "batch_sha256": batch["batch_sha256"],
                "contrast": batch["contrast"],
                "groups": batch["groups"],
                "episodes": len(batch["entries"]),
                "turns": sum(len(e["turns"]) for e in batch["entries"])}
    if not batch["contrast"]:
        pg_event["outcome"] = "no-contrast"
        return None, pg_event
    source_commit = _sp.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    log_jsonl = training / "pg-log.jsonl"
    step_cmd = [PY, str(Path(__file__).resolve().parent /
                        "rl_policy_gradient_step.py"),
                "--parent-checkpoint", state["lineage_path"],
                "--parent-sha256", state["lineage_sha256"],
                "--anchor-mode", "parent",
                "--args-json", str(args.args_json),
                "--batch", str(batch_path),
                "--batch-sha256", batch["batch_sha256"],
                "--output-root", str(training / "checkpoints"),
                "--log-jsonl", str(log_jsonl),
                "--pg-stream", str(paths["training"] / "pg-stream.jsonl"),
                "--source-commit", source_commit,
                "--run-tag", f"lane-{args.lane}-cycle-{cycle:04d}",
                "--steps", "1",
                "--lr", repr(args.pg_lr),
                "--kl-beta", repr(args.pg_kl_beta),
                "--kl-guard", repr(args.pg_kl_guard)]
    completed = None
    with cycle_log.open("a") as log:
        try:
            completed = _sp.run(step_cmd, env=dict(cycle_env),
                                timeout=args.pg_train_timeout,
                                stdout=log, stderr=_sp.STDOUT)
        except _sp.TimeoutExpired:
            pg_event["outcome"] = "stage-timeout"
            return None, pg_event
    if completed is None or completed.returncode != 0:
        rc = completed.returncode if completed is not None else "none"
        # fail-closed: guard trip or step failure — the SFT fallback keeps the
        # lane's update cadence; the guard reason is in the step log
        guard = None
        if log_jsonl.is_file():
            for line in log_jsonl.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") == "guard":
                    guard = row
        pg_event["outcome"] = "guard-trip" if guard is not None else "step-failed"
        pg_event["returncode"] = rc
        if guard is not None:
            pg_event["guard"] = guard
        return None, pg_event
    train_event = None
    for line in log_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "checkpoint":
            train_event = row
    if train_event is None or not train_event.get("checkpoint_sha256"):
        pg_event["outcome"] = "no-checkpoint-event"
        return None, pg_event
    pg_event["outcome"] = "policy-gradient-update"
    for key in ("loss", "grad_norm", "realized_kl_vs_old_mean"):
        for line in log_jsonl.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "step":
                pg_event[key] = row.get(key)
    return train_event, pg_event


def _assemble_lane_metrics(bank, paths, args, cycle, summary, train_row, gpu,
                          *, train_channel=None, pg_event=None,
                          probe_name="re-serve-probe.json", adopted=None) -> None:
    """Per-lane-cycle metrics in the loop's existing cycles/cycle-XXXX format."""
    from rl_receipts import walk_stream

    rows = walk_stream(paths, enc=tiktoken.get_encoding("p50k_base"))
    cycle_rows = [row for row in rows if row["cycle"] == cycle]
    metrics = {
        "schema": "emender-rl-loop-cycle-metrics-v1",
        "cycle": cycle,
        "lane": args.lane,
        "gpu": gpu,
        "policy_checkpoint_sha256": summary["policy_checkpoint_sha256"],
        "attempts": summary["attempts"],
        "outcomes": summary["outcomes"],
        "attempt_pass": sum(1 for o in summary["outcomes"]
                            if o["attempt"]["grade_passed"]),
        "corrections": sum(1 for o in summary["outcomes"] if o["correction"]),
        "teacher_pass": sum(1 for o in summary["outcomes"]
                            if o["correction"] and o["correction"]["grade_passed"]),
        "teacher_calls": summary.get("teacher_calls", 0),
        "receipts": len(cycle_rows),
        "receipt_kinds": {kind: sum(1 for r in cycle_rows if r["kind"] == kind)
                          for kind in sorted({r["kind"] for r in cycle_rows})},
        "targets_queued": sum(r["episode"]["targets"] for r in cycle_rows),
        "spliced_receipts": [r["receipt_sha256"] for r in cycle_rows
                             if r["episode"]["supervise_from"] > 0],
        "receipt_chain_head": rows[-1]["receipt_sha256"] if rows else None,
    }
    if train_row is not None:
        metrics["train"] = train_row
        probe_path = paths["cycles"] / f"cycle-{cycle:04d}" / probe_name
        metrics["probe_frame_valid"] = json.loads(
            probe_path.read_text()).get("frame_valid")
    else:
        metrics["train"] = None
        metrics["probe_frame_valid"] = None
    metrics["train_step_channel"] = train_channel
    metrics["adopted"] = adopted
    if pg_event is not None:
        metrics["pg_event"] = pg_event
    cycle_dir = paths["cycles"] / f"cycle-{cycle:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    out_path = cycle_dir / "metrics.json"
    prior = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
        except Exception:
            prior = {}
    # era-9: both the anchor meal and the receipts seasoning can adopt in
    # one cycle; every train row lands in the "trains" array (read-modify-
    # write so the second adoption does not clobber the first).
    trains = list(prior.get("trains") or [])
    trains.append({"train": metrics.get("train"), "adopted": adopted,
                   "probe_frame_valid": metrics.get("probe_frame_valid"),
                   "train_step_channel": train_channel,
                   **({"pg_event": pg_event} if pg_event is not None else {})})
    metrics["trains"] = trains
    out_path.write_text(
        json.dumps(metrics, indent=1, sort_keys=True) + "\n")


# -------------------------------------------------------------------- init
def lane_init(args: argparse.Namespace) -> None:
    bank = bank_paths(args.bank)
    ensure_bank_layout(bank)
    paths = lane_paths(bank, args.lane)
    from rl_bank import init_lane_state

    checkpoint = Path(args.checkpoint)
    observed = sha256_file(checkpoint)
    if observed != args.checkpoint_sha256:
        raise SystemExit("seed checkpoint sha256 mismatch")
    state = init_lane_state(bank, args.lane, checkpoint=checkpoint,
                            checkpoint_sha256=observed, note=args.note)
    print(f"LANE_INIT lane={args.lane} "
          f"lineage={state['lineage_sha256'][:16]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bank", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", parents=[common])
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--note", default="bank lane seed")

    p = sub.add_parser("collect", parents=[common])
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--cycle", type=int, required=True)
    p.add_argument("--args-json", type=Path, required=True)
    p.add_argument("--teacher", choices=("live", "fixture"), default="live")
    p.add_argument("--teacher-model", default=None,
                   help="override durable bank/correction-config.json")
    p.add_argument("--max-tasks", type=int, default=2)
    p.add_argument("--claim-ttl", type=float, default=3600.0)
    p.add_argument("--teacher-min-interval", type=float, default=10.0)
    p.add_argument("--preclaimed", default=None)

    p = sub.add_parser("run", parents=[common])
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--args-json", type=Path, required=True)
    p.add_argument("--teacher", choices=("live", "fixture"), default="live")
    p.add_argument("--teacher-model", default=None,
                   help="override durable bank/correction-config.json")
    p.add_argument("--max-tasks", type=int, default=2)
    p.add_argument("--claim-ttl", type=float, default=3600.0)
    p.add_argument("--teacher-min-interval", type=float, default=10.0)
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--lease-wait", type=int, default=900)
    p.add_argument("--lease-retry-seconds", type=float, default=30.0)
    p.add_argument("--collect-timeout", type=int, default=7200)
    p.add_argument("--pack-timeout", type=int, default=1800)
    p.add_argument("--train-timeout", type=int, default=3600)
    p.add_argument("--probe-timeout", type=int, default=1800)
    p.add_argument("--max-failures", type=int, default=3)
    p.add_argument("--max-seconds", type=int, default=4 * 3600)
    p.add_argument("--collector-only", action="store_true",
                   default=os.environ.get("COLLECTOR_ONLY", "0") == "1",
                   help="collect receipts at the init seed; never train or adopt merges")
    p.add_argument("--block-streams-root", type=Path,
                   default=os.environ.get("BLOCK_STREAMS_ROOT") or None)
    p.add_argument("--block-mode", choices=("off", "on"),
                   default=os.environ.get("BLOCK_MODE", "off"))
    p.add_argument("--block-min-targets", type=int,
                   default=int(os.environ.get("BLOCK_MIN_TARGETS", "65536")))
    p.add_argument("--block-max-targets", type=int,
                   default=int(os.environ.get("BLOCK_MAX_TARGETS", "131072")))
    p.add_argument("--block-max-wait-cycles", type=int,
                   default=int(os.environ.get("BLOCK_MAX_WAIT_CYCLES", "48")))
    # ---- v3 CORPUS-REPLAY ANCHOR (operator ruling 2026-09-29: RL shaping on
    #      the corpus checkpoint; the merge-5 collapse diagnosis — the gym must
    #      be a shaping layer on a corpus-dominant diet, never the whole meal).
    #      Every anchor-period-th cycle the lane's train step runs on a pack
    #      from the FROZEN admitted corpus (the E4 main prep admission),
    #      rotating through screened passing sampler keys; receipts are still
    #      collected, graded, teacher-corrected and stored as always. Default
    #      0 = OFF (byte-identical v1/v2 behavior).
    p.add_argument("--anchor-period", type=int, default=0,
                   help="train on a corpus pack every N-th cycle (0=off)")
    p.add_argument("--anchor-authority-root", type=Path, default=None)
    p.add_argument("--anchor-authority-sha256", default=None)
    p.add_argument("--anchor-pack-root", type=Path, default=None)
    p.add_argument("--anchor-pack-sha256", default=None)
    p.add_argument("--anchor-keys", default=None,
                   help="comma list of screened passing sampler keys")
    # ---- v2 policy-gradient channel (operator RL-regime ruling 2026-09-27):
    #      flag-gated per-lane train-step selection, default UNCHANGED.
    p.add_argument("--lane-train-step",
                   choices=("sft-receipts", "policy-gradient"),
                   default="sft-receipts")
    p.add_argument("--pg-lr", type=float, default=2e-6,
                   help="screened: operator-selection.json (largest sane lr)")
    p.add_argument("--pg-kl-beta", type=float, default=0.01)
    p.add_argument("--pg-kl-guard", type=float, default=0.056,
                   help="screened: S5 = 3x the selected combo's max realized KL")
    p.add_argument("--pg-window-cycles", type=int, default=16,
                   help="trajectory batch window (this lane's recent cycles)")
    p.add_argument("--pg-max-episodes", type=int, default=8,
                   help="recency cap on batch episodes. PRODUCTION-SCALE "
                        "RECALIBRATION (measured live 2026-09-27): 21-24-"
                        "episode batches at lr 2e-6 moved 0.17-0.99 realized "
                        "k3-KL/token and tripped the guard; the screened "
                        "geometry (8 episodes / 14 rows -> 0.018) is the "
                        "qualified batch scale, so the cap now matches it")
    p.add_argument("--pg-group-by", choices=("task", "family"),
                   default="family",
                   help="GRPO group key: sealed template family (one-shot "
                        "lakeexp instances make per-task groups singletons)")
    p.add_argument("--pg-min-group", type=int, default=2)
    p.add_argument("--pg-batch-timeout", type=int, default=1800)
    p.add_argument("--pg-train-timeout", type=int, default=3600)

    args = parser.parse_args()
    if args.command == "init":
        lane_init(args)
    elif args.command == "collect":
        bank_collect(args)
    elif args.command == "run":
        lane_run(args)


if __name__ == "__main__":
    main()
