#!/usr/bin/env python3
"""Lane-private durable bounded correction spool; only the lane owner publishes.

Admission owns copied episode/workspace bytes and the original task claim.
A service restart fails interrupted executions explicitly (never reuses a
possibly mutated splice). Finished results and publication intents are replayable.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from ndm.e97_atomic import durable_directory, publish_bytes_no_replace
from rl_correction_pool import CorrectionPool


def immutable(path: Path, value: dict) -> None:
    publish_bytes_no_replace(Path(path), (json.dumps(value, sort_keys=True)+"\n").encode(), mode=0o400)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def process_birth(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


class CorrectionService:
    def __init__(self, bank: Path, lane: int):
        self.bank = Path(bank)
        self.lane = int(lane)
        from rl_bank import bank_paths, lane_paths
        self.root = lane_paths(bank_paths(self.bank), self.lane)["root"] / "corrections"
        durable_directory(self.root)

    def jobs(self) -> list[Path]:
        return sorted((p for p in self.root.glob("job-*") if (p / "spec.json").is_file()),
                      key=lambda p: read(p / "spec.json")["admitted_unix"])

    def active(self) -> list[Path]:
        return [p for p in self.jobs() if not (p / "published.json").exists()]

    def submit(self, spec: dict, *, capacity: int, min_interval: float) -> tuple[str | None, dict | None]:
        # Exactly one collect owner admits; service never changes membership.
        if len(self.active()) >= capacity:
            return None, {"passed": False, "error": "pool exhausted"}
        job_id = "job-" + uuid.uuid4().hex
        staging = self.root / f".job-{os.getpid()}-{process_birth(os.getpid())}-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        job = self.root / job_id
        try:
            shutil.copytree(spec["task_dir"], staging / "episode", symlinks=False)
            owned = json.loads(json.dumps(spec))
            owned.update(task_dir=str(job / "episode"), workspace=str(job / "episode" / "attempt" / "workspace"),
                         admitted_unix=time.time(), min_interval_s=float(min_interval), job_id=job_id)
            # Persist every copied file and directory before the admission marker.
            for directory, _, files in os.walk(staging, topdown=False):
                for name in files:
                    with open(Path(directory) / name, "rb") as handle:
                        os.fsync(handle.fileno())
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            owned["owned_files_sha256"] = {
                str(p.relative_to(staging / "episode")): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (staging / "episode").rglob("*") if p.is_file()}
            immutable(staging / "spec.json", owned)
            os.rename(staging, job)
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return job_id, None

    def ensure_running(self, width: int) -> None:
        # Probe the lock without waiting; avoid spawning on every idle poll.
        with (self.root / "service.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        # Competing starts still lose the service lock and exit.
        log = (self.root / "service.log").open("a")
        try:
            subprocess.Popen([sys.executable, __file__, "--bank", str(self.bank),
                "--lane", str(self.lane), "--width", str(width)], stdout=log, stderr=log,
                start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        finally:
            log.close()

    def metrics(self, *, generation_attempts: int = 0, generation_wall_s: float = 0) -> dict:
        jobs = self.jobs()
        active = self.active()
        now = time.time()
        waits = {}
        starts = finishes = closed = receipts = targets = cycle_finishes = cycle_admissions = 0
        admitted = [read(p / "spec.json")["admitted_unix"] for p in jobs]
        for p in jobs:
            spec = read(p / "spec.json")
            if (p / "execution.json").exists():
                starts += 1
                waits[p.name] = max(0, read(p / "execution.json")["started_unix"] - spec["admitted_unix"])
            else:
                waits[p.name] = max(0, now-spec["admitted_unix"])
            if generation_wall_s > 0 and spec["admitted_unix"] >= now-generation_wall_s:
                cycle_admissions += 1
            if (p / "result.json").exists():
                finishes += 1
                result = read(p / "result.json")
                record = result.get("teacher_record") or {}
                closed += record.get("status") == "finished" and bool(record.get("close_verified"))
                if generation_wall_s > 0 and result.get("finished_unix", 0) >= now-generation_wall_s:
                    cycle_finishes += 1
            if (p / "published.json").exists():
                outcome = read(p / "published.json")["outcome"]
                receipts += bool(outcome.get("receipt"))
                targets += outcome.get("targets", 0)
        wall = max(0.001, now-min(admitted)) if admitted else 0
        return {"queue_depth": sum(not (p / "execution.json").exists() for p in active),
                "in_flight": sum((p / "execution.json").exists() and not (p / "result.json").exists() for p in active),
                "ready_to_publish": sum((p / "result.json").exists() for p in active),
                "backlog_depth": len(active), "admitted_total": len(jobs), "started_total": starts,
                "finalized_total": finishes, "native_closed_total": closed,
                "receipts_published_total": receipts, "targets_published_total": targets,
                "wait_s_by_correction": waits, "observation_unix": now,
                "correction_observation_wall_s": wall, "generation_observation_wall_s": generation_wall_s,
                "cycle_correction_admissions_per_h": cycle_admissions*3600/generation_wall_s if generation_wall_s > 0 else 0,
                "cycle_correction_outcomes_per_h": cycle_finishes*3600/generation_wall_s if generation_wall_s > 0 else 0,
                "cycle_backlog_growth_per_h": (cycle_admissions-cycle_finishes)*3600/generation_wall_s if generation_wall_s > 0 else 0,
                "generation_attempts_per_h": generation_attempts*3600/generation_wall_s if generation_wall_s > 0 else 0,
                "correction_arrivals_per_h": len(jobs)*3600/wall if wall else 0,
                "correction_outcomes_per_h": finishes*3600/wall if wall else 0,
                "backlog_growth_per_h": (len(jobs)-finishes)*3600/wall if wall else 0,
                "oldest_pending_age_s": max((now-read(p / "spec.json")["admitted_unix"] for p in active), default=0)}

    def publish_ready(self, build, apply) -> list[dict]:
        published = []
        for job in self.active():
            if not (job / "result.json").exists():
                continue
            spec = read(job / "spec.json")
            intent = job / "publication.json"
            if not intent.exists():
                immutable(intent, build(spec, read(job / "result.json")))
            plan = read(intent)
            apply(spec, plan)
            immutable(job / "published.json", {"published_unix": time.time(), "outcome": plan["outcome"]})
            published.append({**plan, "job_id": job.name})
        return published

    def run(self, width: int) -> None:
        lock = (self.root / "service.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return
        # Staging directories have no admission marker and no owned claim.
        for staging in self.root.glob(".job-*"):
            _, pid, birth, _ = staging.name.split("-", 3)
            if process_birth(int(pid)) != birth:
                shutil.rmtree(staging)
        pool = CorrectionPool(width, self.root / ("run-" + uuid.uuid4().hex))
        pending = {}
        # Parent-death containment kills the old worker/Pi groups. Do not retry
        # an interrupted splice; publish its explicit failure after owner resume.
        for job in self.active():
            if (job / "execution.json").exists() and not (job / "result.json").exists():
                result = {"passed": False, "error": "correction service lost execution claim"}
                metrics = Path(read(job / "spec.json")["task_dir"]) / "correction-api.jsonl"
                if metrics.exists():
                    calls = []
                    for line in metrics.read_text().splitlines():
                        try:
                            calls.append(json.loads(line))
                        except ValueError:
                            result["metrics_partial_tail"] = True
                    result["metrics"] = {"calls": calls}
                immutable(job / "result.json", {**result, "finished_unix": time.time()})
        last_start = max((read(p / "execution.json")["started_unix"] for p in self.jobs()
                          if (p / "execution.json").exists()), default=0)
        last_start_monotonic = time.monotonic() - max(0, time.time()-last_start)
        try:
            while True:
                from rl_correction_config import read_correction_config
                pool.width = read_correction_config(self.bank)["pool_width"]
                width = pool.width
                for sequence, result in pool.poll():
                    job = pending.pop(sequence)
                    immutable(job / "result.json", {**result, "finished_unix": time.time()})
                for job in self.active():
                    if len(pending) >= width:
                        break
                    if (job / "execution.json").exists() or (job / "result.json").exists():
                        continue
                    spec = read(job / "spec.json")
                    now = time.time()
                    if time.monotonic()-last_start_monotonic < spec["min_interval_s"]:
                        break
                    # Durable claim precedes launch: crash in this gap fails closed.
                    immutable(job / "execution.json", {"started_unix": now, "deadline_unix": now+pool.timeout_s})
                    last_start = now
                    last_start_monotonic = time.monotonic()
                    sequence, failure = pool.submit(spec)
                    if failure:
                        immutable(job / "result.json", {**failure, "finished_unix": time.time()})
                    else:
                        pending[sequence] = job
                time.sleep(0.1)
        finally:
            pool.close()
            lock.close()


def claim_has_correction(bank: Path, task: dict) -> bool:
    claim = task.get("claim") or {}
    if "lane" not in claim:
        return False
    from rl_bank import bank_paths, lane_paths
    root = lane_paths(bank_paths(bank), claim["lane"])["root"] / "corrections"
    if not root.is_dir():
        return False
    service = CorrectionService(bank, claim["lane"])
    for job in service.active():
        original = read(job / "spec.json")["task"]
        if (original["task_id"] == task["task_id"] and original.get("task_sha256") == task.get("task_sha256")
                and (original.get("claim") or {}).get("claimed_unix") == claim.get("claimed_unix")):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()
    CorrectionService(args.bank, args.lane).run(args.width)


if __name__ == "__main__":
    main()
