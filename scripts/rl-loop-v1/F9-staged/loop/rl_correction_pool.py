#!/usr/bin/env python3
"""Bounded process-isolated correction I/O; the caller alone publishes receipts.

No unbounded queue: a full pool returns an explicit exhausted failure. Closing
fences the cycle before terminating workers, so late results cannot be admitted.
Each child owns its Pi process group and an immutable, durable specification.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class CorrectionPool:
    def __init__(self, width: int, directory: Path, *, timeout_s: float = 750):
        if not 1 <= width <= 12 or timeout_s <= 0:
            raise ValueError("invalid correction pool bounds")
        self.width = width
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.timeout_s = timeout_s
        self.jobs = {}
        self.closed = False
        self.sequence = 0

    def submit(self, spec: dict) -> tuple[int | None, dict | None]:
        if self.closed or len(self.jobs) >= self.width:
            return None, {"passed": False, "error": "pool closed" if self.closed else "pool exhausted"}
        sequence = self.sequence
        self.sequence += 1
        spec_path = self.directory / f"spec-{sequence:04d}.json"
        result_path = self.directory / f"result-{sequence:04d}.json"
        spec_path.write_text(json.dumps(spec, sort_keys=True) + "\n")
        spec_path.chmod(0o400)
        log = (self.directory / f"worker-{sequence:04d}.log").open("w")
        started = time.monotonic()
        try:
            proc = subprocess.Popen([sys.executable, __file__, "--spec", str(spec_path),
                "--result", str(result_path), "--owner-pid", str(os.getpid()),
                "--deadline", str(started + self.timeout_s)], stdout=log, stderr=log,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.jobs[sequence] = (proc, result_path, log, started)
        return sequence, None

    @staticmethod
    def _kill(proc):
        # Even an already-dead owner may have a live Pi descendant.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)

    def poll(self) -> list[tuple[int, dict]]:
        if self.closed:
            return []
        ready = []
        for sequence, (proc, path, log, started) in list(self.jobs.items()):
            running = proc.poll() is None
            timed_out = running and time.monotonic()-started >= self.timeout_s
            if running and not timed_out:
                continue
            self._kill(proc)
            log.close()
            del self.jobs[sequence]
            result = {"passed": False, "error": "correction timeout" if timed_out else
                      f"correction worker exit {proc.returncode}"}
            if not timed_out and proc.returncode == 0 and path.is_file():
                try:
                    result = json.loads(path.read_text())
                    if not isinstance(result, dict) or "passed" not in result:
                        raise ValueError("malformed worker result")
                except (ValueError, OSError) as exc:
                    result = {"passed": False, "error": f"invalid correction result: {exc}"}
            if result.get("error"):
                # Preserve completed API attempts even when the worker dies
                # before it can return its final metrics (never a receipt).
                spec = json.loads((self.directory / f"spec-{sequence:04d}.json").read_text())
                task_dir = spec.get("task_dir")
                metrics_path = Path(task_dir) / "correction-api.jsonl" if task_dir else None
                if metrics_path is not None and metrics_path.is_file():
                    calls = []
                    for line in metrics_path.read_text().splitlines():
                        try:
                            calls.append(json.loads(line))
                        except ValueError:
                            result["metrics_partial_tail"] = True
                    result["metrics"] = {"calls": calls}
            ready.append((sequence, result))
        return ready  # stable admission-order tie break among available results

    def close(self) -> list[int]:
        self.closed = True  # publication fence precedes process teardown
        cancelled = list(self.jobs)
        for proc, _, log, _ in self.jobs.values():
            self._kill(proc)
            log.close()
        self.jobs.clear()
        return cancelled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    # Linux owner-death fencing prevents orphan Pi jobs even on collect SIGKILL.
    import ctypes
    def terminate_group(signum, frame):
        # All descendants inherit this group; SIGKILL contains stuck tools too.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, terminate_group)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot bind correction parent death")
    if os.getppid() != args.owner_pid:
        terminate_group(signal.SIGTERM, None)
    signal.signal(signal.SIGALRM, terminate_group)
    remaining = args.deadline - time.monotonic()
    if remaining <= 0:
        terminate_group(signal.SIGALRM, None)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    from rl_bank_lane import execute_correction_spec
    spec = json.loads(args.spec.read_text())
    try:
        result = execute_correction_spec(spec)
    except BaseException as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    temp = args.result.with_suffix(".tmp")
    temp.write_text(json.dumps(result, sort_keys=True) + "\n")
    os.replace(temp, args.result)


if __name__ == "__main__":
    main()
