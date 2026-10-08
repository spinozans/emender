#!/usr/bin/env python3
"""F7 CPU-only fresh correction qualification. Never claims or writes a bank.

Freeze archived TRAIN failures routing fresh; each repeat gets a separate real
Pi process/workspace. Capacity denominators include startup, failures and drain.
The isolated pilot is byte-identical to the pre-F7 pilot, not deadline-hardened.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MODELS = ("glm-5.3-flash-background", "glm-5.3-background",
          "deepseek-4.1-flash-background")


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(args):
    import rl_bank_lane as lane
    driver = sys.modules["rl_loop_driver"]
    curriculum = driver.load_curriculum()
    tools = driver.tool_manifest(curriculum)["model_visible_tools"]
    by_template = {}
    seen = set()
    mix = {"fresh": 0, "spliced": 0}
    for summary in sorted(args.bank.glob("lanes/lane-*/episodes/cycle-*/collect-summary.json")):
        for outcome in json.loads(summary.read_text())["outcomes"]:
            if outcome.get("correction") is None:
                continue
            episode = summary.parent / outcome["task_id"] / "correction/episode-private.json"
            if episode.exists():
                record = json.loads(episode.read_text())
                mix["spliced" if record.get("prefix") is not None else "fresh"] += 1
            task_path = args.bank / "pool/done" / (outcome["task_id"] + ".json")
            attempt = summary.parent / outcome["task_id"] / "attempt/episode-private.json"
            if not task_path.exists() or not attempt.exists():
                continue
            task = json.loads(task_path.read_text())
            body = task["body"]
            identity = body.get("task_lake", {}).get("task_identity")
            if (body.get("split") != "train" or not body.get("receipt_eligible")
                    or not identity or identity in seen):
                continue
            policy = json.loads(attempt.read_text())
            if not lane._correction_requires_fresh_solve(body, policy, tools):
                continue
            seen.add(identity)
            by_template.setdefault(body["template"], []).append({
                "task": task, "attempt": policy, "attempt_sha256": digest(attempt),
                "task_source": str(task_path), "task_source_sha256": digest(task_path),
                "attempt_source": str(attempt)})
    chosen = []
    while len(chosen) < 30 and any(by_template.values()):
        for template in sorted(by_template):
            if by_template[template] and len(chosen) < 30:
                chosen.append(by_template[template].pop(0))
    if len(chosen) != 30:
        raise RuntimeError(f"only {len(chosen)} qualifying TRAIN fresh failures")
    args.root.mkdir(parents=True, exist_ok=False)
    pilot_source = driver.PILOT_PATH
    (args.root / "pilot-isolated.py").write_bytes(pilot_source.read_bytes())
    write(args.root / "plan.json", {
        "schema": "emender-f7-fresh-correction-plan-v1", "tasks": chosen,
        "models": MODELS, "preregistered": "DeepSeek receipt rate >= live flash rate - 0.05",
        "archived_mix": mix, "bank_read_only": str(args.bank),
        "pilot_sha256": digest(pilot_source), "driver_sha256": digest(driver.__file__),
        "frozen_unix": time.time(), "convergence_n": 30,
        "ladder": [4, 8, 16, 32], "repeats_per_rung": 64,
        "worker_timeout_s": 1200, "judge_model": MODELS[0]})
    print(json.dumps({"frozen": 30, "templates": sorted(by_template), "mix": mix}))


def worker(args):
    import tiktoken
    import rl_loop_driver as driver
    plan = json.loads((args.root / "plan.json").read_text())
    pilot_path = args.root / "pilot-isolated.py"
    if digest(pilot_path) != plan["pilot_sha256"]:
        raise ValueError("isolated pilot identity drift")
    spec = importlib.util.spec_from_file_location("f7_isolated_pilot", pilot_path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    item = plan["tasks"][args.index % len(plan["tasks"])]
    task, policy = item["task"], item["attempt"]
    body = task["body"]
    if body["split"] != "train" or not body["receipt_eligible"]:
        raise ValueError("TRAIN eligibility required")
    curriculum = driver.load_curriculum()
    manifest = driver.tool_manifest(curriculum)
    tools = manifest["model_visible_tools"]
    enc = tiktoken.get_encoding("p50k_base")
    class Metrics(pilot.Metrics):
        def record(self, entry):
            super().record(entry)
            with (args.output / "api.jsonl").open("a") as f:
                f.write(json.dumps({**entry, "ended_unix": time.time()}) + "\n")
    metrics = Metrics()
    start = time.monotonic()
    row = {"model": args.model, "index": args.index, "task_id": task["task_id"],
           "template": body["template"], "receipt": False, "grade_passed": False,
           "targets": 0, "started_unix": time.time(), "probe_source_sha256": digest(__file__)}
    try:
        workspace = driver.prepare_workspace(args.output / "workspace", body)
        panel = driver.make_panel(curriculum.SYSTEM, tools, {
            **driver.TEACHER_PANEL, "max_turns": int(body["task_lake"]["limits"]["turns"])})
        generate = driver.make_teacher_generate("live", args.model, tools, enc, pilot, metrics)
        record = driver.run_episode(
            episode_dir=args.output, workspace=workspace, prompt=body["prompt"], panel=panel,
            generate=generate, enc=enc, pi_bin=Path(manifest["pi_bin"]),
            manifest_path=driver.MANIFEST_PATH, pilot=pilot, curriculum=curriculum,
            seconds=panel["episode_seconds"], prefix_messages=None, prefix_policy_text=None)
        passed, grade = driver.grade_episode(
            body, workspace, pilot, task, record=record, episode_dir=args.output,
            stage="teacher", system=curriculum.SYSTEM, judge_metrics=metrics,
            judge_model=plan["judge_model"])
        write(args.output / "grade.json", grade)
        row.update(grade_passed=passed, status=record["status"],
                   close_verified=record.get("close_verified", False),
                   teacher_tokens=record.get("tokens", 0), reason=record.get("reason"))
        if passed and record["status"] == "finished" and record.get("close_verified"):
            # Exercise the real canonical encoding/receipt gates; never publish into a bank.
            receipt = driver._receipt_from_episode(
                cycle=0, task=task, kind="teacher-corrected",
                policy_checkpoint={"checkpoint_path": "archived-policy-not-loaded",
                    "checkpoint_sha256": json.loads((Path(item["attempt_source"]).parents[2]
                        / "collect-summary.json").read_text())["policy_checkpoint_sha256"]},
                record=record, supervise_from=0, grade=grade,
                teacher={"mode": "live", "model": args.model},
                attempt_link={"attempt_episode_sha256": item["attempt_sha256"]},
                enc=enc, prev=None)
            write(args.output / "candidate-receipt.json", receipt)
            row.update(receipt=True, targets=receipt["episode"]["targets"])
    except BaseException as exc:
        row.update(error=f"{type(exc).__name__}: {exc}")
    row.update(elapsed_s=time.monotonic()-start, ended_unix=time.time(),
               metrics=metrics.dump(), generator_state=getattr(locals().get("generate"), "state", {}))
    write(args.output / "result.json", row)


def run_case(root, phase, model, index):
    output = root / phase / model / f"case-{index:04d}"
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    start = time.monotonic()
    with (output / "worker.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, __file__, "worker", "--root", str(root),
            "--output", str(output), "--model", model, "--index", str(index)],
            stdout=log, stderr=log, env=env, start_new_session=True)
        try:
            proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    if not (output / "result.json").exists():
        write(output / "result.json", {"model": model, "index": index, "receipt": False,
              "grade_passed": False, "targets": 0, "error": f"worker exit {proc.returncode}",
              "elapsed_s": time.monotonic()-start})
    return json.loads((output / "result.json").read_text())


def run(args):
    models = MODELS if args.phase == "ab" else (MODELS[1], MODELS[2])
    rungs = [4] if args.phase == "ab" else [4, 8, 16, 32]
    if args.phase == "ladder":
        design = args.root / "ladder-design.json"
        if not design.exists():
            write(design, {"models": models, "rungs": rungs, "waves": 2,
                "approval": "supervisor approved two-wave bounded degradation probe",
                "approved_before_first_rung_unix": time.time(),
                "capacity_definition": "burst+drain, finite sample, NOT sustained production limit"})
    for width in rungs:
        count = 30 if args.phase == "ab" else 2*width
        for model in models:
            phase = f"{args.phase}-{width}"
            summary = args.root / phase / model / "cohort.json"
            if summary.exists():
                continue
            start = time.monotonic()
            started_unix = time.time()
            background = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                                        text=True, check=True).stdout
            with ThreadPoolExecutor(max_workers=width) as pool:
                rows = list(pool.map(lambda i: run_case(args.root, phase, model, i), range(count)))
            elapsed = time.monotonic()-start
            write(summary, {"model": model, "concurrency": width, "n": count,
                "wall_s": elapsed, "started_unix": started_unix, "ended_unix": time.time(),
                "background_bank_processes_at_start": [line for line in background.splitlines()
                    if "bank-singlelearner-B" in line and "rl_bank_lane.py collect" in line],
                "capacity_definition": "burst+drain, finite sample, NOT sustained production limit",
                "completed_per_hour": count*3600/elapsed,
                "receipts_per_hour": sum(r["receipt"] for r in rows)*3600/elapsed,
                "targets_per_hour": sum(r["targets"] for r in rows)*3600/elapsed,
                "rows": rows})
            print(json.dumps({"phase": phase, "model": model, "wall_s": elapsed,
                              "receipts": sum(r["receipt"] for r in rows)}), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--phase", choices=("ab", "ladder"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
