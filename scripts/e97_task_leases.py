#!/usr/bin/env python3
"""Typed command line for the local E97 atomic task lease index."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ndm.e97_task_leases import Lease, LeaseError, TaskLeases, validate_lease


def _lease(value: str) -> Lease:
    raw = json.loads(value)
    required = {"task_id", "owner", "attempt", "deadline_ns", "identity"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise LeaseError("lease JSON must contain exactly the sealed lease fields")
    return validate_lease(Lease(
        task_id=raw["task_id"], owner=raw["owner"], attempt=raw["attempt"],
        deadline_ns=raw["deadline_ns"], identity=raw["identity"],
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("--tasks-jsonl", type=Path, required=True)
    claim = commands.add_parser("claim")
    claim.add_argument("--owner", required=True)
    claim.add_argument("--lease-seconds", type=float, required=True)
    beat = commands.add_parser("heartbeat")
    beat.add_argument("--lease-json", required=True)
    beat.add_argument("--lease-seconds", type=float, required=True)
    complete = commands.add_parser("complete")
    complete.add_argument("--lease-json", required=True)
    complete.add_argument("--receipt-json", type=Path, required=True)
    commands.add_parser("status")
    args = parser.parse_args()
    leases = TaskLeases(args.root)
    try:
        if args.command == "initialize":
            leases.initialize([json.loads(line) for line in args.tasks_jsonl.read_text().splitlines() if line.strip()])
            result = {"status": "initialized", "tasks": len(leases.status())}
        elif args.command == "claim":
            item = leases.claim(args.owner, lease_seconds=args.lease_seconds)
            result = None if item is None else item.__dict__
        elif args.command == "heartbeat":
            result = leases.heartbeat(_lease(args.lease_json), lease_seconds=args.lease_seconds).__dict__
        elif args.command == "complete":
            result = {"receipt_sha256": leases.complete(_lease(args.lease_json), json.loads(args.receipt_json.read_text()))}
        else:
            result = leases.status()
        print(json.dumps(result, sort_keys=True))
    except (LeaseError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        leases.close()

if __name__ == "__main__":
    main()
