#!/usr/bin/env python3
"""Typed, bounded first-party task validator invoked from bundle argv."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("focused", "regression"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    args = parser.parse_args()
    spec, terminal = json.loads(args.spec.read_text()), json.loads(args.terminal.read_text())
    actions = terminal.get("actions")
    if not isinstance(actions, list):
        raise SystemExit("terminal actions are required")
    allowed = spec["minefield"]["allowed_tools"]
    if any(item.get("tool_name") not in allowed for item in actions):
        raise SystemExit("minefield tool violation")
    if args.mode == "focused":
        final = next((item.get("content", "") for item in reversed(terminal.get("messages", []))
                      if item.get("role") == "assistant"), "")
        if terminal.get("status") != "success" or final != "Final: " + spec["expected_token"]:
            raise SystemExit("focused outcome failed")
        if not any(item.get("tool_name") == "read" and item.get("arguments", {}).get("path") == spec["required_read_path"] for item in actions):
            raise SystemExit("required read missing")
    else:
        if len(actions) > 12:
            raise SystemExit("regression action limit failed")
        for action in actions:
            path = str(action.get("arguments", {}).get("path", ""))
            if path.startswith("/") or ".." in path.split("/"):
                raise SystemExit("regression path containment failed")
    print(json.dumps({"mode": args.mode, "status": "pass", "action_count": len(actions)}, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
