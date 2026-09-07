"""Synthetic sealed-panel fixtures with the production metadata/record schemas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ndm.e97_protected_overlap import PI_V3_SCHEMA, PI_V4_SCHEMA, REAL_REPO_SCHEMA


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _pi_record(version: str, *, prompt: str) -> dict[str, Any]:
    return {
        "id": f"synthetic-{version}-opaque-read-00000000",
        "source": f"synthetic-{version}",
        "split": 1,
        "kind": "synthetic-opaque-read",
        "user": prompt,
        "task": {
            "fixtures": [
                {"path": "synthetic/value-000001.json", "content": '{"release":"synthetic-token-12345","value":42}\n'},
                {"path": "synthetic/pointer.txt", "content": "target=synthetic/value-000001.json\n"},
            ],
            "expected_calls": [],
            "postconditions": [],
            "final_contains": ["synthetic/value-000001.json", "synthetic-token-12345", "42"],
        },
    }


def _write_pi_panel(root: Path, *, schema: str, version: str, prompt: str) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    records = root / "records.jsonl"
    row = _pi_record(version, prompt=prompt)
    _write_jsonl(records, [row])
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": schema,
        "status": "complete",
        "purpose": "synthetic sealed overlap adapter test",
        "records": 1,
        "seed": 1,
        "kinds": [row["kind"]],
        "kind_counts": {row["kind"]: 1},
        "training_exclusion": "synthetic test only",
        "outputs": {"metadata": _descriptor(records)},
    }, sort_keys=True))
    return manifest, records


def _write_real_repo_panel(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    records = root / "tasks.jsonl"
    row = {
        "id": "synthetic-markupsafe-repair",
        "split": 1,
        "repository": "markupsafe",
        "url": "https://example.invalid/markupsafe.git",
        "commit": "a" * 40,
        "path": "src/markupsafe/example.py",
        "clean": "return decoded_value",
        "mutated": "return raw_value",
        "focused_test": "python -m pytest -q tests/test_example.py",
        "setup_files": {"src/markupsafe/_version.py": '__version__ = "synthetic"\n'},
        "prompt": "Synthetic repository repair prompt with a bounded focused test.",
        "expected_patch": {
            "path": "src/markupsafe/example.py",
            "oldText": "return raw_value",
            "newText": "return decoded_value",
        },
    }
    _write_jsonl(records, [row])
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": REAL_REPO_SCHEMA,
        "status": "complete",
        "purpose": "synthetic sealed overlap adapter test",
        "tasks": 1,
        "repositories": {"markupsafe": {"commit": "a" * 40}},
        "training_exclusion": "synthetic test only",
        "outputs": {"tasks": _descriptor(records)},
    }, sort_keys=True))
    return manifest, records


def write_synthetic_real_schema_panels(root: Path, *, v3_prompt: str = "Synthetic protected V3 prompt.") -> list[tuple[Path, Path]]:
    """Return one V3, V4, and real-repository panel with no protected data."""

    return [
        _write_pi_panel(root / "v3", schema=PI_V3_SCHEMA, version="v3", prompt=v3_prompt),
        _write_pi_panel(root / "v4", schema=PI_V4_SCHEMA, version="v4", prompt="Synthetic protected V4 prompt."),
        _write_real_repo_panel(root / "real-repository"),
    ]
