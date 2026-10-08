"""Durable correction settings, sampled at cycle boundaries (never env-only)."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MODEL = "glm-5.3-flash-background"
DEFAULT_WIDTH = 1  # Eight B lanes, twelve global background slots; leave four for other callers.


def read_correction_config(bank: Path) -> dict:
    path = Path(bank) / "correction-config.json"
    config = {"teacher_model": DEFAULT_MODEL, "async_enabled": False,
              "pool_width": DEFAULT_WIDTH}
    if path.exists():
        supplied = json.loads(path.read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(config):
            raise ValueError("invalid correction config keys")
        config.update(supplied)
    if not isinstance(config["teacher_model"], str) or not config["teacher_model"].strip():
        raise ValueError("teacher_model must be a nonempty model id")
    if type(config["async_enabled"]) is not bool:
        raise ValueError("async_enabled must be boolean")
    if type(config["pool_width"]) is not int or not 1 <= config["pool_width"] <= 12:
        raise ValueError("pool_width must be an integer in 1..12")
    return config


def resolve_correction_config(args) -> dict:
    config = read_correction_config(args.bank)
    if getattr(args, "teacher_model", None) is not None:
        config["teacher_model"] = args.teacher_model
    return config
