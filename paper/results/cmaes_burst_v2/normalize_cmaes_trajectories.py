#!/usr/bin/env python3
"""Normalize inventoried CMA-ES stdout trajectories for the v2 burst figure.

The script intentionally uses only the Python standard library so it can be
rerun from a fresh checkout without installing plotting or dataframe packages.
It reads raw logs in place and writes derived artifacts under this directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1"
DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_INVENTORY = DEFAULT_DIR / "cmaes_log_manifest.json"
DEFAULT_OUT_DIR = DEFAULT_DIR

TRAJECTORY_CSV = "cmaes_trajectory_points.csv.gz"
EVAL_CSV = "cmaes_eval_summary.csv"
GENERATION_CSV = "cmaes_generation_summary.csv"
NORMALIZATION_MANIFEST = "cmaes_normalization_manifest.json"

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(
    rf"^step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>{FLOAT})(?P<rest>.*)$"
)
FIELD_PATTERNS = {
    "lr": re.compile(rf"(?:^|\s)lr\s+(?P<value>{FLOAT})"),
    "grad": re.compile(rf"(?:^|\s)grad\s+(?P<value>{FLOAT})"),
    "tokens_per_second": re.compile(rf"(?:^|\s)tok/s\s+(?P<value>{FLOAT})"),
    "stdout_elapsed_h": re.compile(rf"(?:^|\s)elapsed_h\s+(?P<value>{FLOAT})"),
    "wallclock": re.compile(r"(?:^|\s)time\s+(?P<value>\S+)"),
}
TRAIN_MINUTES_RE = re.compile(r"Time-based training:\s+(?P<minutes>[0-9.]+)\s+min")
OUTPUT_DIR_PREFIX = "Output directory:"

TRAJECTORY_FIELDS = [
    "architecture_family",
    "architecture_model",
    "source_inventory_group",
    "sweep_id",
    "run_id",
    "run_status",
    "trial_id",
    "config_id",
    "seed",
    "wallclock_timestamp_utc",
    "sweep_elapsed_seconds",
    "sweep_elapsed_minutes",
    "candidate_elapsed_seconds",
    "candidate_elapsed_minutes",
    "stdout_elapsed_h",
    "stdout_elapsed_minutes",
    "step",
    "iteration",
    "loss",
    "metric_name",
    "metric_unit",
    "learning_rate",
    "grad_norm",
    "tokens_per_second",
    "cma_loss",
    "final_loss",
    "actual_params",
    "batch_size",
    "target_batch_size",
    "train_minutes",
    "chunk_size",
    "tokenizer",
    "generation",
    "warm_start_index",
    "generation_wallclock_utc",
    "generation_elapsed_minutes",
    "generation_eval_counter",
    "is_results_best_loss",
    "is_results_best_final_loss",
    "is_observed_best_loss",
    "is_observed_best_final_loss",
    "selected_best_flag_source",
    "source_log_path",
    "source_eval_dir",
    "params_json_path",
    "done_json_path",
    "args_json_path",
]

EVAL_FIELDS = [
    "architecture_family",
    "architecture_model",
    "source_inventory_group",
    "sweep_id",
    "run_id",
    "run_status",
    "trial_id",
    "config_id",
    "seed",
    "trajectory_rows",
    "first_wallclock_timestamp_utc",
    "last_wallclock_timestamp_utc",
    "wallclock_span_minutes",
    "first_step",
    "last_step",
    "final_step_loss",
    "min_step_loss",
    "cma_loss",
    "final_loss",
    "actual_params",
    "batch_size",
    "target_batch_size",
    "train_minutes",
    "chunk_size",
    "tokenizer",
    "generation",
    "warm_start_index",
    "generation_wallclock_utc",
    "generation_elapsed_minutes",
    "is_results_best_loss",
    "is_results_best_final_loss",
    "is_observed_best_loss",
    "is_observed_best_final_loss",
    "selected_best_flag_source",
    "source_log_path",
    "source_eval_dir",
    "params_json_path",
    "done_json_path",
    "args_json_path",
]

GENERATION_FIELDS = [
    "architecture_family",
    "architecture_model",
    "source_inventory_group",
    "sweep_id",
    "run_id",
    "run_status",
    "generation",
    "warm_start_index",
    "generation_wallclock_utc",
    "generation_elapsed_minutes",
    "popsize",
    "n_valid_this_gen",
    "total_generated_this_gen",
    "eval_counter",
    "eval_id_start",
    "eval_id_end",
    "gen_best_loss",
    "best_loss_so_far",
    "generations_without_improvement",
    "sigma",
    "source_log_path",
]


@dataclass(frozen=True)
class RunSpec:
    source_inventory_group: str
    sweep_id: str
    family: str
    model: str
    status: str
    run_dir: Path
    top_log: str | None
    results_json: Path | None
    generations_jsonl: Path | None
    inventory_counts: dict[str, Any]


@dataclass
class EvalParse:
    eval_id: int
    eval_dir: Path
    stdout_path: Path
    params_json_path: Path
    done_json_path: Path
    args_json_path: Path | None
    params_record: dict[str, Any] | None
    done_record: dict[str, Any] | None
    args_record: dict[str, Any] | None
    train_minutes_from_stdout: float | None
    output_dir: str | None
    points: list[dict[str, Any]]
    exclusions: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize CMA-ES per-eval stdout loss trajectories."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="Path to cmaes_log_manifest.json from inventory-cma-es.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for derived CSV/JSON outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return value


def iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def eval_sort_key(path: Path) -> tuple[int, str]:
    eval_id = parse_eval_id(path)
    return (eval_id if eval_id is not None else 10**12, path.name)


def parse_eval_id(path: Path) -> int | None:
    if not path.name.startswith("eval_"):
        return None
    suffix = path.name.split("_", 1)[1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def read_inventory_runs(inventory: dict[str, Any]) -> list[RunSpec]:
    runs: list[RunSpec] = []

    for raw in inventory.get("fresh_corrected_1300m", {}).get("runs", []):
        runs.append(raw_run_to_spec(raw, "fresh_corrected_1300m"))

    for raw in inventory.get("runs", []):
        runs.append(raw_run_to_spec(raw, "primary_or_failed_v2"))

    seen: set[str] = set()
    unique: list[RunSpec] = []
    for run in runs:
        if run.sweep_id in seen:
            continue
        seen.add(run.sweep_id)
        unique.append(run)
    return unique


def raw_run_to_spec(raw: dict[str, Any], group: str) -> RunSpec:
    results = raw.get("results_json")
    generations = raw.get("generations_jsonl")
    return RunSpec(
        source_inventory_group=group,
        sweep_id=str(raw["id"]),
        family=str(raw.get("family") or raw.get("model") or raw["id"]),
        model=str(raw.get("model") or ""),
        status=str(raw.get("status") or ""),
        run_dir=Path(raw["run_dir"]),
        top_log=raw.get("top_log"),
        results_json=Path(results) if results else None,
        generations_jsonl=Path(generations) if generations else None,
        inventory_counts=dict(raw.get("counts") or {}),
    )


def finite_metric(records: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for record in records:
        eval_id = as_int(record.get("eval_id"))
        value = as_float(record.get(key))
        if eval_id is not None and value is not None:
            out.append((eval_id, value))
    return out


def min_eval_id(records: list[dict[str, Any]], key: str) -> int | None:
    metrics = finite_metric(records, key)
    if not metrics:
        return None
    return min(metrics, key=lambda item: (item[1], item[0]))[0]


def load_results_by_eval(path: Path | None) -> tuple[dict[int, dict[str, Any]], int | None, int | None]:
    if not path or not path.exists():
        return {}, None, None
    data = load_json(path)
    all_results = list(data.get("all_results") or [])
    by_eval: dict[int, dict[str, Any]] = {}
    for record in all_results:
        eval_id = as_int(record.get("eval_id"))
        if eval_id is not None:
            by_eval[eval_id] = dict(record)
    return by_eval, min_eval_id(all_results, "loss"), min_eval_id(all_results, "final_loss")


def parse_generations(
    run: RunSpec,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not run.generations_jsonl or not run.generations_jsonl.exists():
        return {}, [], ["missing_generations_jsonl"]

    generation_rows: list[dict[str, Any]] = []
    generation_by_eval: dict[int, dict[str, Any]] = {}
    exclusions: list[str] = []
    first_wallclock: datetime | None = None
    previous_eval_counter = 0

    with run.generations_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                exclusions.append(f"generation_json_decode_error_line_{line_number}")
                continue

            wallclock = parse_time(raw.get("wallclock_utc"))
            if first_wallclock is None and wallclock is not None:
                first_wallclock = wallclock
            elapsed_minutes = None
            if first_wallclock is not None and wallclock is not None:
                elapsed_minutes = (wallclock - first_wallclock).total_seconds() / 60.0

            eval_counter = as_int(raw.get("eval_counter"))
            n_valid = as_int(raw.get("n_valid_this_gen"))
            if n_valid is None:
                n_valid = len(raw.get("gen_fitnesses") or [])
            if eval_counter is None:
                eval_id_start = previous_eval_counter
                eval_id_end = previous_eval_counter + n_valid - 1 if n_valid else previous_eval_counter - 1
            else:
                eval_id_end = eval_counter - 1
                eval_id_start = max(previous_eval_counter, eval_counter - n_valid)
                previous_eval_counter = eval_counter

            normalized = {
                "architecture_family": run.family,
                "architecture_model": run.model,
                "source_inventory_group": run.source_inventory_group,
                "sweep_id": run.sweep_id,
                "run_id": run.sweep_id,
                "run_status": run.status,
                "generation": as_int(raw.get("gen")),
                "warm_start_index": as_int(raw.get("ws_idx")),
                "generation_wallclock_utc": iso_utc(wallclock),
                "generation_elapsed_minutes": elapsed_minutes,
                "popsize": as_int(raw.get("popsize")),
                "n_valid_this_gen": n_valid,
                "total_generated_this_gen": as_int(raw.get("total_generated_this_gen")),
                "eval_counter": eval_counter,
                "eval_id_start": eval_id_start,
                "eval_id_end": eval_id_end,
                "gen_best_loss": as_float(raw.get("gen_best_loss")),
                "best_loss_so_far": as_float(raw.get("best_loss_so_far")),
                "generations_without_improvement": as_int(raw.get("generations_without_improvement")),
                "sigma": as_float(raw.get("sigma")),
                "source_log_path": str(run.generations_jsonl),
            }
            generation_rows.append(normalized)

            if eval_id_start <= eval_id_end:
                for eval_id in range(eval_id_start, eval_id_end + 1):
                    generation_by_eval[eval_id] = normalized

    return generation_by_eval, generation_rows, exclusions


def extract_stdout_metadata(stdout_path: Path) -> tuple[str | None, float | None]:
    output_dir: str | None = None
    train_minutes: float | None = None
    with stdout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(OUTPUT_DIR_PREFIX):
                output_dir = stripped.split(OUTPUT_DIR_PREFIX, 1)[1].strip()
            match = TRAIN_MINUTES_RE.search(stripped)
            if match:
                train_minutes = as_float(match.group("minutes"))
            if output_dir is not None and train_minutes is not None:
                break
    return output_dir, train_minutes


def find_args_json(eval_dir: Path, output_dir: str | None) -> Path | None:
    args_paths = sorted(eval_dir.glob("*/args.json"), key=lambda path: str(path))
    if not args_paths:
        return None

    if output_dir:
        output_path = Path(output_dir)
        if output_path.is_absolute():
            direct = output_path / "args.json"
            if direct.exists():
                return direct

        output_basename = output_path.name
        for args_path in args_paths:
            if args_path.parent.name == output_basename:
                return args_path

    if len(args_paths) == 1:
        return args_paths[0]
    return None


def parse_step_line(line: str) -> dict[str, Any] | None:
    match = STEP_RE.match(line.strip())
    if not match:
        return None

    rest = match.group("rest")
    fields: dict[str, Any] = {
        "step": int(match.group("step")),
        "loss": float(match.group("loss")),
        "learning_rate": None,
        "grad_norm": None,
        "tokens_per_second": None,
        "stdout_elapsed_h": None,
        "wallclock_dt": None,
    }

    for key, pattern in FIELD_PATTERNS.items():
        field_match = pattern.search(rest)
        if not field_match:
            continue
        value = field_match.group("value")
        if key == "wallclock":
            fields["wallclock_dt"] = parse_time(value)
        elif key == "lr":
            fields["learning_rate"] = as_float(value)
        elif key == "grad":
            fields["grad_norm"] = as_float(value)
        else:
            fields[key] = as_float(value)
    return fields


def parse_eval(eval_dir: Path) -> EvalParse:
    eval_id = parse_eval_id(eval_dir)
    if eval_id is None:
        raise ValueError(f"unexpected eval directory name: {eval_dir}")

    stdout_path = eval_dir / "stdout.txt"
    params_json_path = eval_dir / "params.json"
    done_json_path = eval_dir / ".done"
    exclusions: list[str] = []
    points: list[dict[str, Any]] = []
    output_dir: str | None = None
    train_minutes_from_stdout: float | None = None
    args_json_path: Path | None = None
    args_record: dict[str, Any] | None = None

    params_record = load_json_if_exists(params_json_path)
    if params_record is None:
        exclusions.append("missing_params_json")

    done_record = load_json_if_exists(done_json_path)
    if done_record is None:
        exclusions.append("missing_done_json")

    if not stdout_path.exists():
        exclusions.append("missing_stdout_txt")
        return EvalParse(
            eval_id=eval_id,
            eval_dir=eval_dir,
            stdout_path=stdout_path,
            params_json_path=params_json_path,
            done_json_path=done_json_path,
            args_json_path=None,
            params_record=params_record,
            done_record=done_record,
            args_record=None,
            train_minutes_from_stdout=None,
            output_dir=None,
            points=[],
            exclusions=exclusions,
        )

    output_dir, train_minutes_from_stdout = extract_stdout_metadata(stdout_path)
    args_json_path = find_args_json(eval_dir, output_dir)
    if args_json_path is not None:
        args_record = load_json_if_exists(args_json_path)
    else:
        exclusions.append("missing_or_ambiguous_args_json")

    with stdout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            point = parse_step_line(line)
            if point is None:
                continue
            point["line_number"] = line_number
            points.append(point)

    if not points:
        exclusions.append("no_step_loss_lines")

    return EvalParse(
        eval_id=eval_id,
        eval_dir=eval_dir,
        stdout_path=stdout_path,
        params_json_path=params_json_path,
        done_json_path=done_json_path,
        args_json_path=args_json_path,
        params_record=params_record,
        done_record=done_record,
        args_record=args_record,
        train_minutes_from_stdout=train_minutes_from_stdout,
        output_dir=output_dir,
        points=points,
        exclusions=exclusions,
    )


def metadata_for_eval(
    parsed: EvalParse,
    result_record: dict[str, Any] | None,
    results_best_loss_eval_id: int | None,
    results_best_final_loss_eval_id: int | None,
    observed_best_loss_eval_id: int | None,
    observed_best_final_loss_eval_id: int | None,
) -> dict[str, Any]:
    done = parsed.done_record or {}
    params_record = parsed.params_record or {}
    args = parsed.args_record or {}
    result = result_record or {}

    cma_loss = as_float(done.get("loss"))
    if cma_loss is None:
        cma_loss = as_float(result.get("loss"))

    final_loss = as_float(done.get("final_loss"))
    if final_loss is None:
        final_loss = as_float(result.get("final_loss"))

    actual_params = as_int(done.get("actual_params"))
    if actual_params is None:
        actual_params = as_int(result.get("actual_params"))

    batch_size = as_int(done.get("batch_size"))
    if batch_size is None:
        batch_size = as_int(result.get("batch_size"))

    target_batch_size = as_int(done.get("target_batch_size"))
    if target_batch_size is None:
        target_batch_size = as_int(result.get("target_batch_size"))

    train_minutes = as_float(args.get("train_minutes"))
    if train_minutes is None:
        train_minutes = parsed.train_minutes_from_stdout

    chunk_size = as_int(args.get("chunk_size"))
    tokenizer = args.get("tokenizer")
    seed = as_int(args.get("seed"))
    if seed is None:
        seed = as_int(params_record.get("seed"))

    is_results_best_loss = (
        results_best_loss_eval_id is not None and parsed.eval_id == results_best_loss_eval_id
    )
    is_results_best_final_loss = (
        results_best_final_loss_eval_id is not None
        and parsed.eval_id == results_best_final_loss_eval_id
    )
    is_observed_best_loss = (
        observed_best_loss_eval_id is not None and parsed.eval_id == observed_best_loss_eval_id
    )
    is_observed_best_final_loss = (
        observed_best_final_loss_eval_id is not None
        and parsed.eval_id == observed_best_final_loss_eval_id
    )

    flag_sources: list[str] = []
    if is_results_best_loss:
        flag_sources.append("results_json_best_loss")
    if is_results_best_final_loss:
        flag_sources.append("results_json_best_final_loss")
    if is_observed_best_loss and not is_results_best_loss:
        flag_sources.append("observed_done_min_loss")
    if is_observed_best_final_loss and not is_results_best_final_loss:
        flag_sources.append("observed_done_min_final_loss")

    return {
        "seed": seed,
        "cma_loss": cma_loss,
        "final_loss": final_loss,
        "actual_params": actual_params,
        "batch_size": batch_size,
        "target_batch_size": target_batch_size,
        "train_minutes": train_minutes,
        "chunk_size": chunk_size,
        "tokenizer": tokenizer,
        "is_results_best_loss": is_results_best_loss,
        "is_results_best_final_loss": is_results_best_final_loss,
        "is_observed_best_loss": is_observed_best_loss,
        "is_observed_best_final_loss": is_observed_best_final_loss,
        "selected_best_flag_source": ";".join(flag_sources),
    }


def done_records_for_observed_bests(parsed_evals: list[EvalParse]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for parsed in parsed_evals:
        if parsed.done_record:
            record = dict(parsed.done_record)
            record.setdefault("eval_id", parsed.eval_id)
            records.append(record)
    return records


def build_rows_for_run(
    run: RunSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_summary: dict[str, Any] = {
        "sweep_id": run.sweep_id,
        "family": run.family,
        "model": run.model,
        "status": run.status,
        "run_dir": str(run.run_dir),
        "top_log": run.top_log,
        "results_json": str(run.results_json) if run.results_json else None,
        "generations_jsonl": str(run.generations_jsonl) if run.generations_jsonl else None,
        "inventory_counts": run.inventory_counts,
        "observed_counts": {},
        "axis": None,
        "exclusions": [],
    }

    if not run.run_dir.exists():
        run_summary["exclusions"].append(
            {"scope": "run", "reason": "missing_run_dir", "path": str(run.run_dir)}
        )
        return [], [], [], run_summary

    eval_dirs = sorted(
        [path for path in run.run_dir.glob("eval_*") if path.is_dir()],
        key=eval_sort_key,
    )
    run_summary["observed_counts"]["eval_dirs"] = len(eval_dirs)

    generation_by_eval, generation_rows, generation_exclusions = parse_generations(run)
    for reason in generation_exclusions:
        run_summary["exclusions"].append({"scope": "generation", "reason": reason})

    results_by_eval, results_best_loss_eval_id, results_best_final_loss_eval_id = (
        load_results_by_eval(run.results_json)
    )

    parsed_evals = [parse_eval(eval_dir) for eval_dir in eval_dirs]
    observed_best_records = done_records_for_observed_bests(parsed_evals)
    observed_best_loss_eval_id = min_eval_id(observed_best_records, "loss")
    observed_best_final_loss_eval_id = min_eval_id(observed_best_records, "final_loss")

    trajectory_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    all_wallclock_dts = [
        point["wallclock_dt"]
        for parsed in parsed_evals
        for point in parsed.points
        if point.get("wallclock_dt") is not None
    ]
    run_first_wallclock = min(all_wallclock_dts) if all_wallclock_dts else None

    for parsed in parsed_evals:
        for reason in parsed.exclusions:
            run_summary["exclusions"].append(
                {
                    "scope": "eval",
                    "eval_id": parsed.eval_id,
                    "reason": reason,
                    "path": str(parsed.eval_dir),
                }
            )

        if not parsed.points:
            continue

        metadata = metadata_for_eval(
            parsed,
            results_by_eval.get(parsed.eval_id),
            results_best_loss_eval_id,
            results_best_final_loss_eval_id,
            observed_best_loss_eval_id,
            observed_best_final_loss_eval_id,
        )
        generation = generation_by_eval.get(parsed.eval_id, {})
        config_id = f"eval_{parsed.eval_id}"
        candidate_wallclocks = [
            point["wallclock_dt"] for point in parsed.points if point.get("wallclock_dt")
        ]
        candidate_first_wallclock = (
            min(candidate_wallclocks) if candidate_wallclocks else None
        )
        candidate_last_wallclock = (
            max(candidate_wallclocks) if candidate_wallclocks else None
        )

        for point in parsed.points:
            wallclock = point.get("wallclock_dt")
            sweep_elapsed_seconds = None
            if run_first_wallclock is not None and wallclock is not None:
                sweep_elapsed_seconds = (wallclock - run_first_wallclock).total_seconds()
            candidate_elapsed_seconds = None
            if candidate_first_wallclock is not None and wallclock is not None:
                candidate_elapsed_seconds = (
                    wallclock - candidate_first_wallclock
                ).total_seconds()
            stdout_elapsed_h = point.get("stdout_elapsed_h")

            row = {
                "architecture_family": run.family,
                "architecture_model": run.model,
                "source_inventory_group": run.source_inventory_group,
                "sweep_id": run.sweep_id,
                "run_id": run.sweep_id,
                "run_status": run.status,
                "trial_id": parsed.eval_id,
                "config_id": config_id,
                "seed": metadata["seed"],
                "wallclock_timestamp_utc": iso_utc(wallclock),
                "sweep_elapsed_seconds": sweep_elapsed_seconds,
                "sweep_elapsed_minutes": (
                    sweep_elapsed_seconds / 60.0
                    if sweep_elapsed_seconds is not None
                    else None
                ),
                "candidate_elapsed_seconds": candidate_elapsed_seconds,
                "candidate_elapsed_minutes": (
                    candidate_elapsed_seconds / 60.0
                    if candidate_elapsed_seconds is not None
                    else None
                ),
                "stdout_elapsed_h": stdout_elapsed_h,
                "stdout_elapsed_minutes": (
                    stdout_elapsed_h * 60.0 if stdout_elapsed_h is not None else None
                ),
                "step": point["step"],
                "iteration": point["step"],
                "loss": point["loss"],
                "metric_name": "loss",
                "metric_unit": "natural_log_cross_entropy",
                "learning_rate": point.get("learning_rate"),
                "grad_norm": point.get("grad_norm"),
                "tokens_per_second": point.get("tokens_per_second"),
                "cma_loss": metadata["cma_loss"],
                "final_loss": metadata["final_loss"],
                "actual_params": metadata["actual_params"],
                "batch_size": metadata["batch_size"],
                "target_batch_size": metadata["target_batch_size"],
                "train_minutes": metadata["train_minutes"],
                "chunk_size": metadata["chunk_size"],
                "tokenizer": metadata["tokenizer"],
                "generation": generation.get("generation"),
                "warm_start_index": generation.get("warm_start_index"),
                "generation_wallclock_utc": generation.get("generation_wallclock_utc"),
                "generation_elapsed_minutes": generation.get("generation_elapsed_minutes"),
                "generation_eval_counter": generation.get("eval_counter"),
                "is_results_best_loss": metadata["is_results_best_loss"],
                "is_results_best_final_loss": metadata["is_results_best_final_loss"],
                "is_observed_best_loss": metadata["is_observed_best_loss"],
                "is_observed_best_final_loss": metadata["is_observed_best_final_loss"],
                "selected_best_flag_source": metadata["selected_best_flag_source"],
                "source_log_path": str(parsed.stdout_path),
                "source_eval_dir": str(parsed.eval_dir),
                "params_json_path": (
                    str(parsed.params_json_path) if parsed.params_json_path.exists() else None
                ),
                "done_json_path": (
                    str(parsed.done_json_path) if parsed.done_json_path.exists() else None
                ),
                "args_json_path": str(parsed.args_json_path)
                if parsed.args_json_path
                else None,
            }
            trajectory_rows.append(row)

        steps = [point["step"] for point in parsed.points]
        losses = [point["loss"] for point in parsed.points]
        final_step_loss = losses[-1] if losses else None
        min_step_loss = min(losses) if losses else None
        wallclock_span_minutes = None
        if candidate_first_wallclock is not None and candidate_last_wallclock is not None:
            wallclock_span_minutes = (
                candidate_last_wallclock - candidate_first_wallclock
            ).total_seconds() / 60.0

        eval_rows.append(
            {
                "architecture_family": run.family,
                "architecture_model": run.model,
                "source_inventory_group": run.source_inventory_group,
                "sweep_id": run.sweep_id,
                "run_id": run.sweep_id,
                "run_status": run.status,
                "trial_id": parsed.eval_id,
                "config_id": config_id,
                "seed": metadata["seed"],
                "trajectory_rows": len(parsed.points),
                "first_wallclock_timestamp_utc": iso_utc(candidate_first_wallclock),
                "last_wallclock_timestamp_utc": iso_utc(candidate_last_wallclock),
                "wallclock_span_minutes": wallclock_span_minutes,
                "first_step": min(steps) if steps else None,
                "last_step": max(steps) if steps else None,
                "final_step_loss": final_step_loss,
                "min_step_loss": min_step_loss,
                "cma_loss": metadata["cma_loss"],
                "final_loss": metadata["final_loss"],
                "actual_params": metadata["actual_params"],
                "batch_size": metadata["batch_size"],
                "target_batch_size": metadata["target_batch_size"],
                "train_minutes": metadata["train_minutes"],
                "chunk_size": metadata["chunk_size"],
                "tokenizer": metadata["tokenizer"],
                "generation": generation.get("generation"),
                "warm_start_index": generation.get("warm_start_index"),
                "generation_wallclock_utc": generation.get("generation_wallclock_utc"),
                "generation_elapsed_minutes": generation.get("generation_elapsed_minutes"),
                "is_results_best_loss": metadata["is_results_best_loss"],
                "is_results_best_final_loss": metadata["is_results_best_final_loss"],
                "is_observed_best_loss": metadata["is_observed_best_loss"],
                "is_observed_best_final_loss": metadata["is_observed_best_final_loss"],
                "selected_best_flag_source": metadata["selected_best_flag_source"],
                "source_log_path": str(parsed.stdout_path),
                "source_eval_dir": str(parsed.eval_dir),
                "params_json_path": (
                    str(parsed.params_json_path) if parsed.params_json_path.exists() else None
                ),
                "done_json_path": (
                    str(parsed.done_json_path) if parsed.done_json_path.exists() else None
                ),
                "args_json_path": str(parsed.args_json_path)
                if parsed.args_json_path
                else None,
            }
        )

    run_summary["observed_counts"].update(
        {
            "stdout_txt_files": sum(1 for parsed in parsed_evals if parsed.stdout_path.exists()),
            "done_files": sum(1 for parsed in parsed_evals if parsed.done_json_path.exists()),
            "params_json_files": sum(1 for parsed in parsed_evals if parsed.params_json_path.exists()),
            "args_json_files_resolved": sum(1 for parsed in parsed_evals if parsed.args_json_path),
            "parsed_eval_trajectories": len(eval_rows),
            "trajectory_rows": len(trajectory_rows),
            "generation_rows": len(generation_rows),
        }
    )
    if all_wallclock_dts:
        run_summary["axis"] = {
            "primary": "wallclock_timestamp_utc",
            "sweep_elapsed_minutes": "minutes since first parsed stdout step timestamp in this sweep",
            "candidate_elapsed_minutes": "minutes since first parsed stdout step timestamp in this eval",
            "fallback": "stdout_elapsed_h or step if a future log lacks the time field",
            "first_wallclock_timestamp_utc": iso_utc(min(all_wallclock_dts)),
            "last_wallclock_timestamp_utc": iso_utc(max(all_wallclock_dts)),
        }
    else:
        run_summary["axis"] = {
            "primary": "step",
            "fallback_reason": "no parseable stdout time fields",
        }

    return trajectory_rows, eval_rows, generation_rows, run_summary


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_inventory_group") or "",
        row.get("sweep_id") or "",
        int(row.get("trial_id") or -1),
        int(row.get("step") or -1),
        row.get("source_log_path") or "",
    )


def generation_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_inventory_group") or "",
        row.get("sweep_id") or "",
        int(row.get("warm_start_index") or 0),
        int(row.get("generation") or -1),
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if path.suffix == ".gz":
        raw_handle = path.open("wb")
        gzip_handle = gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        )
        handle = io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="")
    else:
        raw_handle = None
        gzip_handle = None
        handle = path.open("w", encoding="utf-8", newline="")

    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})
    finally:
        handle.close()
        if gzip_handle is not None:
            gzip_handle.close()
        if raw_handle is not None:
            raw_handle.close()


def check_monotonic(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sweep_id"]), int(row["trial_id"]))].append(row)

    errors: list[str] = []
    for (sweep_id, trial_id), group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item["step"]))
        steps = [int(row["step"]) for row in ordered]
        if any(b <= a for a, b in zip(steps, steps[1:])):
            errors.append(f"{sweep_id}/eval_{trial_id}: non-increasing steps")
            continue
        times = [
            parse_time(row.get("wallclock_timestamp_utc"))
            for row in ordered
            if row.get("wallclock_timestamp_utc")
        ]
        if len(times) > 1 and any(b < a for a, b in zip(times, times[1:])):
            errors.append(f"{sweep_id}/eval_{trial_id}: decreasing wallclock timestamps")
    return not errors, errors[:20]


def sanity_checks(
    rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    runs: list[RunSpec],
) -> dict[str, Any]:
    monotonic_ok, monotonic_errors = check_monotonic(rows)
    losses = [as_float(row.get("loss")) for row in rows]
    actual_families = sorted({str(row["architecture_family"]) for row in rows})
    expected_families = sorted({run.family for run in runs})
    source_paths = sorted({str(row["source_log_path"]) for row in rows})

    checks = {
        "nonempty_rows": {"passed": bool(rows), "row_count": len(rows)},
        "nonempty_eval_trajectories": {
            "passed": bool(eval_rows),
            "eval_trajectory_count": len(eval_rows),
        },
        "expected_architecture_labels": {
            "passed": set(expected_families).issubset(set(actual_families)),
            "expected": expected_families,
            "actual": actual_families,
        },
        "monotonic_step_and_time_within_trajectory": {
            "passed": monotonic_ok,
            "sample_errors": monotonic_errors,
        },
        "finite_loss_values": {
            "passed": all(value is not None for value in losses),
            "checked": len(losses),
        },
        "source_log_paths_exist": {
            "passed": all(Path(path).exists() for path in source_paths),
            "checked": len(source_paths),
        },
    }
    checks["all_passed"] = all(
        check["passed"] for key, check in checks.items() if isinstance(check, dict)
    )
    return checks


def excluded_sources_from_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    for root in inventory.get("historical_or_diagnostic_roots", []):
        excluded.append(
            {
                "path": root.get("path"),
                "reason": root.get("note")
                or "historical or diagnostic root, not a v2 burst source",
                "accessible": root.get("accessible"),
                "families": root.get("families"),
            }
        )
    for item in inventory.get("missing_or_incomplete", []):
        excluded.append(
            {
                "path": item.get("path"),
                "reason": f"inventory status: {item.get('status')}",
                "accessible": False,
            }
        )
    return excluded


def build_manifest(
    inventory_path: Path,
    out_dir: Path,
    inventory: dict[str, Any],
    runs: list[RunSpec],
    run_summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = sanity_checks(rows, eval_rows, runs)
    return {
        "normalizer": {
            "script": display_path(Path(__file__)),
            "version": SCRIPT_VERSION,
            "deterministic_outputs": True,
            "uses_only_python_standard_library": True,
        },
        "source_inventory": {
            "path": display_path(inventory_path),
            "generated_at_utc": inventory.get("generated_at_utc"),
            "task_id": inventory.get("task_id"),
        },
        "raw_logs_modified": False,
        "axis_policy": {
            "primary": "wallclock_timestamp_utc parsed from stdout step lines",
            "sweep_elapsed_minutes": "computed from first parsed step timestamp per sweep",
            "candidate_elapsed_minutes": "computed from first parsed step timestamp per eval",
            "fallback": "stdout_elapsed_h and step remain populated for logs without wallclock timestamps",
            "exact_eval_start_end_limitation": inventory.get("duration_assessment", {}).get(
                "reason_exact_per_eval_wallclock_not_verified"
            ),
        },
        "metric_policy": {
            "loss": "natural-log cross-entropy loss from stdout step lines",
            "bpb": "not stored in CMA-ES logs; downstream can derive from loss if needed",
            "cma_loss": ".done/results.json average-loss fitness when available",
            "final_loss": ".done/results.json final or FINAL_LOSS_LAST100 value when available",
        },
        "outputs": {
            "trajectory_points_csv": display_path(out_dir / TRAJECTORY_CSV),
            "trajectory_points_format": "deterministic gzip-compressed CSV",
            "eval_summary_csv": display_path(out_dir / EVAL_CSV),
            "generation_summary_csv": display_path(out_dir / GENERATION_CSV),
            "normalization_manifest_json": display_path(
                out_dir / NORMALIZATION_MANIFEST
            ),
        },
        "counts": {
            "runs_in_inventory_subset": len(runs),
            "runs_with_parsed_trajectories": sum(
                1 for summary in run_summaries if summary["observed_counts"].get("parsed_eval_trajectories", 0) > 0
            ),
            "trajectory_rows": len(rows),
            "eval_trajectories": len(eval_rows),
            "generation_rows": len(generation_rows),
            "source_stdout_logs": len({row["source_log_path"] for row in rows}),
        },
        "families": sorted({row["architecture_family"] for row in rows}),
        "runs": run_summaries,
        "excluded_sources": excluded_sources_from_inventory(inventory),
        "unsupported_formats": [
            {
                "format": "historical_or_diagnostic_roots",
                "reason": "not part of the final v2 2K/corrected 1.3B burst campaign per inventory",
            },
            {
                "format": "runs without stdout.txt step loss lines",
                "reason": "no per-step trajectory can be normalized; listed under per-run exclusions",
            },
        ],
        "sanity_checks": checks,
    }


def main() -> int:
    args = parse_args()
    inventory_path = args.inventory.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_json(inventory_path)
    runs = read_inventory_runs(inventory)

    all_rows: list[dict[str, Any]] = []
    all_eval_rows: list[dict[str, Any]] = []
    all_generation_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run in runs:
        rows, eval_rows, generation_rows, run_summary = build_rows_for_run(run)
        all_rows.extend(rows)
        all_eval_rows.extend(eval_rows)
        all_generation_rows.extend(generation_rows)
        run_summaries.append(run_summary)

    all_rows.sort(key=row_sort_key)
    all_eval_rows.sort(key=lambda row: row_sort_key({**row, "step": 0}))
    all_generation_rows.sort(key=generation_sort_key)

    write_csv(out_dir / TRAJECTORY_CSV, TRAJECTORY_FIELDS, all_rows)
    write_csv(out_dir / EVAL_CSV, EVAL_FIELDS, all_eval_rows)
    write_csv(out_dir / GENERATION_CSV, GENERATION_FIELDS, all_generation_rows)

    manifest = build_manifest(
        inventory_path=inventory_path,
        out_dir=out_dir,
        inventory=inventory,
        runs=runs,
        run_summaries=run_summaries,
        rows=all_rows,
        eval_rows=all_eval_rows,
        generation_rows=all_generation_rows,
    )
    with (out_dir / NORMALIZATION_MANIFEST).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if not manifest["sanity_checks"]["all_passed"]:
        print(
            json.dumps(manifest["sanity_checks"], indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    print(
        f"wrote {len(all_rows)} trajectory rows, {len(all_eval_rows)} eval summaries, "
        f"{len(all_generation_rows)} generation rows to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
