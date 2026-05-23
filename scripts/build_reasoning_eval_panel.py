#!/usr/bin/env python3
"""Build a multiple-choice reasoning eval panel for racer checkpoints.

The output schema matches ``scripts/racer_eval_suite.py``:

  prompt, choices, answer, category, name

This is a pragmatic intelligence/reasoning tracking panel. It avoids visual
ARC/Raven-style tasks and free-form generation tasks so results remain directly
comparable with the continuation-NLL multiple-choice harness.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable


def continuation(text: Any) -> str:
    s = str(text).strip()
    return s if s.startswith((" ", "\n", "\t")) else " " + s


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_first_available(candidates: list[tuple[str, tuple[str, ...], str]]):
    from datasets import load_dataset

    errors: list[str] = []
    for name, configs, split in candidates:
        try:
            if configs:
                return load_dataset(name, *configs, split=split)
            return load_dataset(name, split=split)
        except Exception as exc:  # pragma: no cover - depends on local HF cache/network
            errors.append(f"{name}{configs or ''}:{split}: {type(exc).__name__}: {exc}")
    raise RuntimeError("all dataset candidates failed:\n" + "\n".join(errors))


def sample_records(ds, per_task: int, seed: int) -> list[dict[str, Any]]:
    n = len(ds)
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return [dict(ds[i]) for i in idxs[: min(per_task, n)]]


def label_to_index(answer: Any, labels: list[Any], texts: list[Any] | None = None) -> int:
    key = str(answer).strip()
    str_labels = [str(label).strip() for label in labels]
    if key in str_labels:
        return str_labels.index(key)
    if key.startswith("(") and key.endswith(")") and len(key) == 3:
        key = key[1:-1]
    if key in str_labels:
        return str_labels.index(key)
    if texts is not None:
        lowered = [str(text).strip().lower() for text in texts]
        if key.lower() in lowered:
            return lowered.index(key.lower())
    if key.isdigit():
        value = int(key)
        if 0 <= value < len(labels):
            return value
        if 1 <= value <= len(labels):
            return value - 1
    if len(key) == 1 and key.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        value = ord(key.upper()) - ord("A")
        if 0 <= value < len(labels):
            return value
    raise ValueError(f"cannot map answer {answer!r} to labels {labels!r}")


def parse_bbh_options(text: str) -> tuple[str, list[str], list[str]]:
    """Split a BIG-Bench Hard prompt into stem plus answer labels/texts."""
    if "Options:" not in text:
        return text.strip(), [], []

    stem, options_blob = text.split("Options:", 1)
    labels: list[str] = []
    choices: list[str] = []
    bullet_ix = 0
    for raw in options_blob.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\(([A-Z])\)\s*(.+)$", line)
        if m:
            labels.append(m.group(1))
            choices.append(m.group(2).strip())
            continue
        m = re.match(r"^-\s*(.+)$", line)
        if m:
            choice = m.group(1).strip()
            labels.append(choice)
            choices.append(choice)
            bullet_ix += 1
            continue
        # Some BBH tasks wrap long options. Attach wrapped text to the prior
        # choice rather than silently dropping it.
        if choices:
            choices[-1] = f"{choices[-1]} {line}".strip()
        else:
            labels.append(str(bullet_ix))
            choices.append(line)
            bullet_ix += 1
    return stem.strip(), labels, choices


def reclor_rows(per_task: int, seed: int) -> list[dict[str, Any]]:
    ds = load_first_available(
        [
            ("metaeval/reclor", (), "validation"),
            # Legacy scripted datasets are kept as fallbacks for environments
            # with older datasets versions.
            ("reclor", (), "validation"),
        ]
    )
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        rows.append(
            {
                "category": "reclor",
                "name": f"reclor_{i:04d}",
                "prompt": (
                    f"\x1ePassage: {row['context'].strip()}\n"
                    f"Question: {row['question'].strip()}\nAnswer:"
                ),
                "choices": [continuation(text) for text in row["answers"]],
                "answer": int(row["label"]),
            }
        )
    return rows


def folio_rows(per_task: int, seed: int) -> list[dict[str, Any]]:
    ds = load_first_available(
        [
            ("tasksource/folio", (), "validation"),
            ("yale-nlp/FOLIO", (), "validation"),
        ]
    )
    label_order = ["True", "False", "Uncertain"]
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        rows.append(
            {
                "category": "folio",
                "name": f"folio_{i:04d}",
                "prompt": (
                    f"\x1ePremises:\n{row['premises'].strip()}\n"
                    f"Conclusion: {row['conclusion'].strip()}\n"
                    "Answer:"
                ),
                "choices": [continuation(text) for text in label_order],
                "answer": label_to_index(row["label"], label_order),
            }
        )
    return rows


def bbh_rows(per_task: int, seed: int, config: str) -> list[dict[str, Any]]:
    ds = load_first_available([("lukaemon/bbh", (config,), "test")])
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        text = str(row["input"]).strip()
        target = str(row["target"]).strip()
        stem, labels, choices = parse_bbh_options(text)

        if not choices:
            if target in {"True", "False"}:
                labels = ["False", "True"]
                choices = ["False", "True"]
            elif target in {"No", "Yes"}:
                labels = ["No", "Yes"]
                choices = ["No", "Yes"]
            else:
                raise ValueError(f"BBH config {config} row {i} is not multiple choice: target={target!r}")

        rows.append(
            {
                "category": f"bbh_{config}",
                "name": f"bbh_{config}_{i:04d}",
                "prompt": f"\x1e{stem}\nAnswer:",
                "choices": [continuation(choice) for choice in choices],
                "answer": label_to_index(target, labels, choices),
            }
        )
    return rows


BBH_DEFAULTS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
]

TASKS: dict[str, Callable[[int, int], list[dict[str, Any]]]] = {
    "reclor": reclor_rows,
    "folio": folio_rows,
}
for bbh_config in BBH_DEFAULTS:
    TASKS[f"bbh_{bbh_config}"] = lambda n, s, cfg=bbh_config: bbh_rows(n, s, cfg)

DEFAULT_TASKS = ["reclor", "folio"] + [f"bbh_{cfg}" for cfg in BBH_DEFAULTS]


def cap_rows(rows: list[dict[str, Any]], limit_total: int, seed: int) -> list[dict[str, Any]]:
    if limit_total <= 0 or len(rows) <= limit_total:
        return rows
    rng = random.Random(seed)
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    keep = sorted(indexed[:limit_total], key=lambda item: item[0])
    return [row for _, row in keep]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/tmp/racer_eval_panels/reasoning_panel.jsonl")
    p.add_argument("--per_task", type=int, default=160)
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--limit_total", type=int, default=2048)
    p.add_argument("--keep_going", action="store_true", help="Skip tasks that fail to load.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    successful_tasks: list[str] = []
    pre_cap_counts: dict[str, int] = {}
    for offset, task in enumerate(args.tasks):
        if task not in TASKS:
            raise SystemExit(f"unknown task {task!r}; available: {', '.join(TASKS)}")
        try:
            task_rows = TASKS[task](args.per_task, args.seed + offset * 997)
        except Exception as exc:
            if not args.keep_going:
                raise
            failures[task] = f"{type(exc).__name__}: {exc}"
            continue
        rows.extend(task_rows)
        successful_tasks.append(task)
        pre_cap_counts[task] = len(task_rows)
        print(f"{task}: {len(task_rows)}")

    rows = cap_rows(rows, args.limit_total, args.seed + 99991)
    out = Path(args.out)
    write_jsonl(out, rows)

    post_cap_counts: dict[str, int] = {}
    for row in rows:
        post_cap_counts[row["category"]] = post_cap_counts.get(row["category"], 0) + 1

    manifest = {
        "out": str(out),
        "n": len(rows),
        "per_task": args.per_task,
        "limit_total": args.limit_total,
        "seed": args.seed,
        "tasks": args.tasks,
        "successful_tasks": successful_tasks,
        "failures": failures,
        "pre_cap_counts": pre_cap_counts,
        "post_cap_counts": dict(sorted(post_cap_counts.items())),
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
