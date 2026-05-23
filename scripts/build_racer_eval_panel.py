#!/usr/bin/env python3
"""Build a small multiple-choice eval panel for racer checkpoints.

The output is the JSONL schema consumed by ``scripts/racer_eval_suite.py``:

  prompt, choices, answer, category, name

This is intentionally a pragmatic tracking panel, not a benchmark replacement.
It samples deterministic subsets of common validation splits so we can watch
knowledge/reasoning competence move as the racers converge.
"""

from __future__ import annotations

import argparse
import json
import random
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


def label_to_index(answer: Any, labels: list[Any]) -> int:
    key = str(answer).strip()
    str_labels = [str(label).strip() for label in labels]
    if key in str_labels:
        return str_labels.index(key)
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


def arc_rows(per_task: int, seed: int, challenge: bool) -> list[dict[str, Any]]:
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    category = "arc_challenge" if challenge else "arc_easy"
    ds = load_first_available(
        [
            ("ai2_arc", (config,), "validation"),
            ("allenai/ai2_arc", (config,), "validation"),
        ]
    )
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        rows.append(
            {
                "category": category,
                "name": f"{category}_{i:04d}",
                "prompt": f"\x1eQuestion: {row['question'].strip()}\nAnswer:",
                "choices": [continuation(text) for text in texts],
                "answer": label_to_index(row["answerKey"], labels),
            }
        )
    return rows


def piqa_rows(per_task: int, seed: int) -> list[dict[str, Any]]:
    ds = load_first_available([("piqa", (), "validation")])
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        rows.append(
            {
                "category": "piqa",
                "name": f"piqa_{i:04d}",
                "prompt": f"\x1eQuestion: {row['goal'].strip()}\nAnswer:",
                "choices": [continuation(row["sol1"]), continuation(row["sol2"])],
                "answer": int(row["label"]),
            }
        )
    return rows


def hellaswag_rows(per_task: int, seed: int) -> list[dict[str, Any]]:
    ds = load_first_available([("Rowan/hellaswag", (), "validation"), ("hellaswag", (), "validation")])
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        prompt = str(row.get("ctx") or row.get("ctx_a") or "").strip()
        rows.append(
            {
                "category": "hellaswag",
                "name": f"hellaswag_{i:04d}",
                "prompt": "\x1e" + prompt,
                "choices": [continuation(text) for text in row["endings"]],
                "answer": int(row["label"]),
            }
        )
    return rows


def sciq_rows(per_task: int, seed: int) -> list[dict[str, Any]]:
    ds = load_first_available([("sciq", (), "validation")])
    rows = []
    rng = random.Random(seed + 17)
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        choices = [
            row["correct_answer"],
            row["distractor1"],
            row["distractor2"],
            row["distractor3"],
        ]
        indexed = list(enumerate(choices))
        rng.shuffle(indexed)
        rows.append(
            {
                "category": "sciq",
                "name": f"sciq_{i:04d}",
                "prompt": f"\x1eQuestion: {row['question'].strip()}\nAnswer:",
                "choices": [continuation(text) for _, text in indexed],
                "answer": [j for j, (orig, _) in enumerate(indexed) if orig == 0][0],
            }
        )
    return rows


def openbookqa_rows(per_task: int, seed: int) -> list[dict[str, Any]]:
    ds = load_first_available([("openbookqa", ("main",), "validation"), ("allenai/openbookqa", ("main",), "validation")])
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        rows.append(
            {
                "category": "openbookqa",
                "name": f"openbookqa_{i:04d}",
                "prompt": f"\x1eQuestion: {row['question_stem'].strip()}\nAnswer:",
                "choices": [continuation(text) for text in texts],
                "answer": label_to_index(row["answerKey"], labels),
            }
        )
    return rows


def boolq_rows(per_task: int, seed: int) -> list[dict[str, Any]]:
    ds = load_first_available([("google/boolq", (), "validation"), ("boolq", (), "validation")])
    rows = []
    for i, row in enumerate(sample_records(ds, per_task, seed)):
        rows.append(
            {
                "category": "boolq",
                "name": f"boolq_{i:04d}",
                "prompt": f"\x1ePassage: {row['passage'].strip()}\nQuestion: {row['question'].strip()}\nAnswer:",
                "choices": [" no", " yes"],
                "answer": 1 if bool(row["answer"]) else 0,
            }
        )
    return rows


TASKS: dict[str, Callable[[int, int], list[dict[str, Any]]]] = {
    "arc_easy": lambda n, s: arc_rows(n, s, challenge=False),
    "arc_challenge": lambda n, s: arc_rows(n, s, challenge=True),
    "piqa": piqa_rows,
    "hellaswag": hellaswag_rows,
    "sciq": sciq_rows,
    "openbookqa": openbookqa_rows,
    "boolq": boolq_rows,
}
DEFAULT_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "sciq", "openbookqa", "boolq"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/tmp/racer_eval_panels/knowledge_panel.jsonl")
    p.add_argument("--per_task", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--keep_going", action="store_true", help="Skip tasks that fail to load.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    successful_tasks: list[str] = []
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
        print(f"{task}: {len(task_rows)}")

    out = Path(args.out)
    write_jsonl(out, rows)
    manifest = {
        "out": str(out),
        "n": len(rows),
        "per_task": args.per_task,
        "seed": args.seed,
        "tasks": args.tasks,
        "successful_tasks": successful_tasks,
        "failures": failures,
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
