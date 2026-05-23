#!/usr/bin/env python3
"""Batched continuation evals for racer checkpoints.

This runner scores multiple-choice items by continuation NLL.  It uses the
repo-native racer checkpoint loader and p50k-aware tokenizer setup from
``generate_racer_samples.py`` and reuses the built-in 40 item probe from
``knowledge_continuation_probe.py``.

Input probe files may be JSON arrays or JSONL records.  Expected fields are:

  prompt, choices, answer, category, name

Common converted-dataset aliases are also accepted, including question/query,
options/endings, label/gold/target, task, id/question_id.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ndm" / "cuda"))

from generate_racer_samples import (  # noqa: E402
    apply_schedulefree_train_weights,
    build_model,
    load_args,
    resolve_checkpoint,
)
from knowledge_continuation_probe import PROBES as BUILTIN_PROBES  # noqa: E402


@dataclass(frozen=True)
class EvalItem:
    category: str
    name: str
    prompt: str
    choices: list[str]
    answer: int
    source_index: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class ScoreRequest:
    key: tuple[str, int]
    prefix: tuple[int, ...]
    continuation: tuple[int, ...]


def encode_text(enc: Any, text: str) -> list[int]:
    if enc is None:
        return list(text.encode("utf-8"))
    return enc.encode(text, disallowed_special=())


def decode_tokens(enc: Any, tokens: Iterable[int]) -> str:
    toks = list(tokens)
    if enc is None:
        return bytes([t for t in toks if 0 <= t < 256]).decode("utf-8", errors="replace")
    return enc.decode(toks)


def load_probe_records(path: str | None) -> list[dict[str, Any]]:
    if not path or path == "built-in":
        return [dict(row) for row in BUILTIN_PROBES]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    text = p.read_text()
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("rows", "items", "examples", "data"):
                if key in obj and isinstance(obj[key], list):
                    rows = obj[key]
                    break
            else:
                raise ValueError(f"{p} is a JSON object without rows/items/examples/data")
        elif isinstance(obj, list):
            rows = obj
        else:
            raise ValueError(f"{p} must contain a JSON list, JSON object with rows, or JSONL records")

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{p} contains non-object probe rows")
    return rows


def first_present(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def normalize_choices(raw_choices: Any) -> tuple[list[str], list[str | None]]:
    if isinstance(raw_choices, dict):
        if "text" in raw_choices and isinstance(raw_choices["text"], list):
            texts = [str(x) for x in raw_choices["text"]]
            labels_raw = raw_choices.get("label") or raw_choices.get("labels")
            labels = [str(x) for x in labels_raw] if isinstance(labels_raw, list) else [None] * len(texts)
            return texts, labels
        labels = [str(k) for k in raw_choices.keys()]
        return [str(v) for v in raw_choices.values()], labels

    if not isinstance(raw_choices, list):
        raise ValueError("choices/options/endings must be a list or mapping")
    if not raw_choices:
        raise ValueError("choices list is empty")

    if all(isinstance(choice, dict) for choice in raw_choices):
        texts: list[str] = []
        labels: list[str | None] = []
        for choice in raw_choices:
            text = first_present(choice, ("text", "value", "choice", "ending", "answer"))
            if text is None:
                raise ValueError(f"choice object lacks text/value/choice/ending: {choice}")
            texts.append(str(text))
            label = first_present(choice, ("label", "key", "id"))
            labels.append(str(label) if label is not None else None)
        return texts, labels

    return [str(choice) for choice in raw_choices], [None] * len(raw_choices)


def normalize_answer(raw_answer: Any, choices: list[str], labels: list[str | None]) -> int:
    if isinstance(raw_answer, bool):
        return int(raw_answer)
    if isinstance(raw_answer, int):
        answer = raw_answer
    elif isinstance(raw_answer, float) and raw_answer.is_integer():
        answer = int(raw_answer)
    elif isinstance(raw_answer, str):
        value = raw_answer.strip()
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            answer = int(value)
        else:
            label_map = {label: i for i, label in enumerate(labels) if label is not None}
            if value in label_map:
                answer = label_map[value]
            elif len(value) == 1 and value.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                answer = ord(value.upper()) - ord("A")
            else:
                lowered = [choice.strip().lower() for choice in choices]
                if value.lower() in lowered:
                    answer = lowered.index(value.lower())
                else:
                    raise ValueError(f"cannot map answer {raw_answer!r} to choices")
    else:
        raise ValueError(f"unsupported answer value: {raw_answer!r}")

    if not 0 <= answer < len(choices):
        raise ValueError(f"answer index {answer} is outside {len(choices)} choices")
    return answer


def normalize_probe_rows(rows: list[dict[str, Any]]) -> list[EvalItem]:
    items: list[EvalItem] = []
    for i, row in enumerate(rows):
        prompt = first_present(row, ("prompt", "query", "question", "ctx", "context"))
        if prompt is None:
            raise ValueError(f"row {i} lacks prompt/query/question/ctx/context")
        raw_choices = first_present(row, ("choices", "options", "endings", "answers"))
        if raw_choices is None:
            raise ValueError(f"row {i} lacks choices/options/endings/answers")
        raw_answer = first_present(row, ("answer", "label", "gold", "target"))
        if raw_answer is None:
            raise ValueError(f"row {i} lacks answer/label/gold/target")

        choices, labels = normalize_choices(raw_choices)
        answer = normalize_answer(raw_answer, choices, labels)
        category = str(first_present(row, ("category", "task", "subject", "dataset"), "default"))
        name = str(first_present(row, ("name", "id", "question_id", "uid"), f"item_{i:06d}"))
        items.append(
            EvalItem(
                category=category,
                name=name,
                prompt=str(prompt),
                choices=choices,
                answer=answer,
                source_index=i,
                raw=row,
            )
        )
    return items


def pad_2d(seqs: list[list[int]], pad_token: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in seqs)
    x = torch.full((len(seqs), max_len), pad_token, dtype=torch.long, device=device)
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return x, lengths


def gather_continuation_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -(token_logp * mask).sum(dim=1)


@torch.no_grad()
def score_full_sequence_batch(
    model: torch.nn.Module,
    requests: list[ScoreRequest],
    device: str,
    pad_token: int,
    batch_size: int,
) -> dict[tuple[str, int], tuple[float, int]]:
    out: dict[tuple[str, int], tuple[float, int]] = {}
    # Some sequence kernels are pad-sensitive even when we only read causal
    # logits before the padded suffix. Group by exact model input length so
    # full-sequence batching remains numerically equivalent to scalar scoring.
    by_input_len: dict[int, list[ScoreRequest]] = defaultdict(list)
    for req in requests:
        by_input_len[len(req.prefix) + len(req.continuation) - 1].append(req)

    for _, group in sorted(by_input_len.items()):
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            seqs = [list(req.prefix + req.continuation) for req in batch]
            if any(len(req.prefix) < 1 for req in batch):
                raise ValueError("all prompts must encode to at least one token")
            if any(len(req.continuation) < 1 for req in batch):
                raise ValueError("all choices must encode to at least one token")

            input_seqs = [seq[:-1] for seq in seqs]
            x = torch.tensor(input_seqs, dtype=torch.long, device=device)
            logits = model(x, return_loss=False)

            max_cont = max(len(req.continuation) for req in batch)
            targets = torch.full((len(batch), max_cont), pad_token, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), max_cont), dtype=torch.float32, device=device)
            selected = []
            for row, req in enumerate(batch):
                prefix_len = len(req.prefix)
                cont = list(req.continuation)
                selected.append(logits[row, prefix_len - 1 : prefix_len - 1 + len(cont), :])
                targets[row, : len(cont)] = torch.tensor(cont, dtype=torch.long, device=device)
                mask[row, : len(cont)] = 1.0
            selected_logits = torch.zeros(
                (len(batch), max_cont, logits.size(-1)),
                dtype=logits.dtype,
                device=device,
            )
            for row, row_logits in enumerate(selected):
                selected_logits[row, : row_logits.size(0), :] = row_logits

            nlls = gather_continuation_nll(selected_logits, targets, mask)
            for req, nll in zip(batch, nlls.tolist()):
                out[req.key] = (float(nll), len(req.continuation))
    return out


def nested_batch_expand(value: Any, batch_size: int) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.size(0) != 1:
            raise ValueError(f"cannot expand hidden state with batch dimension {value.size(0)}")
        return value.expand(batch_size, *value.shape[1:]).contiguous()
    if isinstance(value, list):
        return [nested_batch_expand(item, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(nested_batch_expand(item, batch_size) for item in value)
    raise TypeError(f"unsupported hidden state type: {type(value).__name__}")


@torch.no_grad()
def score_stateful_prefix_group(
    model: torch.nn.Module,
    prefix: tuple[int, ...],
    requests: list[ScoreRequest],
    device: str,
    pad_token: int,
    batch_size: int,
) -> dict[tuple[str, int], tuple[float, int]]:
    if len(prefix) < 1:
        raise ValueError("all prompts must encode to at least one token")
    prefix_x = torch.tensor([list(prefix)], dtype=torch.long, device=device)
    prefix_logits, (hiddens, _) = model(
        prefix_x,
        return_loss=False,
        return_prev_hiddens=True,
        prev_hiddens=None,
    )
    first_logp = F.log_softmax(prefix_logits[0, -1].float(), dim=-1)

    out: dict[tuple[str, int], tuple[float, int]] = {}
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        totals = torch.empty(len(batch), dtype=torch.float32, device=device)
        tail_reqs: list[tuple[int, ScoreRequest]] = []
        tail_inputs: list[list[int]] = []

        for row, req in enumerate(batch):
            cont = list(req.continuation)
            if not cont:
                raise ValueError("all choices must encode to at least one token")
            totals[row] = -first_logp[cont[0]]
            if len(cont) > 1:
                tail_reqs.append((row, req))
                tail_inputs.append(cont[:-1])

        if tail_inputs:
            x, _ = pad_2d(tail_inputs, pad_token, device)
            expanded_hiddens = nested_batch_expand(hiddens, len(tail_inputs))
            tail_logits = model(
                x,
                return_loss=False,
                return_prev_hiddens=False,
                prev_hiddens=expanded_hiddens,
            )
            max_tail = max(len(req.continuation) - 1 for _, req in tail_reqs)
            targets = torch.full((len(tail_reqs), max_tail), pad_token, dtype=torch.long, device=device)
            mask = torch.zeros((len(tail_reqs), max_tail), dtype=torch.float32, device=device)
            for tail_row, (_, req) in enumerate(tail_reqs):
                target = list(req.continuation[1:])
                targets[tail_row, : len(target)] = torch.tensor(target, dtype=torch.long, device=device)
                mask[tail_row, : len(target)] = 1.0
            tail_nlls = gather_continuation_nll(tail_logits[:, :max_tail, :], targets, mask)
            for tail_nll, (row, _) in zip(tail_nlls, tail_reqs):
                totals[row] += tail_nll

        for row, req in enumerate(batch):
            out[req.key] = (float(totals[row].item()), len(req.continuation))
    return out


@torch.no_grad()
def score_stateful_prefix_batches(
    model: torch.nn.Module,
    requests: list[ScoreRequest],
    device: str,
    pad_token: int,
    batch_size: int,
) -> dict[tuple[str, int], tuple[float, int]]:
    grouped: dict[tuple[int, ...], list[ScoreRequest]] = defaultdict(list)
    for req in requests:
        grouped[req.prefix].append(req)

    out: dict[tuple[str, int], tuple[float, int]] = {}
    for prefix, group in grouped.items():
        out.update(score_stateful_prefix_group(model, prefix, group, device, pad_token, batch_size))
    return out


def can_use_stateful_prefix(model_args: dict[str, Any]) -> bool:
    level = str(model_args.get("level", "")).lower()
    if level not in {"e88", "m2rnn"}:
        return False
    if bool(model_args.get("use_conv", 0)):
        return False
    return True


def choose_score_mode(requested: str, model_args: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    return "stateful-prefix" if can_use_stateful_prefix(model_args) else "full-sequence"


def build_requests(
    items: list[EvalItem],
    enc: Any,
    neutral_prefix: str,
    compute_pmi: bool,
) -> tuple[list[ScoreRequest], dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    requests: list[ScoreRequest] = []
    conditional: dict[tuple[int, int], int] = {}
    neutral: dict[tuple[int, int], int] = {}
    neutral_tokens = tuple(encode_text(enc, neutral_prefix))

    for item_idx, item in enumerate(items):
        prefix = tuple(encode_text(enc, item.prompt))
        for choice_idx, choice in enumerate(item.choices):
            cont = tuple(encode_text(enc, choice))
            key = (item_idx, choice_idx)
            conditional[key] = len(requests)
            requests.append(ScoreRequest(key=("cond", len(requests)), prefix=prefix, continuation=cont))
            if compute_pmi:
                neutral[key] = len(requests)
                requests.append(
                    ScoreRequest(key=("base", len(requests)), prefix=neutral_tokens, continuation=cont)
                )
    return requests, conditional, neutral


def score_items(
    model: torch.nn.Module,
    items: list[EvalItem],
    enc: Any,
    device: str,
    pad_token: int,
    batch_size: int,
    score_mode: str,
    neutral_prefix: str,
    compute_pmi: bool,
) -> list[dict[str, Any]]:
    requests, conditional_ix, neutral_ix = build_requests(items, enc, neutral_prefix, compute_pmi)

    if score_mode == "stateful-prefix":
        scored = score_stateful_prefix_batches(model, requests, device, pad_token, batch_size)
    elif score_mode == "full-sequence":
        scored = score_full_sequence_batch(model, requests, device, pad_token, batch_size)
    else:
        raise ValueError(f"unknown score mode: {score_mode}")

    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items):
        scores = []
        for choice_idx, choice in enumerate(item.choices):
            key = (item_idx, choice_idx)
            req_ix = conditional_ix[key]
            nll, ntok = scored[("cond", req_ix)]  # type: ignore[index]
            rec = {
                "choice": choice,
                "nll": nll,
                "avg_nll": nll / ntok,
                "tokens": ntok,
            }
            if compute_pmi:
                base_ix = neutral_ix[key]
                base_nll, _ = scored[("base", base_ix)]  # type: ignore[index]
                rec.update(
                    {
                        "base_nll": base_nll,
                        "base_avg_nll": base_nll / ntok,
                        "pmi": nll - base_nll,
                        "pmi_avg": (nll - base_nll) / ntok,
                    }
                )
            scores.append(rec)

        preds = {
            "nll": min(range(len(scores)), key=lambda i: scores[i]["nll"]),
            "avg_nll": min(range(len(scores)), key=lambda i: scores[i]["avg_nll"]),
        }
        if compute_pmi:
            preds["pmi_avg"] = min(range(len(scores)), key=lambda i: scores[i]["pmi_avg"])

        rows.append(
            {
                "category": item.category,
                "name": item.name,
                "source_index": item.source_index,
                "prompt": item.prompt,
                "choices": item.choices,
                "answer": item.answer,
                "preds": preds,
                "scores": scores,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]], score_keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for score_key in score_keys:
        correct = sum(int(row["preds"][score_key] == row["answer"]) for row in rows)
        cats: dict[str, dict[str, Any]] = {}
        for row in rows:
            cat = row["category"]
            rec = cats.setdefault(cat, {"correct": 0, "n": 0, "accuracy": 0.0})
            rec["n"] += 1
            rec["correct"] += int(row["preds"][score_key] == row["answer"])
        for rec in cats.values():
            rec["accuracy"] = rec["correct"] / rec["n"] if rec["n"] else math.nan
        summary[score_key] = {
            "correct": correct,
            "n": len(rows),
            "accuracy": correct / len(rows) if rows else math.nan,
            "category_summary": dict(sorted(cats.items())),
        }
    return summary


def markdown_report(results: list[dict[str, Any]], primary_score: str) -> str:
    lines = [
        "# Racer Continuation Eval",
        "",
        "| label | level | step | mode | score | correct | n | accuracy |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for result in results:
        primary = result["summary"][primary_score]
        lines.append(
            f"| {result['label']} | {result['level']} | {result.get('step')} | "
            f"{result['score_mode']} | {primary_score} | {primary['correct']} | "
            f"{primary['n']} | {primary['accuracy']:.3f} |"
        )

    lines.extend(["", "## Category Accuracy", ""])
    for result in results:
        lines.append(f"### {result['label']}")
        lines.append("")
        lines.append("| category | correct | n | accuracy |")
        lines.append("|---|---:|---:|---:|")
        for cat, rec in result["summary"][primary_score]["category_summary"].items():
            lines.append(f"| {cat} | {rec['correct']} | {rec['n']} | {rec['accuracy']:.3f} |")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Scores are continuation NLLs over tokenized choice text.",
            "- `avg_nll` normalizes by choice token count.",
            "- `pmi_avg` subtracts the same choice scored after the neutral prefix when PMI scoring is enabled.",
            "- `stateful-prefix` reuses recurrent prefix state only for models with simple tensor states; other models use length-grouped full-sequence batches to avoid padding-sensitive score drift.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_results(path: Path, results: list[dict[str, Any]], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        path.write_text(json.dumps(results, indent=2) + "\n")
        return
    with path.open("w") as f:
        for result in results:
            summary = {k: v for k, v in result.items() if k != "rows"}
            f.write(json.dumps({"type": "summary", **summary}) + "\n")
            for row in result["rows"]:
                f.write(json.dumps({"type": "row", "label": result["label"], **row}) + "\n")


def infer_label(path: str) -> str:
    p = Path(path)
    if p.name:
        parent = p.parent.name
        stem = p.stem
        return f"{parent}_{stem}" if parent else stem
    return str(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path. Repeat for multiple checkpoints.")
    p.add_argument("--label", action="append", default=[], help="Label for each checkpoint. Repeat to match --checkpoint.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--probes", default="built-in", help="built-in, JSON, or JSONL probe file")
    p.add_argument("--limit", type=int, default=0, help="Optional first-N smoke limit")
    p.add_argument("--out", default="/tmp/racer_eval/results.json")
    p.add_argument("--report", default=None, help="Markdown report path; defaults beside --out")
    p.add_argument("--format", choices=["json", "jsonl"], default="json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--score_mode", choices=["auto", "full-sequence", "stateful-prefix"], default="auto")
    p.add_argument("--primary_score", choices=["avg_nll", "nll", "pmi_avg"], default="avg_nll")
    p.add_argument("--neutral_prefix", default="\x1e")
    p.add_argument("--no_pmi", action="store_true", help="Skip neutral-prefix scores and pmi_avg")
    p.add_argument("--pad_token", type=int, default=0)
    p.add_argument("--dry_run", action="store_true", help="Validate probes and print counts without loading a checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = load_probe_records(args.probes)
    items = normalize_probe_rows(raw_rows)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise ValueError("no eval items loaded")

    categories = defaultdict(int)
    for item in items:
        categories[item.category] += 1

    if args.dry_run:
        print(json.dumps({"n": len(items), "categories": dict(sorted(categories.items()))}, indent=2))
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --dry_run is set")
    if args.label and len(args.label) != len(args.checkpoint):
        raise SystemExit("--label must be repeated once per --checkpoint")
    labels = args.label or [infer_label(path) for path in args.checkpoint]

    compute_pmi = not args.no_pmi
    if args.primary_score == "pmi_avg" and not compute_pmi:
        raise SystemExit("--primary_score pmi_avg requires PMI scoring; remove --no_pmi")
    score_keys = ["nll", "avg_nll"] + (["pmi_avg"] if compute_pmi else [])

    results: list[dict[str, Any]] = []
    for label, checkpoint in zip(labels, args.checkpoint):
        ckpt_path = resolve_checkpoint(checkpoint)
        model_args = load_args(ckpt_path)

        tokenizer_name = model_args.get("tokenizer")
        if tokenizer_name:
            import tiktoken

            enc = tiktoken.get_encoding(tokenizer_name)
            vocab_size = enc.n_vocab
        else:
            enc = None
            vocab_size = 256

        model = build_model(model_args, vocab_size)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        used_sf_swap = apply_schedulefree_train_weights(model, ckpt, model_args)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        model = model.to(args.device)
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
        model.eval()

        score_mode = choose_score_mode(args.score_mode, model_args)
        t0 = time.time()
        rows = score_items(
            model=model,
            items=items,
            enc=enc,
            device=args.device,
            pad_token=args.pad_token,
            batch_size=args.batch_size,
            score_mode=score_mode,
            neutral_prefix=args.neutral_prefix,
            compute_pmi=compute_pmi,
        )
        elapsed = time.time() - t0
        summary = summarize_rows(rows, score_keys)
        result = {
            "label": label,
            "checkpoint": str(ckpt_path),
            "step": ckpt.get("step"),
            "loss": ckpt.get("loss"),
            "level": model_args["level"],
            "tokenizer": tokenizer_name or "byte",
            "schedulefree_train_weight_swap": used_sf_swap,
            "score_mode": score_mode,
            "primary_score": args.primary_score,
            "batch_size": args.batch_size,
            "neutral_prefix_tokens": encode_text(enc, args.neutral_prefix) if compute_pmi else None,
            "n": len(rows),
            "elapsed_s": elapsed,
            "items_per_s": len(rows) / max(elapsed, 1e-9),
            "summary": summary,
            "rows": rows,
        }
        results.append(result)

        primary = summary[args.primary_score]
        print(
            f"{label} step={result['step']} mode={score_mode} primary={args.primary_score} "
            f"acc={primary['correct']}/{primary['n']} ({primary['accuracy']:.3f}) "
            f"elapsed={elapsed:.1f}s"
        )

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    write_results(out_path, results, args.format)
    report_path = Path(args.report) if args.report else out_path.with_suffix(".md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(results, args.primary_score))
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
