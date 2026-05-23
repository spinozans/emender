#!/usr/bin/env python3
"""Forced-choice continuation probe for racer checkpoints.

The goal is not to benchmark "truth" exhaustively. It is to track when a model
starts assigning high probability to ordinary continuations: common facts, code
idioms, format completions, and simple concept bindings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_racer_samples import (  # noqa: E402
    apply_schedulefree_train_weights,
    build_model,
    load_args,
    resolve_checkpoint,
)


PROBES = [
    # Common facts
    {"category": "facts", "name": "capital_france", "prompt": "\x1eQ: What is the capital of France?\nA:", "choices": [" London", " Paris", " Berlin", " Madrid"], "answer": 1},
    {"category": "facts", "name": "capital_japan", "prompt": "\x1eQ: What is the capital of Japan?\nA:", "choices": [" Beijing", " Seoul", " Tokyo", " Kyoto"], "answer": 2},
    {"category": "facts", "name": "capital_italy", "prompt": "\x1eThe capital city of Italy is", "choices": [" Milan", " Venice", " Rome", " Naples"], "answer": 2},
    {"category": "facts", "name": "largest_planet", "prompt": "\x1eThe largest planet in the solar system is", "choices": [" Mars", " Earth", " Jupiter", " Venus"], "answer": 2},
    {"category": "facts", "name": "water_freezes", "prompt": "\x1eAt standard pressure, water freezes at", "choices": [" 100 degrees Celsius", " 37 degrees Celsius", " 0 degrees Celsius", " 273 degrees Celsius"], "answer": 2},
    {"category": "facts", "name": "earth_orbits", "prompt": "\x1eThe Earth orbits the", "choices": [" Moon", " Sun", " North Pole", " Milky Way"], "answer": 1},
    {"category": "facts", "name": "romeo", "prompt": "\x1eRomeo and Juliet was written by", "choices": [" Jane Austen", " William Shakespeare", " Charles Darwin", " Isaac Newton"], "answer": 1},
    {"category": "facts", "name": "darwin_origin", "prompt": "\x1eOn the Origin of Species was written by", "choices": [" Charles Darwin", " William Shakespeare", " Albert Einstein", " Jane Austen"], "answer": 0},

    # Concept bindings
    {"category": "concepts", "name": "mitochondria", "prompt": "\x1eThe mitochondria is the", "choices": [" largest planet", " programming language", " powerhouse of the cell", " capital of France"], "answer": 2},
    {"category": "concepts", "name": "dna", "prompt": "\x1eDNA stores", "choices": [" electrical charge", " genetic information", " ocean currents", " compiled bytecode"], "answer": 1},
    {"category": "concepts", "name": "photosynthesis", "prompt": "\x1ePlants use photosynthesis to convert sunlight into", "choices": [" chemical energy", " magnetic fields", " database tables", " prime numbers"], "answer": 0},
    {"category": "concepts", "name": "gravity", "prompt": "\x1eGravity is a force that", "choices": [" repels all masses", " attracts masses", " sorts arrays", " prints text"], "answer": 1},
    {"category": "concepts", "name": "bachelor", "prompt": "\x1eA bachelor is an unmarried", "choices": [" woman", " child", " man", " city"], "answer": 2},
    {"category": "concepts", "name": "opposite_hot", "prompt": "\x1eThe opposite of hot is", "choices": [" warm", " cold", " heavy", " square"], "answer": 1},
    {"category": "concepts", "name": "rectangle", "prompt": "\x1eA rectangle has four", "choices": [" wheels", " corners", " oceans", " alphabets"], "answer": 1},
    {"category": "concepts", "name": "doctor_patient", "prompt": "\x1eA doctor treats a", "choices": [" patient", " compiler", " theorem", " planet"], "answer": 0},

    # Code and technical idioms
    {"category": "code", "name": "python_list_comp", "prompt": "\x1eIn Python, a list comprehension looks like", "choices": [" SELECT * FROM table", " public static void main", " [x for x in xs]", " \\begin{document}"], "answer": 2},
    {"category": "code", "name": "python_sort", "prompt": "\x1eQ: How do I sort a list in Python?\nA:", "choices": [" Use sort() or sorted().", " Use SELECT and FROM.", " Use \\begin{document}.", " Use the capital of France."], "answer": 0},
    {"category": "code", "name": "html_link", "prompt": "\x1eAn HTML link is written with the", "choices": [" <script> tag", " <img> tag", " <a> tag", " <table> tag"], "answer": 2},
    {"category": "code", "name": "sql_select", "prompt": "\x1eA basic SQL query starts with", "choices": [" SELECT", " printf", " import", " \\section"], "answer": 0},
    {"category": "code", "name": "js_log", "prompt": "\x1eIn JavaScript, a common way to print a value is", "choices": [" console.log(x)", " SELECT x FROM t", " \\frac{x}{y}", " x for x in xs"], "answer": 0},
    {"category": "code", "name": "git_commit", "prompt": "\x1eAfter editing files in git, a typical command to save a snapshot is", "choices": [" git commit", " git divide", " python select", " html orbit"], "answer": 0},
    {"category": "code", "name": "c_header", "prompt": "\x1eIn C, printf is declared in", "choices": [" stdio.h", " vector.py", " index.html", " package.json"], "answer": 0},
    {"category": "code", "name": "json_shape", "prompt": "\x1eA JSON object is usually written with", "choices": [" curly braces", " angle brackets only", " musical notes", " roman numerals"], "answer": 0},

    # Math and symbolic continuations
    {"category": "math", "name": "two_plus_two", "prompt": "\x1e2 + 2 =", "choices": [" 3", " 4", " 5", " 22"], "answer": 1},
    {"category": "math", "name": "derivative", "prompt": "\x1eThe derivative of x^2 is", "choices": [" x", " 2x", " x^3", " 0"], "answer": 1},
    {"category": "math", "name": "pythagorean", "prompt": "\x1eFor a right triangle, the Pythagorean theorem says", "choices": [" a^2 + b^2 = c^2", " E = mc^2", " SELECT * FROM t", " F = ma"], "answer": 0},
    {"category": "math", "name": "prime_after_five", "prompt": "\x1eThe prime number after 5 is", "choices": [" 6", " 7", " 8", " 9"], "answer": 1},
    {"category": "math", "name": "one_half", "prompt": "\x1eOne half plus one half equals", "choices": [" one", " two", " zero", " three"], "answer": 0},
    {"category": "math", "name": "square_root_nine", "prompt": "\x1eThe square root of 9 is", "choices": [" 2", " 3", " 9", " 81"], "answer": 1},

    # Format / genre
    {"category": "format", "name": "stackoverflow_answer", "prompt": "\x1eQ: How do I reverse a string in Python?\nA:", "choices": [" Use slicing like s[::-1].", " The proof follows by contradiction.", " Paris is the capital.", " The mitochondria is an organelle."], "answer": 0},
    {"category": "format", "name": "paper_abstract", "prompt": "\x1eAbstract\nWe present", "choices": [" a new method for", " the capital city of", " a stack trace at", " a recipe with"], "answer": 0},
    {"category": "format", "name": "latex_document", "prompt": "\x1eA minimal LaTeX document begins with", "choices": [" \\begin{document}", " SELECT * FROM", " console.log", " Dear Sir"], "answer": 0},
    {"category": "format", "name": "theorem_proof", "prompt": "\x1eTheorem. Let n be an even integer.\nProof.", "choices": [" Since n is even,", " The patient was given", " SELECT the rows", " Once upon a time"], "answer": 0},
    {"category": "format", "name": "patent_claim", "prompt": "\x1eWhat is claimed is:", "choices": [" A system comprising", " Dear editor,", " Q: How do I", " The capital is"], "answer": 0},
    {"category": "format", "name": "markdown_heading", "prompt": "\x1eIn Markdown, a section heading often starts with", "choices": [" #", " SELECT", " printf", " DNA"], "answer": 0},

    # Local coherence / discourse
    {"category": "coherence", "name": "cause_effect", "prompt": "\x1eThe road was wet because", "choices": [" it had rained.", " it was made of numbers.", " the code compiled.", " France is Paris."], "answer": 0},
    {"category": "coherence", "name": "pronoun_mary", "prompt": "\x1eMary dropped the glass, and it", "choices": [" shattered.", " graduated.", " compiled.", " orbited."], "answer": 0},
    {"category": "coherence", "name": "hungry_food", "prompt": "\x1eBecause he was hungry, he", "choices": [" ate dinner.", " wrote a checksum.", " froze water.", " became a city."], "answer": 0},
    {"category": "coherence", "name": "question_answer", "prompt": "\x1eQ: What color is the sky on a clear day?\nA:", "choices": [" Blue.", " Seven.", " Shakespeare.", " SELECT."], "answer": 0},
]


def encode_text(enc, text: str) -> list[int]:
    if enc is None:
        return list(text.encode("utf-8"))
    return enc.encode(text, disallowed_special=())


@torch.no_grad()
def continuation_nll(model, prefix: list[int], continuation: list[int], device: str) -> tuple[float, int]:
    seq = torch.tensor([prefix + continuation], dtype=torch.long, device=device)
    logits = model(seq[:, :-1], return_loss=False)
    logp = F.log_softmax(logits.float(), dim=-1)
    start = len(prefix) - 1
    total = 0.0
    for i, tok in enumerate(continuation):
        total += float(logp[0, start + i, tok].item())
    return -total, len(continuation)


def load_probes(path: str | None) -> list[dict]:
    if not path:
        return PROBES
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, list):
        raise ValueError("probe file must contain a JSON list")
    return obj


def category_summary(rows: list[dict], pred_key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        cat = row["category"]
        rec = out.setdefault(cat, {"correct": 0, "n": 0, "accuracy": 0.0})
        rec["n"] += 1
        rec["correct"] += int(row[pred_key] == row["answer"])
    for rec in out.values():
        rec["accuracy"] = rec["correct"] / rec["n"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/racer_generations_20260518/knowledge_probe.json")
    p.add_argument("--probes", default=None, help="Optional JSON file containing probe list")
    p.add_argument(
        "--score",
        choices=["avg_nll", "nll", "pmi_avg"],
        default="avg_nll",
        help="Primary prediction score. pmi_avg subtracts an unconditional continuation prior.",
    )
    args = p.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
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
    sf_swap = apply_schedulefree_train_weights(model, ckpt, model_args)
    model = model.to(args.device).bfloat16().eval()

    rows = []
    probes = load_probes(args.probes)
    correct = {"nll": 0, "avg_nll": 0, "pmi_avg": 0}
    neutral_prefix = encode_text(enc, "\x1e")
    for probe in probes:
        prefix = encode_text(enc, probe["prompt"])
        scores = []
        for choice in probe["choices"]:
            continuation = encode_text(enc, choice)
            nll, ntok = continuation_nll(model, prefix, continuation, args.device)
            base_nll, _ = continuation_nll(model, neutral_prefix, continuation, args.device)
            scores.append(
                {
                    "choice": choice,
                    "nll": nll,
                    "avg_nll": nll / ntok,
                    "base_nll": base_nll,
                    "base_avg_nll": base_nll / ntok,
                    "pmi": nll - base_nll,
                    "pmi_avg": (nll - base_nll) / ntok,
                    "tokens": ntok,
                }
            )
        preds = {
            "nll": min(range(len(scores)), key=lambda i: scores[i]["nll"]),
            "avg_nll": min(range(len(scores)), key=lambda i: scores[i]["avg_nll"]),
            "pmi_avg": min(range(len(scores)), key=lambda i: scores[i]["pmi_avg"]),
        }
        for key, pred in preds.items():
            correct[key] += int(pred == probe["answer"])
        rows.append({**probe, "pred": preds[args.score], "preds": preds, "scores": scores})

    result = {
        "label": args.label,
        "checkpoint": str(ckpt_path),
        "step": ckpt.get("step"),
        "loss": ckpt.get("loss"),
        "level": model_args["level"],
        "schedulefree_train_weight_swap": sf_swap,
        "primary_score": args.score,
        "accuracy": correct[args.score] / len(probes),
        "correct": correct[args.score],
        "n": len(probes),
        "accuracies": {key: val / len(probes) for key, val in correct.items()},
        "correct_by_score": correct,
        "rows": rows,
    }
    result["category_summary"] = {
        score_key: category_summary(
            [{**row, "_pred": row["preds"][score_key]} for row in rows], "_pred"
        )
        for score_key in ["nll", "avg_nll", "pmi_avg"]
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.exists():
        existing = json.loads(out.read_text())
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2) + "\n")

    print(
        f"{args.label} step={result['step']} primary={args.score} "
        f"acc={correct[args.score]}/{len(probes)} "
        f"(nll={correct['nll']}, avg={correct['avg_nll']}, pmi_avg={correct['pmi_avg']})"
    )
    cats = result["category_summary"][args.score]
    for cat in sorted(cats):
        rec = cats[cat]
        print(f"  {cat:10s} {rec['correct']:2d}/{rec['n']:2d} {rec['accuracy']:.3f}")
    for row in rows:
        best = row["scores"][row["preds"][args.score]]
        gold = row["scores"][row["answer"]]
        status = "OK" if row["preds"][args.score] == row["answer"] else "MISS"
        print(
            f"{status:4s} {row['category']:10s} {row['name']:22s} pred={best['choice']!r} "
            f"gold={gold['choice']!r} pred_{args.score}={best[args.score]:.2f} "
            f"gold_{args.score}={gold[args.score]:.2f}"
        )


if __name__ == "__main__":
    main()
