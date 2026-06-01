#!/usr/bin/env python3
"""Read the three fine-tune result JSONs and emit accuracy-vs-T markdown tables
plus a mechanical verdict on the S3/S5 separation and E88-vs-M2RNN.

No fabrication: every number printed is read straight from the run JSONs.
"""
import json
import sys
from pathlib import Path

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/review/s3_s5_finetune_data")
MODELS = ["e88", "gdn", "m2rnn"]
LABEL = {"e88": "E88", "gdn": "GDN (fla-gdn)", "m2rnn": "M2RNN"}
TASKS = ["parity", "s3_permutation", "s5_permutation"]
TASKLAB = {"parity": "Parity (S2, solvable)",
           "s3_permutation": "S3 (solvable)",
           "s5_permutation": "S5 (NON-solvable / NC1-complete)"}


def load():
    out = {}
    for m in MODELS:
        p = DATA / f"{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def table_for_task(res, task):
    # union of eval lengths
    Ts = None
    for m in MODELS:
        if m in res and task in res[m]["tasks"]:
            Ts = [int(k) for k in res[m]["tasks"][task]["acc_vs_T"].keys()]
            break
    if Ts is None:
        return f"_(no data for {task})_\n"
    Ts = sorted(Ts)
    base = None
    train_max = None
    lines = []
    hdr = "| model | " + " | ".join(f"T={t}" for t in Ts) + " |"
    sep = "|" + "---|" * (len(Ts) + 1)
    lines += [hdr, sep]
    for m in MODELS:
        if m not in res or task not in res[m]["tasks"]:
            continue
        rec = res[m]["tasks"][task]
        base = rec["random_baseline_acc"]
        train_max = rec["train_max_T"]
        row = [LABEL[m]]
        for t in Ts:
            a = rec["acc_vs_T"].get(str(t))
            row.append(f"{a:.3f}" if a is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    note = f"\n_baseline (chance) = {base:.4f}; trained on T≤{train_max} (cols beyond are extrapolation)_\n"
    return "\n".join(lines) + "\n" + note


def short_acc(rec):
    """Competence at the max trained length."""
    return rec["acc_vs_T"].get(str(rec["train_max_T"]))


def acc_at(rec, T):
    return rec["acc_vs_T"].get(str(T))


def main():
    res = load()
    print("# data loaded for:", list(res.keys()))
    for task in TASKS:
        print(f"\n## {TASKLAB[task]}\n")
        print(table_for_task(res, task))

    # mechanical verdict on S5 length-gen
    print("\n## Mechanical signals (S5)\n")
    if all(m in res and "s5_permutation" in res[m]["tasks"] for m in MODELS):
        Ts = sorted(int(k) for k in res["e88"]["tasks"]["s5_permutation"]["acc_vs_T"])
        train_max = res["e88"]["tasks"]["s5_permutation"]["train_max_T"]
        far = max(Ts)
        for m in MODELS:
            rec = res[m]["tasks"]["s5_permutation"]
            print(f"- {LABEL[m]}: S5 acc@T{train_max}(trained)={short_acc(rec):.3f}  "
                  f"acc@T{far}(8x)={acc_at(rec, far):.3f}  "
                  f"baseline={rec['random_baseline_acc']:.4f}")


if __name__ == "__main__":
    main()
