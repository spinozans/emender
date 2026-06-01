#!/usr/bin/env python3
"""Merge the three per-task M2RNN to-competence runs (all at the SAME recipe:
lr=5e-5 const, grad_clip=0.5, warmup=300, 12000 steps) into a single results
JSON shaped like the other model files, for analyze_s3_s5.py / plot_s3_s5.py."""
import json
from pathlib import Path

SRC = {
    "parity":         "/tmp/m2rnn_parity_lr5e5const.json",
    "s3_permutation": "/tmp/m2rnn_s3_push_lr5e5const.json",
    "s5_permutation": "/tmp/m2rnn_s5_besteffort_lr5e5const.json",
}
OUT = "paper/review/s3_s5_finetune_v03_data_tocomp/m2rnn.json"

merged = None
tasks = {}
for tname, path in SRC.items():
    d = json.loads(Path(path).read_text())
    if merged is None:
        merged = {k: v for k, v in d.items() if k != "tasks"}
    assert tname in d["tasks"], f"{tname} missing in {path}"
    tasks[tname] = d["tasks"][tname]
merged["tasks"] = tasks
merged["note"] = ("M2RNN to-competence: per-task runs at IDENTICAL recipe "
                  "lr=5e-5 const, grad_clip=0.5, warmup=300, 12000 steps "
                  "(the recipe that brings M2RNN to parity competence; matched "
                  "lr=2e-4 left it under-fit).")
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUT).write_text(json.dumps(merged, indent=2))
print(f"wrote {OUT}")
for t, rec in tasks.items():
    a = rec["acc_vs_T"]
    print(f"  {t:16s} T16={a['16']:.3f} T32={a['32']:.3f} T64={a['64']:.3f} "
          f"T128={a['128']:.3f} T256={a['256']:.3f} T512={a['512']:.3f}")
