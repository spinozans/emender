# X37 head differentiation

Read RECEIPTS.md for the pretraining predictions and REPORT.md for outcomes.
The experiment imports the X36 optimized lane, X31 cell contract and X34
skeleton/data/probes without editing their source files. All new artifacts
live here; checkpoints under runs/ are ignored.

From the repository root, use the exact project interpreter and broker:

```bash
experiments/expressivity_tasks/x37_head_differentiation/run_leased.sh parity -q \
  --junitxml=experiments/expressivity_tasks/x37_head_differentiation/receipts/oracle.xml
experiments/expressivity_tasks/x37_head_differentiation/run_leased.sh test -q \
  --junitxml=experiments/expressivity_tasks/x37_head_differentiation/receipts/mechanism.xml
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x37_head_differentiation.cli seal-gate
experiments/expressivity_tasks/x37_head_differentiation/run_leased.sh all
```

`all` audits the existing control, trains b/c/d sequentially (resumes only an
identical fingerprint), selects the validation-NLL winner, and probes it.
`train --arm c` runs one arm. Import/source drift is rejected at an arm
boundary. The initial passing source manifest is gate_sources.json; the
lossless gated_sources.tar.gz archive captures it for reproduction in the
presence of concurrent X36 development. It does not modify upstream files.

The completed summary and report can be rebuilt without GPU use:

```bash
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x37_head_differentiation.cli summarize
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x37_head_differentiation/receipts/analyze.py
```

A fresh reproduction must prepare the X34 data/tokenizer in advance and pass
fresh parity gates against its chosen imported sources. Do not overwrite
historical receipts when re-registering a different implementation. CPU source
and data hashing are allowed outside the broker; GPU execution is not.

The delivered run hit an upstream source-change gate after b completed.
Its exact continuation uses the archived parity-gated sources, without
patching imported files or retraining b:

```bash
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x37_head_differentiation/receipts/resume_pinned.py --check
experiments/expressivity_tasks/x37_head_differentiation/receipts/pinned/run_leased.sh all
```

The loader verifies every archived file against gate_sources.json, refuses
changes to X37's own gated files, and resolves runtime imports to those exact
bytes. Source identity checks use that verified manifest; numerical and data
gates remain intact. Source drift and the failed boundary launch are retained
in pinned_execution.json and training_console.txt. Use the archive also for
CPU-only summary rebuilding if current upstream imports have changed:

```bash
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x37_head_differentiation/receipts/resume_pinned.py summarize
```
